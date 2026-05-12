/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/evictionPolicy.h"

using namespace tensorrt_llm::batch_manager::kv_cache_manager;

// This implements priority-based eviction.
// Blocks are assigned priority levels, with blocks at a lower priority evicted before blocks at a higher priority.
// New priority values always override the previous value.

namespace tensorrt_llm::batch_manager::eviction_policy
{

auto const kMinPriority = executor::KvCacheRetentionConfig::kMinRetentionPriority;
auto const kMaxPriority = executor::KvCacheRetentionConfig::kMaxRetentionPriority;
auto const kNumPriorities = kMaxPriority - kMinPriority + 1;

auto const kDefaultPriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority;
executor::RetentionPriority const kDefaultSecondaryOffloadMinPriority = 30;

int const kNumCacheLevels = 2;
int const kPlaceholderLevel = kNumCacheLevels; // placeholder blocks live at level 2

namespace
{
SizeType32 getCacheLevel(BlockPtr const& block)
{
    if (block->isPlaceholder())
    {
        return kPlaceholderLevel;
    }
    return block->isPrimary() ? 0 : 1;
}

constexpr SizeType32 getPriorityIdx(executor::RetentionPriority priority)
{
    return priority - kMinPriority;
}

constexpr auto defaultPriorityIdx = getPriorityIdx(kDefaultPriority);
} // namespace

void LRUEvictionPolicy::initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> sizes,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority)
{
    SizeType32 startIdx = 0;

    // Create queues for all levels: primary, secondary, and placeholder (initially empty).
    mFreeQueues.resize(kPlaceholderLevel + 1, std::vector<FreeBlocksQueue>(kNumPriorities));
    mFreeBlockIterators.positive.resize(mAllBlocksById.size());
    mFreeBlockIteratorPositions.positive.resize(mAllBlocksById.size());
    mNumFreeBlocksPerLevel.resize(kPlaceholderLevel + 1, 0);

    for (SizeType32 cacheLevel = 0; cacheLevel < kNumCacheLevels; cacheLevel++)
    {
        auto& freeQueue = mFreeQueues[cacheLevel][defaultPriorityIdx];

        for (SizeType32 blockId = 0; blockId < sizes[cacheLevel]; blockId++)
        {
            // Initialize all blocks to be the default priority level
            mFreeBlockIterators[startIdx + blockId]
                = freeQueue.insert(freeQueue.end(), mAllBlocksById[startIdx + blockId]);
            mFreeBlockIteratorPositions[startIdx + blockId] = FreeBlockQueuePosition{cacheLevel, defaultPriorityIdx};
        }

        mNumFreeBlocksPerLevel[cacheLevel] = sizes[cacheLevel];
        startIdx += sizes[cacheLevel];
    }

    mSecondaryOffloadMinPriority = secondaryOffloadMinPriority.value_or(kDefaultSecondaryOffloadMinPriority);
}

void LRUEvictionPolicy::initializePlaceholders(std::vector<BlockPtr>& allPlaceholderBlocksById)
{
    auto const len = static_cast<SizeType32>(allPlaceholderBlocksById.size());

    // Placeholder IDs -2, -3, ... map to indices 2, 3, ... via abs(id).
    // Indices 0 and 1 are unused (0 is invalid, 1 corresponds to kCachedBlocksRootId).
    mFreeBlockIterators.negative.resize(len);
    mFreeBlockIteratorPositions.negative.resize(len);

    auto& freeQueue = mFreeQueues[kPlaceholderLevel][defaultPriorityIdx];

    for (auto const& block : allPlaceholderBlocksById)
    {
        if (block)
        {
            mFreeBlockIterators[block->getBlockId()] = freeQueue.insert(freeQueue.end(), block);
            mFreeBlockIteratorPositions[block->getBlockId()]
                = FreeBlockQueuePosition{kPlaceholderLevel, defaultPriorityIdx};
            mNumFreeBlocksPerLevel[kPlaceholderLevel]++;
        }
    }
}

bool LRUEvictionPolicy::verifyQueueIntegrity() const
{
    static char const* const levelToStr[] = {"primary", "secondary", "placeholder"};
    static std::function<bool(BlockPtr const&)> const levelValidators[]
        = {[](BlockPtr const& block) { return block->isPrimary(); },
            [](BlockPtr const& block) { return !block->isPrimary(); },
            [](BlockPtr const& block) { return block->isPlaceholder(); }};
    bool queueCompromised = false;
    for (SizeType32 queueLevel = 0; queueLevel < kNumCacheLevels + 1; queueLevel++)
    {
        for (SizeType32 pri = 0; pri < kNumPriorities; pri++)
        {
            for (auto const& block : mFreeQueues[queueLevel][pri])
            {
                bool const valid = levelValidators[queueLevel](block);
                if (!valid)
                {
                    TLLM_LOG_WARNING("Block (id %d) has level=%s, but misplaced at queueLevel %s", block->getBlockId(),
                        levelToStr[queueLevel], levelToStr[queueLevel]);
                    queueCompromised = true;
                }
                if (block->hasRefs())
                {
                    TLLM_LOG_WARNING("Found block (id %d) with references at queueLevel %s", block->getBlockId(),
                        levelToStr[queueLevel]);
                    queueCompromised = true;
                }
            }
        }
    }
    TLLM_LOG_DEBUG("LRUEvictionPolicy queues are %s", queueCompromised ? "compromised" : "not compromised");
    return !queueCompromised;
}

std::tuple<BlockPtr, bool> LRUEvictionPolicy::getFreeBlock(SizeType32 cacheLevel, bool wantPlaceholder)
{
    SizeType32 const level = wantPlaceholder ? kPlaceholderLevel : cacheLevel;

    for (SizeType32 pri = 0; pri < kNumPriorities; pri++)
    {
        // Find the first non-empty queue, and return the first block.
        if (!mFreeQueues[level][pri].empty())
        {
            auto block = mFreeQueues[level][pri].front();
            TLLM_CHECK_WITH_INFO(block != nullptr,
                "LRUEvictionPolicy free queue contains null block at level %d priorityIdx %d", level, pri);
            auto const blockId = block->getBlockId();
            TLLM_CHECK_WITH_INFO(mFreeBlockIterators[blockId] != std::nullopt,
                "LRUEvictionPolicy free queue contains block %d at level %d priorityIdx %d without iterator", blockId,
                level, pri);
            auto const& position = mFreeBlockIteratorPositions[blockId];
            TLLM_CHECK_WITH_INFO(position.has_value(),
                "LRUEvictionPolicy free queue contains block %d at level %d priorityIdx %d without iterator position",
                blockId, level, pri);
            TLLM_CHECK_WITH_INFO(position->cacheLevel == level && position->priorityIdx == pri,
                "LRUEvictionPolicy free queue front mismatch for block %d: queued at level %d priorityIdx %d but "
                "iterator position says level %d priorityIdx %d",
                blockId, level, pri, position->cacheLevel, position->priorityIdx);
            auto const& iteratorBlock = **mFreeBlockIterators[blockId];
            TLLM_CHECK_WITH_INFO(iteratorBlock != nullptr,
                "LRUEvictionPolicy free queue iterator for block %d points to null block", blockId);
            TLLM_CHECK_WITH_INFO(iteratorBlock.get() == block.get(),
                "LRUEvictionPolicy free queue front mismatch for block %d: iterator points to block %d", blockId,
                iteratorBlock->getBlockId());
            TLLM_CHECK_WITH_INFO(!block->hasRefs(),
                "LRUEvictionPolicy free queue contains referenced block %d at level %d priorityIdx %d", blockId, level,
                pri);

            // mFreeQueues only contains leaf blocks, so no need to iterate through the next block pointers.
            // It's possible to have a primary block with children in secondary memory. We handle this
            // by freeing all descendants in WindowBlockManager::getFreeBlock. This is done either by
            // offloading (preferred method) or explicitly.
            bool const canOffload
                = !wantPlaceholder && cacheLevel == 0 && pri >= getPriorityIdx(mSecondaryOffloadMinPriority);
            return std::make_tuple(block, canOffload);
        }
    }
    TLLM_THROW("No free block found. This shouldn't happen!");
}

void LRUEvictionPolicy::releaseBlock(BlockPtr block)
{
    releaseBlock(block, false);
}

void LRUEvictionPolicy::releaseBlock(BlockPtr block, bool toFront)
{
    // The dummy root block (kCachedBlocksRootId) is permanently attached to the lookup tree
    // via setAsRoot() and must never enter the eviction queue — it is not a real cache block.
    TLLM_CHECK_WITH_INFO(
        block->getBlockId() != tensorrt_llm::batch_manager::kv_cache_manager::KVCacheBlock::kCachedBlocksRootId,
        "Attempted to release the cached-blocks root into the eviction queue");
    // SWA on-demand placeholders are transient sentinels created by createPlaceholder() and
    // are not part of the pooled placeholder free queues. Skip re-inserting only those
    // sentinels; pooled linear-attention placeholders must fall through and return to the
    // placeholder queue at kPlaceholderLevel.
    if (block->isPlaceholder() && block->getBlockId() == KVCacheBlock::kPlaceholderBlockId)
    {
        return;
    }
    SizeType32 const cacheLevel = getCacheLevel(block);
    SizeType32 const id = block->getBlockId();
    SizeType32 const priorityIdx = getPriorityIdx(block->getPriority());

    TLLM_CHECK_WITH_INFO(mFreeBlockIterators[id] == std::nullopt,
        "LRUEvictionPolicy duplicate release for block %d: existing position level %d priorityIdx %d, new position "
        "level %d priorityIdx %d, hasRefs=%d, isPrimary=%d, isPlaceholder=%d",
        id, mFreeBlockIteratorPositions[id].has_value() ? mFreeBlockIteratorPositions[id]->cacheLevel : -1,
        mFreeBlockIteratorPositions[id].has_value() ? mFreeBlockIteratorPositions[id]->priorityIdx : -1, cacheLevel,
        priorityIdx, block->hasRefs(), block->isPlaceholder() ? 0 : block->isPrimary(), block->isPlaceholder());

    // If there are no children, this is a leaf block. Insert into a queue.
    auto& q = mFreeQueues[cacheLevel][priorityIdx];
    if (toFront)
    {
        mFreeBlockIterators[id] = q.insert(q.begin(), block);
    }
    else
    {
        mFreeBlockIterators[id] = q.insert(q.end(), block);
    }
    mFreeBlockIteratorPositions[id] = FreeBlockQueuePosition{cacheLevel, priorityIdx};

    mNumFreeBlocksPerLevel[cacheLevel]++;

    if (block->getDurationMs().has_value()
        && block->getPriority() != executor::KvCacheRetentionConfig::kDefaultRetentionPriority)
    {
        auto expirationTime = getTime() + *block->getDurationMs();
        block->setExpirationTime(expirationTime);
        mExpiringBlockHeap.emplace(block);
    }
}

SizeType32 LRUEvictionPolicy::getNumFreeBlocks(SizeType32 cacheLevel)
{
    return mNumFreeBlocksPerLevel[cacheLevel];
}

void LRUEvictionPolicy::claimBlock(BlockPtr block)
{
    claimBlock(block, std::nullopt, std::nullopt);
}

void LRUEvictionPolicy::claimBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
    std::optional<std::chrono::milliseconds> durationMs)
{
    SizeType32 const id = block->getBlockId();
    SizeType32 const cacheLevel = getCacheLevel(block);
    SizeType32 const priorityIdx = getPriorityIdx(block->getPriority());

    if (mFreeBlockIterators[id] != std::nullopt)
    {
        TLLM_CHECK_WITH_INFO(mFreeBlockIteratorPositions[id].has_value(),
            "LRUEvictionPolicy iterator for block %d has no recorded queue position", id);
        auto const position = *mFreeBlockIteratorPositions[id];
        if (position.cacheLevel != cacheLevel || position.priorityIdx != priorityIdx)
        {
            TLLM_LOG_WARNING(
                "LRUEvictionPolicy queue position mismatch while claiming block %d: recorded level %d priorityIdx %d, "
                "current level %d priorityIdx %d. Erasing from recorded queue.",
                id, position.cacheLevel, position.priorityIdx, cacheLevel, priorityIdx);
        }
        TLLM_CHECK_WITH_INFO(mNumFreeBlocksPerLevel[position.cacheLevel] > 0,
            "LRUEvictionPolicy free-block counter underflow while claiming block %d: recorded level %d priorityIdx %d",
            id, position.cacheLevel, position.priorityIdx);
        auto const& iteratorBlock = **mFreeBlockIterators[id];
        TLLM_CHECK_WITH_INFO(iteratorBlock != nullptr,
            "LRUEvictionPolicy iterator for block %d points to null block at recorded level %d priorityIdx %d", id,
            position.cacheLevel, position.priorityIdx);
        TLLM_CHECK_WITH_INFO(iteratorBlock.get() == block.get(),
            "LRUEvictionPolicy iterator mismatch while claiming block %d: iterator points to block %d at recorded "
            "level %d priorityIdx %d",
            id, iteratorBlock->getBlockId(), position.cacheLevel, position.priorityIdx);
        TLLM_LOG_DEBUG(
            "LRUEvictionPolicy claiming block %d from level %d priorityIdx %d: queueSize=%zu freeCount=%d "
            "currentLevel=%d "
            "currentPriorityIdx=%d hasRefs=%d isPrimary=%d isPlaceholder=%d",
            id, position.cacheLevel, position.priorityIdx,
            mFreeQueues[position.cacheLevel][position.priorityIdx].size(), mNumFreeBlocksPerLevel[position.cacheLevel],
            cacheLevel, priorityIdx, block->hasRefs(), block->isPlaceholder() ? 0 : block->isPrimary(),
            block->isPlaceholder());
        mFreeQueues[position.cacheLevel][position.priorityIdx].erase(*mFreeBlockIterators[id]);
        mNumFreeBlocksPerLevel[position.cacheLevel] -= 1;
    }

    mFreeBlockIterators[id] = std::nullopt;
    mFreeBlockIteratorPositions[id] = std::nullopt;

    if (priority.has_value())
    {
        block->setPriority(*priority);
    }

    mExpiringBlockHeap.erase(block);
    block->setDurationMs(durationMs);
}

std::chrono::steady_clock::time_point::duration LRUEvictionPolicy::getTime() const
{
    return std::chrono::steady_clock::now().time_since_epoch();
}

void LRUEvictionPolicy::refresh()
{
    while (!mExpiringBlockHeap.empty())
    {
        auto const block = *mExpiringBlockHeap.begin();
        if (block->getExpirationTime() > getTime())
        {
            break;
        }

        auto const id = block->getBlockId();
        auto const level = getCacheLevel(block);

        mExpiringBlockHeap.erase(mExpiringBlockHeap.begin());

        if (mFreeBlockIterators[id] != std::nullopt)
        {
            // This is already in another queue. Delete it, and bring it down to the default queue
            TLLM_CHECK_WITH_INFO(mFreeBlockIteratorPositions[id].has_value(),
                "LRUEvictionPolicy expiring block %d has iterator but no recorded queue position", id);
            auto const position = *mFreeBlockIteratorPositions[id];
            if (position.cacheLevel != level || position.priorityIdx != getPriorityIdx(block->getPriority()))
            {
                TLLM_LOG_WARNING(
                    "LRUEvictionPolicy queue position mismatch while expiring block %d: recorded level %d priorityIdx "
                    "%d, current level %d priorityIdx %d. Erasing from recorded queue.",
                    id, position.cacheLevel, position.priorityIdx, level, getPriorityIdx(block->getPriority()));
            }
            mFreeQueues[position.cacheLevel][position.priorityIdx].erase(*mFreeBlockIterators[id]);
            auto& q = mFreeQueues[level][defaultPriorityIdx];
            mFreeBlockIterators[id] = q.insert(q.end(), block);
            mFreeBlockIteratorPositions[id] = FreeBlockQueuePosition{level, defaultPriorityIdx};
        }
        block->setPriority(kDefaultPriority);
    }
}

} // namespace tensorrt_llm::batch_manager::eviction_policy
