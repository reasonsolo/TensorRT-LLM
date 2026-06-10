# Copyright (c) 2025-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

from tensorrt_llm.bindings.internal.batch_manager import BlockKey as _NativeBlockKey
from tensorrt_llm.bindings.internal.batch_manager import BlockKeyHasher as _NativeBlockKeyHasher
from tensorrt_llm.runtime import kv_cache_hash
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import Block as V2Block
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import ReuseScope
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import RootBlock as V2RootBlock

KV_CACHE_HASH_ALGO_DEFAULT = kv_cache_hash.KV_CACHE_HASH_ALGO_DEFAULT
KV_CACHE_HASH_ALGO_V1 = kv_cache_hash.KV_CACHE_HASH_ALGO_V1
KV_CACHE_HASH_ALGO_V2 = kv_cache_hash.KV_CACHE_HASH_ALGO_V2
KV_CACHE_HASH_ALGO_V2_SHA256_64 = kv_cache_hash.KV_CACHE_HASH_ALGO_V2_SHA256_64
get_cache_salt_id = kv_cache_hash.get_cache_salt_id
hash_v1_block_key = kv_cache_hash.hash_v1_block_key
truncate_sha256_hash_to_int64 = kv_cache_hash.truncate_sha256_hash_to_int64

BlockHash = Union[int, str]


def block_key_hasher(
    token_ids: list[int], parent_hash: Optional[int] = None, cache_salt_id: Optional[int] = None
) -> int:
    parent = 0 if parent_hash is None else parent_hash
    # Fast path: the native C++ BlockKeyHasher is bit-exact with
    # hash_v1_block_key and avoids the per-token Python loop. Its hash() binding
    # takes no cache_salt_id, so fall back to Python only when a salt is set
    # (rare opt-in; never in the unsalted agent/chat completion path).
    if cache_salt_id is None:
        return _NativeBlockKeyHasher.hash(_NativeBlockKey(token_ids), parent)
    return hash_v1_block_key(token_ids, parent_hash=parent, cache_salt_id=cache_salt_id)


def v2_sha256_block_hasher(
    token_ids: list[int], parent_hash: Optional[str] = None, cache_salt_id: Optional[int] = None
) -> str:
    parent_key = (
        V2RootBlock.make_key(ReuseScope(salt=cache_salt_id))
        if parent_hash is None
        else bytes.fromhex(parent_hash)
    )
    return V2Block.make_key(parent_key, token_ids).hex()


def compute_token_ids_block_hashes(
    token_lists: list[list[int]],
    tokens_per_block: int,
    hash_algo: str = KV_CACHE_HASH_ALGO_DEFAULT,
    cache_salt_id: Optional[int] = None,
) -> list[list[BlockHash]]:
    if hash_algo == KV_CACHE_HASH_ALGO_V1:
        block_hasher = block_key_hasher
    elif hash_algo == KV_CACHE_HASH_ALGO_V2:
        block_hasher = v2_sha256_block_hasher
    elif hash_algo == KV_CACHE_HASH_ALGO_V2_SHA256_64:
        reuse_scope = ReuseScope(salt=cache_salt_id)
        block_hashes: list[list[BlockHash]] = []
        for token_list in token_lists:
            hash_list = []
            parent_key = V2RootBlock.make_key(reuse_scope)
            # in KvCacheManager, the last token is not included in the block key
            for t in range(0, len(token_list) - 1, tokens_per_block):
                t_end = min(t + tokens_per_block, len(token_list) - 1)
                parent_key = V2Block.make_key(parent_key, token_list[t:t_end])
                hash_list.append(truncate_sha256_hash_to_int64(parent_key))
            block_hashes.append(hash_list)
        return block_hashes
    else:
        raise ValueError(f"Unsupported KV cache hash algorithm: {hash_algo}")

    block_hashes: list[list[BlockHash]] = []
    for token_list in token_lists:
        hash_list = []
        # in KvCacheManager, the last token is not included in the block key
        for t in range(0, len(token_list) - 1, tokens_per_block):
            t_end = min(t + tokens_per_block, len(token_list) - 1)
            hash_list.append(
                block_hasher(token_list[t:t_end], None if t == 0 else hash_list[-1], cache_salt_id)
            )
        block_hashes.append(hash_list)
    return block_hashes


__all__ = [
    "BlockHash",
    "KV_CACHE_HASH_ALGO_DEFAULT",
    "KV_CACHE_HASH_ALGO_V1",
    "KV_CACHE_HASH_ALGO_V2",
    "KV_CACHE_HASH_ALGO_V2_SHA256_64",
    "block_key_hasher",
    "compute_token_ids_block_hashes",
    "get_cache_salt_id",
    "hash_v1_block_key",
    "truncate_sha256_hash_to_int64",
    "v2_sha256_block_hasher",
]
