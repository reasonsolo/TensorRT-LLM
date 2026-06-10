# Backward compatibility shim - the block hash helpers live in
# tensorrt_llm.block_hash. All imports from tensorrt_llm.llmapi.block_hash will
# continue to work.
from tensorrt_llm.block_hash import (
    KV_CACHE_HASH_ALGO_DEFAULT,
    KV_CACHE_HASH_ALGO_V1,
    KV_CACHE_HASH_ALGO_V2,
    KV_CACHE_HASH_ALGO_V2_SHA256_64,
    BlockHash,
    block_key_hasher,
    compute_token_ids_block_hashes,
    get_cache_salt_id,
    hash_v1_block_key,
    truncate_sha256_hash_to_int64,
    v2_sha256_block_hasher,
)

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
