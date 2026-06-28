"""Block allocator: reservation, location, fragmentation, free/reuse."""

from __future__ import annotations

import torch

from engine.blocks import BlockPool, PagedKVCache


def make_pool(num_blocks=4, block_size=4):
    return BlockPool(
        num_blocks=num_blocks, n_layers=2, n_kv_heads=2, head_dim=8,
        block_size=block_size, device=torch.device("cpu"), dtype=torch.float32,
    )


def test_storage_shape():
    pool = make_pool()
    assert pool.k[0].shape == (4, 2, 4, 8)
    assert len(pool.k) == 2 and len(pool.v) == 2


def test_reserve_and_fragmentation():
    pool = make_pool(num_blocks=4, block_size=4)
    req = PagedKVCache(pool)
    req.reserve(5)                       # 5 tokens -> ceil(5/4) = 2 blocks
    assert len(req.block_table) == 2
    assert req.reserved_tokens() == 8
    assert req.live_tokens() == 5
    assert req.fragmentation() == 3
    assert pool.num_used == 2


def test_locate():
    pool = make_pool(block_size=4)
    req = PagedKVCache(pool)
    req.reserve(6)
    b0, o0 = req.locate(0)
    b5, o5 = req.locate(5)
    assert (b0, o0) == (req.block_table[0], 0)
    assert (b5, o5) == (req.block_table[1], 1)


def test_exhaustion():
    pool = make_pool(num_blocks=2, block_size=4)
    req = PagedKVCache(pool)
    try:
        req.reserve(100)                 # needs 25 blocks, only 2 exist
        assert False, "expected exhaustion"
    except RuntimeError:
        pass


def test_release_reuse():
    pool = make_pool(num_blocks=4, block_size=4)
    req = PagedKVCache(pool)
    req.reserve(8)
    assert pool.num_free == 2
    req.release()
    assert pool.num_free == 4            # all blocks back
    assert req.block_table == [] and req.seq_len == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("BLOCKS OK")
