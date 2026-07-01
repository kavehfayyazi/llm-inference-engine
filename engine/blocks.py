"""Paged KV: a shared block pool and per-request block tables."""

from __future__ import annotations

import math

import torch


class BlockPool:
    """Preallocated physical KV blocks shared by all requests.

    Storage per layer: K and V tensors of shape
    [num_blocks, n_kv_heads, block_size, head_dim]. A physical block id indexes
    the same slot in every layer (the block table is shared across layers since
    a token sits at the same position in all of them). A free list hands out and
    reclaims block ids.
    """

    def __init__(self, num_blocks, n_layers, n_kv_heads, head_dim, block_size, device, dtype):
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks, n_kv_heads, block_size, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.free = list(range(num_blocks))
        self.peak_used = 0  # high-water mark of concurrently held blocks

    def allocate(self) -> int:
        # Hand out one physical block id.
        if not self.free:
            raise RuntimeError("block pool exhausted")
        block_id = self.free.pop()
        self.peak_used = max(self.peak_used, self.num_used)
        return block_id

    def free_block(self, block_id: int):
        # Return one block id to the pool.
        self.free.append(block_id)

    @property
    def num_free(self) -> int:
        return len(self.free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free)


class PagedKVCache:
    """One request's view: logical block -> physical block id."""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_table: list = []  # logical index -> physical block id
        self.seq_len = 0

    def reserve(self, total_len: int):
        # Ensure enough blocks for total_len tokens, then record the length.
        need = math.ceil(total_len / self.pool.block_size)
        while len(self.block_table) < need:
            self.block_table.append(self.pool.allocate())
        self.seq_len = total_len

    def locate(self, pos: int):
        # Token position -> (physical block id, offset within block).
        return self.block_table[pos // self.pool.block_size], pos % self.pool.block_size

    def release(self):
        # Hand all blocks back to the pool.
        for block_id in self.block_table:
            self.pool.free_block(block_id)
        self.block_table = []
        self.seq_len = 0

    def reserved_tokens(self) -> int:
        # Token slots held (blocks * block_size).
        return len(self.block_table) * self.pool.block_size

    def live_tokens(self) -> int:
        # Token slots actually used.
        return self.seq_len

    def fragmentation(self) -> int:
        # Reserved but unused slots (internal fragmentation).
        return self.reserved_tokens() - self.live_tokens()
