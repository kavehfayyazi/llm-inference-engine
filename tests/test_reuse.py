"""KV reuse: a pool too small to hold all requests still serves them correctly."""

from __future__ import annotations

import math

import torch

from engine.blocks import BlockPool, PagedKVCache
from engine.config import EngineConfig
from engine.generate import generate_paged
from engine.model import load
from engine.request import Request
from engine.scheduler import ContinuousScheduler

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In 1969, the first humans landed on the Moon and",
    "def add(a, b):",
]
ARRIVALS = [0, 0, 2, 4]
MAX_NEW = 24
MAX_BATCH = 2


@torch.no_grad()
def test_small_pool_reuse():
    lm = load(EngineConfig(dtype="float32"))
    refs = [generate_paged(lm, p, max_new_tokens=MAX_NEW)[0][0].tolist() for p in PROMPTS]

    bs = lm.cfg.block_size
    per_req = math.ceil((max(len(lm.tokenizer(p).input_ids) for p in PROMPTS) + MAX_NEW) / bs)
    # Size for peak concurrency only -- NOT all requests. Without release this
    # would exhaust; reuse is what makes it fit.
    num_blocks = MAX_BATCH * per_req + 1
    total_if_no_reuse = len(PROMPTS) * per_req
    assert num_blocks < total_if_no_reuse, "pool must be smaller than the no-reuse footprint"

    pool = BlockPool(num_blocks, lm.dims.n_layers, lm.dims.n_kv_heads, lm.dims.head_dim,
                     bs, lm.device, lm.dtype)
    reqs = [Request(id=i, prompt_ids=lm.tokenizer(p, return_tensors="pt").input_ids.to(lm.device),
                    kv=PagedKVCache(pool), max_new=MAX_NEW, arrival=ARRIVALS[i])
            for i, p in enumerate(PROMPTS)]

    done, _ = ContinuousScheduler(lm, pool, MAX_BATCH).run(reqs)
    by_id = {r.id: r for r in done}
    assert len(by_id) == len(PROMPTS)
    for i, ref in enumerate(refs):
        assert by_id[i].full_ids() == ref, f"request {i} diverged"
    assert pool.num_used == 0, "all blocks should be freed after the stream drains"
    assert pool.peak_used <= num_blocks


if __name__ == "__main__":
    test_small_pool_reuse()
    print("REUSE OK: small pool serves the full stream correctly, all blocks freed")
