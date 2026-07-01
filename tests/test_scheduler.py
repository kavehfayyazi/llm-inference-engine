"""All three schedulers serve concurrent requests with correct per-request output."""

from __future__ import annotations

import math

import torch

from engine.blocks import BlockPool, PagedKVCache
from engine.config import EngineConfig
from engine.generate import generate_paged
from engine.model import load
from engine.request import Request
from engine.scheduler import ContinuousScheduler, DynamicScheduler, StaticScheduler

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In 1969, the first humans landed on the Moon and",
    "def add(a, b):",
]
ARRIVALS = [0, 0, 3, 3]
MAX_NEW = 24
MAX_BATCH = 2


def _fresh_pool_and_requests(lm):
    total = sum(len(lm.tokenizer(p).input_ids) + MAX_NEW for p in PROMPTS)
    num_blocks = math.ceil(total / lm.cfg.block_size) + len(PROMPTS)
    pool = BlockPool(num_blocks, lm.dims.n_layers, lm.dims.n_kv_heads, lm.dims.head_dim,
                     lm.cfg.block_size, lm.device, lm.dtype)
    reqs = []
    for i, p in enumerate(PROMPTS):
        ids = lm.tokenizer(p, return_tensors="pt").input_ids.to(lm.device)
        reqs.append(Request(id=i, prompt_ids=ids, kv=PagedKVCache(pool), max_new=MAX_NEW, arrival=ARRIVALS[i]))
    return pool, reqs


@torch.no_grad()
def _check(lm, refs, scheduler_cls):
    pool, reqs = _fresh_pool_and_requests(lm)
    done, _ = scheduler_cls(lm, pool, MAX_BATCH).run(reqs)
    by_id = {r.id: r for r in done}
    assert len(by_id) == len(PROMPTS), "some requests never finished"
    for i, ref in enumerate(refs):
        assert by_id[i].full_ids() == ref, f"{scheduler_cls.__name__}: request {i} diverged"


@torch.no_grad()
def test_schedulers_match_single():
    lm = load(EngineConfig(dtype="float32"))
    refs = [generate_paged(lm, p, max_new_tokens=MAX_NEW)[0][0].tolist() for p in PROMPTS]
    _check(lm, refs, StaticScheduler)
    _check(lm, refs, DynamicScheduler)
    _check(lm, refs, ContinuousScheduler)


if __name__ == "__main__":
    test_schedulers_match_single()
    print("SCHEDULER OK: static + dynamic + continuous match single-request output")
