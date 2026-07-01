"""Batched execution: each request in a batch matches its single-request output."""

from __future__ import annotations

import math

import torch

from engine.batch import decode_step, prefill
from engine.blocks import BlockPool, PagedKVCache
from engine.config import EngineConfig
from engine.generate import _eos_ids, generate_paged
from engine.model import load
from engine.request import Request

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In 1969, the first humans landed on the Moon and",
    "def add(a, b):",
]
MAX_NEW = 24


def _finished(req, eos_ids, max_new):
    return len(req.generated) >= max_new or req.last_token in eos_ids


@torch.no_grad()
def test_batch_matches_single():
    lm = load(EngineConfig(dtype="float32"))
    eos_ids = _eos_ids(lm)

    # Single-request references (already gated against HF elsewhere).
    refs = [generate_paged(lm, p, max_new_tokens=MAX_NEW)[0][0].tolist() for p in PROMPTS]

    # One shared pool for all requests -- the paging payoff.
    total = sum(len(lm.tokenizer(p).input_ids) + MAX_NEW for p in PROMPTS)
    num_blocks = math.ceil(total / lm.cfg.block_size) + len(PROMPTS)
    pool = BlockPool(num_blocks, lm.dims.n_layers, lm.dims.n_kv_heads, lm.dims.head_dim,
                     lm.cfg.block_size, lm.device, lm.dtype)

    reqs = []
    for i, p in enumerate(PROMPTS):
        ids = lm.tokenizer(p, return_tensors="pt").input_ids.to(lm.device)
        reqs.append(Request(id=i, prompt_ids=ids, kv=PagedKVCache(pool), max_new=MAX_NEW))

    for r in reqs:
        prefill(lm, pool, r)
    running = [r for r in reqs if not _finished(r, eos_ids, MAX_NEW)]

    while running:
        decode_step(lm, pool, running)
        running = [r for r in running if not _finished(r, eos_ids, MAX_NEW)]

    for r, ref in zip(reqs, refs):
        assert r.full_ids() == ref, f"request {r.id} diverged from single-request run"


if __name__ == "__main__":
    test_batch_matches_single()
    print(f"BATCH OK: {len(PROMPTS)} concurrent requests match single-request output")
