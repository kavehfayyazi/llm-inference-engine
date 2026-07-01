"""Verify the triton paged kernels match reference + HF (run on CUDA)."""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.blocks import BlockPool, PagedKVCache
from engine.config import EngineConfig
from engine.generate import generate_paged
from engine.model import load
from engine.request import Request
from engine.scheduler import ContinuousScheduler
from tests.test_reference import MAX_NEW, PROMPTS, first_divergence, hf_greedy_ids


def _single_request(lm):
    # Single-request paged path (uses the per-request kernel under triton).
    for prompt in PROMPTS:
        lm.cfg.attention_backend = "reference"
        ref, _ = generate_paged(lm, prompt, max_new_tokens=MAX_NEW)
        lm.cfg.attention_backend = "triton"
        tri, _ = generate_paged(lm, prompt, max_new_tokens=MAX_NEW)
        hf = hf_greedy_ids(lm, prompt)
        ref, tri = ref[0].tolist(), tri[0].tolist()
        assert first_divergence(tri, ref) is None, f"single: triton != reference for {prompt!r}"
        assert first_divergence(tri, hf) is None, f"single: triton != HF for {prompt!r}"


def _continuous(lm, backend):
    total = sum(len(lm.tokenizer(p).input_ids) + MAX_NEW for p in PROMPTS)
    num_blocks = math.ceil(total / lm.cfg.block_size) + len(PROMPTS)
    pool = BlockPool(num_blocks, lm.dims.n_layers, lm.dims.n_kv_heads, lm.dims.head_dim,
                     lm.cfg.block_size, lm.device, lm.dtype)
    reqs = [Request(id=i, prompt_ids=lm.tokenizer(p, return_tensors="pt").input_ids.to(lm.device),
                    kv=PagedKVCache(pool), max_new=MAX_NEW) for i, p in enumerate(PROMPTS)]
    lm.cfg.attention_backend = backend
    done, _ = ContinuousScheduler(lm, pool, len(PROMPTS)).run(reqs)
    return {r.id: r.full_ids() for r in done}


def _batched(lm):
    # Batched scheduler path (uses the fused batched kernel under triton).
    ref = _continuous(lm, "reference")
    tri = _continuous(lm, "triton")
    for i in ref:
        assert ref[i] == tri[i], f"batched: triton != reference for request {i}"


def main():
    assert torch.cuda.is_available(), "triton kernels need CUDA; run this on a GPU box"
    lm = load(EngineConfig(dtype="float32"))
    _single_request(lm)
    _batched(lm)
    print(f"TRITON OK: single + batched kernels match reference/HF on {len(PROMPTS)} prompts")


if __name__ == "__main__":
    main()
