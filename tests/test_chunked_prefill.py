"""Chunked prefill: multi-chunk prompts still produce single-request output."""

from __future__ import annotations

import math

import torch

from engine.blocks import BlockPool, PagedKVCache
from engine.config import EngineConfig
from engine.generate import generate_paged
from engine.model import load
from engine.request import Request
from engine.scheduler import ContinuousScheduler

# Prompts longer than the chunk size so prefill spans several steps.
PROMPTS = [
    "In 1969, the first humans landed on the Moon and it was a giant leap for all of mankind because",
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant rolling hills",
    "def add(a, b):",
]
MAX_NEW = 20
CHUNK = 8


@torch.no_grad()
def test_chunked_matches_single():
    lm = load(EngineConfig(dtype="float32", prefill_chunk=CHUNK))
    refs = [generate_paged(lm, p, max_new_tokens=MAX_NEW)[0][0].tolist() for p in PROMPTS]

    total = sum(len(lm.tokenizer(p).input_ids) + MAX_NEW for p in PROMPTS)
    num_blocks = math.ceil(total / lm.cfg.block_size) + len(PROMPTS)
    pool = BlockPool(num_blocks, lm.dims.n_layers, lm.dims.n_kv_heads, lm.dims.head_dim,
                     lm.cfg.block_size, lm.device, lm.dtype)
    reqs = [Request(id=i, prompt_ids=lm.tokenizer(p, return_tensors="pt").input_ids.to(lm.device),
                    kv=PagedKVCache(pool), max_new=MAX_NEW, arrival=i)
            for i, p in enumerate(PROMPTS)]

    done, _ = ContinuousScheduler(lm, pool, max_batch=2).run(reqs)
    by_id = {r.id: r for r in done}
    assert len(by_id) == len(PROMPTS)
    for i, ref in enumerate(refs):
        assert by_id[i].full_ids() == ref, f"request {i} diverged under chunked prefill"


if __name__ == "__main__":
    test_chunked_matches_single()
    print(f"CHUNKED PREFILL OK: multi-chunk prompts match single-request (chunk={CHUNK})")
