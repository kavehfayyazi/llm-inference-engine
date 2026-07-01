"""Benchmark: sweep the three schedulers x batch sizes, print serving metrics."""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.blocks import BlockPool, PagedKVCache
from engine.config import EngineConfig
from engine.metrics import summarize
from engine.model import load
from engine.request import Request
from engine.scheduler import ContinuousScheduler, DynamicScheduler, StaticScheduler

# Heterogeneous workload: high gen-length variance so admission policy matters.
SPECS = [
    ("The capital of France is", 8),
    ("Once upon a time,", 40),
    ("def add(a, b):", 12),
    ("In 1969, the first humans landed on the Moon and", 36),
    ("The three primary colors are", 8),
    ("Write a haiku about autumn:", 40),
    ("The speed of light is approximately", 12),
    ("A brief history of Rome:", 36),
    ("List the planets in order:", 8),
    ("Explain gravity simply:", 40),
    ("Translate hello to French:", 12),
    ("The meaning of life is", 36),
]
BATCH_SIZES = [1, 2, 4, 8]
SCHEDULERS = [("static", StaticScheduler), ("dynamic", DynamicScheduler), ("continuous", ContinuousScheduler)]


def build(lm):
    total = sum(len(lm.tokenizer(p).input_ids) + n for p, n in SPECS)
    num_blocks = math.ceil(total / lm.cfg.block_size) + len(SPECS)
    pool = BlockPool(num_blocks, lm.dims.n_layers, lm.dims.n_kv_heads, lm.dims.head_dim,
                     lm.cfg.block_size, lm.device, lm.dtype)
    reqs = []
    for i, (p, n) in enumerate(SPECS):
        ids = lm.tokenizer(p, return_tensors="pt").input_ids.to(lm.device)
        reqs.append(Request(id=i, prompt_ids=ids, kv=PagedKVCache(pool), max_new=n, arrival=0))
    return pool, reqs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="reference", choices=["reference", "triton"])
    args = ap.parse_args()

    lm = load(EngineConfig(attention_backend=args.backend))
    print(f"device={lm.device} dtype={lm.dtype} backend={args.backend} requests={len(SPECS)}\n")

    # Warmup: compile kernels / autotune / cudnn before timing.
    pool, reqs = build(lm)
    ContinuousScheduler(lm, pool, max(BATCH_SIZES)).run(reqs)
    header = f"{'scheduler':<11} {'batch':>5} {'steps':>6} {'ttft_ms':>9} {'tpot_ms':>9} {'p99_ms':>9} {'tok/s':>7}"
    print(header)
    print("-" * len(header))
    for name, cls in SCHEDULERS:
        for bs in BATCH_SIZES:
            pool, reqs = build(lm)
            done, stats = cls(lm, pool, bs).run(reqs)
            m = summarize(done, stats["wall_s"])
            print(f"{name:<11} {bs:>5} {stats['steps']:>6} {m['ttft_ms_mean']:>9} {m['tpot_ms_mean']:>9} "
                  f"{m['latency_ms_p99']:>9} {m['throughput_tok_s']:>7}")


if __name__ == "__main__":
    main()
