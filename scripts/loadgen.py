"""Open-loop load generator: Poisson arrivals, measure queueing per scheduler.

Time unit is one scheduler step (a discrete-event model). Requests arrive over
time via exponential inter-arrivals, so a full queue builds up -- which is where
continuous batching's admit/evict advantage actually shows, unlike the offline
(all-at-t=0) sweep in bench.py.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.blocks import BlockPool, PagedKVCache
from engine.config import EngineConfig
from engine.metrics import percentile
from engine.model import load
from engine.request import Request
from engine.scheduler import ContinuousScheduler, DynamicScheduler, StaticScheduler

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "def add(a, b):",
    "In 1969, the first humans landed on the Moon and",
]
GEN_LENS = [8, 40, 12, 36]
SCHEDULERS = [("static", StaticScheduler), ("dynamic", DynamicScheduler), ("continuous", ContinuousScheduler)]


def make_arrivals(n, rate, seed):
    # Exponential inter-arrivals (steps); rate = mean arrivals per step.
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(int(t))
    return out


def build(lm, arrivals, n):
    per = [(PROMPTS[i % len(PROMPTS)], GEN_LENS[i % len(GEN_LENS)]) for i in range(n)]
    total = sum(len(lm.tokenizer(p).input_ids) + g for p, g in per)
    num_blocks = math.ceil(total / lm.cfg.block_size) + n
    pool = BlockPool(num_blocks, lm.dims.n_layers, lm.dims.n_kv_heads, lm.dims.head_dim,
                     lm.cfg.block_size, lm.device, lm.dtype)
    reqs = []
    for i, (p, g) in enumerate(per):
        ids = lm.tokenizer(p, return_tensors="pt").input_ids.to(lm.device)
        reqs.append(Request(id=i, prompt_ids=ids, kv=PagedKVCache(pool), max_new=g, arrival=arrivals[i]))
    return pool, reqs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--rate", type=float, default=0.5, help="arrivals per step")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--backend", default="reference", choices=["reference", "triton"])
    args = ap.parse_args()

    lm = load(EngineConfig(attention_backend=args.backend))
    arrivals = make_arrivals(args.n, args.rate, seed=0)
    print(f"device={lm.device} backend={args.backend} n={args.n} rate={args.rate}/step batch={args.batch}\n")

    header = f"{'scheduler':<11} {'makespan':>9} {'ttft_stp':>9} {'lat_p99':>8} {'lat_mean':>9} {'req/step':>9}"
    print(header)
    print("-" * len(header))
    for name, cls in SCHEDULERS:
        pool, reqs = build(lm, arrivals, args.n)
        done, stats = cls(lm, pool, args.batch).run(reqs)
        ttft = [r.s_first - r.arrival for r in done]
        latency = [r.s_finish - r.arrival for r in done]
        makespan = stats["steps"]
        print(f"{name:<11} {makespan:>9} {sum(ttft) / len(ttft):>9.1f} "
              f"{percentile(latency, 99):>8} {sum(latency) / len(latency):>9.1f} "
              f"{args.n / makespan:>9.3f}")


if __name__ == "__main__":
    main()
