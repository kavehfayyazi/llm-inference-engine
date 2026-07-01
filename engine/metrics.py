"""Timing helpers and latency/throughput aggregation."""

from __future__ import annotations

import time

import torch


def sync(device: torch.device):
    # Block until queued GPU work finishes, so timers measure real compute.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def now(device: torch.device) -> float:
    # Device-synced wall clock in seconds.
    sync(device)
    return time.perf_counter()


def percentile(values, p: float) -> float:
    # p-th percentile (0-100) via nearest-rank on a sorted copy.
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def summarize(requests, wall_time: float) -> dict:
    # Aggregate per-request stamps into serving metrics.
    ttft = [r.t_first for r in requests if r.t_first is not None]
    latency = [r.t_finish for r in requests if r.t_finish is not None]
    tpot = [
        (r.t_finish - r.t_first) / max(1, len(r.generated) - 1)
        for r in requests
        if r.t_first is not None and r.t_finish is not None and len(r.generated) > 1
    ]
    out_tokens = sum(len(r.generated) for r in requests)
    return {
        "requests": len(requests),
        "wall_s": round(wall_time, 4),
        "ttft_ms_mean": round(1000 * (sum(ttft) / len(ttft)) if ttft else 0.0, 2),
        "tpot_ms_mean": round(1000 * (sum(tpot) / len(tpot)) if tpot else 0.0, 2),
        "latency_ms_p99": round(1000 * percentile(latency, 99), 2),
        "throughput_req_s": round(len(requests) / wall_time, 3) if wall_time else 0.0,
        "throughput_tok_s": round(out_tokens / wall_time, 1) if wall_time else 0.0,
    }
