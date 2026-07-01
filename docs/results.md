# Benchmark results

Workload: 12 concurrent requests, generation lengths 8-40 (high variance), all
arriving at t=0. Model: Llama-3.2-1B. Reference paged attention path.
Device: Apple MPS, fp16. Reproduce with `PYTHONPATH=. python scripts/bench.py`.

Absolute numbers are illustrative (small model, MPS, unfused per-request
attention). The **relative trends** are the point.

```
scheduler   batch  steps   ttft_ms   tpot_ms    p99_ms   tok/s
--------------------------------------------------------------
static          1    276   4467.91     32.23   9560.55    30.1
static          2    222    3589.0     47.21   8459.98    34.0
static          4    117   2653.42     84.24   7663.61    37.6
static          8     78   1741.42    136.41   7411.37    38.9
dynamic         1    276   3862.95     30.59   8853.05    32.5
dynamic         2    222   3572.93     46.75   8438.94    34.1
dynamic         4    117   2659.44     84.29   7688.33    37.5
dynamic         8     78   1818.53    137.81   7620.68    37.8
continuous      1    276   4028.25     31.46   9055.64    31.8
continuous      2    145   2968.00     54.42   7914.24    36.4
continuous      4     85   2082.69     99.17   7510.64    38.3
continuous      8     46    808.11    186.15   7278.06    39.6
```

## What the numbers say

**Batch size is the primary lever (the core batching tradeoff).** Within any
scheduler, growing the batch:
- cuts **TTFT** (155ms-class waits shrink as more requests are served together),
- raises **TPOT** (a bigger batch = more work per decode step, so each token is slower),
- raises **throughput** and cuts **p99**.
This is the fundamental latency-vs-throughput knob: you trade per-token speed for
work-per-GPU-second.

**Continuous beats static/dynamic on slot efficiency.** The `steps` column is a
clean, timing-free signal (one step = one decode pass). At batch 8, continuous
finishes the workload in **46 steps vs 78** for static -- 41% fewer. Reason:
static runs a batch to completion, so every slot waits for that batch's *longest*
request (head-of-line waste); continuous refills a freed slot from the queue the
moment a short request finishes, keeping the batch full. Fewer wasted slot-steps
-> lower TTFT (808ms vs 1741ms at batch 8), lower p99, higher tok/s.

**Static == dynamic here** because all 12 requests arrive at t=0. Dynamic's
timeout only helps when requests *trickle in* (fire a partial batch instead of
waiting for N). Under a saturated offline arrival, the two are identical.

## Caveats / next steps

- **Arrival model.** This is an offline/saturated sweep. A wall-clock load
  generator with a real arrival *rate* would surface dynamic's timeout benefit
  and widen continuous's lead under bursty traffic.
- **Attention path.** The reference read gathers KV per request each step; the
  Triton kernel (CUDA) removes that copy and would raise tok/s, especially at
  larger batch and sequence length.
- **KV release.** Blocks are freed at request end but not reused mid-run here;
  a persistent shared pool across a longer stream would show packing gains.
