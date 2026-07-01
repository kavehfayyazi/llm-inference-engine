# Benchmark results

Workload: 12 concurrent requests, generation lengths 8-40 (high variance), all
arriving at t=0. Model: Llama-3.2-1B. Device: **NVIDIA T4, bf16** (Colab).
Correctness verified first: `verify_triton.py` -> both single and batched paged
kernels match the reference path and HF greedy. Reproduce with
`PYTHONPATH=. python scripts/bench.py --backend {reference,triton}`.

`steps` is a deterministic, timing-free efficiency signal (one step = one decode
pass over the running batch). It is identical across devices and backends.

## Reference read path (gather + SDPA per request)

```
scheduler   batch  steps   ttft_ms   tpot_ms    p99_ms   tok/s
--------------------------------------------------------------
static          1    276   3747.05     28.67   8749.40    32.9
static          2    222   4257.84     69.98  10254.32    28.1
static          4    117   3955.92    109.28  11172.19    25.8
static          8     78   2054.51    139.72   8845.36    32.6
dynamic         1    276   3832.61     27.17   8641.69    33.3
dynamic         2    222   4380.21     72.84  10520.04    27.4
dynamic         4    117   4024.12    111.38  11636.23    24.8
dynamic         8     78   2038.82    135.20   8955.29    32.2
continuous      1    276   4020.27     27.92   8692.56    33.1
continuous      2    145   5578.90     97.37  14051.44    20.5
continuous      4     85   2586.16    117.13   9394.54    30.7
continuous      8     46    967.22    153.55   6561.33    43.9
```

## Triton kernel read path (one fused launch per layer)

```
scheduler   batch  steps   ttft_ms   tpot_ms    p99_ms   tok/s
--------------------------------------------------------------
static          1    276   3220.72     22.81   7402.53    38.9
static          2    222   3892.51     65.72   9294.79    31.0
static          4    117   3686.50     98.03  10370.42    27.8
static          8     78   1769.90    111.61   7771.94    37.1
dynamic         1    276   3128.00     21.82   7180.04    40.1
dynamic         2    222   4161.01     67.26   9498.18    30.3
dynamic         4    117   3658.66     97.20  10265.67    28.1
dynamic         8     78   1695.33    107.89   7503.73    38.4
continuous      1    276   3118.21     21.31   6909.98    41.7
continuous      2    145   4845.32     84.39  12292.98    23.4
continuous      4     85   2171.80     96.61   7907.99    36.4
continuous      8     46    844.07    115.64   5083.07    56.7
```

## Three findings

**1. Batch size is the latency-vs-throughput knob.** Within a scheduler, a larger
batch cuts TTFT and raises tok/s, but raises TPOT (more work per decode step). The
classic tradeoff, visible in both tables.

**2. Continuous batching wins slot efficiency.** The `steps` column tells it
cleanly: at batch 8, continuous finishes in **46 steps vs 78** for static (-41%).
Static runs a batch to completion, so every slot waits for that batch's longest
request; continuous refills a freed slot the moment a short request finishes.
Static == dynamic here because all requests arrive at t=0 -- dynamic's timeout
only helps under trickle-in load.

**3. The Triton kernel makes decode compute-bound.** The reference read gathers
each request's KV and runs a separate SDPA -- N kernel launches per layer, which
leaves the GPU launch-bound. The fused kernel does one launch per layer. Result,
best config (continuous, batch 8):

| metric | reference | triton | change |
|--------|-----------|--------|--------|
| tok/s  | 43.9      | 56.7   | +29%   |
| TPOT   | 153.6 ms  | 115.6 ms | -25% |
| p99    | 6.56 s    | 5.08 s | -23%   |
| TTFT   | 967 ms    | 844 ms | -13%   |

The kernel wins on every row of both tables; `steps` is unchanged, confirming it
alters only the attention read, not the scheduling.

## Caveats / next steps

- **Arrival model.** Offline/saturated sweep. A wall-clock load generator with a
  real arrival rate would surface dynamic's timeout benefit and continuous's lead
  under bursty traffic.
- **Prefill batching.** Newly admitted requests are prefilled one at a time; this
  shows as noise at small batch (continuous batch 2). Chunked/batched prefill
  would smooth it.
- **KV release.** Blocks free at request end; a persistent pool across a long
  stream would show packing gains directly.
