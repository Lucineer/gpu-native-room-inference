# Suite #66: Progressive Refinement (INT8 → FP16)

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Full Pipeline (16384 rooms)

| Method | μs | M qps | vs Full FP16 |
|---|---|---|---|
| Full FP16 | 141.4 | 115.9 | 1.00× |
| Full INT8 | 125.2 | 130.9 | **1.13×** |
| Progressive (K=256) | 131.1 | 125.0 | 1.08× |

## INT8 Top-K Recall (how often INT8 ranking matches FP16 ranking)

| K | Recall |
|---|---|
| 16 | **100%** |
| 32 | 96.9% |
| 64 | 98.4% |
| 128 | 99.2% |
| 256 | 99.2% |
| 512 | 99.4% |
| 1024 | 99.6% |

## Key Findings

1. **INT8 alone is the best strategy** — 1.13× faster than FP16, no refinement needed. The FP16 refinement step adds back latency that negates the INT8 speedup.

2. **INT8 ranking is highly accurate** — 100% recall at K=16, 99.2% at K=256. The quantization error doesn't significantly affect relative ordering of rooms. For ranking applications, INT8 is sufficient.

3. **Progressive refinement only helps for very small K** — At K=16 with 100% recall, you save 5μs on the 256-room FP16 refinement. But the sort step on CPU costs ~6μs, wiping out the gain.

4. **Sort is the bottleneck for progressive** — Finding top-K from 16K rooms requires a partial sort (~6μs on CPU). A GPU radix sort could reduce this to <1μs, making progressive viable.

## Rule #61: Use INT8 Directly, Skip Progressive Refinement

For room ranking, INT8 quantization is sufficient (99.2% top-256 recall). Progressive refinement (INT8→FP16) adds overhead from sorting and weight gathering that negates the INT8 speedup. Only use progressive refinement when exact top-K scores are required AND a GPU sort is available to eliminate the CPU sort bottleneck.
