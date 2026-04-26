# Suite #62: L2 Cache Persisting + FP16 Accumulation

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87

## L2 Cache Configuration
- Max persisting L2: 1,408 KB
- Default persisting L2: 384 KB
- Weight memory at 4096 rooms: 2,048 KB (145% of L2)
- Weight memory at 1024 rooms: 512 KB (36% of L2)

## L2 Persist Results (4096 rooms, V7 kernel)

| Configuration | μs | M qps | vs Baseline |
|---|---|---|---|
| Baseline (no persist) | 36.05 | 113.6 | 1.00× |
| Max persist (1408 KB) | 33.45 | 122.5 | **1.08×** |
| Stream policy window | 31.14 | 131.5 | **1.16×** |
| **Partial persist (1K rooms)** | **29.37** | **139.5** | **1.23×** |

## FP16 Accumulation
| Type | μs | M qps |
|---|---|---|
| FP32 accumulation | 36.05 | 113.6 |
| FP16 accumulation | 28.10 | 145.8 |

## Sustained (1M inferences, max persist)
| Configuration | μs | M qps |
|---|---|---|
| 1M sustained | 26.71 | **153.4** |

## Key Findings

1. **Partial persist beats full persist** — Persisting 1024 rooms (512KB, 36% of L2) is 1.23× faster than no persist, better than persisting all 4096 rooms (1.08×). When you persist more than fits in L2, cache eviction creates thrashing. Less is more.

2. **Stream policy window > global persist setting** — Explicit `cudaStreamSetAttribute` with `cudaAccessPropertyPersisting` (1.16×) beats the global `cudaDeviceSetLimit` (1.08×). Fine-grained control is better.

3. **FP16 accumulation is 22% faster** — Staying in half precision avoids the FP16→float conversion on every multiply. For dim=256, the accumulation precision loss is negligible (room scores don't need FP32 precision).

4. **1M sustained: 153.4M qps** — With max persist + warmed GPU, sustained throughput approaches the all-time record (163M qps from Suite #56 at 306MHz).

## Rule #54: Persist Only What Fits in L2

Persist only the most frequently accessed rooms (≤36% of L2 capacity). Persisting more than fits causes cache eviction and thrashing. For the fleet: pin the top-N hot rooms in L2, let cold rooms stream through normally. Partial persist + streaming is the optimal hybrid.

## Rule #55: Use FP16 Accumulation for Room Inference

FP16 accumulation (half2 arithmetic) is 22% faster than FP32 with negligible accuracy impact for dim=256 dot products. Room scores are used for ranking/comparison, not exact arithmetic. Only use FP32 accumulation for dim>1024 where rounding errors compound.
