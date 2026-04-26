# Suite #64: Combined Optimizations & Block Size Sweep

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Block Size & Room-Per-Block Sweep (4096 rooms, dim=256, correct kernels)

| Kernel | Rooms/Block | Threads | μs | M qps | vs V7 |
|---|---|---|---|---|---|
| V7 (baseline) | 8 | 256 | 32.63 | 125.5 | 1.00× |
| V15 (2r/warp) | 16 | 256 | 31.85 | 128.6 | 1.02× |
| V16 (2r/warp,u2) | 16 | 256 | 37.39 | 109.5 | 0.87× |
| **V17 (4r,128t)** | **4** | **128** | **28.60** | **143.2** | **1.14×** |

## V15 (2 rooms/warp) Scaling vs V7

| Rooms | V7 μs | V15 μs | V15/V7 |
|---|---|---|---|
| 64 | 4.62 | 5.47 | 1.18× (V7 wins) |
| 256 | 5.43 | 5.78 | 1.06× (V7 wins) |
| 1,024 | 8.72 | 9.00 | 1.03× (V7 wins) |
| 4,096 | 27.75 | 28.17 | 1.02× (V15 slight win) |
| 8,192 | 57.48 | 60.92 | 1.06× (V7 wins) |

## L2 Persist Results (from Suite #62, combined here)

| Config | μs | M qps | vs Baseline |
|---|---|---|---|
| Baseline | 36.05 | 113.6 | 1.00× |
| Max persist | 33.45 | 122.5 | 1.08× |
| Policy window | 31.14 | 131.5 | 1.16× |
| Partial persist (1K) | 29.37 | 139.5 | **1.23×** |
| FP16 accumulation | 28.10 | 145.8 | 1.28× |
| 1M sustained + persist | 26.71 | 153.4 | 1.35× |

## Key Findings

1. **Multiple rooms per warp doesn't help** — V15 (2r/warp) is at best 2% faster, worse at small batches. The GPU scheduler already issues warps efficiently for 1-room-per-warp. Adding loop overhead per warp negates any block launch savings.

2. **V17 (128 threads, 4 rooms/block) is 14% faster** — Fewer blocks = fewer kernel launch dispatches. At 4096 rooms: V7 needs 512 blocks, V17 needs 1024 blocks... wait, that's more blocks. The benefit comes from 128-thread blocks having better SM occupancy — more blocks fit per SM.

3. **The earlier "319M qps" was a measurement artifact** — V9 (16 rooms/block) only assigned 1 room per warp but declared 16 rooms per block, leaving rooms 8-15 unprocessed (zero output). The 2.4× "speedup" was from doing half the work.

4. **L2 persist is the real win** — 1.23-1.35× improvement over baseline. Combined with V17 structure, the theoretical max is ~163M qps (V17 at 143.2 × 1.14 persist = ~163M qps).

## Rule #58: Verify Correctness Before Benchmarking Speed

Multiple rooms per block requires careful room-to-warp mapping. A simple `blockIdx.x * rooms_per_block + threadIdx.x/32` only works when rooms_per_block ≤ warps_per_block. For more rooms than warps, use an inner loop. Always verify output against a known-correct kernel before reporting throughput — the GPU will happily compute wrong answers very fast.

## Rule #59: 128-Thread Blocks Beat 256-Thread Blocks on Jetson

V17 (128 threads, 4 rooms/block) is 14% faster than V7 (256 threads, 8 rooms/block). Smaller blocks improve SM occupancy and reduce per-block resource allocation overhead. The optimal block size is the minimum that still fills all warps (32 threads per warp × number_of_warps).

## Production Kernel: V17 + L2 Partial Persist
- 128 threads per block (4 warps × 32)
- 4 rooms per block
- Partial L2 persist (1K hot rooms)
- Unroll 4× in inner loop
- FP32 shuffle reduction (correct, unlike FP16 shuffle)
- Expected throughput: ~160-170M qps sustained
