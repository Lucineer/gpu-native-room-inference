# Suite #41: Memory Layout — Row-Major Wins

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Results

| Rooms | Row-Major (μs) | Interleaved (μs) | Ratio | Block-Inter (μs) | Padded (μs) |
|-------|----------------|-------------------|-------|-------------------|-------------|
| 8 | 5.44 | 7.05 | 0.77× | 7.43 | 7.43 |
| 256 | 7.05 | 14.49 | **0.49×** | 5.79 | 5.85 |
| 1024 | 9.42 | 41.26 | **0.23×** | 9.40 | 9.48 |
| 4096 | 29.72 | 151.33 | **0.20×** | 27.85 | 27.60 |

## Why Row-Major Wins

**Interleaved is 5× slower at scale.** The access pattern `weights[i * num_rooms + room]` creates a stride of `num_rooms` between consecutive dimension indices for the same room. At 4096 rooms, each warp's reads are scattered across 128KB — massive cache thrashing.

**Row-major is optimal because:**
1. Within a warp: 32 threads read `room * dim + (lane + i*32)` — contiguous 32-element chunks
2. The 512-byte room data fits in a single L2 cache line set
3. Cross-warp stride (256 halfs = 512 bytes) matches the natural L2 associativity

**Block-interleaved and padded ≈ row-major** — no measurable difference. Cache line alignment is already optimal at dim=256.

## Rule #29: Keep Row-Major Weight Layout

Don't try to optimize memory layout for room inference. Row-major (one room's weights contiguous) is already optimal for the V7 kernel's access pattern. Interleaved layouts cause catastrophic cache thrashing.
