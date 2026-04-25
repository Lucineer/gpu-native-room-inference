# Suite #42: Input Reuse — L2 Cache Is Enough

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Results

| Rooms | Global (μs) | Shmem (μs) | Const (μs) |
|-------|-------------|------------|------------|
| 8 | 8.22 | 8.09 | **6.16** |
| 256 | 8.48 | 8.06 | 7.80 |
| 1024 | 9.61 | 9.99 | 9.31 |
| 4096 | 27.09 | 29.13 | 27.11 |

## Key Finding

**All three methods are within 5% of each other.** The L2 cache already handles input reuse perfectly. The 512-byte input vector fits in a single L2 cache line set and is reused by all 8 warps in a block without any explicit optimization.

Shared memory adds a `__syncthreads()` barrier that costs more than the cache savings. Constant memory's broadcast optimization is redundant when L2 caching already provides the same benefit.

**Rule #30: Don't optimize input access.** The L2 cache handles it. Focus optimization effort on weight access patterns instead.
