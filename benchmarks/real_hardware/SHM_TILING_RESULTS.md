# Suite #61: Shared Memory Tiling vs V7

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Results (dim=256)

| Rooms | V7 μs | SHM8 μs | SHM/V7 |
|---|---|---|---|
| 64 | 5.09 | 5.81 | 1.14× (V7 wins) |
| 256 | 5.89 | 7.41 | 1.26× (V7 wins) |
| 1,024 | 10.23 | 13.63 | 1.33× (V7 wins) |
| 4,096 | 28.50 | 37.74 | 1.32× (V7 wins) |
| 8,192 | 70.60 | 75.20 | 1.07× (V7 wins) |

Shared memory: 4,608 bytes (3.5% of 128KB). Only 3.5% of budget used.

## Key Finding

**Shared memory tiling is always slower.** The copy from global→shared memory adds 13-33% overhead without any benefit. On Jetson's unified memory architecture, L2 cache (2MB, 128 bytes/line) already provides the same locality as shared memory for sequential access patterns. The V7 kernel reads weights directly from L2 cache with no intermediate copy.

Shared memory only helps when:
1. Data is reused across multiple compute phases (not our case — single pass)
2. Global memory access patterns are uncoalesced (not our case — sequential)
3. There's bank conflict in shared that's worse than global latency (not applicable)

For room inference's simple single-pass dot product, the extra shared memory load/store is pure overhead.

## Rule #53: Shared Memory Tiling Is Counterproductive for Single-Pass Dot Products

On Jetson's unified memory architecture, L2 cache provides equivalent locality to shared memory for sequential access patterns. The explicit copy to shared memory adds 13-33% overhead with zero benefit. Only use shared memory tiling for multi-pass algorithms where data is reused between phases. For room inference, V7's direct L2 reads are optimal.
