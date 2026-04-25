# Suite #46: L2 Cache Bank Conflict Analysis

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Part 1: Input Access Pattern (8 rooms/block)

| Rooms | Baseline (μs) | Staggered | Ratio | Shmem | Ratio |
|-------|---------------|-----------|-------|-------|-------|
| 8 | 7.46 | 7.90 | 0.95× | **5.75** | **1.30×** |
| 256 | 8.64 | **7.34** | **1.18×** | 7.84 | 1.10× |
| 1024 | 10.53 | **8.18** | **1.29×** | 10.14 | 1.04× |
| 4096 | 28.31 | **17.32** | **1.63×** | 29.29 | 0.97× |

**Staggered input wins at scale** — each warp reads different input elements, reducing L2 contention when many blocks compete. At 4096 rooms: 1.63× speedup.

## Part 2: Rooms per Block (256 rooms)

| Config | Time (μs) | qps (M) |
|--------|-----------|---------|
| 1 room (32 threads) | 9.22 | 27.8 |
| 1 room (256 threads) | 10.61 | 24.1 |
| 2 rooms (64 threads) | 6.01 | 42.6 |
| **4 rooms (128 threads)** | **5.78** | **44.3** |
| 8 rooms (256 threads) | 5.95 | 43.0 |
| 16 rooms (512 threads) | 6.30 | 40.6 |

**4 rooms/block wins at 256 rooms** — slightly better than 8 rooms. Too few rooms = low occupancy; too many = register pressure.

## Key Findings

1. **Staggered input access helps at large batch** — 1.63× at 4096 rooms by reducing L2 set contention
2. **Block size depends on batch** — 4 rooms/block at 256 rooms, 8 rooms/block at 1024+ rooms
3. **Shmem only helps at small batch** — syncthreads overhead kills the advantage at scale
4. **L2 contention is real** — when 512+ blocks compete for the same input cache lines, staggering reads reduces conflict misses

## Rule #32: Stagger Input Reads at Large Batch
At batch > 1024 rooms, assign each warp a different input offset to reduce L2 set contention. 1.63× speedup at 4096 rooms.
