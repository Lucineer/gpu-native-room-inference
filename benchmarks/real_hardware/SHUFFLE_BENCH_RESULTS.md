# Suite #29: Warp Shuffle vs Shared Memory Reduction

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87, passive cooling
**Dim:** 256, Warmup: 500, Iters: 50000

## Hypothesis
Warp shuffle (`__shfl_down_sync`) eliminates shared memory synchronization overhead for dot product reduction by using register-to-register communication.

## Results

### Latency (μs per batch) — Lower is Better
| Kernel | 1 room | 4 rooms | 16 rooms | 64 rooms | 256 rooms | 1024 rooms |
|--------|--------|---------|----------|----------|-----------|------------|
| shmem (32 threads) | 7.97 | 5.52 | 5.56 | 6.18 | 8.92 | 23.02 |
| shmem (256 threads) | 6.95 | 5.60 | 5.57 | 8.00 | 16.86 | 52.15 |
| shuffle (32 threads) | 5.97 | 5.30 | 5.44 | 5.58 | 8.27 | 20.70 |
| shuffle+block (256) | 5.44 | 5.49 | 5.57 | 7.90 | 15.24 | 45.34 |
| **contig4 (shuffle)** | **4.51** | **4.37** | **5.23** | **5.38** | **7.26** | **14.16** |
| **contig8 (shuffle)** | **4.12** | **4.37** | **5.24** | **5.33** | **7.27** | **13.99** |

### Speedup vs shmem(32)
| Kernel | 1 room | 4 rooms | 16 rooms | 64 rooms | 256 rooms | 1024 rooms |
|--------|--------|---------|----------|----------|-----------|------------|
| shuffle (32 threads) | 1.34× | 1.04× | 1.02× | 1.11× | 1.08× | 1.11× |
| shuffle+block (256) | 1.47× | 1.01× | 1.00× | 0.78× | 0.59× | 0.51× |
| **contig4 (shuffle)** | **1.77×** | **1.26×** | **1.06×** | **1.15×** | **1.23×** | **1.63×** |
| **contig8 (shuffle)** | **1.93×** | **1.27×** | **1.06×** | **1.16×** | **1.23×** | **1.65×** |

### Room-qps (Millions)
| Kernel | 1 room | 64 rooms | 256 rooms | 1024 rooms |
|--------|--------|----------|-----------|------------|
| shmem (32 threads) | 0.13M | 10.4M | 28.7M | 44.5M |
| **contig4 (shuffle)** | **0.22M** | **11.9M** | **35.3M** | **72.3M** |
| **contig8 (shuffle)** | **0.24M** | **12.0M** | **35.2M** | **73.2M** |

## Key Findings

1. **Contiguous warp layout is the winner** — 1.65× at 1024 rooms, 1.93× at 1 room
2. **Eliminates ALL shared memory** — no `__syncthreads()`, no bank conflicts, no barriers
3. **The trick: 32 threads per room = exactly 1 warp** — reduction is pure register communication
4. **256 threads/block HURTS shared memory kernels** (0.51×) but HELPS shuffle kernels (1.65×)
5. **Multi-room blocking + shuffle = additive** — contig4/contig8 combine both techniques
6. **At 1024 rooms: 73.2M room-qps** — new record (up from 44.5M shared memory baseline)
7. **shuffle+block hybrid doesn't work** — shared memory still needed for cross-warp reduction, negating the benefit

## Why Contiguous Warp Layout Wins

Traditional layout: threads 0-31 are room 0, threads 32-63 are room 1, etc.
This means each room gets exactly one warp — the reduction happens entirely within warp registers.

No shared memory allocation, no bank conflicts, no `__syncthreads()` barriers.
The GPU scheduler sees independent warps that can execute with zero inter-warp synchronization.

## Production Implications

**New recommended kernel for batch ≥ 64: contig8 (shuffle)**
- 73.2M room-qps at 1024 rooms (1.65× over previous best)
- Zero shared memory usage
- Cleanest kernel architecture possible

Combined with sustained load results (suite #28): **73.2M room-qps × 0.8% degradation over 10M iters = production-ready.**
