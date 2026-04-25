# Suite #31: Real-World Fleet Throughput Simulator

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87, passive cooling

## Purpose
Previous suites measured kernel-only latency. This measures ACTUAL end-to-end throughput including weight uploads, kernel inference, output retrieval, and CPU-GPU synchronization.

## Component Costs

| Operation | Latency | Notes |
|-----------|---------|-------|
| Weight upload H2D (512B, 1 room) | 15.85 μs | **Dominates cold latency** |
| Weight upload H2D (128KB, 256 rooms) | 41.85 μs | Batch upload |
| V7 kernel (256 rooms) | 10.93 μs | Fastest part! |
| D2H copy (1KB, 256 floats) | 15.48 μs | Eliminated by zero-copy |
| Zero-copy read (1 float) | 0.001 μs | Negligible |
| **Total overhead (launch + sync)** | **~20 μs** | **Independent of kernel** |

## Scenario Results

| Scenario | Latency (μs) | Room-qps | p99/p50 |
|----------|-------------|----------|---------|
| 1 room, hot | 23.3 | 42.9K | 1.06× |
| 16 rooms, hot | 25.6 | 625K | 1.30× |
| 64 rooms, hot | 28.6 | 2.24M | 1.02× |
| 256 rooms, hot | 29.9 | **8.56M** | 1.03× |
| 256 rooms, 50% cold | 30.6 | 8.37M | 1.04× |
| 256 rooms, all cold | 30.5 | 8.40M | 1.04× |
| 1024 rooms, hot | 43.5 | **23.5M** | 1.01× |

## Key Findings

1. **Launch+sync overhead (~20μs) dominates at small batch** — at 1 room, the kernel takes ~5μs but the total is 23μs
2. **Cold rooms barely matter** — at 256 rooms, adding 128 cold uploads adds only 0.7μs (H2D overlaps with kernel)
3. **Real-world throughput is 4.5× lower than kernel-only** — 23.5M vs 105M room-qps at 1024 rooms
4. **Weight upload is the hidden tax** — 15.85μs per cold room upload, but batched upload of 256 rooms is only 42μs
5. **Zero-copy output is essential** — D2H copy costs 15.48μs, zero-copy costs 0.001μs
6. **256 rooms is the sweet spot for real-world** — 8.56M room-qps with excellent jitter (1.03×)

## Production Implications

The kernel is NOT the bottleneck. The CPU-GPU communication is. To reach the theoretical 105M room-qps:

1. **Keep ALL active weights GPU-resident** (128KB for 256 rooms — fits in L2)
2. **Use zero-copy output** (saves 15.48μs per batch)
3. **Batch requests aggressively** (amortize 20μs overhead)
4. **Pre-load weights during idle time** (eliminate cold penalty entirely)

With these optimizations, the practical fleet throughput approaches the kernel-only numbers.
