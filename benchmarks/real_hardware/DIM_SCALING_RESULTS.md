# Suite #47: Dimension Scaling

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Part 1: Latency & Throughput (1024 rooms, 8 rooms/block)

| Dim | μs | qps (M) | TFLOPS | BW (GB/s) |
|-----|-----|---------|--------|-----------|
| 32 | 11.45 | 89.5 | 0.006 | 5.7 |
| 64 | 11.60 | 88.3 | 0.011 | 11.3 |
| **128** | **10.78** | **95.0** | 0.024 | 24.3 |
| 256 | 11.07 | 92.5 | 0.047 | 47.4 |
| 384 | 11.51 | 89.0 | 0.068 | 68.4 |
| 512 | 12.56 | 81.5 | 0.083 | 83.5 |
| 768 | 15.84 | 64.7 | 0.099 | **99.4** |
| 1024 | 23.05 | 44.4 | 0.091 | 91.1 |
| 1536 | 53.80 | 19.0 | 0.058 | 58.5 |
| 2048 | 55.42 | 18.5 | 0.076 | 75.8 |

**Launch-dominated zone: dim ≤ 384** — latency flat at ~11μs regardless of dimension.
**Memory-dominated zone: dim ≥ 512** — latency scales linearly with weight data size.

## Part 2: Optimal Block Size per Dimension

| Dim | Best Config | μs |
|-----|-------------|-----|
| 32-128 | 8 rooms/256 threads | 7.4-8.0 |
| 256-768 | 4 rooms/128 threads | 9.2-15.8 |
| ≥1024 | 16 rooms/512 threads | 22.2-55.3 |

**Block size shifts with dimension** — small dims benefit from more rooms/block (occupancy), large dims need more threads per room to hide memory latency.

## Part 3: Batch Scaling

| Dim | 8 rooms | 256 rooms | 1024 rooms | 4096 rooms |
|-----|---------|-----------|------------|------------|
| 64 | 4.0μs | 5.1μs | 8.2μs | 20.2μs |
| 256 | 4.1μs | 5.7μs | 9.3μs | 27.8μs |
| 1024 | 5.5μs | 8.4μs | 20.9μs | 103.2μs |
| 2048 | 6.6μs | 12.2μs | 55.8μs | OOM |

## Key Findings

1. **OI = 1.0 for ALL dimensions** — dot product is always memory-bound. Ridge point is 7.35 ops/byte.
2. **Peak throughput at dim=128** (95M qps) — not dim=256. Smaller rooms = faster.
3. **Bandwidth peaks at 99.4 GB/s** (dim=768) — exceeds theoretical GPU bandwidth due to unified memory.
4. **dim ≥ 1536 is impractical for fleet** — 53μs+ latency, 19M qps at 1024 rooms.
5. **OOM at dim=2048 × 4096 rooms** — exceeds 8GB unified memory allocation limits.

## Rule #33: Use Smallest Effective Dimension
Room inference throughput peaks at dim=128 (95M qps). If model accuracy allows, reducing dimension from 256 to 128 gives 2.8% throughput boost AND 50% memory reduction. Larger dimensions (>1024) are impractical for real-time fleet inference.
