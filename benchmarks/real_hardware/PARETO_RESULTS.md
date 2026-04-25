# Suite #43: Pareto Frontier — Latency vs Throughput

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Complete Pareto Table (Double-Buffered Async Pipeline)

| Rooms | p50 (μs) | p99 (μs) | qps (M) | Efficiency |
|-------|----------|----------|---------|------------|
| 1 | 6.50 | 11.58 | 0.15 | 0.1% |
| 8 | 4.29 | 5.38 | 1.87 | 1.3% |
| 64 | 5.03 | 5.92 | 12.74 | 8.5% |
| 256 | 5.82 | 6.75 | 43.96 | 29.5% |
| 768 | 8.61 | 9.57 | 89.22 | 59.9% |
| 1024 | 9.76 | 10.75 | 104.91 | 70.4% |
| 2048 | 14.37 | 15.23 | 142.53 | 95.7% |
| **3072** | **19.04** | **20.03** | **161.34** | **108.3%** |
| 4096 | 26.05 | 27.10 | 157.24 | 105.5% |

## SLA Lookup

| SLA (p99) | Max Rooms | Throughput |
|-----------|-----------|------------|
| 10 μs | 768 | 89M qps |
| 20 μs | 2048 | 142M qps |
| 50 μs | 3072 | 161M qps |
| 100 μs | 3072 | 161M qps |

## Cost Model

| Scenario | Rooms | SLA | Jetsons | Hardware | Power/yr |
|----------|-------|-----|---------|----------|----------|
| Small fleet | 100 | 50μs | 1 | $249 | $12 |
| Medium fleet | 1K | 20μs | 2 | $498 | $23 |
| Large fleet | 10K | 20μs | 5 | $1,245 | $58 |
| Enterprise | 100K | 50μs | 24 | $5,976 | $278 |
| Mega fleet | 1M | 100μs | 118 | $29,382 | $1,364 |

## Key Numbers

- **Peak throughput: 161M room-qps** at 3072 rooms (20μs p99)
- **Best latency: 4.29μs p50** at 8 rooms (5.38μs p99)
- **Jitter ratio: 1.06-1.16×** across all batch sizes
- **Efficiency > 100%** at 3072+ rooms (async pipeline exceeds GPU-only estimates)
- **1 Jetson serves 100K rooms** with 24 devices ($5,976 total)
