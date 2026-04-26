# Suite #53: Room Capacity & Memory Scaling

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Part 1: GPU Memory Status

| Metric | Value |
|---|---|
| Total GPU memory | 7,619.9 MB |
| Free at idle | 3,833.7 MB |
| Usable for weights (95%) | ~3,642 MB |

## Part 2: Memory Per Room (dim=256)

| Component | Size |
|---|---|
| Weights (FP16) | 512 bytes (0.5 KB) |
| Output (FP32) | 4 bytes |
| Sparse CSR (max) | 1,540 bytes (1.5 KB) |

## Part 3: Max Room Capacity

| Dimension | Max Rooms | Memory Used | Weight Size |
|---|---|---|---|
| 32 | 43.6M | 2,664 MB | 64 B |
| 64 | 22.5M | 2,744 MB | 128 B |
| 128 | 11.4M | 2,787 MB | 256 B |
| **256** | **5.75M** | **2,808 MB** | **512 B** |
| 384 | 3.84M | 2,815 MB | 768 B |
| 512 | 2.89M | 2,819 MB | 1 KB |
| 768 | 1.93M | 2,823 MB | 1.5 KB |
| 1024 | 1.45M | 2,825 MB | 2 KB |

**Finding:** Memory is NOT the bottleneck. 5.75M rooms at dim=256 use only 2.8 GB. The Jetson can hold millions of rooms.

## Part 4: Throughput vs Room Count

| Rooms | μs | M room-qps | Memory MB | Efficiency |
|---|---|---|---|---|
| 64 | 10.79 | 5.9 | 0.0 | 3.7% |
| 256 | 7.97 | 32.1 | 0.1 | 19.9% |
| 1,024 | 19.00 | 53.9 | 0.5 | 33.5% |
| 4,096 | 36.57 | 112.0 | 2.0 | 69.6% |
| 8,192 | 71.86 | 114.0 | 4.0 | 70.8% |
| 16,384 | 125.26 | 130.8 | 8.0 | 81.2% |
| 32,768 | 246.48 | 132.9 | 16.0 | 82.6% |
| **65,536** | **488.58** | **134.1** | **32.0** | **83.3%** |

**Finding:** Throughput scales sublinearly but consistently. At 65K rooms we hit 83.3% efficiency. The GPU is compute-bound at large batches, not memory-bound.

## Part 5: Multi-Model Serving (Mixed Dimensions)

| Configuration | μs | M room-qps | Memory |
|---|---|---|---|
| Uniform dim=256 (4096 rooms) | 27.24 | 150.4 | 2.0 MB |
| Variable-dim (50/30/20 mix) | 1,087.15 | 3.8 | 1.4 MB |

**Variable-dim is 40× SLOWER.** The per-room cumulative weight offset loop (`for r < room: offset += dims[r]`) is O(n²) across all rooms.

Memory savings from variable-dim: 30.5% (2.0 → 1.4 MB). Not worth the throughput loss.

## Key Findings

1. **Memory is not the bottleneck** — 5.75M rooms fit at dim=256 (2.8 GB). One Jetson can hold millions of room weights.
2. **Compute is the bottleneck** — 134M qps peak at 65K rooms (83.3% efficiency). More rooms = better utilization.
3. **Variable-dim serving is catastrophic** — 40× slower due to O(n²) weight offset computation. Never use heterogeneous dimensions in a batch.
4. **Always pad to uniform dimension** — pad smaller rooms with zeros. The 30% memory savings from variable-dim doesn't justify the 40× throughput loss.
5. **Sweet spot: 4K-16K rooms per batch** — 70-81% efficiency, low latency (37-125μs), practical memory usage.
6. **For fleet deployment:** Standardize room dimensions. Use dim=256 as the fleet-wide standard. Pad smaller models, split larger ones across multiple inferences.

## Rule #38: Standardize Room Dimensions

Heterogeneous room dimensions in a single batch cause 40× throughput degradation due to cumulative weight offset computation. Always pad all rooms to a uniform dimension (fleet standard: dim=256). Memory is abundant (5.75M rooms fit at dim=256) — the constraint is compute throughput, not memory capacity.

## Rule #39: Batch 4K-16K Rooms for Production

The 4K-16K room range provides 70-81% GPU efficiency with practical latency (37-125μs). Smaller batches underutilize the GPU (<70% efficiency). Larger batches add diminishing returns and increase latency. For a 100K-room fleet, batch in groups of 8K-16K.
