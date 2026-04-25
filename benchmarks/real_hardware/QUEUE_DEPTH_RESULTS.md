# Suite #34: GPU Command Queue Depth & Multi-Stream Saturation

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Results

### Queue Depth Scaling (256 rooms, launch N then sync)
| Depth | Per-kernel (μs) | Room-qps | vs depth=1 |
|-------|-----------------|----------|------------|
| 1 | 31.5 | 8.1M | 1.00× |
| 32 | 11.9 | 21.5M | 2.65× |
| 256 | 8.2 | 31.4M | 3.87× |
| 1024 | 6.4 | 40.1M | 4.94× |

### N-Buffer Pipeline (256 rooms)
| Buffers | Per-iter (μs) | Room-qps | vs 2-buf |
|---------|---------------|----------|----------|
| 2 | 6.03 | 42.5M | 1.00× |
| 4 | 5.94 | 43.1M | 1.01× |
| 64 | 5.87 | 43.6M | 1.03× |

### Multiple Independent Pipelines
| Streams | Per-iter (μs) | Total Room-qps | vs 1 stream |
|---------|---------------|----------------|-------------|
| 1 | 5.97 | 42.9M | 1.00× |
| 2 | 7.65 | 66.9M | 1.56× |
| 4 | 15.6 | 65.6M | 1.53× |
| 8 | 30.7 | 66.6M | 1.55× |

## Key Findings

1. **GPU saturates at 2 buffers** — adding more buffers gives only 1.03× improvement
2. **Single kernel takes 5.9μs at 256 rooms** — this is the hardware floor
3. **Max total throughput: ~67M room-qps** regardless of stream count
4. **2 streams is optimal for multi-tenant** — 1.56× over single stream
5. **4+ streams add contention, not throughput** — SM occupancy maxed
6. **Queue depth helps** — depth 1024 gives 4.94× over depth 1, but double-buffer already reaches the floor

## Production Architecture Decision

**Single stream + double buffer = 42.5M room-qps (optimal per-pipeline)**
**2 streams + double buffer = 66.9M room-qps (optimal total)**

No reason to use more than 2 streams. The Orin's 1024 CUDA cores and 16 blocks/SM limit mean only ~2 medium kernels can run in parallel. Beyond that, it's pure contention.

For a fleet of 256 rooms: one stream, two output buffers, fire-and-forget launches. 42.5M room-qps. Done.
