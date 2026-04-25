# Suite #30: Ultimate Production Kernel — Shuffle + Multi-Room Comparison

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87, passive cooling
**Warmup:** 500, **Iters:** 50000

## Hypothesis
Combining warp shuffle (suite #29) with multi-room blocking (suite #20) yields the ultimate production kernel.

## Kernels Tested
- **V4:** Shared memory, 4 rooms/block, 128 threads (previous champion)
- **V5:** Contig8 shuffle, stride-8 unrolled (suite #29 winner)
- **V6:** Contig8 shuffle, stride-8, fused GELU
- **V7:** Contig8 shuffle, general dim loop (stride-32)
- **V8:** Contig16 shuffle, 512 threads/block

## Key Results (dim=256)

### Room-qps (Millions)
| Rooms | V4 (shmem) | V5 (stride8) | V6 (fused) | **V7 (general)** | V8 (16r/block) |
|-------|-----------|-------------|-----------|-----------------|---------------|
| 1 | 0.22 | 0.22 | 0.24 | 0.23 | 0.22 |
| 4 | 0.87 | 0.87 | 0.90 | 0.90 | 0.87 |
| 16 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 |
| 64 | 11.4 | 11.7 | 11.9 | **12.0** | 11.3 |
| 256 | 38.3 | 34.6 | 34.8 | **44.4** | 34.7 |
| 1024 | 90.3 | 72.1 | 73.0 | **105.0** | 72.4 |

### Key Finding: V7 (General Loop) is the New Champion

**V7 at 1024 rooms, dim=256: 105M room-qps** — new all-time record.

Why V7 wins at large batches:
1. **Stride-32 loop** (`for i = lane; i < dim; i += 32`) gives the compiler better scheduling freedom
2. **Stride-8 unroll** in V5/V6 causes register pressure at 256 threads/block (32 regs × 8 = 256 per thread)
3. **Register spilling** in V5/V6 at large batches negates the unroll benefit
4. V7 at 256 rooms is **44.4M room-qps vs V4's 38.3M** — 1.16× improvement

### Dim Sensitivity (1024 rooms)

| Dim | V4 (shmem) | V5 (stride8) | **V7 (general)** | Best |
|-----|-----------|-------------|-----------------|------|
| 64 | 100.8M | 111.5M | 116.9M | V7 |
| 128 | 100.6M | 99.9M | **117.5M** | V7 |
| 256 | 90.3M | 72.1M | **105.0M** | V7 |
| 512 | 71.4M | 72.4M | **82.8M** | V7 |

### Contig16 (V8) Doesn't Help
V8 (16 rooms/block, 512 threads) is consistently **slower** than V7/V5/V6. 512 threads exceeds the 16 blocks/SM limit (fewer blocks fit = lower occupancy).

## Production Recommendation

**V7 (contig8 general shuffle) is the new production kernel:**
- 105M room-qps at 1024 rooms, dim=256
- 117.5M room-qps at 1024 rooms, dim=128
- Works with any dim (general loop, no hardcoded stride)
- Zero shared memory, pure register communication
- Simple, clean kernel architecture

### Selection Rule (Updated from Suite #27)
| Batch Size | Recommended Kernel | Reason |
|-----------|-------------------|--------|
| ≤ 32 | V4 (shmem 4r/block) | Lower launch overhead |
| ≥ 64 | **V7 (contig8 general)** | Register scheduling freedom |
| ≥ 64, dim ≤ 128 | V7 (117M+ room-qps) | Memory-bound, shuffle dominates |

## All-Time Records

| Metric | Previous | New | Improvement |
|--------|----------|-----|-------------|
| Max room-qps | 93.8M (suite #28, 1024r) | **117.5M** (suite #30, 1024r, dim=128) | 1.25× |
| Max room-qps (dim=256) | 93.8M (suite #28) | **105.0M** (suite #30, V7) | 1.12× |
| Max room-qps (dim=512) | 81.8M (suite #20) | **82.8M** (suite #30, V7) | 1.01× |
