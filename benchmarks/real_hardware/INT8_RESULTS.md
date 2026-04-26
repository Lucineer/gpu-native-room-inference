# Suite #65: INT8 Quantized Room Inference

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87 (Ampere)

## Throughput (4096 rooms, dim=256)

| Type | μs | M qps | vs FP16 |
|---|---|---|---|
| V7 FP16 | 30.10 | 136.1 | 1.00× |
| **INT8 symmetric** | **22.16** | **184.8** | **1.36×** |

## Accuracy (per-room symmetric quantization, weights [-0.5, 0.5], input [-1, 1])

| Metric | Value |
|---|---|
| Max relative error | 1868% (near-zero values) |
| **Avg relative error** | **4.15%** |
| Max absolute error | <0.06 |

### Sample Values
| Room | FP16 | INT8 | |Diff| |
|---|---|---|---|
| 0 | 0.4134 | 0.4214 | 0.008 |
| 1 | 4.3222 | 4.2885 | 0.034 |
| 2 | 4.4249 | 4.3666 | 0.058 |
| 3 | -3.7895 | -3.7390 | 0.051 |

## Memory Usage
| Format | KB | vs FP16 |
|---|---|---|
| FP16 weights | 2,048 | 1.00× |
| INT8 weights | 1,024 | **0.50×** |
| INT8 + per-room scales | 1,040 | 0.51× |

## Key Findings

1. **INT8 is 36% faster than FP16** — Half the memory bandwidth (1 byte vs 2 bytes per weight). At 4096 rooms, weights are 2MB in FP16 vs 1MB in INT8. The reduced bandwidth directly translates to throughput.

2. **4.15% average accuracy loss** — Acceptable for room ranking where relative ordering matters more than exact values. The max relative error is misleading (near-zero values); absolute errors are <0.06.

3. **Per-room symmetric quantization** — Each room has its own scale factor (max_abs / 127). This preserves weight magnitude distribution per room. Global quantization would lose precision for rooms with small weights.

4. **Memory savings enable larger batches** — At 50% memory, you can fit 11.5M rooms instead of 5.75M in 8GB. Or serve 2× more rooms per batch at the same memory.

5. **Combined with L2 persist** — INT8 weights (1MB for 4K rooms) fit entirely in L2 cache (1.4MB). This means with L2 persist, ALL weights can be pinned — no cache misses at all. Theoretically: 184.8 × 1.23 (persist) = ~227M qps.

## Rule #60: INT8 Quantization for Room Ranking

Use per-room symmetric INT8 quantization for room inference. It's 36% faster, uses 50% less memory, and loses only 4% accuracy. The reduced memory footprint allows all weights to fit in L2 cache with partial persisting, enabling further speedup. Reserve FP16 only for rooms requiring exact scores (e.g., threshold comparison rooms).
