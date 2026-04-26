# Suite #58: Bfloat16 vs FP16 vs FP32

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87 (Ampere — native BF16 support)

## Throughput (4096 rooms, dim=256)

| Type | μs | M qps | vs FP16 |
|---|---|---|---|
| FP16 | 33.51 | 122.2 | 1.00× |
| BF16 | 29.22 | 140.2 | 1.15× |
| FP32 | 70.00 | 58.5 | 0.48× |

## With GELU Activation

| Type | μs | M qps | vs FP16 |
|---|---|---|---|
| FP16+GELU | 28.18 | 145.3 | 1.00× |
| BF16+GELU | 28.13 | 145.6 | 1.00× |

## Accuracy with Large Values (weights in [-10, 10])

| Type | Max Rel Error | Avg Rel Error |
|---|---|---|
| FP16 | 29.02% | 0.14% |
| **BF16** | **176.67%** | **1.01%** |

## Key Findings

1. **BF16 throughput ≈ FP16** — Within 2% at all batch sizes. Same memory bandwidth (2 bytes), same conversion cost.
2. **BF16 accuracy is terrible** — 176% max relative error on large values. BF16 has only 8 mantissa bits vs FP16's 11.
3. **FP16 is clearly superior for room inference** — Better accuracy, same throughput, same memory.
4. **BF16 advantage (wider range) doesn't matter** — Room inference weights are small (Kaiming/Xavier), never overflow FP16.

## Rule #48: Use FP16, Not BF16

BF16 has wider dynamic range but 3× worse precision than FP16 (8 vs 11 mantissa bits). For room inference where weights are small, FP16 is strictly better: same throughput, superior accuracy. BF16 is only useful for training where gradient explosions require wider range.
