# Suite #35: Adaptive Room Weights — Negative Result

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Question
Can we reduce room weight memory (128KB for 256 rooms × dim 256) by sharing base weights with per-room adaptations?

## Results

### Scale+Bias Model: 51× Memory Reduction, 128% Error
- Memory: 128KB → 2.5KB (51.2× reduction)
- **Accuracy: 128.57% relative error — unusable**
- The 2-parameter model (scale + bias) can't capture the full room weight diversity
- Latency: 8.25μs vs 9.88μs (faster, but accuracy makes this irrelevant)

### LoRA Model: No Memory Reduction
| Rank | Latency (μs) | Memory | vs Full |
|------|-------------|--------|---------|
| 1 | 10.46 | 132KB | **1.0× (worse!)** |
| 2 | 11.12 | 258KB | 0.5× (worse!) |
| 4 | 13.02 | 515KB | 0.25× (worse!) |

LoRA increases memory because `A[rooms][rank][dim]` is always larger than `W[rooms][dim]` for rank ≥ 1. Only the B vector is small.

### Adaptive Scaling: Slower Than Full Weights
| Rooms | Full (μs) | Adaptive (μs) | Ratio |
|-------|-----------|---------------|-------|
| 16 | 5.23 | 5.38 | 0.97× |
| 256 | 5.77 | 6.49 | **0.89×** |
| 1024 | 9.39 | 10.01 | 0.94× |

The extra memory loads (scale + bias per room) offset any savings from shared base weights.

## Conclusion

**Adaptive weight compression doesn't work for room inference.**

1. Scale+bias: too few parameters to maintain accuracy
2. LoRA: increases memory, not reduces it
3. Even when memory is reduced, latency is worse (extra indirection)
4. The full-weight model at 128KB fits in L2 cache anyway — no benefit from compression

**Keep full per-room weights.** 128KB for 256 rooms is already optimal.
