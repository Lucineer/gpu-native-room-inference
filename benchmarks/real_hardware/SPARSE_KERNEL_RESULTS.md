# Suite #52: Sparse Kernel — Skip Zero Weights

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87
**Previous:** Suite #51 showed dense kernel ignores sparsity (1.00× across all levels)

## Part 1: Sparse vs Dense Across Sparsity Levels (1024 rooms, dim=256)

| Sparsity | Dense (μs) | Sparse v1 | v1 Ratio | Coop v2 | v2 Ratio | Sorted SoA | SoA Ratio |
|---|---|---|---|---|---|---|---|
| 0% | 13.90 | 19.82 | 0.701× | 20.37 | 0.682× | 13.28 | 1.046× |
| 25% | 9.83 | 14.35 | 0.685× | 14.86 | 0.662× | 13.36 | 0.736× |
| 50% | 9.85 | 12.94 | 0.761× | 13.06 | 0.754× | 12.63 | 0.780× |
| 75% | 9.83 | 10.85 | 0.906× | 10.32 | 0.952× | 10.83 | 0.907× |
| **90%** | **9.83** | **9.32** | **1.054×** | **8.99** | **1.093×** | **9.34** | **1.052×** |
| **95%** | **9.79** | **8.75** | **1.119×** | **8.53** | **1.148×** | **8.76** | **1.118×** |
| **99%** | **9.79** | **8.68** | **1.128×** | **8.43** | **1.162×** | **8.73** | **1.121×** |

## Part 2: Memory Usage

| Sparsity | nnz/room | Sparse Mem | Dense Mem | Savings |
|---|---|---|---|---|
| 0% | 256 | 1540 KB | 512 KB | 0.3× (worse) |
| 50% | 128 | 772 KB | 512 KB | 0.7× |
| 75% | 64 | 388 KB | 512 KB | 1.3× |
| 90% | 25 | 154 KB | 512 KB | 3.3× |
| 95% | 12 | 76 KB | 512 KB | 6.7× |
| 99% | 2 | 16 KB | 512 KB | **32.0×** |

## Part 3: Correctness (50% sparse)

| Room | Dense | Sparse | Diff |
|---|---|---|---|
| 0 | -0.00450637 | -0.00450637 | 1.86e-09 |
| 1 | 0.00273731 | 0.00273731 | 9.31e-10 |
| 2 | 0.00114393 | 0.00114392 | 1.86e-09 |
| 3 | -0.01275562 | -0.01275561 | 1.86e-09 |

## Key Findings

1. **Dense wins below ~87% sparsity** — random index lookups destroy L2 cache locality that dense sequential access enjoys
2. **Sparse crossover at ~87% sparsity** — only then does skipping zero compute outweigh index lookup cost
3. **Cooperative (v2) is best sparse kernel** — warp-cooperative iteration gets 1.162× at 99% vs 1.128× for naive
4. **Sorted SoA (v3) worst for sparse** — despite being 1.046× at 0% (extra padding helps dense access), it degrades faster than AoS
5. **Maximum speedup is modest: 1.16×** — even at 99% sparsity, indirect access overhead limits gains
6. **Memory savings are the real win** — 32× less memory at 99% sparsity enables more rooms on-device
7. **Correctness excellent** — FP16 error ~1e-9 (near machine epsilon)

## Rule #37: Use Dense Unless >90% Sparse

Dense V7 kernel dominates at typical sparsity levels. Sparse kernels only justify their index lookup overhead at >90% sparsity (1.05-1.16×). The real value of sparsity is memory savings (up to 32×), not compute speedup. For heavily pruned MoE experts (>95% sparse), use cooperative sparse v2. For everything else, keep dense.
