# Suite #60: Custom Kernel vs cuBLAS vs cuSPARSE

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87 (Ampere)

## Head-to-Head (4096 rooms, dim=256)

| Method | μs | M qps | vs V7 |
|---|---|---|---|
| **V7 Custom** | **43.45** | **94.3** | **1.00×** |
| cuBLAS GEMM | 55.94 | 73.2 | 0.78× |
| cuSPARSE 50% sparse | 60.99 | 67.2 | 0.71× |
| cuSPARSE dense CSR | 116.74 | 35.1 | 0.37× |

## cuBLAS vs V7 at Various Batch Sizes

| Rooms | V7 μs | cuBLAS μs | cuBLAS/V7 |
|---|---|---|---|
| 64 | 5.13 | 7.45 | 1.45× (V7 wins) |
| 256 | 6.61 | 10.51 | 1.59× (V7 wins) |
| 1,024 | 10.22 | 10.93 | 1.07× (V7 wins) |
| 4,096 | 30.92 | 33.88 | 1.10× (V7 wins) |
| **8,192** | **68.74** | **54.46** | **0.79× (cuBLAS wins)** |

## Key Findings

1. **Custom V7 kernel beats cuBLAS at all practical batch sizes** — up to 1.59× faster at 256 rooms. cuBLAS only wins at 8K+ rooms where tensor cores have enough work to amortize overhead.

2. **cuSPARSE is catastrophically slow** — Dense CSR SpMV is 2.7× slower than custom. The SpMV framework overhead (format parsing, index indirection, scatter-gather) dominates for small matrices.

3. **50% sparse SpMV barely helps** — Despite half the data, SpMV is still 40% slower than custom dense. The sparse format overhead (indices + indirection) cancels out the memory savings.

4. **Sparse memory actually LARGER** — 3088 KB sparse vs 2048 KB dense! CSR needs 4-byte indices per nonzero, so 50% sparse = 50% weights (1KB) + 50% indices (1KB) + row pointers (16KB) = larger than dense (2KB). The "memory savings" from Suite #52 were misleading — only COO format saves memory for extreme sparsity.

5. **The crossover is at ~6000 rooms** — Below this, custom kernel wins. Above, cuBLAS tensor cores take over. For the fleet's standard 4K rooms, custom is clearly faster.

## Rule #51: Custom Kernels Beat Libraries for Room Inference

At dim=256 and ≤6000 rooms, hand-written CUDA kernels are 10-59% faster than cuBLAS and 40-170% faster than cuSPARSE. cuBLAS tensor cores only win at large batch sizes (8K+) where they can amortize launch and setup overhead. For edge inference with small matrices, the V7 warp-shuffle kernel is optimal. Library calls add framework overhead that dominates compute for small matrix-vector products.

## Rule #52: CSR Sparsity is Counterproductive on Jetson

Storing sparse weights in CSR format requires 4-byte column indices per nonzero, making 50% sparse matrices actually 50% LARGER in memory than dense. The index indirection also destroys L2 cache locality. Only COO or bitmap formats save memory, and only at >90% sparsity (see Suite #52 Rule #37). For structured 2:4 sparsity on Jetson, stay dense.
