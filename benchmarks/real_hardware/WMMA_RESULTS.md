# Suite #37: Tensor Core WMMA for Room Inference — Negative Result

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Question
Can WMMA (Warp Matrix Multiply Accumulate) Tensor Cores accelerate room dot products?

## Results

| Rooms | V7 (μs) | WMMA (μs) | V7/WMMA |
|-------|---------|-----------|---------|
| 16 | 8.43 | 19.75 | **0.43×** (WMMA 2.3× slower) |
| 64 | 5.37 | 15.59 | **0.34×** (WMMA 2.9× slower) |
| 256 | 5.80 | 24.17 | **0.24×** (WMMA 4.2× slower) |
| 1024 | 9.40 | 80.95 | **0.12×** (WMMA 8.5× slower) |

## Why WMMA Fails for Dot Products

1. **Tensor cores are 16×16×16** — a dim=256 dot product uses only 1/16th of the hardware
2. **Matrix-vector is the wrong shape** — we compute W[M×K] × x[K×1], but WMMA wants W[M×K] × X[K×K]
3. **B fragment broadcast wastes cycles** — 15 out of 16 rows are duplicates
4. **Fragment manipulation overhead** — manual B fill is slower than direct FP16 multiply
5. **Numerical error: 0.043** — slightly different precision (acceptable but pointless)

## Conclusion

**Tensor Cores don't help for room inference.** The workload is a simple dot product, not a matrix multiplication. Regular FP16 arithmetic (V7 kernel) is the right tool. Tensor Cores are for batched GEMM (large M, K, N) — not for matrix-vector products.

**Confirmed:** Suite #5's finding that custom TC kernels are 19× slower than cuBLAS extends to dot products too.
