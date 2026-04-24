# TC GEMM Benchmark Results
**Date:** 2026-04-24 09:10 AKDT
**Hardware:** Jetson Orin Nano 8GB, sm_87, CUDA 12.6, Tensor Cores

## Results

| Matrix Size | TC WMMA (ms) | cuBLAS (ms) | TC GFLOPS | cuBLAS GFLOPS | TC/cuBLAS |
|-------------|-------------|-------------|-----------|--------------|-----------|
| 16x16 | 0.044 | 0.023 | 0.2 | 0.4 | 0.52x |
| 32x32 | 0.059 | 0.019 | 1.1 | 3.4 | 0.33x |
| 64x64 | 0.062 | 0.017 | 8.5 | 30.3 | 0.28x |
| 128x128 | 0.084 | 0.016 | 49.7 | 256.0 | 0.19x |
| 256x256 | 0.344 | 0.018 | 97.6 | 1,869.3 | 0.05x |
| 64x128x256 | 0.081 | 0.015 | 52.0 | 275.5 | 0.19x |
| 256x128x64 | 0.081 | 0.015 | 52.0 | 275.5 | 0.19x |

## Analysis

### cuBLAS is extremely optimized
cuBLAS achieves 1,869 GFLOPS on 256x256 (4.7% of theoretical 40 TFLOPS peak). It uses:
- Multi-CTA tiling
- Shared memory tiling across K dimension
- Software pipelining
- Tensor core instructions with optimal register allocation

### Our TC kernel is naive — by design
Our kernel assigns one warp per 16x16 output tile. For 256x256, that's 256 warps — far more than the 8 SMs can handle efficiently. The kernel launch overhead alone dominates.

### Where TC actually wins
Tensor cores don't win on small matrices. They win on:
1. **Large batch sizes** — amortize launch overhead
2. **Large K dimension** — more WMMA operations per launch
3. **Multiple tiles per CTA** — better SM utilization

### Key lesson for deckboss
For room inference (small matrices), **don't use custom TC kernels** — use cuBLAS or TensorRT. They're already optimal. Custom TC kernels only make sense for specialized workloads that don't fit standard GEMM patterns.

## Next Steps
- Test CUDA Graphs to eliminate launch overhead (could make TC competitive at small sizes)
- Test with batched GEMM (cuBLAS strided batched)
- Focus edge optimization effort on orchestration, not kernel writing
