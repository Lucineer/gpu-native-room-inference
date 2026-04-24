# CUDA Graphs Benchmark — Jetson Orin Nano
**Date:** 2026-04-24 09:20 AKDT
**Hardware:** Jetson Orin Nano 8GB, sm_87, CUDA 12.6

## Results

| Test | Latency | QPS | vs Standard |
|------|---------|-----|-------------|
| Standard (single kernel) | 7.7 μs | 129,109 | 1.00x |
| CUDA Graph (single kernel) | 6.7 μs | 149,449 | 1.16x |
| CUDA Graph (pipeline: infer + scale) | 6.7 μs | 148,545 | 1.15x |
| Standard (pipeline: infer + scale) | 9.0 μs | 110,587 | 0.86x |

## Key Findings

### 1. CUDA launch overhead is ~1-2μs per kernel
Not 35μs as estimated from TensorRT numbers. The TRT overhead includes framework overhead (memory management, tensor conversion) on top of CUDA launch.

### 2. Pipeline graph eliminates inter-kernel overhead
Standard pipeline (2 kernels): 9.0μs
Graph pipeline (2 kernels): 6.7μs
**Speedup: 1.34x**

The second kernel's launch overhead is completely eliminated in the graph.

### 3. Single-kernel graphs: marginal improvement
1.16x for a single kernel — not worth the complexity for simple cases.

### 4. The real win is multi-kernel pipelines
For production workloads with 3-5 kernels per inference (matmul → activation → normalization → output), CUDA Graphs could give:
- 2 kernels: 1.34x
- 3 kernels: ~1.5x (estimated)
- 5 kernels: ~2x (estimated)

### 5. TensorRT's 0.041ms vs our CUDA's 0.007ms
TensorRT adds ~34μs of framework overhead per inference call. CUDA Graphs alone can't close this gap — it's the TensorRT runtime, not CUDA launch.

## Implications for deckboss

- **CUDA Graphs are worth it** for multi-step inference pipelines
- **TensorRT's framework overhead dominates** for small models — consider raw CUDA + graphs for latency-critical paths
- **Hybrid approach**: TRT for model flexibility, CUDA Graphs for hot-path room inference
- **129K qps raw CUDA** vs **17K qps TRT** — the gap is all framework overhead
