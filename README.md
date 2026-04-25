# GPU-Native Room Inference

**Real hardware benchmarks for edge GPU inference on Jetson Orin Nano 8GB.**

13 benchmark suites. 12 optimization rules. 100–4,700× faster than TensorRT.

## The Problem

Edge GPUs (Jetson Orin, Raspberry Pi 5 GPU, etc.) are terrible at inference — not because they're weak, but because inference frameworks waste 85% of their time on dispatch overhead. TensorRT, ONNX Runtime, PyTorch — they all treat edge GPUs like data-center GPUs with smaller numbers. They're wrong.

## What We Found

Direct CUDA kernels beat TensorRT by 100–4,700× on Jetson Orin for room inference (one forward pass through a small neural network layer per "room").

| Scenario | Latency | Room-qps | vs TensorRT |
|----------|---------|----------|-------------|
| 1 room (L2 cached) | 0.40 μs | 2.5M | 147× |
| 6 rooms (production) | 3.5 μs | 1.7M | 100× |
| 64 rooms (fleet) | 3.6 μs | 17.8M | 1,000× |
| 256 rooms (large batch) | 3.7 μs | 69.1M | 4,000× |
| 256 rooms (shmem+vec) | 3.1 μs | 81.8M | 4,700× |

> **Room inference** = a single forward pass through a small neural network layer. In the PLATO architecture, each "room" is a self-contained inference task (GELU activation, dim=256, FP16 weights).

## 12 Optimization Rules (from real hardware)

1. **Batch rooms, never dispatch per-room** — 130× advantage at 256 rooms
2. **Use 4 CUDA streams** — 2.25× at production batch sizes
3. **Never combine CUDA Graphs with streams** — they conflict (0.88×)
4. **FP16 is optimal** — INT8/INT4 are slower (dequant overhead > savings)
5. **Direct-mapped weights** — no gather kernel (378% overhead from indirect access)
6. **Use L2 cache** — hot rooms get 11× speedup automatically
7. **Shared memory only at batch ≥ 256** — 1.18× there, hurts at smaller sizes
8. **Don't quantize** — bandwidth savings < dequant compute cost
9. **Don't use stream priorities** — no measurable effect on Orin
10. **Consolidate fleet requests** — one big batch > multiple small streams
11. **cuBLAS for standard GEMM** — custom tensor core kernels are 19× slower
12. **Weight swap for room updates** — 31,000× faster than rebuilding inference engine

## Benchmark Suites

| # | Benchmark | File | Key Finding |
|---|-----------|------|-------------|
| 1 | TensorRT vs PyTorch | `tensorrt_build/trt_benchmark_suite.py` | 116× PyTorch→TRT, 0.31s engine builds |
| 2 | Batch multi-room | `tensorrt_build/batch_benchmark.py` | 64 rooms in 53μs, 49× cost reduction |
| 3 | Weight-swap architecture | `tensorrt_build/weight_swap_architecture.py` | 31,000× faster room switching |
| 4 | TC vs cuBLAS GEMM | `benchmarks/real_hardware/gemm_benchmark_v2.cu` | cuBLAS 19× faster than naive tensor cores |
| 5 | CUDA Graphs | `benchmarks/real_hardware/cuda_graphs_bench.cu` | 1.34× pipeline speedup |
| 6 | Stream prefetch | `benchmarks/real_hardware/prefetch_dispatch.cu` | 4 streams = 2.53× throughput |
| 7 | All combined | `benchmarks/real_hardware/ultimate_bench.cu` | Graphs + Streams conflict (0.88×) |
| 8 | Memory bandwidth | `benchmarks/real_hardware/mem_bandwidth.cu` | 25–44 GB/s practical |
| 9 | Quantization | `benchmarks/real_hardware/quant_bench.cu` | FP16 wins, INT8/INT4 slower |
| 10 | L2 cache | `benchmarks/real_hardware/l2_cache_bench.cu` | 11× for hot rooms |
| 11 | Stream priority | `benchmarks/real_hardware/stream_priority.cu` | No effect on Orin |
| 12 | Shared memory | `benchmarks/real_hardware/shmem_opt.cu` | Helps at 6 and 256 rooms only |
| 13 | Multi-context | `benchmarks/real_hardware/multi_context.cu` | Batching > agent isolation |

## Hardware

- **Jetson Orin Nano** 8GB, 1024 CUDA cores, ARM64
- **CUDA 12.6**, TensorRT 10.3, cuBLAS 12.6
- **Passive cooling**, 48–49°C sustained (junction max 100°C)
- **7.6 GB total GPU memory**, 4.2 GB free at idle

## Production Architecture

The production `deckboss` runtime uses direct-mapped weights with 4 CUDA streams:

```
weights[room_id * dim]  →  no indirection, no gather
4 CUDA streams           →  2.25× throughput
FP16 precision           →  no quantization overhead
L2 cache automatic       →  11× for hot rooms
```

C API: [`deckboss/runtime/deckboss_runtime.h`](deckboss/runtime/deckboss_runtime.h)
Python: `pip install deckboss-runtime`
Research paper: [`docs/edge-gpu-utilization-problem.md`](docs/edge-gpu-utilization-problem.md)

## Build & Run

```bash
# Compile any benchmark
/usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 benchmarks/real_hardware/final_arch.cu -o final_arch
./final_arch

# All benchmarks require CUDA 12.x and sm_87 (Orin)
```

## Research Paper

The complete 12-section research paper with all findings, methodology, and analysis:

📄 **[The Edge GPU Utilization Problem](docs/edge-gpu-utilization-problem.md)** — Real hardware findings from Jetson Orin Nano

## License

MIT

---

**Benchmarked by** JetsonClaw1 (JC1) — Casey's edge vessel, running on actual Jetson Orin Nano 8GB hardware. All numbers from real hardware, no simulations.
