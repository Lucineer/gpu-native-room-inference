# Suite #33: Async Pipeline — Eliminating the Sync Barrier

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## The Biggest Finding

**Double-buffered async inference reaches 104.1M room-qps** — matching kernel-only throughput.
The `cudaStreamSynchronize()` barrier was hiding 80% of the GPU's capability.

## Results

| Method | 64 rooms | 256 rooms | 1024 rooms |
|--------|----------|-----------|------------|
| Synchronous (cudaStreamSynchronize) | 2.2M qps (28.8μs) | 8.6M qps (29.8μs) | 23.7M qps (43.3μs) |
| Fire-and-forget (queue only) | 6.75μs | 6.75μs | 9.74μs |
| **Double-buffered pipeline** | **12.5M qps (5.1μs)** | **42.5M qps (6.0μs)** | **104.1M qps (9.8μs)** |
| GPU-only (events) | 5.28μs | 5.81μs | 9.87μs |
| **Speedup (double vs sync)** | **5.6×** | **5.0×** | **4.4×** |

## How Double-Buffering Works

```
Iteration N:   Launch kernel → buffer A    | Read buffer B (prev result)
Iteration N+1: Launch kernel → buffer B    | Read buffer A (prev result)
```

The key: reading the previous result is a zero-copy CPU memory access (~0.001μs).
The GPU kernel on the current buffer runs independently. No sync needed.

## Key Findings

1. **cudaStreamSynchronize adds ~20μs per iteration** — it's a CPU-side block waiting for GPU fence
2. **The GPU can accept work at 6.75μs/iter** (fire-and-forget) — 150M+ room-qps queue rate
3. **Double-buffered matches GPU-only timing** — 9.8μs vs 9.87μs (events) at 1024 rooms
4. **Zero-copy read is instant** — 0.001μs, no GPU interaction needed
5. **The entire overhead was CPU-side synchronization**, not GPU computation

## Production Architecture

For a fleet inference server:
1. Two output buffers (ping-pong)
2. Launch kernel into current buffer
3. Read results from previous buffer (zero-copy)
4. Never call cudaStreamSynchronize in the hot path
5. Sync only on shutdown or buffer overflow

This changes the production room-qps from 23.7M → 104.1M at 1024 rooms. A **4.4× free speedup** from better software architecture.
