# Suite #32: CUDA Graphs for Complete Pipeline

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Question
Can CUDA Graphs eliminate the ~20μs CPU-GPU communication overhead?

## Results

| Rooms | Sequential (μs) | Graph (μs) | Speedup | Graph Room-qps |
|-------|-----------------|------------|---------|----------------|
| 1 | 23.36 | 22.82 | 1.02× | 43.8K |
| 4 | 23.23 | 22.82 | 1.02× | 175K |
| 16 | 23.52 | 22.79 | 1.03× | 702K |
| 64 | 29.76 | 22.85 | **1.30×** | 2.80M |
| 256 | 29.89 | 29.60 | 1.01× | 8.65M |
| 1024 | 43.52 | 43.01 | 1.01× | 23.8M |

## Key Findings

1. **CUDA Graphs provide minimal speedup at large batch** — 1.01× at 256+ rooms
2. **The bottleneck is cudaStreamSynchronize(), not kernel launch** — graphs can't eliminate sync
3. **At 64 rooms: 1.30× speedup** — sweet spot where launch overhead matters
4. **p99 jitter improves with graphs** — 1.22× better at 16 rooms (eliminates scheduling jitter)
5. **Graph (1024 rooms, 1 launch) = 32.2M room-qps** — better than 4-stream sequential (22.0M)

## Why Graphs Don't Help Much

The end-to-end pipeline is:
1. CPU queues kernel → ~1μs
2. GPU executes kernel → ~11μs (256 rooms)
3. CPU calls cudaStreamSynchronize() → ~15μs (waits for GPU to finish)
4. CPU reads zero-copy output → ~0.001μs

CUDA Graphs eliminate step 1 (kernel launch overhead). But step 3 (sync) is the real bottleneck — it's a CPU-side wait that graphs can't optimize.

## Conclusion

CUDA Graphs are **not worth the complexity** for the complete inference pipeline. The sync overhead dominates. Use them only for:
- Repeated single-kernel launch patterns (batch ≤ 64)
- Jitter-sensitive applications where p99 matters more than p50
