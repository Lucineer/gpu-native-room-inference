# Suite #55: Double-Buffered Weight Pipeline

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87
**Previous:** Rule #41 says avoid memory ops during inference

## Part 1: Weight Upload Strategies (1024 rooms, dim=256, continuous updates)

| Strategy | Total μs | Upload μs | Infer μs | M qps |
|---|---|---|---|---|
| Naive (sequential) | 187.34 | 149.07 | 38.27 | 5.5 |
| Double-buffer | 181.34 | overlapped | overlapped | 5.6 |
| Triple-buffer | 181.42 | overlapped | overlapped | 5.6 |
| **No upload (baseline)** | **16.38** | **0.00** | **16.38** | **62.5** |

## Part 2: Partial Weight Update (subset of rooms change)

| Update % | Total μs | M qps | vs Full |
|---|---|---|---|
| 1% | 29.63 | 34.6 | 6.1× |
| 5% | 36.12 | 28.4 | 5.0× |
| 10% | 44.44 | 23.0 | 4.1× |
| 25% | 67.41 | 15.2 | 2.7× |
| 50% | 96.88 | 10.6 | 1.9× |
| 100% | 180.25 | 5.7 | 1.0× |

## Part 3: Upload Cost at Various Batch Sizes

| Rooms | Naive μs | Upload μs | Weight KB |
|---|---|---|---|
| 64 | 14.91 | 4.97 | 32 |
| 256 | 21.08 | 17.23 | 128 |
| 1,024 | 76.98 | 47.73 | 512 |
| 4,096 | 187.71 | 178.58 | 2,048 |
| 16,384 | 1,009.68 | 844.56 | 8,192 |

## Key Findings

1. **Double-buffering provides only 3.2% speedup** — Jetson's unified memory architecture already overlaps H2D transfers with GPU compute via the hardware page migration engine. Explicit double-buffering adds negligible value.

2. **Weight upload dominates: 3.9× slower than inference** — Uploading 512KB (1024 rooms × dim=256) takes 149μs vs 38μs for inference. The fleet is upload-bound, not compute-bound, when weights change every batch.

3. **Pure inference is 11.4× faster** — 62.5M qps with cached weights vs 5.5M qps with per-batch uploads. This is the most important number: keeping weights on GPU is the #1 optimization.

4. **Partial updates scale nearly linearly** — Updating 1% of rooms costs 29.6μs (6.1× faster than full update). Selective update is the practical solution.

5. **Upload cost scales linearly with batch size** — 5μs for 64 rooms, 845μs for 16K rooms. At large batches, upload becomes the sole bottleneck.

6. **Triple-buffer ≈ double-buffer** — Two-ahead provides zero additional benefit. The memory bus is the bottleneck, not pipeline depth.

## Rule #43: Cache Weights on GPU, Upload Only Deltas

Never upload all weights every batch. Keep a persistent weight buffer on GPU and only upload rooms that changed. 1% partial update is 6.1× faster than full upload. Combined with pure inference (62.5M qps cached vs 5.5M qps with upload), selective delta updates are the most impactful fleet optimization.

## Rule #44: Double-Buffering Doesn't Help on Unified Memory

Jetson's shared CPU/GPU memory already overlaps transfers with compute. Explicit double-buffering provides only 3.2% speedup. Don't add pipeline complexity for negligible gain. Instead, minimize upload volume (Rule #43).
