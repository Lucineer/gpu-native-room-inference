# Suite #48: Multi-Layer Fusion Scaling

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Part 1: Fused vs Unfused (1024 rooms, dim=256)

| Layers | Fused (μs) | Unfused (μs) | Speedup | OI | BW (GB/s) |
|--------|-----------|--------------|---------|-----|-----------|
| 1 | 15.26 | 12.84 | 0.84× | 2.0 | 34.4 |
| 2 | 15.51 | 19.97 | **1.29×** | 4.0 | 67.7 |
| 3 | 20.19 | 29.40 | **1.46×** | 6.0 | 77.9 |
| 4 | 33.97 | 43.32 | 1.28× | 8.0 | 61.8 |
| 5 | 40.06 | 56.39 | **1.41×** | 10.0 | 65.5 |
| 6 | 46.52 | 67.67 | **1.45×** | 12.0 | 67.6 |
| 7 | 53.01 | 78.92 | **1.49×** | 14.0 | 69.2 |
| 8 | 60.01 | 90.29 | **1.50×** | 16.0 | 69.9 |

## Part 2: Fusion Benefit by Dimension (4 layers)

| Dim | Fused (μs) | Unfused (μs) | Speedup |
|-----|-----------|--------------|---------|
| **64** | 16.84 | 32.82 | **1.95×** |
| 128 | 19.13 | 31.43 | 1.64× |
| 256 | 30.39 | 41.31 | 1.36× |
| 512 | 56.21 | 71.42 | 1.27× |

## Key Findings

1. **Fusion plateaus at 1.50×** — far less than suite #18's 3.69×. The nlayer kernel's loop overhead and register pressure eat into savings.
2. **1 layer is slower fused** (0.84×) — general-purpose nlayer kernel has overhead vs specialized 1-layer kernel.
3. **Fusion helps MORE at small dimensions** — 1.95× at dim=64 vs 1.27× at dim=512. Launch overhead dominates at small dims.
4. **Bandwidth utilization scales with layers** — 34 GB/s (1L) → 70 GB/s (8L). Fusion keeps the memory bus busier.
5. **No register spilling observed** — 8 layers fused still works correctly on sm_87.

## Rule #34: Fuse at Small Dimensions, Specialize at Large
For dim ≤ 128, fusion gives 1.6-2.0× speedup (launch overhead dominant). For dim ≥ 512, fusion gives only 1.3× (memory dominates). At large dimensions, prefer kernel specialization over generic fusion.
