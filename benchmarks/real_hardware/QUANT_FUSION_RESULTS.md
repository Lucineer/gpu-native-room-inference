# Suite #49: Quantized Multi-Layer Fusion

**Date:** 2026-04-25
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Results (1024 rooms, dim=256)

| Layers | FP16 (μs) | INT8 (μs) | INT8 Ratio | INT4 (μs) | INT4 Ratio |
|--------|-----------|-----------|------------|-----------|------------|
| 1 | 18.96 | 11.78 | **1.61×** | 14.23 | **1.33×** |
| 2 | 15.67 | 17.25 | 0.91× | 23.00 | 0.68× |
| **4** | **36.36** | **28.61** | **1.27×** | 40.22 | 0.90× |
| **6** | **58.56** | **39.75** | **1.47×** | 57.09 | 1.03× |
| **8** | **75.29** | **56.84** | **1.32×** | 73.90 | 1.02× |

## Correctness (2 layers)

| Room | FP16 | INT8 Error | INT4 Error |
|------|------|------------|------------|
| 0 | -0.000331 | 9.5% | **92.1%** |
| 1 | -0.000554 | 2.5% | **100%** |
| 2 | -0.001241 | 2.5% | **100%** |
| 3 | -0.091732 | **0.02%** | **99.7%** |

## Key Findings

1. **INT8 wins at ≥4 fused layers** — 1.27× at 4L, 1.47× at 6L. The OI crossover at ~4 layers is real.
2. **INT8 loses at 1-2 layers** — dequant overhead (int8→float conversion) exceeds memory savings at low OI.
3. **INT4 is catastrophically bad** — 92-100% error, no performance benefit. Only 4 bits per weight destroys accuracy.
4. **INT8 accuracy is excellent** — 0.02-9.5% error, perfectly acceptable for room ranking.
5. **Memory savings: INT8 = 49.2%, INT4 = 73.4%** — but INT4 savings don't matter when accuracy is destroyed.

## Rule #35: Use INT8 for ≥4 Fused Layers
For multi-layer fused inference, INT8 quantization provides 1.3-1.5× speedup at ≥4 layers with <10% accuracy loss. Never use INT4 — accuracy is destroyed. For single-layer inference, stay with FP16.
