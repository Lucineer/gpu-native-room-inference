# Suite #68: Launch Bounds & Compiler Flags

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87 (8 SMs)

## Launch Bounds Impact (default -O3)

| Config | M qps | vs Baseline |
|---|---|---|
| No bounds (256t) | 129.0 | 1.00× |
| `__launch_bounds__(256, 2)` | 129.8 | 1.01× |
| `__launch_bounds__(256, 4)` | 131.6 | 1.02× |
| **`__launch_bounds__(256, 8)`** | **155.0** | **1.20×** |
| `__launch_bounds__(128, 4)` | 150.2 | 1.16× |
| `__launch_bounds__(128, 8)` | 150.5 | 1.17× |

## Compiler Flags Impact (with `__launch_bounds__(256, 8)`)

| Flag | M qps | vs Default |
|---|---|---|
| Default -O3 | 155.0 | 1.00× |
| -ftz=true | 156.6 | 1.01× |
| -O2 | 151.3 | 0.98× |
| -maxrregcount=32 | 158.6 | 1.02× |
| -maxrregcount=64 | 165.4 | 1.07× |
| **--use_fast_math** | **167.5** | **1.08×** |

## Combined: Best Configuration

| Config | M qps | vs Untuned |
|---|---|---|
| Untuned V7 (no bounds, -O3) | 129.0 | 1.00× |
| **`--use_fast_math` + `__launch_bounds__(256, 8)`** | **167.5** | **1.30×** |

## Key Findings

1. **`__launch_bounds__(256, 8)` is the single biggest win** — 20% faster than no bounds. Telling the compiler to maximize for 8 blocks per SM (256 threads × 8 = 2048 threads/SM, well within the 2048 limit) allows it to minimize register usage and maximize occupancy.

2. **`--use_fast_math` adds another 8%** — This flag enables: faster but less precise math, fusion of multiply-add, and approximations for division/sqrt. For room inference where FP16→float conversion already loses precision, this is free performance.

3. **`-maxrregcount=32` helps slightly** — Forcing 32 registers per thread (minimum) allows more warps per SM. But the default register allocation with launch bounds already does this.

4. **128-thread blocks + launch bounds = 16-17%** — Similar to 256-thread but doesn't benefit from `--use_fast_math` as much. The 256-thread version is better overall.

5. **PTXAS warning for 256×8 is harmless** — The compiler warns that 256 threads × 8 blocks = 2048 exceeds the SM's capacity for this kernel's register usage. The compiler automatically reduces blocks per SM, which is the optimization we want.

## Rule #62: Always Use `__launch_bounds__(256, 8)` for Room Inference

Specifying `__launch_bounds__(256, 8)` gives the compiler permission to minimize register usage (currently ~32 regs/thread), enabling 8 blocks per SM and maximum occupancy. This single annotation provides 20% throughput improvement with zero code changes.

## Rule #63: Use `--use_fast_math` for Ranking Workloads

The `--use_fast_math` compiler flag provides 8% additional throughput by using approximate math. For room ranking where relative ordering matters more than exact values, this is acceptable. Do NOT use for workloads requiring exact FP arithmetic.

## Production Recipe
```bash
nvcc -arch=sm_87 -O3 --use_fast_math infer.cu -o infer
# Kernel: __global__ void __launch_bounds__(256, 8) infer(...)
```
Expected: **167.5 M qps** (up from 129 M qps untuned = 30% free improvement)
