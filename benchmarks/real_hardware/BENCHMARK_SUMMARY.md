# Jetson Orin Nano Real Hardware Benchmarks

## GPU: Orin sm_87, 8 SMs, 1020 MHz, 7620 MB unified

## Results Summary

### Room Inference (12 rooms × 16 neurons × 256 weights)

| Implementation | Latency | QPS | GFLOPS | vs Baseline |
|---|---|---|---|---|
| Thread (256 thr) | 0.0771 ms | 12,973/s | 1.28 | — |
| Warp shuffle (baseline) | 0.0174 ms | 57,444/s | 5.66 | 1.00× |
| Warp + shared mem | 0.0176 ms | 56,685/s | 5.60 | 0.99× |
| **Warp + half2 vectorized** | **0.0143 ms** | **69,704/s** | **6.85** | **1.21×** |
| Warp + reg blocking | 0.0150 ms | 66,672/s | 6.57 | 1.16× |
| TC mat-vec (16 WMMA) | 0.0551 ms | 18,149/s | 1.78 | 0.31× |
| TC mat-vec + GELU (2 kern) | 0.0605 ms | 16,529/s | 1.63 | 0.29× |

**Winner: Warp + half2 vectorized (1.21× over baseline)**

### Matrix Multiply (16×K × K×16) — TC Domain

| K | Warp (ms) | TC (ms) | Speedup | TC GFLOPS |
|---|---|---|---|---|
| 16 | 0.0944 | 0.0054 | **17.48×** | 1.5 |
| 32 | 0.0875 | 0.0066 | **13.26×** | 2.5 |
| 64 | 0.1079 | 0.0094 | **11.46×** | 3.5 |
| 128 | 0.1241 | 0.0147 | **8.42×** | 4.4 |
| 256 | 0.1492 | 0.0252 | **5.93×** | 5.2 |

**Winner: Tensor Core (store_matrix_sync) — always faster for matmul**

### Key Findings

1. **WMMA + tanh compiler bug**: WMMA intrinsics and tanh() in same kernel cause incorrect results at -O2/-O3. Fix: separate kernels.

2. **Fragment layout**: 16×16 result distributed across 32 lanes, 8 elements each. `store_matrix_sync` handles extraction correctly.

3. **TC for mat-vec**: Not worth it. Shared memory load overhead dominates for 256-element dot products.

4. **TC for matmul**: Dominant. `store_matrix_sync` + single WMMA per K-chunk = 6-17× speedup.

5. **Production recommendation**: 
   - Room inference (mat-vec): Warp + half2 vectorized
   - Training transforms (matmul): Tensor core native
   - Activation functions: Separate kernel from WMMA operations

## Room Pipeline Benchmarks (Multi-Room Inference Chain)

| Operation | Latency | QPS |
|---|---|---|
| Single room inference | 0.0135 ms | 74,333/s |
| All 12 rooms parallel | 0.0190 ms | 52,704/s |
| Room selection | 0.0064 ms | 155,733/s |
| Input projection (16→256) | 0.0070 ms | 141,873/s |
| Pipeline: infer→select | 0.0209 ms | 47,940/s |
| Room weight switch (D2D) | 0.0053 ms | 188,887/s |
| Multi-hop chain (3 rooms, 5 infers) | 0.0536 ms | 18,667/s |

### Deckboss Product Targets — ALL PASS

| Target | Requirement | Actual | Status |
|---|---|---|---|
| Room switch | < 200 ms | **0.005 ms** | ✅ 40,000× under |
| Inference | < 1.0 ms | **0.014 ms** | ✅ 74× under |
| 12 rooms in 8GB | 12 rooms | **1,248 KB** | ✅ 6,400× headroom |
| Theoretical max rooms | — | **1,024 rooms** | 🚀 |

### Hop Latency
- Single hop (infer + select + project): **0.032 ms** (30,823 hops/sec)
- 3-room chain: **0.054 ms** (56,000 hops/sec)
- Multi-room conversations can chain rooms at 30K+ hops/sec
