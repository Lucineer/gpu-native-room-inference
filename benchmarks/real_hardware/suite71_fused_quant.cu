/**
 * Suite #71 (v2): Fused Quantization + Inference — FP16 Upload, INT8 Compute on GPU
 * 
 * FIXED: Race condition in shared memory. Each warp computes its own room's
 * absmax and quantization independently, no shared state between warps needed.
 * 
 * Key insight: since each warp handles exactly 1 room (32 threads × 8 rooms/block),
 * we can do everything in registers + warp shuffle. No shared memory needed.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>

#define DIM 256
#define ITERS 10000
#define WARMUP 500

// Baseline: pre-quantized INT8
__global__ void __launch_bounds__(256, 8) int8_baseline(
    const signed char* __restrict__ weights,
    const signed char* __restrict__ input,
    const float* __restrict__ scales,
    float iscale,
    float* __restrict__ output,
    int dim,
    int num_rooms
) {
    int gi = blockIdx.x * 8 + threadIdx.x / 32;
    if (gi >= num_rooms) return;
    int lane = threadIdx.x % 32;
    
    int sum = 0;
    #pragma unroll 8
    for (int i = lane; i < dim; i += 32)
        sum += (int)weights[(size_t)gi * dim + i] * (int)input[i];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) output[gi] = (float)sum * scales[gi] * iscale;
}

// Fused: FP16 weights → on-the-fly INT8 quantize → dot product
// All within one warp, no shared memory
__global__ void __launch_bounds__(256, 8) fused_fp16_int8(
    const half* __restrict__ fp16_weights,  // [num_rooms * dim]
    const half* __restrict__ fp16_input,     // [dim]
    float* __restrict__ output,
    int dim,
    int num_rooms
) {
    int gi = blockIdx.x * 8 + threadIdx.x / 32;
    if (gi >= num_rooms) return;
    int lane = threadIdx.x % 32;
    
    // Step 1: Find this room's weight absmax (warp reduction)
    float wmax = 0.0f;
    #pragma unroll 4
    for (int i = lane; i < dim; i += 32) {
        float v = fabsf(__half2float(fp16_weights[(size_t)gi * dim + i]));
        if (v > wmax) wmax = v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_down_sync(0xffffffff, wmax, o);
        if (other > wmax) wmax = other;
    }
    float wscale = wmax > 0 ? wmax / 127.0f : 1.0f;
    float winv = wmax > 0 ? 127.0f / wmax : 0.0f;
    
    // Step 2: Find input absmax (same for all warps — every warp computes it)
    float imax = 0.0f;
    #pragma unroll 4
    for (int i = lane; i < dim; i += 32) {
        float v = fabsf(__half2float(fp16_input[i]));
        if (v > imax) imax = v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_down_sync(0xffffffff, imax, o);
        if (other > imax) imax = other;
    }
    float iscale = imax > 0 ? imax / 127.0f : 1.0f;
    float iinv = imax > 0 ? 127.0f / imax : 0.0f;
    
    // Step 3: Fused quantize + dot product in one pass
    // Quantize FP16→INT8 in registers, multiply, accumulate
    int sum = 0;
    #pragma unroll 4
    for (int i = lane; i < dim; i += 32) {
        // Quantize weight
        float wv = __half2float(fp16_weights[(size_t)gi * dim + i]) * winv;
        int wq = (int)wv;
        if (wq > 127) wq = 127;
        if (wq < -127) wq = -127;
        
        // Quantize input
        float iv = __half2float(fp16_input[i]) * iinv;
        int iq = (int)iv;
        if (iq > 127) iq = 127;
        if (iq < -127) iq = -127;
        
        sum += wq * iq;
    }
    
    // Step 4: Warp shuffle reduction
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    
    // Step 5: Dequantize result
    if (lane == 0) {
        output[gi] = (float)sum * wscale * iscale;
    }
}

// Variant: Skip input quantization — only quantize weights, keep input as FP16 math
__global__ void __launch_bounds__(256, 8) fused_wq_only(
    const half* __restrict__ fp16_weights,
    const half* __restrict__ fp16_input,
    float* __restrict__ output,
    int dim,
    int num_rooms
) {
    int gi = blockIdx.x * 8 + threadIdx.x / 32;
    if (gi >= num_rooms) return;
    int lane = threadIdx.x % 32;
    
    // Weight absmax
    float wmax = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float v = fabsf(__half2float(fp16_weights[(size_t)gi * dim + i]));
        if (v > wmax) wmax = v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_down_sync(0xffffffff, wmax, o);
        if (other > wmax) wmax = other;
    }
    float wscale = wmax > 0 ? wmax / 127.0f : 1.0f;
    float winv = wmax > 0 ? 127.0f / wmax : 0.0f;
    
    // Dot product: INT8 weight × FP16 input
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = lane; i < dim; i += 32) {
        float wv = __half2float(fp16_weights[(size_t)gi * dim + i]) * winv;
        int wq = (int)wv;
        if (wq > 127) wq = 127;
        if (wq < -127) wq = -127;
        
        // Input stays FP16 — no quantization
        float iv = __half2float(fp16_input[i]);
        sum += (float)wq * iv;
    }
    
    // FP32 warp reduction
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    
    if (lane == 0) output[gi] = sum * wscale;
}

int main() {
    printf("=== Suite #71 v2: Fused Quantization + Inference ===\n");
    printf("=== FP16 Upload -> GPU On-the-fly INT8 -> Compute ===\n\n");
    
    int R = 4096;
    
    // Generate FP32 data
    std::vector<float> hw((size_t)R * DIM), hi(DIM);
    srand(42);
    for (size_t i = 0; i < (size_t)R * DIM; i++)
        hw[i] = 0.5f * (2.0f * (float)rand() / RAND_MAX - 1.0f);
    for (int i = 0; i < DIM; i++)
        hi[i] = cosf((float)i / DIM * 6.2832f);
    
    // Convert to FP16
    std::vector<half> hw16((size_t)R * DIM), hi16(DIM);
    for (size_t i = 0; i < (size_t)R * DIM; i++)
        hw16[i] = __float2half(hw[i]);
    for (int i = 0; i < DIM; i++)
        hi16[i] = __float2half(hi[i]);
    
    // Pre-quantize to INT8 (for baseline)
    std::vector<signed char> wq((size_t)R * DIM), iq(DIM);
    std::vector<float> ws(R);
    float iscale;
    {
        float imax = 0;
        for (int i = 0; i < DIM; i++) { float v = fabsf(hi[i]); if (v > imax) imax = v; }
        iscale = imax / 127.0f;
        float iinv = imax > 0 ? 127.0f / imax : 0;
        for (int i = 0; i < DIM; i++) {
            float v = hi[i] * iinv;
            iq[i] = (signed char)(v > 127 ? 127 : (v < -127 ? -127 : (int)v));
        }
        for (int r = 0; r < R; r++) {
            float wmax = 0;
            for (int d = 0; d < DIM; d++) {
                float v = fabsf(hw[r * DIM + d]);
                if (v > wmax) wmax = v;
            }
            ws[r] = wmax / 127.0f;
            float inv = wmax > 0 ? 127.0f / wmax : 0;
            for (int d = 0; d < DIM; d++) {
                float v = hw[r * DIM + d] * inv;
                wq[r * DIM + d] = (signed char)(v > 127 ? 127 : (v < -127 ? -127 : (int)v));
            }
        }
    }
    
    // GPU memory — baseline
    signed char *dw, *di;
    float *ds, *dout;
    cudaMalloc(&dw, (size_t)R * DIM);
    cudaMalloc(&di, DIM);
    cudaMalloc(&ds, R * sizeof(float));
    cudaMalloc(&dout, R * sizeof(float));
    cudaMemcpy(dw, wq.data(), (size_t)R * DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(di, iq.data(), DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, ws.data(), R * sizeof(float), cudaMemcpyHostToDevice);
    
    // GPU memory — fused (FP16)
    half *dw16, *di16;
    cudaMalloc(&dw16, (size_t)R * DIM * sizeof(half));
    cudaMalloc(&di16, DIM * sizeof(half));
    cudaMemcpy(dw16, hw16.data(), (size_t)R * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(di16, hi16.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    
    // ===== Correctness first =====
    printf("--- Correctness (first 5 rooms) ---\n");
    {
        int n = 5;
        dim3 g(1);
        
        // Baseline
        int8_baseline<<<g, block>>>(dw, di, ds, iscale, dout, DIM, n);
        cudaDeviceSynchronize();
        float r_base[5];
        cudaMemcpy(r_base, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Fused full INT8
        fused_fp16_int8<<<g, block>>>(dw16, di16, dout, DIM, n);
        cudaDeviceSynchronize();
        float r_fused[5];
        cudaMemcpy(r_fused, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Weight-quant-only
        fused_wq_only<<<g, block>>>(dw16, di16, dout, DIM, n);
        cudaDeviceSynchronize();
        float r_wq[5];
        cudaMemcpy(r_wq, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int t = 0; t < n; t++) {
            float ref = 0;
            for (int d = 0; d < DIM; d++) ref += hw[t * DIM + d] * hi[d];
            float eb = fabsf(r_base[t] - ref) / (fabsf(ref) + 1e-10f);
            float ef = fabsf(r_fused[t] - ref) / (fabsf(ref) + 1e-10f);
            float ew = fabsf(r_wq[t] - ref) / (fabsf(ref) + 1e-10f);
            printf("  Room %d: ref=%.4f  base=%.4f (%.2f%%)  fused=%.4f (%.2f%%)  wq_only=%.4f (%.2f%%)\n",
                   t, ref, r_base[t], eb * 100, r_fused[t], ef * 100, r_wq[t], ew * 100);
        }
    }
    
    // ===== Performance =====
    printf("\n--- Throughput Sweep ---\n");
    printf("%8s  %10s  %12s  %10s  %12s  %10s  %12s  %8s\n",
           "Rooms", "Baseline", "Baseline", "Fused", "Fused", "WQ-Only", "WQ-Only", "Fused/B");
    printf("%8s  %10s  %12s  %10s  %12s  %10s  %12s  %8s\n",
           "------", "-------", "--------", "-----", "-----", "-------", "-------", "-------");
    
    int batches[] = {64, 256, 1024, 4096};
    int nb = sizeof(batches) / sizeof(batches[0]);
    
    for (int b = 0; b < nb; b++) {
        int n = batches[b];
        dim3 g((n + 7) / 8);
        
        // Baseline
        for (int w = 0; w < WARMUP; w++)
            int8_baseline<<<g, block>>>(dw, di, ds, iscale, dout, DIM, n);
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; i++)
            int8_baseline<<<g, block>>>(dw, di, ds, iscale, dout, DIM, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        float base_us = ms / ITERS * 1000;
        cudaEventDestroy(s); cudaEventDestroy(e);
        
        // Fused full
        for (int w = 0; w < WARMUP; w++)
            fused_fp16_int8<<<g, block>>>(dw16, di16, dout, DIM, n);
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; i++)
            fused_fp16_int8<<<g, block>>>(dw16, di16, dout, DIM, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        float fused_us = ms / ITERS * 1000;
        cudaEventDestroy(s); cudaEventDestroy(e);
        
        // Weight-quant-only
        for (int w = 0; w < WARMUP; w++)
            fused_wq_only<<<g, block>>>(dw16, di16, dout, DIM, n);
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; i++)
            fused_wq_only<<<g, block>>>(dw16, di16, dout, DIM, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        float wq_us = ms / ITERS * 1000;
        cudaEventDestroy(s); cudaEventDestroy(e);
        
        printf("%8d  %8.2f us  %8.1f Mq  %8.2f us  %8.1f Mq  %8.2f us  %8.1f Mq  %6.2fx\n",
               n, base_us, (float)n / base_us * 1000,
               fused_us, (float)n / fused_us * 1000,
               wq_us, (float)n / wq_us * 1000,
               fused_us / base_us);
    }
    
    // ===== End-to-end: Upload + Compute =====
    printf("\n--- End-to-End: Upload(4096 rooms) + Compute ---\n");
    {
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        
        // Path 1: FP32 upload + CPU quantize + INT8 upload + INT8 compute
        // (Skip CPU quant timing — just measure transfer + GPU)
        cudaEventRecord(s);
        cudaMemcpy(dw, wq.data(), (size_t)R * DIM, cudaMemcpyHostToDevice);
        cudaMemcpy(di, iq.data(), DIM, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, ws.data(), R * sizeof(float), cudaMemcpyHostToDevice);
        int8_baseline<<<(R+7)/8, block>>>(dw, di, ds, iscale, dout, DIM, R);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        printf("  INT8 pre-quant path: %.2f us\n", ms * 1000);
        
        // Path 2: FP16 upload + fused compute
        cudaEventRecord(s);
        cudaMemcpy(dw16, hw16.data(), (size_t)R * DIM * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(di16, hi16.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
        fused_fp16_int8<<<(R+7)/8, block>>>(dw16, di16, dout, DIM, R);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("  FP16 fused path:     %.2f us\n", ms * 1000);
        
        cudaEventDestroy(s); cudaEventDestroy(e);
    }
    
    cudaFree(dw); cudaFree(di); cudaFree(ds); cudaFree(dout);
    cudaFree(dw16); cudaFree(di16);
    
    printf("\n=== Suite #71 v2 Complete ===\n");
    return 0;
}
