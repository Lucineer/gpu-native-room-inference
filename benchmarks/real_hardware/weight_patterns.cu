/**
 * Suite #51: Neural Network Weight Pattern Impact
 * 
 * All previous suites used synthetic weights (sin/cos patterns).
 * Real neural network weights have specific statistical properties:
 * - Trained weights follow approximately Gaussian distributions
 * - Layer norm weights have variance ~1/dim
 * - Attention weights have structured sparsity patterns
 * - Embedding weights may have high-magnitude outliers
 * 
 * Questions:
 * 1. Does weight distribution affect inference throughput?
 * 2. Do outlier weights (magnitude >> mean) cause numerical issues in half precision?
 * 3. Does weight sparsity change the optimal kernel configuration?
 * 4. Are there numerical stability differences between FP16, BF16, and FP32?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 weight_patterns.cu -o weight_patterns
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#define DIM 256
#define WARMUP 500
#define ITERS 10000

__global__ void infer_v7(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room_base + room_idx] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

// FP32 inference for numerical comparison
__global__ void infer_fp32(
    const float* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += weights[(room_base + room_idx) * dim + i] * input[i];
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room_base + room_idx] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

// Kahan summation for numerical accuracy comparison
__global__ void infer_kahan(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f, c = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float w = __half2float(weights[(room_base + room_idx) * dim + i]);
        float x = __half2float(input[i]);
        float y = w * x - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room_base + room_idx] = sum;
}

// Box-Muller transform for Gaussian random numbers
float randn(unsigned int* state) {
    float u1 = (float)rand_r(state) / RAND_MAX;
    float u2 = (float)rand_r(state) / RAND_MAX;
    while (u1 < 1e-10f) u1 = (float)rand_r(state) / RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

int main() {
    printf("=== Suite #51: Neural Network Weight Pattern Impact ===\n\n");
    
    int rooms = 1024;
    
    half *d_weights, *d_input;
    float *d_weights_f32, *d_input_f32, *d_out;
    cudaMalloc(&d_weights, rooms * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_weights_f32, rooms * DIM * sizeof(float));
    cudaMalloc(&d_input_f32, DIM * sizeof(float));
    cudaMalloc(&d_out, rooms * sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    dim3 grid((rooms + 7) / 8);
    
    // ===== Part 1: Throughput by Weight Distribution =====
    printf("--- Part 1: Throughput by Weight Distribution (1024 rooms, dim=256) ---\n");
    printf("%-25s | %-10s | %-10s | %-10s\n", "Distribution", "μs", "qps(M)", "vs Uniform");
    printf("---------------------------|------------|------------|------------\n");
    
    struct DistTest {
        const char* name;
        std::vector<half> weights;
    };
    
    std::vector<DistTest> dists;
    unsigned int seed = 42;
    
    // Uniform [-1, 1]
    {
        std::vector<half> w(rooms * DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = __float2half(2.0f * (float)rand_r(&seed) / RAND_MAX - 1.0f);
        dists.push_back({"Uniform [-1,1]", w});
    }
    
    // Gaussian N(0, 0.01) — typical trained weights
    {
        std::vector<half> w(rooms * DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = __float2half(0.01f * randn(&seed));
        dists.push_back({"Gaussian N(0,0.01)", w});
    }
    
    // Gaussian N(0, 1.0) — large variance
    {
        std::vector<half> w(rooms * DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = __float2half(randn(&seed));
        dists.push_back({"Gaussian N(0,1.0)", w});
    }
    
    // Sparse (80% zeros)
    {
        std::vector<half> w(rooms * DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = ((float)rand_r(&seed) / RAND_MAX < 0.8f) ? __float2half(0.0f) : __float2half(0.01f * randn(&seed));
        dists.push_back({"Sparse 80%%", w});
    }
    
    // Sparse (95% zeros)
    {
        std::vector<half> w(rooms * DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = ((float)rand_r(&seed) / RAND_MAX < 0.95f) ? __float2half(0.0f) : __float2half(0.01f * randn(&seed));
        dists.push_back({"Sparse 95%%", w});
    }
    
    // Outlier-heavy (Gaussian + 1% outliers at 100x magnitude)
    {
        std::vector<half> w(rooms * DIM);
        for (int i = 0; i < rooms * DIM; i++) {
            float v = 0.01f * randn(&seed);
            if ((float)rand_r(&seed) / RAND_MAX < 0.01f) v *= 100.0f;
            w[i] = __float2half(v);
        }
        dists.push_back({"Outlier-heavy (1%%)", w});
    }
    
    // Xavier initialization (uniform [-sqrt(6/dim), sqrt(6/dim)])
    {
        std::vector<half> w(rooms * DIM);
        float scale = sqrtf(6.0f / DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = __float2half(2.0f * scale * (float)rand_r(&seed) / RAND_MAX - scale);
        dists.push_back({"Xavier uniform", w});
    }
    
    // Kaiming initialization (normal, sqrt(2/dim))
    {
        std::vector<half> w(rooms * DIM);
        float scale = sqrtf(2.0f / DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = __float2half(scale * randn(&seed));
        dists.push_back({"Kaiming normal", w});
    }
    
    // Sin/cos (our default)
    {
        std::vector<half> w(rooms * DIM);
        for (int i = 0; i < rooms * DIM; i++)
            w[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
        dists.push_back({"Sin/Cos (default)", w});
    }
    
    float baseline_us = 0;
    for (auto& d : dists) {
        cudaMemcpy(d_weights, d.weights.data(), rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
        
        float us;
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,stream>>>(d_weights,d_input,d_out,rooms,DIM);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,stream>>>(d_weights,d_input,d_out,rooms,DIM);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        if (baseline_us == 0) baseline_us = us;
        float qps = rooms / (us / 1e6f) / 1e6f;
        
        printf("%-25s | %8.2f   | %8.2f   | %.3fx\n", d.name, us, qps, baseline_us / us);
    }
    
    // ===== Part 2: Numerical Accuracy — FP16 vs FP32 vs Kahan =====
    printf("\n--- Part 2: Numerical Accuracy (Gaussian weights, 1024 rooms) ---\n");
    
    // Generate Gaussian weights in both FP16 and FP32
    seed = 42;
    std::vector<half> hw16(rooms * DIM);
    std::vector<float> hw32(rooms * DIM);
    for (int i = 0; i < rooms * DIM; i++) {
        float v = 0.01f * randn(&seed);
        hw16[i] = __float2half(v);
        hw32[i] = v;
    }
    
    std::vector<half> hi16(DIM);
    std::vector<float> hi32(DIM);
    for (int i = 0; i < DIM; i++) {
        float v = 0.5f * cosf((float)i / DIM * 6.2832f);
        hi16[i] = __float2half(v);
        hi32[i] = v;
    }
    
    cudaMemcpy(d_weights, hw16.data(), rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi16.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_f32, hw32.data(), rooms * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_f32, hi32.data(), DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    // FP16 inference
    infer_v7<<<grid, 256, 0, stream>>>(d_weights, d_input, d_out, rooms, DIM);
    std::vector<float> out_fp16(rooms);
    cudaMemcpy(out_fp16.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
    
    // FP32 inference
    infer_fp32<<<grid, 256, 0, stream>>>(d_weights_f32, d_input_f32, d_out, rooms, DIM);
    std::vector<float> out_fp32(rooms);
    cudaMemcpy(out_fp32.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Kahan summation (FP32 accumulation of FP16 weights)
    infer_kahan<<<grid, 256, 0, stream>>>(d_weights, d_input, d_out, rooms, DIM);
    std::vector<float> out_kahan(rooms);
    cudaMemcpy(out_kahan.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    float max_err_16_32 = 0, max_err_kahan_32 = 0;
    float avg_err_16_32 = 0, avg_err_kahan_32 = 0;
    
    for (int r = 0; r < rooms; r++) {
        float e16 = fabsf(out_fp16[r] - out_fp32[r]);
        float ek = fabsf(out_kahan[r] - out_fp32[r]);
        float rel16 = (fabsf(out_fp32[r]) > 1e-10f) ? e16 / fabsf(out_fp32[r]) : e16;
        float relk = (fabsf(out_fp32[r]) > 1e-10f) ? ek / fabsf(out_fp32[r]) : ek;
        max_err_16_32 = std::max(max_err_16_32, rel16);
        max_err_kahan_32 = std::max(max_err_kahan_32, relk);
        avg_err_16_32 += rel16;
        avg_err_kahan_32 += relk;
    }
    avg_err_16_32 /= rooms;
    avg_err_kahan_32 /= rooms;
    
    printf("  FP16 vs FP32:   max_rel_err=%.6f%%, avg_rel_err=%.6f%%\n", max_err_16_32 * 100, avg_err_16_32 * 100);
    printf("  Kahan vs FP32:  max_rel_err=%.6f%%, avg_rel_err=%.6f%%\n", max_err_kahan_32 * 100, avg_err_kahan_32 * 100);
    
    // Test with outlier weights
    printf("\n  --- With Outlier Weights (1%% at 100x magnitude) ---\n");
    seed = 42;
    for (int i = 0; i < rooms * DIM; i++) {
        float v = 0.01f * randn(&seed);
        if ((float)rand_r(&seed) / RAND_MAX < 0.01f) v *= 100.0f;
        hw16[i] = __float2half(v);
        hw32[i] = v;
    }
    
    cudaMemcpy(d_weights, hw16.data(), rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_f32, hw32.data(), rooms * DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    infer_v7<<<grid, 256, 0, stream>>>(d_weights, d_input, d_out, rooms, DIM);
    cudaMemcpy(out_fp16.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
    
    infer_fp32<<<grid, 256, 0, stream>>>(d_weights_f32, d_input_f32, d_out, rooms, DIM);
    cudaMemcpy(out_fp32.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
    
    max_err_16_32 = 0; avg_err_16_32 = 0;
    for (int r = 0; r < rooms; r++) {
        float rel = (fabsf(out_fp32[r]) > 1e-10f) ? fabsf(out_fp16[r] - out_fp32[r]) / fabsf(out_fp32[r]) : fabsf(out_fp16[r] - out_fp32[r]);
        max_err_16_32 = std::max(max_err_16_32, rel);
        avg_err_16_32 += rel;
    }
    avg_err_16_32 /= rooms;
    printf("  FP16 vs FP32:   max_rel_err=%.4f%%, avg_rel_err=%.6f%%\n", max_err_16_32 * 100, avg_err_16_32 * 100);
    
    // Show some sample values
    printf("\n  Sample outputs (outlier weights):\n");
    for (int r = 0; r < 5; r++) {
        printf("    Room %d: FP16=%.8f, FP32=%.8f, diff=%.2e\n",
               r, out_fp16[r], out_fp32[r], fabsf(out_fp16[r] - out_fp32[r]));
    }
    
    // ===== Part 3: Sparsity Benefit =====
    printf("\n--- Part 3: Sparsity Benefit (can we skip zero weights?) ---\n");
    printf("If weights are sparse, can we skip zeros for faster inference?\n\n");
    
    printf("%-12s | %-10s | %-10s | %-10s | %-10s\n", "Sparsity", "μs", "qps(M)", "vs Dense", "Non-zero");
    printf("--------------|------------|------------|------------|------------\n");
    
    float sparse_us[6];
    float sparsity_levels[] = {0.0f, 0.5f, 0.8f, 0.9f, 0.95f, 0.99f};
    
    for (int s = 0; s < 6; s++) {
        float sparsity = sparsity_levels[s];
        
        std::vector<half> w(rooms * DIM);
        seed = 42;
        int non_zero = 0;
        for (int i = 0; i < rooms * DIM; i++) {
            if ((float)rand_r(&seed) / RAND_MAX >= sparsity) {
                w[i] = __float2half(0.01f * randn(&seed));
                non_zero++;
            } else {
                w[i] = __float2half(0.0f);
            }
        }
        cudaMemcpy(d_weights, w.data(), rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
        
        float us;
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,stream>>>(d_weights,d_input,d_out,rooms,DIM);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,stream>>>(d_weights,d_input,d_out,rooms,DIM);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        sparse_us[s] = us;
        printf("%-12.0f%% | %8.2f   | %8.2f   | %.3fx     | %d\n",
               sparsity * 100, us, rooms / (us / 1e6f) / 1e6f,
               sparse_us[0] / us, non_zero);
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input);
    cudaFree(d_weights_f32); cudaFree(d_input_f32); cudaFree(d_out);
    
    printf("\n=== Suite #51 Complete ===\n");
    return 0;
}
