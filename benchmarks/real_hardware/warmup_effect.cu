/**
 * Suite #45: GPU Warmup Effects on Inference Latency
 * 
 * Suite #44 showed background compute reduces inference p99.
 * This isolates the warmup effect: does pre-running kernels 
 * before inference reduce latency?
 * 
 * Test: vary warmup kernel count before inference measurement.
 * Also test: warmup with different kernel types (compute, memory, mixed).
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 warmup_effect.cu -o warmup_effect
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#define DIM 256
#define ITERS 50000

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

__global__ void warmup_compute(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = data[idx];
        for (int i = 0; i < 100; i++) v = v * 0.999f + sinf(v) * 0.001f;
        data[idx] = v;
    }
}

__global__ void warmup_memory(float* __restrict__ src, float* __restrict__ dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] = src[idx] * 2.0f + 1.0f;
}

int run_bench(const char* label, half* d_w, half* d_in, float* d_zc[2], float* h_out[2],
              cudaStream_t stream, int rooms, int warmup_count, int warmup_type,
              float* bg_data, int bg_size) {
    dim3 grid((rooms + 7) / 8);
    dim3 bg_grid(64);
    
    // Warmup phase
    for (int i = 0; i < warmup_count; i++) {
        if (warmup_type == 1) warmup_compute<<<bg_grid, 256, 0, stream>>>(bg_data, bg_size);
        else if (warmup_type == 2) warmup_memory<<<bg_grid, 256, 0, stream>>>(bg_data, bg_data + bg_size, bg_size);
        else infer_v7<<<grid, dim3(256), 0, stream>>>(d_w, d_in, d_zc[i%2], rooms, DIM);
    }
    cudaStreamSynchronize(stream);
    
    // Measure
    std::vector<float> lats(ITERS);
    for (int i = 0; i < ITERS; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        infer_v7<<<grid, dim3(256), 0, stream>>>(d_w, d_in, d_zc[i%2], rooms, DIM);
        if (i > 0) volatile float sink = h_out[(i-1)%2][0];
        auto t1 = std::chrono::high_resolution_clock::now();
        lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
    }
    cudaStreamSynchronize(stream);
    
    std::sort(lats.begin(), lats.end());
    float p50 = lats[ITERS/2];
    float p99 = lats[(int)(ITERS*0.99)];
    float p999 = lats[(int)(ITERS*0.999)];
    
    printf("  %-40s | %6.2f  | %6.2f  | %7.2f  | %.3fx\n",
           label, p50, p99, p999, p50 > 0 ? 16.0f / p50 : 0);
    
    return 0;
}

int main() {
    printf("=== Suite #45: GPU Warmup Effects ===\n\n");
    
    int rooms = 256;
    
    half *d_weights, *d_input;
    float *d_out[2], *h_out[2], *d_zc[2];
    cudaMalloc(&d_weights, 1024 * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    for (int b = 0; b < 2; b++) {
        cudaMalloc(&d_out[b], 1024 * sizeof(float));
        cudaHostAlloc(&h_out[b], 1024 * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_zc[b], h_out[b], 0);
    }
    
    std::vector<half> hw(1024 * DIM), hi(DIM);
    for (int i = 0; i < 1024 * DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    cudaMemcpy(d_weights, hw.data(), 1024 * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    int bg_size = 1024 * 1024;
    float *d_bg;
    cudaMalloc(&d_bg, bg_size * 2 * sizeof(float));
    cudaMemset(d_bg, 0, bg_size * 2 * sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    printf("%-40s | %-6s  | %-6s  | %-7s  | %-6s\n",
           "Warmup", "p50(us)", "p99(us)", "p999(us)", "p99/baseline");
    printf("------------------------------------------|---------|---------|----------|--------\n");
    
    // No warmup
    run_bench("No warmup (cold start)", d_weights, d_input, d_zc, h_out,
              stream, rooms, 0, 0, d_bg, bg_size);
    
    // Inference warmup: 10, 100, 1000
    run_bench("10 inference warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 10, 0, d_bg, bg_size);
    run_bench("100 inference warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 100, 0, d_bg, bg_size);
    run_bench("1000 inference warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 1000, 0, d_bg, bg_size);
    
    // Compute warmup: 10, 100, 1000
    run_bench("10 compute warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 10, 1, d_bg, bg_size);
    run_bench("100 compute warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 100, 1, d_bg, bg_size);
    run_bench("1000 compute warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 1000, 1, d_bg, bg_size);
    
    // Memory warmup: 10, 100, 1000
    run_bench("10 memory warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 10, 2, d_bg, bg_size);
    run_bench("100 memory warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 100, 2, d_bg, bg_size);
    run_bench("1000 memory warmups", d_weights, d_input, d_zc, h_out,
              stream, rooms, 1000, 2, d_bg, bg_size);
    
    // Continuous background (like suite #44)
    printf("\n--- Continuous Background (parallel stream) ---\n\n");
    
    cudaStream_t bg_stream;
    cudaStreamCreate(&bg_stream);
    
    // No background
    run_bench("No background", d_weights, d_input, d_zc, h_out,
              stream, rooms, 0, 0, d_bg, bg_size);
    
    // Continuous compute background
    for (int i = 0; i < ITERS; i++) {
        warmup_compute<<<64, 256, 0, bg_stream>>>(d_bg, bg_size);
    }
    // Don't sync — let it run concurrently
    // Actually we need to measure inference during background execution
    // Let me restructure this
    
    cudaStreamSynchronize(bg_stream);
    
    // Launch continuous background, then measure inference
    for (int burst = 0; burst < 100; burst++) {
        warmup_compute<<<64, 256, 0, bg_stream>>>(d_bg, bg_size);
    }
    
    std::vector<float> lats(ITERS);
    dim3 grid((rooms + 7) / 8);
    for (int i = 0; i < ITERS; i++) {
        // Keep background alive
        if (i % 10 == 0) warmup_compute<<<64, 256, 0, bg_stream>>>(d_bg, bg_size);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc[i%2], rooms, DIM);
        if (i > 0) volatile float sink = h_out[(i-1)%2][0];
        auto t1 = std::chrono::high_resolution_clock::now();
        lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
    }
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(bg_stream);
    
    std::sort(lats.begin(), lats.end());
    printf("  %-40s | %6.2f  | %6.2f  | %7.2f  | %.3fx\n",
           "Continuous compute bg (every 10th)", lats[ITERS/2],
           lats[(int)(ITERS*0.99)], lats[(int)(ITERS*0.999)], 16.0f / lats[ITERS/2]);
    
    // Continuous memory background
    for (int burst = 0; burst < 100; burst++) {
        warmup_memory<<<64, 256, 0, bg_stream>>>(d_bg, d_bg + bg_size, bg_size);
    }
    
    for (int i = 0; i < ITERS; i++) {
        if (i % 10 == 0) warmup_memory<<<64, 256, 0, bg_stream>>>(d_bg, d_bg + bg_size, bg_size);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc[i%2], rooms, DIM);
        if (i > 0) volatile float sink = h_out[(i-1)%2][0];
        auto t1 = std::chrono::high_resolution_clock::now();
        lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
    }
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(bg_stream);
    
    std::sort(lats.begin(), lats.end());
    printf("  %-40s | %6.2f  | %6.2f  | %7.2f  | %.3fx\n",
           "Continuous memory bg (every 10th)", lats[ITERS/2],
           lats[(int)(ITERS*0.99)], lats[(int)(ITERS*0.999)], 16.0f / lats[ITERS/2]);
    
    cudaStreamDestroy(stream);
    cudaStreamDestroy(bg_stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_bg);
    for (int b = 0; b < 2; b++) { cudaFree(d_out[b]); cudaFreeHost(h_out[b]); }
    
    printf("\n=== Suite #45 Complete ===\n");
    return 0;
}
