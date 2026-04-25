/**
 * Suite #44: GPU Contention Under Mixed Workloads
 * 
 * Suite #21 measured contention between inference kernels.
 * This measures contention between inference AND other GPU work:
 * - Memory copies (H2D/D2H from other processes)
 * - Compute kernels (simulating training, encoding, etc.)
 * - Display/rendering (simulated via memory writes)
 * 
 * Question: How much does background work degrade inference latency?
 * What's the p99 impact? Is there a "noisy neighbor" problem?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 contention_mixed.cu -o contention_mixed
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

// Background compute kernel (simulates training/encoding workload)
__global__ void bg_compute(float* data, int size, float intensity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float val = data[idx];
    // Do some FP32 work proportional to intensity
    for (int i = 0; i < (int)(intensity * 100); i++)
        val = val * 0.999f + 0.001f * sinf(val);
    data[idx] = val;
}

// Background memory kernel (simulates memory-intensive workload)
__global__ void bg_memory(float* __restrict__ src, float* __restrict__ dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = src[idx] * 2.0f + 1.0f;
}

int main() {
    printf("=== Suite #44: GPU Contention Under Mixed Workloads ===\n\n");
    
    int infer_rooms = 256;
    dim3 infer_grid((infer_rooms + 7) / 8);
    
    // Allocate inference buffers
    half *d_weights, *d_input;
    float *d_out[2], *h_out[2], *d_zc[2];
    cudaMalloc(&d_weights, 1024 * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    for (int b = 0; b < 2; b++) {
        cudaMalloc(&d_out[b], 1024 * sizeof(float));
        cudaHostAlloc(&h_out[b], 1024 * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_zc[b], h_out[b], 0);
    }
    
    // Init
    std::vector<half> hw(1024 * DIM), hi(DIM);
    for (int i = 0; i < 1024 * DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    cudaMemcpy(d_weights, hw.data(), 1024 * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Background workload buffers
    int bg_size = 4 * 1024 * 1024;  // 16MB
    float *d_bg_src, *d_bg_dst;
    cudaMalloc(&d_bg_src, bg_size * sizeof(float));
    cudaMalloc(&d_bg_dst, bg_size * sizeof(float));
    cudaMemset(d_bg_src, 0, bg_size * sizeof(float));
    
    cudaStream_t infer_stream, bg_stream;
    cudaStreamCreate(&infer_stream);
    cudaStreamCreate(&bg_stream);
    
    // ===== Baseline: No background =====
    printf("--- Baseline: Inference Only ---\n");
    {
        std::vector<float> lats(ITERS);
        for (int i = 0; i < WARMUP; i++) {
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
        }
        cudaStreamSynchronize(infer_stream);
        
        for (int i = 0; i < ITERS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
        }
        cudaStreamSynchronize(infer_stream);
        
        std::sort(lats.begin(), lats.end());
        printf("  p50=%.2f p90=%.2f p99=%.2f p999=%.2f max=%.2f (us)\n",
               lats[ITERS/2], lats[(int)(ITERS*0.90)], lats[(int)(ITERS*0.99)],
               lats[(int)(ITERS*0.999)], lats[ITERS-1]);
    }
    
    // ===== Background: Memory copies =====
    printf("\n--- Background: Async H2D Memory Copies ---\n");
    
    float intensities[] = {0.1f, 0.25f, 0.5f, 1.0f};
    const char* intensity_names[] = {"Light (10%%)", "Medium (25%%)", "Heavy (50%%)", "Max (100%%)"};
    
    for (int t = 0; t < 4; t++) {
        float intensity = intensities[t];
        int copy_size = (int)(bg_size * intensity * sizeof(float));
        
        std::vector<float> host_buf(copy_size, 1.0f);
        float *d_copy_dst;
        cudaMalloc(&d_copy_dst, copy_size);
        
        std::vector<float> lats(ITERS);
        
        for (int i = 0; i < WARMUP; i++) {
            cudaMemcpyAsync(d_copy_dst, host_buf.data(), copy_size, cudaMemcpyHostToDevice, bg_stream);
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
        }
        cudaStreamSynchronize(infer_stream);
        cudaStreamSynchronize(bg_stream);
        
        for (int i = 0; i < ITERS; i++) {
            // Launch background copy
            cudaMemcpyAsync(d_copy_dst, host_buf.data(), copy_size, cudaMemcpyHostToDevice, bg_stream);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
        }
        cudaStreamSynchronize(infer_stream);
        cudaStreamSynchronize(bg_stream);
        
        std::sort(lats.begin(), lats.end());
        printf("  %s (%dKB): p50=%.2f p99=%.2f p999=%.2f max=%.2f (us)\n",
               intensity_names[t], copy_size/1024,
               lats[ITERS/2], lats[(int)(ITERS*0.99)], lats[(int)(ITERS*0.999)], lats[ITERS-1]);
        
        cudaFree(d_copy_dst);
    }
    
    // ===== Background: Compute kernels =====
    printf("\n--- Background: Compute Kernels (on separate stream) ---\n");
    
    for (int t = 0; t < 4; t++) {
        float intensity = intensities[t];
        int bg_blocks = (int)(intensity * 64);
        
        std::vector<float> lats(ITERS);
        
        for (int i = 0; i < WARMUP; i++) {
            bg_compute<<<bg_blocks, 256, 0, bg_stream>>>(d_bg_src, bg_size, intensity);
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
        }
        cudaStreamSynchronize(infer_stream);
        cudaStreamSynchronize(bg_stream);
        
        for (int i = 0; i < ITERS; i++) {
            bg_compute<<<bg_blocks, 256, 0, bg_stream>>>(d_bg_src, bg_size, intensity);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
        }
        cudaStreamSynchronize(infer_stream);
        cudaStreamSynchronize(bg_stream);
        
        std::sort(lats.begin(), lats.end());
        printf("  %s (%d blocks): p50=%.2f p99=%.2f p999=%.2f max=%.2f (us)\n",
               intensity_names[t], bg_blocks,
               lats[ITERS/2], lats[(int)(ITERS*0.99)], lats[(int)(ITERS*0.999)], lats[ITERS-1]);
    }
    
    // ===== Background: Memory bandwidth saturation =====
    printf("\n--- Background: Memory Bandwidth Saturation ---\n");
    
    for (int t = 0; t < 4; t++) {
        float intensity = intensities[t];
        int bg_blocks = (int)(intensity * 256);
        
        std::vector<float> lats(ITERS);
        
        for (int i = 0; i < WARMUP; i++) {
            bg_memory<<<bg_blocks, 256, 0, bg_stream>>>(d_bg_src, d_bg_dst, bg_size);
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
        }
        cudaStreamSynchronize(infer_stream);
        cudaStreamSynchronize(bg_stream);
        
        for (int i = 0; i < ITERS; i++) {
            bg_memory<<<bg_blocks, 256, 0, bg_stream>>>(d_bg_src, d_bg_dst, bg_size);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            infer_v7<<<infer_grid, dim3(256), 0, infer_stream>>>(d_weights, d_input, d_zc[i%2], infer_rooms, DIM);
            if (i > 0) volatile float sink = h_out[(i-1)%2][0];
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
        }
        cudaStreamSynchronize(infer_stream);
        cudaStreamSynchronize(bg_stream);
        
        std::sort(lats.begin(), lats.end());
        printf("  %s (%d blocks): p50=%.2f p99=%.2f p999=%.2f max=%.2f (us)\n",
               intensity_names[t], bg_blocks,
               lats[ITERS/2], lats[(int)(ITERS*0.99)], lats[(int)(ITERS*0.999)], lats[ITERS-1]);
    }
    
    cudaStreamDestroy(infer_stream);
    cudaStreamDestroy(bg_stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_bg_src); cudaFree(d_bg_dst);
    for (int b = 0; b < 2; b++) { cudaFree(d_out[b]); cudaFreeHost(h_out[b]); }
    
    printf("\n=== Suite #44 Complete ===\n");
    return 0;
}
