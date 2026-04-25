/**
 * Suite #40: The Absolute Floor — GPU-Only Latency
 * 
 * Every previous suite included some CPU overhead.
 * This suite measures PURE GPU execution time using CUDA events.
 * 
 * Questions:
 * 1. What is the absolute minimum time for V7 inference?
 * 2. How does it scale with dim (64, 128, 256, 512, 1024)?
 * 3. How does it scale with rooms (1 to 8192)?
 * 4. What's the theoretical compute vs measured ratio?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 absolute_floor.cu -o absolute_floor
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

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

void bench_gpu_only(const half* d_w, const half* d_in, float* d_out,
                    cudaStream_t stream, int rooms, int dim, int iters) {
    dim3 grid((rooms + 7) / 8);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 100; i++)
        infer_v7<<<grid, dim3(256), 0, stream>>>(d_w, d_in, d_out, rooms, dim);
    cudaStreamSynchronize(stream);
    
    // Measure
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++)
        infer_v7<<<grid, dim3(256), 0, stream>>>(d_w, d_in, d_out, rooms, dim);
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float us = ms / iters * 1000;
    
    float qps = rooms / (us / 1e6f);
    printf("%-8d | %10.3f   | %12.0f   | %10.1f", rooms, us, qps, qps / 1e6f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Suite #40: The Absolute Floor — GPU-Only Latency ===\n\n");
    printf("Pure GPU execution time (CUDA events, zero CPU overhead)\n\n");
    
    int max_rooms = 8192;
    int max_dim = 1024;
    int iters = 50000;
    
    half *d_weights, *d_input;
    float *d_output;
    cudaMalloc(&d_weights, max_rooms * max_dim * sizeof(half));
    cudaMalloc(&d_input, max_dim * sizeof(half));
    cudaMalloc(&d_output, max_rooms * sizeof(float));
    
    std::vector<half> hw(max_rooms * max_dim), hi(max_dim);
    for (int i = 0; i < max_rooms * max_dim; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < max_dim; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / max_dim * 6.2832f));
    
    cudaMemcpy(d_weights, hw.data(), max_rooms * max_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), max_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // ========================================
    // Part 1: Room scaling at dim=256
    // ========================================
    printf("--- Part 1: Room Scaling (dim=256) ---\n");
    printf("%-8s | %-12s | %-14s | %-10s | %-14s\n",
           "Rooms", "GPU-only(μs)", "Room-qps", "M room-qps", "qps/room");
    printf("---------|--------------|----------------|------------|----------------\n");
    
    int room_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    for (int r = 0; r < 14; r++) {
        int rooms = room_sizes[r];
        bench_gpu_only(d_weights, d_input, d_output, stream, rooms, 256, iters);
        // Calculate per-room throughput
        // (need to re-measure to get us value — let me restructure)
        printf("\n");
    }
    
    // Redo with captured values
    printf("\n--- Part 1 (clean): Room Scaling (dim=256) ---\n");
    printf("%-8s | %-12s | %-14s\n", "Rooms", "GPU-only(μs)", "Room-qps");
    printf("---------|--------------|----------------\n");
    
    for (int r = 0; r < 14; r++) {
        int rooms = room_sizes[r];
        dim3 grid((rooms + 7) / 8);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        for (int i = 0; i < 100; i++)
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, 256);
        cudaStreamSynchronize(stream);
        
        cudaEventRecord(start, stream);
        for (int i = 0; i < iters; i++)
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, 256);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float us = ms / iters * 1000;
        float qps = rooms / (us / 1e6f);
        
        printf("%-8d | %10.3f   | %12.0f\n", rooms, us, qps);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // ========================================
    // Part 2: Dim scaling at 1024 rooms
    // ========================================
    printf("\n--- Part 2: Dim Scaling (1024 rooms) ---\n");
    printf("%-8s | %-12s | %-14s\n", "Dim", "GPU-only(μs)", "Room-qps");
    printf("---------|--------------|----------------\n");
    
    int dims[] = {64, 128, 256, 512, 1024};
    for (int d = 0; d < 5; d++) {
        int dim = dims[d];
        dim3 grid(128);  // 1024 rooms
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        for (int i = 0; i < 100; i++)
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, 1024, dim);
        cudaStreamSynchronize(stream);
        
        cudaEventRecord(start, stream);
        for (int i = 0; i < iters; i++)
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, 1024, dim);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float us = ms / iters * 1000;
        float qps = 1024.0f / (us / 1e6f);
        
        printf("%-8d | %10.3f   | %12.0f\n", dim, us, qps);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // ========================================
    // Part 3: Theoretical vs Measured
    // ========================================
    printf("\n--- Part 3: Compute Analysis (dim=256, 1024 rooms) ---\n");
    
    int dim = 256;
    int rooms = 1024;
    dim3 grid(128);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < 100; i++)
        infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, dim);
    cudaStreamSynchronize(stream);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++)
        infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, dim);
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float us = ms / iters * 1000;
    
    // Theoretical analysis
    // Per room: 256 FP16 multiplies + 255 FP16 adds + 1 GELU
    // Total: 1024 rooms × (256 mul + 255 add + ~10 GELU ops)
    // Orin FP16 TFLOPS: 40 (theoretical peak)
    // Operations: 1024 × 256 × 2 = 524,288 FP16 ops
    // Time = ops / TFLOPS = 524288 / (40e12) = 13.1 ns
    // But memory bound: 1024 × 256 × 2 bytes (weights) + 256 × 2 bytes (input) = 524,544 bytes
    // BW: 68 GB/s → time = 524544 / 68e9 = 7.7 us
    
    long long ops = (long long)rooms * dim * 2;  // multiply + add
    long long bytes = (long long)rooms * dim * 2 + (long long)dim * 2;  // weights + input (half)
    
    printf("  Operations: %lld FLOPs\n", ops);
    printf("  Memory:     %lld bytes (%.1f KB)\n", bytes, bytes / 1024.0);
    printf("  Measured:   %.3f μs\n", us);
    printf("  Compute bound: %.3f μs (at 40 TFLOPS)\n", ops / 40e6);
    printf("  Memory bound: %.3f μs (at 68 GB/s)\n", bytes / 68e3);
    printf("  Actual: %.3f μs\n", us);
    printf("  Efficiency: %.1f%% of memory bandwidth\n", (bytes / 68e3) / us * 100);
    printf("  Arithmetic intensity: %.2f FLOP/byte\n", (double)ops / bytes);
    printf("  Roofline: memory-bound (AI < 100)\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_output);
    
    printf("\n=== Suite #40 Complete ===\n");
    return 0;
}
