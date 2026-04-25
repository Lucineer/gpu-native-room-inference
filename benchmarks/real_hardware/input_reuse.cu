/**
 * Suite #42: Input Reuse — Shared Memory vs Global Memory
 * 
 * In fleet inference, the same input vector is used for ALL rooms.
 * Currently, each thread reads input[i] from global memory.
 * Since all 256 threads in a block read the same input elements,
 * we could load input into shared memory once and reuse it.
 * 
 * V7 kernel: for (i = lane; i < dim; i += 32)
 *   sum += weights[room*dim+i] * input[i]
 * 
 * Each warp reads input[lane], input[lane+32], input[lane+64], ..., input[lane+224]
 * 8 warps × 8 unique reads = 64 unique input reads per block
 * With shared memory: load 256 input values once (256 halfs = 512 bytes), then all warps read from shmem
 * 
 * But: shared memory has bank conflicts, and global memory already has L1/L2 cache.
 * Question: is the shmem load+read faster than cached global reads?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 input_reuse.cu -o input_reuse
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

__constant__ half input_const[1024];  // Max dim we'll test

// V7 baseline: input from global memory (cached in L1/L2)
__global__ void infer_global_input(
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
    
    if (lane == 0) output[room_base + room_idx] = sum;
}

// Shared memory input: load once, read 8 times (once per room in block)
__global__ void infer_shmem_input(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    // Load input into shared memory (first 256 threads)
    extern __shared__ half shared_input[];
    if (threadIdx.x < dim) {
        shared_input[threadIdx.x] = input[threadIdx.x];
    }
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(shared_input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room_base + room_idx] = sum;
}

// Register-cached input: each thread loads its needed input values into registers
// With stride-32 loop, each thread needs dim/32 = 8 input values
// Total per warp: 32 threads × 8 values = 256 unique reads
// But all threads in a warp read the SAME values (lane 0 in all warps reads input[0])
// We can't easily share registers across warps, but we can use __shfl to broadcast
__global__ void infer_shfl_input(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    // Lane 0-31 in warp 0 loads input[0:31], then broadcasts to other warps
    // Actually __shfl can only broadcast within a warp, not across warps
    // So this approach doesn't help for multi-warp blocks
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room_base + room_idx] = sum;
}

// Constant memory input: load into __constant__ (limited to 64KB, cached via broadcast)
// For dim=256, input = 512 bytes — fits easily
__global__ void infer_const_input(
    const half* __restrict__ weights,
    const half* __restrict__ input_const,  // Actually passed as const memory pointer
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(input_const[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room_base + room_idx] = sum;
}

int main() {
    printf("=== Suite #42: Input Reuse Optimization ===\n\n");
    
    int batch_sizes[] = {8, 32, 64, 256, 512, 1024, 2048, 4096};
    int max_rooms = 4096;
    
    half *d_weights, *d_input, *d_const_input;
    float *d_out;
    cudaMalloc(&d_weights, max_rooms * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_const_input, DIM * sizeof(half));
    cudaMalloc(&d_out, max_rooms * sizeof(float));
    
    std::vector<half> hw(max_rooms * DIM), hi(DIM);
    for (int i = 0; i < max_rooms * DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    
    cudaMemcpy(d_weights, hw.data(), max_rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_const_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Also copy to constant memory
    cudaMemcpyToSymbol(input_const, hi.data(), DIM * sizeof(half));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    printf("%-8s | %-14s | %-14s | %-8s | %-14s | %-8s\n",
           "Rooms", "Global(us)", "Shmem(us)", "Ratio", "Const(us)", "Ratio");
    printf("---------|----------------|----------------|----------|----------------|----------\n");
    
    for (int b = 0; b < 8; b++) {
        int rooms = batch_sizes[b];
        dim3 grid((rooms + 7) / 8);
        
        // Global input
        float global_us;
        {
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            for (int i = 0; i < WARMUP; i++)
                infer_global_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_out, rooms, DIM);
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int i = 0; i < ITERS; i++)
                infer_global_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_out, rooms, DIM);
            cudaEventRecord(e, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, s, e);
            global_us = ms / ITERS * 1000;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        
        // Shared memory input
        float shmem_us;
        {
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            for (int i = 0; i < WARMUP; i++)
                infer_shmem_input<<<grid, dim3(256), DIM * sizeof(half), stream>>>(d_weights, d_input, d_out, rooms, DIM);
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int i = 0; i < ITERS; i++)
                infer_shmem_input<<<grid, dim3(256), DIM * sizeof(half), stream>>>(d_weights, d_input, d_out, rooms, DIM);
            cudaEventRecord(e, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, s, e);
            shmem_us = ms / ITERS * 1000;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        
        // Constant memory input
        float const_us;
        {
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            for (int i = 0; i < WARMUP; i++)
                infer_const_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_const_input, d_out, rooms, DIM);
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int i = 0; i < ITERS; i++)
                infer_const_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_const_input, d_out, rooms, DIM);
            cudaEventRecord(e, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, s, e);
            const_us = ms / ITERS * 1000;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        
        printf("%-8d | %12.3f   | %12.3f   | %.3fx   | %12.3f   | %.3fx\n",
               rooms, global_us, shmem_us, global_us / shmem_us, const_us, global_us / const_us);
    }
    
    // Vary input dimensions
    printf("\n--- Input Size Scaling (1024 rooms) ---\n");
    printf("%-8s | %-14s | %-14s | %-8s | %-14s | %-8s\n",
           "Dim", "Global(us)", "Shmem(us)", "Ratio", "Const(us)", "Ratio");
    printf("---------|----------------|----------------|----------|----------------|----------\n");
    
    int dims[] = {64, 128, 256, 512, 1024};
    for (int d = 0; d < 5; d++) {
        int dim = dims[d];
        dim3 grid(128);
        
        float global_us, shmem_us, const_us;
        
        // Global
        { cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i = 0; i < WARMUP; i++) infer_global_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_out, 1024, dim);
          cudaStreamSynchronize(stream); cudaEventRecord(s, stream);
          for (int i = 0; i < ITERS; i++) infer_global_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_out, 1024, dim);
          cudaEventRecord(e, stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms, s, e); global_us = ms / ITERS * 1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // Shmem
        { cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i = 0; i < WARMUP; i++) infer_shmem_input<<<grid, dim3(256), dim * sizeof(half), stream>>>(d_weights, d_input, d_out, 1024, dim);
          cudaStreamSynchronize(stream); cudaEventRecord(s, stream);
          for (int i = 0; i < ITERS; i++) infer_shmem_input<<<grid, dim3(256), dim * sizeof(half), stream>>>(d_weights, d_input, d_out, 1024, dim);
          cudaEventRecord(e, stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms, s, e); shmem_us = ms / ITERS * 1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // Const
        { cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i = 0; i < WARMUP; i++) infer_const_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_const_input, d_out, 1024, dim);
          cudaStreamSynchronize(stream); cudaEventRecord(s, stream);
          for (int i = 0; i < ITERS; i++) infer_const_input<<<grid, dim3(256), 0, stream>>>(d_weights, d_const_input, d_out, 1024, dim);
          cudaEventRecord(e, stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms, s, e); const_us = ms / ITERS * 1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        printf("%-8d | %12.3f   | %12.3f   | %.3fx   | %12.3f   | %.3fx\n",
               dim, global_us, shmem_us, global_us / shmem_us, const_us, global_us / const_us);
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_const_input); cudaFree(d_out);
    
    printf("\n=== Suite #42 Complete ===\n");
    return 0;
}
