/**
 * Suite #47: Dimension Scaling — How Throughput Scales with Room Size
 * 
 * All previous suites used dim=256. Real models have varying dimensions:
 * - Small embeddings: 64-128
 * - Standard: 256-512
 * - Large: 1024-2048
 * - XL: 4096+
 * 
 * Questions:
 * 1. How does latency scale with dimension?
 * 2. Where does the compute-bound vs memory-bound crossover occur?
 * 3. Is V7 still optimal at large dimensions?
 * 4. What's the optimal block size at each dimension?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 dim_scaling.cu -o dim_scaling
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#define MAX_DIM 4096
#define WARMUP 500
#define ITERS 10000

// General V7-style kernel with configurable rooms per block
__global__ void infer_general(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim, int rooms_per_block)
{
    int room_base = blockIdx.x * rooms_per_block;
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

int bench(const char* label, half* d_w, half* d_in, float* d_out,
          cudaStream_t stream, int rooms, int dim, int rooms_per_block, int threads) {
    dim3 grid((rooms + rooms_per_block - 1) / rooms_per_block);
    float us;
    
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < WARMUP; i++)
        infer_general<<<grid, dim3(threads), 0, stream>>>(d_w, d_in, d_out, rooms, dim, rooms_per_block);
    cudaStreamSynchronize(stream);
    cudaEventRecord(s, stream);
    for (int i = 0; i < ITERS; i++)
        infer_general<<<grid, dim3(threads), 0, stream>>>(d_w, d_in, d_out, rooms, dim, rooms_per_block);
    cudaEventRecord(e, stream);
    cudaStreamSynchronize(stream);
    float ms; cudaEventElapsedTime(&ms, s, e);
    us = ms / ITERS * 1000;
    cudaEventDestroy(s); cudaEventDestroy(e);
    
    float qps = rooms / (us / 1e6f) / 1e6f;
    float ops = (float)rooms * dim * 2;  // multiply-add = 2 ops
    float tflops = ops / (us / 1e6f) / 1e12f;
    float mem_bytes = (float)rooms * dim * 2 + dim * 2;  // weights + input (halfs)
    float bw_gb = mem_bytes / (us / 1e6f) / 1e9f;
    float arith_intensity = ops / mem_bytes;  // ops/byte
    
    printf("  %-20s | %6.2f  | %8.2f  | %6.3f  | %6.1f  | %4.1f\n",
           label, us, qps, tflops, bw_gb, arith_intensity);
    
    return 0;
}

int main() {
    printf("=== Suite #47: Dimension Scaling ===\n\n");
    
    int dims[] = {32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048};
    int num_dims = sizeof(dims) / sizeof(dims[0]);
    int rooms = 1024;
    
    half *d_weights, *d_input;
    float *d_out;
    
    // Allocate for max dim
    cudaMalloc(&d_weights, rooms * MAX_DIM * sizeof(half));
    cudaMalloc(&d_input, MAX_DIM * sizeof(half));
    cudaMalloc(&d_out, rooms * sizeof(float));
    
    std::vector<half> hw(rooms * MAX_DIM), hi(MAX_DIM);
    for (int i = 0; i < rooms * MAX_DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < MAX_DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / MAX_DIM * 6.2832f));
    
    cudaMemcpy(d_weights, hw.data(), rooms * MAX_DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), MAX_DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // ===== Part 1: Latency & Throughput vs Dimension =====
    printf("--- Part 1: Latency, Throughput, TFLOPS, Bandwidth (1024 rooms, 8 rooms/block, 256 threads) ---\n");
    printf("%-20s | %-6s  | %-8s  | %-6s  | %-6s  | %-4s\n",
           "Dimension", "us", "qps(M)", "TFLOPS", "BW(GB/s)", "OI");
    printf("----------------------|--------|----------|--------|--------|------\n");
    
    for (int d = 0; d < num_dims; d++) {
        int dim = dims[d];
        char label[32];
        snprintf(label, sizeof(label), "dim=%d", dim);
        bench(label, d_weights, d_input, d_out, stream, rooms, dim, 8, 256);
    }
    
    // ===== Part 2: Block Size Optimization per Dimension =====
    printf("\n--- Part 2: Optimal Block Size per Dimension (1024 rooms) ---\n");
    printf("%-8s | %-12s | %-12s | %-12s | %-12s\n",
           "Dim", "4r/128t(us)", "8r/256t(us)", "16r/512t(us)", "Best");
    printf("---------|--------------|--------------|--------------|----------\n");
    
    for (int d = 0; d < num_dims; d++) {
        int dim = dims[d];
        float t4, t8, t16;
        
        // 4 rooms/block, 128 threads
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          dim3 g((rooms+3)/4);
          for (int i=0;i<WARMUP;i++) infer_general<<<g,128,0,stream>>>(d_weights,d_input,d_out,rooms,dim,4);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_general<<<g,128,0,stream>>>(d_weights,d_input,d_out,rooms,dim,4);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); t4=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // 8 rooms/block, 256 threads
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          dim3 g((rooms+7)/8);
          for (int i=0;i<WARMUP;i++) infer_general<<<g,256,0,stream>>>(d_weights,d_input,d_out,rooms,dim,8);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_general<<<g,256,0,stream>>>(d_weights,d_input,d_out,rooms,dim,8);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); t8=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // 16 rooms/block, 512 threads
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          dim3 g((rooms+15)/16);
          for (int i=0;i<WARMUP;i++) infer_general<<<g,512,0,stream>>>(d_weights,d_input,d_out,rooms,dim,16);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_general<<<g,512,0,stream>>>(d_weights,d_input,d_out,rooms,dim,16);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); t16=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        float best = std::min({t4, t8, t16});
        const char* best_name = t4 == best ? "4r" : (t8 == best ? "8r" : "16r");
        
        printf("%-8d | %10.3f   | %10.3f   | %10.3f   | %s (%.2f)\n",
               dim, t4, t8, t16, best_name, best);
    }
    
    // ===== Part 3: Batch Size Scaling per Dimension =====
    printf("\n--- Part 3: Batch Size Scaling (best block size) ---\n");
    printf("%-8s | ", "Dim");
    int batches[] = {8, 64, 256, 1024, 4096};
    for (int b = 0; b < 5; b++) printf("%-8d rooms | ", batches[b]);
    printf("\n");
    printf("---------|");
    for (int b = 0; b < 5; b++) printf("--------------|");
    printf("\n");
    
    int test_dims[] = {64, 256, 512, 1024, 2048};
    for (int d = 0; d < 5; d++) {
        int dim = test_dims[d];
        printf("%-8d | ", dim);
        for (int b = 0; b < 5; b++) {
            int r = batches[b];
            int rpb = (dim <= 256) ? 8 : 4;  // fewer rooms per block at large dim
            int thr = rpb * 32;
            dim3 g((r + rpb - 1) / rpb);
            float us;
            { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
              for (int i=0;i<WARMUP;i++) infer_general<<<g,dim3(thr),0,stream>>>(d_weights,d_input,d_out,r,dim,rpb);
              cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
              for (int i=0;i<ITERS;i++) infer_general<<<g,dim3(thr),0,stream>>>(d_weights,d_input,d_out,r,dim,rpb);
              cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
              float ms; cudaEventElapsedTime(&ms,s,e); us=ms/ITERS*1000;
              cudaEventDestroy(s); cudaEventDestroy(e); }
            printf("%8.2f us  | ", us);
        }
        printf("\n");
    }
    
    // ===== Part 4: Arithmetic Intensity Analysis =====
    printf("\n--- Part 4: Compute vs Memory Bound Analysis ---\n");
    printf("Jetson Orin theoretical: ~68 GB/s bandwidth, ~0.5 TFLOPS FP16 (1024 cores)\n");
    printf("Ridge point: ops/byte where compute_time = memory_time\n");
    printf("  memory_time = bytes / 68e9\n");
    printf("  compute_time = ops / 0.5e12\n");
    printf("  ridge = 0.5e12 / 68e9 = 7.35 ops/byte\n\n");
    
    for (int d = 0; d < num_dims; d++) {
        int dim = dims[d];
        float ops = (float)rooms * dim * 2;
        float bytes = (float)rooms * dim * 2 + dim * 2;
        float oi = ops / bytes;
        const char* bound = oi < 7.35f ? "MEMORY" : "COMPUTE";
        printf("  dim=%4d: OI=%.2f ops/byte -> %s bound\n", dim, oi, bound);
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_out);
    
    printf("\n=== Suite #47 Complete ===\n");
    return 0;
}
