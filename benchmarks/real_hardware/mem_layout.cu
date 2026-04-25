/**
 * Suite #41: Memory Layout — Row-Major vs Interleaved vs Transposed
 * 
 * All previous suites stored weights as: W[room0_dim0, room0_dim1, ..., room1_dim0, ...]
 * This is row-major: each room's 512 bytes are contiguous.
 * 
 * Alternative layouts:
 * 1. Interleaved: W[room0_dim0, room1_dim0, ..., room0_dim1, room1_dim1, ...]
 *    Better for coalescing when multiple rooms access same dimension index
 * 2. Transposed: W[dim0_room0, dim0_room1, ..., dim1_room0, ...]
 *    Each warp reads 32 consecutive rooms at same dim index
 * 
 * The V7 kernel reads: weights[room * dim + lane + i*32]
 * With 8 rooms per block (threads 0-7 = room 0, 8-15 = room 0, ...):
 * Thread 0 reads room0[0], thread 32 reads room1[0], thread 64 reads room2[0], etc.
 * These are room0*256+0, room1*256+0, room2*256+0 = 0, 256, 512, 768, 1024, 1536, 2048, 2304
 * NOT contiguous! Stride of 256 (512 bytes) between room0's first element and room1's.
 * 
 * Interleaved layout would make these contiguous:
 * room0_dim0, room1_dim0, ..., room7_dim0, room0_dim1, room1_dim1, ..., room7_dim1, ...
 * Thread 0 reads [0], thread 32 reads [1], thread 64 reads [2], etc. — CONTIGUOUS!
 * 
 * But wait — within a warp (threads 0-31), each thread reads a different room.
 * Thread 0 = room 0, thread 1 = room 0, ..., thread 31 = room 0 (same room, stride-32)
 * No — V7 uses threadIdx.x / 32 = room_idx, threadIdx.x % 32 = lane
 * So threads 0-31 are ALL room 0 (different lanes)
 * Threads 32-63 are ALL room 1
 * 
 * Within a warp, all 32 threads read the SAME room, different dimension indices.
 * Thread 0: room*dim + 0, thread 1: room*dim + 1, ..., thread 31: room*dim + 31
 * These ARE contiguous! Row-major is already optimal for single-warp reads.
 * 
 * BUT — across warps in the same block (8 warps = 8 rooms):
 * Warp 0 reads room0: addresses 0-31
 * Warp 1 reads room1: addresses 256-287 (stride = dim = 256)
 * Not contiguous across warps. If warps execute concurrently (they do on sm_87),
 * the memory controller sees accesses to 0-31 and 256-287 simultaneously.
 * 
 * Interleaved layout: warp 0 reads 0-31, warp 1 reads 32-63 — PERFECT coalescing!
 * 
 * This should improve L2 cache hit rate and memory throughput for multi-room blocks.
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 mem_layout.cu -o mem_layout
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

// Row-major: W[room * dim + i]
__global__ void infer_rowmajor(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

// Interleaved: W[i * num_rooms + room]
// Memory layout: [room0_d0, room1_d0, ..., roomN_d0, room0_d1, room1_d1, ..., roomN_d1, ...]
__global__ void infer_interleaved(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[i * num_rooms + room]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

// Block-interleaved: W[block * dim * 8 + room_in_block * dim + i]
// Each 8-room block is contiguous, but blocks may not be adjacent
// Same as row-major for 8-room blocks, but tests cache line alignment
__global__ void infer_block_interleaved(
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
    // Offset within the 8-room block
    int block_offset = blockIdx.x * 8 * dim + room_idx * dim;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[block_offset + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room_base + room_idx] = sum;
}

// Padded row-major: each room padded to 512 bytes (256 halfs → 256 halfs, no padding needed at dim=256)
// But test with 320-byte padding to see if cache line alignment helps
__global__ void infer_padded(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim, int stride)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * stride + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

int main() {
    printf("=== Suite #41: Memory Layout Optimization ===\n\n");
    
    int batch_sizes[] = {1, 8, 32, 64, 256, 512, 1024, 2048, 4096};
    int max_rooms = 4096;
    
    // Allocate
    half *d_rowmajor, *d_interleaved, *d_block_inter, *d_padded;
    half *d_input;
    float *d_out;
    cudaMalloc(&d_rowmajor, max_rooms * DIM * sizeof(half));
    cudaMalloc(&d_interleaved, max_rooms * DIM * sizeof(half));
    cudaMalloc(&d_block_inter, max_rooms * DIM * sizeof(half));
    cudaMalloc(&d_padded, max_rooms * 512 * sizeof(half));  // 512 half stride (1024 bytes)
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_out, max_rooms * sizeof(float));
    
    // Init data
    std::vector<half> hw(max_rooms * DIM), hi(DIM);
    for (int i = 0; i < max_rooms * DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    
    // Row-major: W[room * dim + i]
    cudaMemcpy(d_rowmajor, hw.data(), max_rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Interleaved: W[i * num_rooms + room]
    std::vector<half> hw_inter(max_rooms * DIM);
    for (int i = 0; i < DIM; i++)
        for (int r = 0; r < max_rooms; r++)
            hw_inter[i * max_rooms + r] = hw[r * DIM + i];
    cudaMemcpy(d_interleaved, hw_inter.data(), max_rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Block-interleaved: same as row-major for sequential blocks
    cudaMemcpy(d_block_inter, hw.data(), max_rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Padded: W[room * 512 + i] (pad to 512 halfs = 1024 bytes per room)
    std::vector<half> hw_padded(max_rooms * 512, __float2half(0.0f));
    for (int r = 0; r < max_rooms; r++)
        for (int i = 0; i < DIM; i++)
            hw_padded[r * 512 + i] = hw[r * DIM + i];
    cudaMemcpy(d_padded, hw_padded.data(), max_rooms * 512 * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    printf("%-8s | %-14s | %-14s | %-8s | %-14s | %-8s | %-14s | %-8s\n",
           "Rooms", "Row-Major(us)", "Interleaved", "Ratio", "Block-Inter", "Ratio", "Padded(512)", "Ratio");
    printf("---------|----------------|----------------|----------|----------------|----------|----------------|----------\n");
    
    float baseline_rm = 0;
    for (int b = 0; b < 9; b++) {
        int rooms = batch_sizes[b];
        dim3 grid((rooms + 7) / 8);
        
        // Row-major
        {
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            for (int i = 0; i < WARMUP; i++)
                infer_rowmajor<<<grid, dim3(256), 0, stream>>>(d_rowmajor, d_input, d_out, rooms, DIM);
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int i = 0; i < ITERS; i++)
                infer_rowmajor<<<grid, dim3(256), 0, stream>>>(d_rowmajor, d_input, d_out, rooms, DIM);
            cudaEventRecord(e, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, s, e);
            baseline_rm = ms / ITERS * 1000;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        
        // Interleaved
        float inter_us;
        {
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            for (int i = 0; i < WARMUP; i++)
                infer_interleaved<<<grid, dim3(256), 0, stream>>>(d_interleaved, d_input, d_out, rooms, DIM);
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int i = 0; i < ITERS; i++)
                infer_interleaved<<<grid, dim3(256), 0, stream>>>(d_interleaved, d_input, d_out, rooms, DIM);
            cudaEventRecord(e, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, s, e);
            inter_us = ms / ITERS * 1000;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        
        // Block-interleaved (same layout, different access pattern)
        float block_us;
        {
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            for (int i = 0; i < WARMUP; i++)
                infer_block_interleaved<<<grid, dim3(256), 0, stream>>>(d_block_inter, d_input, d_out, rooms, DIM);
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int i = 0; i < ITERS; i++)
                infer_block_interleaved<<<grid, dim3(256), 0, stream>>>(d_block_inter, d_input, d_out, rooms, DIM);
            cudaEventRecord(e, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, s, e);
            block_us = ms / ITERS * 1000;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        
        // Padded (stride=512)
        float padded_us;
        {
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            for (int i = 0; i < WARMUP; i++)
                infer_padded<<<grid, dim3(256), 0, stream>>>(d_padded, d_input, d_out, rooms, DIM, 512);
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int i = 0; i < ITERS; i++)
                infer_padded<<<grid, dim3(256), 0, stream>>>(d_padded, d_input, d_out, rooms, DIM, 512);
            cudaEventRecord(e, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, s, e);
            padded_us = ms / ITERS * 1000;
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        
        printf("%-8d | %12.3f   | %12.3f   | %.3fx   | %12.3f   | %.3fx   | %12.3f   | %.3fx\n",
               rooms, baseline_rm, inter_us, baseline_rm / inter_us,
               block_us, baseline_rm / block_us, padded_us, baseline_rm / padded_us);
    }
    
    // Verify correctness
    printf("\n--- Correctness ---\n");
    {
        int rooms = 8;
        dim3 grid(1);
        
        infer_rowmajor<<<grid, dim3(256), 0, stream>>>(d_rowmajor, d_input, d_out, rooms, DIM);
        infer_interleaved<<<grid, dim3(256), 0, stream>>>(d_interleaved, d_input, d_out + rooms, rooms, DIM);
        cudaStreamSynchronize(stream);
        
        std::vector<float> o1(rooms), o2(rooms);
        cudaMemcpy(o1.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(o2.data(), d_out + rooms, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_err = 0;
        for (int i = 0; i < rooms; i++)
            max_err = std::max(max_err, fabsf(o1[i] - o2[i]));
        printf("  Row-major vs Interleaved max error: %.8f\n", max_err);
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_rowmajor); cudaFree(d_interleaved); cudaFree(d_block_inter);
    cudaFree(d_padded); cudaFree(d_input); cudaFree(d_out);
    
    printf("\n=== Suite #41 Complete ===\n");
    return 0;
}
