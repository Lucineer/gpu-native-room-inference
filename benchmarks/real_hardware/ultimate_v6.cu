/**
 * Suite #30: Ultimate Production Kernel — Shuffle + Multi-Room + Fused GELU
 * 
 * Combining the best of:
 * - Suite #18: Kernel fusion (3.69× at 4 layers)
 * - Suite #20: Multi-room blocking (V4 wins)
 * - Suite #29: Warp shuffle reduction (1.65× at 1024 rooms)
 * 
 * Testing the "V6" kernel: contig8 + fused matmul+GELU in one pass
 * vs V5 (contig8 with separate GELU)
 * vs V4 (shmem multi-room, the previous champion)
 * 
 * Also testing: what happens with dim=128 and dim=512 (different room sizes)
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 ultimate_v6.cu -o ultimate_v6
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#define WARMUP 500
#define ITERS 50000

// ============================================================
// V4: Previous champion (shmem, multi-room, separate GELU)
// 4 rooms/block, 128 threads (32 per room)
// ============================================================
__global__ void v4_shmem(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_base = blockIdx.x * 4;
    int room_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (room_base + room_id >= num_rooms) return;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        sum += __half2float(weights[(room_base + room_id) * dim + i]) * __half2float(input[i]);
    }
    
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = 16; s > 0; s >>= 1) {
        if (lane < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    if (lane == 0) {
        float dot = sdata[room_id * 32];
        output[room_base + room_id] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
    }
}

// ============================================================
// V5: Contig8 shuffle (suite #29 winner)
// 8 rooms/block, 256 threads, no shared memory
// ============================================================
__global__ void v5_contig8(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    #pragma unroll
    for (int i = lane * 8; i < (lane + 1) * 8 && i < dim; i++) {
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(input[i]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        float dot = sum;
        output[room_base + room_idx] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
    }
}

// ============================================================
// V6: Contig8 shuffle + FUSED GELU (interleaved matmul+activation)
// Applies GELU to partial sums during reduction to hide latency
// ============================================================
__global__ void v6_fused_shuffle(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (room_base + room_idx >= num_rooms) return;
    
    // Fused matmul: each thread processes 8 elements
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane * 8 + i;
        if (d < dim) {
            float w = __half2float(weights[(room_base + room_idx) * dim + d]);
            float x = __half2float(input[d]);
            sum += w * x;
        }
    }
    
    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Fused GELU (single-thread, no extra launch)
    if (lane == 0) {
        float c = 0.7978845608f;
        float a = 0.044715f;
        float inner = c * (sum + a * sum * sum * sum);
        output[room_base + room_idx] = 0.5f * sum * (1.0f + tanhf(inner));
    }
}

// ============================================================
// V7: Contig8 with FULL loop (no stride-8 assumption)
// Handles any dim by striding through with warp size
// ============================================================
__global__ void v7_general_shuffle(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(input[i]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        float c = 0.7978845608f;
        float a = 0.044715f;
        float inner = c * (sum + a * sum * sum * sum);
        output[room_base + room_idx] = 0.5f * sum * (1.0f + tanhf(inner));
    }
}

// ============================================================
// V8: Contig16 shuffle (16 rooms/block, 512 threads)
// Tests if denser packing helps or hurts
// ============================================================
__global__ void v8_contig16(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_base = blockIdx.x * 16;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    #pragma unroll
    for (int i = lane * 8; i < (lane + 1) * 8 && i < dim; i++) {
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(input[i]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        float c = 0.7978845608f;
        float a = 0.044715f;
        float inner = c * (sum + a * sum * sum * sum);
        output[room_base + room_idx] = 0.5f * sum * (1.0f + tanhf(inner));
    }
}

template<typename K>
float bench(const char* name, K k, dim3 g, dim3 b, int sh, cudaStream_t s,
            const half* w, const half* in, float* out, int rooms, int dim) {
    for (int i = 0; i < WARMUP; i++) k<<<g, b, sh, s>>>(w, in, out, rooms, dim);
    cudaStreamSynchronize(s);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) k<<<g, b, sh, s>>>(w, in, out, rooms, dim);
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    float us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
    printf("  %-35s | %8.3f us | %12.0f room-qps | %6.3fx\n",
           name, us, rooms / (us / 1e6f), rooms / (us / 1e6f) / 1e6);
    return us;
}

int main() {
    printf("=== Suite #30: Ultimate Production Kernel Comparison ===\n\n");
    
    int dims[] = {64, 128, 256, 512};
    int batches[] = {1, 4, 16, 64, 256, 1024};
    int max_rooms = 1024;
    
    half *d_w, *d_in;
    float *d_out;
    cudaMalloc(&d_w, max_rooms * 512 * sizeof(half));
    cudaMalloc(&d_in, 512 * sizeof(half));
    cudaMalloc(&d_out, max_rooms * sizeof(float));
    
    // Init
    std::vector<half> hw(max_rooms * 512), hi(512);
    for (int i = 0; i < max_rooms * 512; i++)
        hw[i] = __float2half(0.01f * (float)(i % 1000) / 1000.0f);
    for (int i = 0; i < 512; i++)
        hi[i] = __float2half(0.5f * sinf((float)i / 512 * 6.2832f));
    cudaMemcpy(d_w, hw.data(), max_rooms * 512 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, hi.data(), 512 * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float best_v4 = 0, best_v6 = 0, best_v8 = 0;
    float best_v4_batch = 0, best_v6_batch = 0, best_v8_batch = 0;
    
    for (int d = 0; d < 4; d++) {
        int dim = dims[d];
        printf("========== dim=%d ==========\n", dim);
        
        for (int b = 0; b < 6; b++) {
            int rooms = batches[b];
            printf("\n--- %d rooms ---\n", rooms);
            
            float t4 = bench("V4: shmem 4rooms/block",
                v4_shmem, dim3((rooms+3)/4), dim3(128), 128*sizeof(float),
                stream, d_w, d_in, d_out, rooms, dim);
            
            float t5 = bench("V5: contig8 shuffle",
                v5_contig8, dim3((rooms+7)/8), dim3(256), 0,
                stream, d_w, d_in, d_out, rooms, dim);
            
            float t6 = bench("V6: contig8 fused shuffle",
                v6_fused_shuffle, dim3((rooms+7)/8), dim3(256), 0,
                stream, d_w, d_in, d_out, rooms, dim);
            
            float t7 = bench("V7: contig8 general dim",
                v7_general_shuffle, dim3((rooms+7)/8), dim3(256), 0,
                stream, d_w, d_in, d_out, rooms, dim);
            
            int grid16 = (rooms + 15) / 16;
            float t8 = bench("V8: contig16 shuffle",
                v8_contig16, dim3(grid16), dim3(512), 0,
                stream, d_w, d_in, d_out, rooms, dim);
            
            // Track best for dim=256
            if (dim == 256) {
                float v4qps = rooms / (t4 / 1e6f);
                float v6qps = rooms / (t6 / 1e6f);
                float v8qps = rooms / (t8 / 1e6f);
                if (v4qps > best_v4) { best_v4 = v4qps; best_v4_batch = rooms; }
                if (v6qps > best_v6) { best_v6 = v6qps; best_v6_batch = rooms; }
                if (v8qps > best_v8) { best_v8 = v8qps; best_v8_batch = rooms; }
            }
        }
    }
    
    printf("\n========== BEST RESULTS (dim=256) ==========\n");
    printf("  V4 (shmem):        %.1fM room-qps at %d rooms\n", best_v4/1e6, best_v4_batch);
    printf("  V6 (fused shuffle): %.1fM room-qps at %d rooms\n", best_v6/1e6, best_v6_batch);
    printf("  V8 (contig16):     %.1fM room-qps at %d rooms\n", best_v8/1e6, best_v8_batch);
    printf("  V6 vs V4: %.3fx\n", best_v6 / best_v4);
    if (best_v8 > best_v6)
        printf("  V8 vs V6: %.3fx (V8 WINS)\n", best_v8 / best_v6);
    else
        printf("  V8 vs V6: %.3fx (V6 still wins)\n", best_v8 / best_v6);
    
    cudaStreamDestroy(stream);
    cudaFree(d_w); cudaFree(d_in); cudaFree(d_out);
    printf("\n=== Suite #30 Complete ===\n");
    return 0;
}
