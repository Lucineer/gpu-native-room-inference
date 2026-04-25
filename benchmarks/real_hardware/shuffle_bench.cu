/**
 * Suite #29: Warp Shuffle vs Shared Memory Reduction
 * 
 * Hypothesis: Warp shuffle (__shfl_down_sync) eliminates shared memory
 * synchronization overhead for dot product reduction.
 * 
 * On Orin (sm_87), warps are 32 threads. For dim=256, each room needs
 * 256 threads × 1 element = one warp per room with 8 elements per thread.
 * 
 * Compare:
 * 1. Shared memory reduction (baseline from V4)
 * 2. Warp shuffle reduction (no shared mem, no __syncthreads)
 * 3. Mixed: warp shuffle + block-level reduction
 * 4. Warp shuffle with 4 rooms/block (V4 style)
 * 5. Warp shuffle with 8 rooms/block (denser packing)
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 shuffle_bench.cu -o shuffle_bench
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include <cuda_fp16.h>

#define DIM 256
#define WARMUP 500
#define ITERS 50000

// ============================================================
// Kernel 1: Shared memory reduction (baseline)
// ============================================================
__global__ void dot_shmem(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_id = blockIdx.x;
    if (room_id >= num_rooms) return;
    
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    
    // Each thread processes dim/num_threads elements
    int elements = (dim + blockDim.x - 1) / blockDim.x;
    float sum = 0.0f;
    for (int i = 0; i < elements; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < dim) {
            sum += __half2float(weights[room_id * dim + idx]) * __half2float(input[idx]);
        }
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) {
        float dot = sdata[0];
        output[room_id] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
    }
}

// ============================================================
// Kernel 2: Warp shuffle reduction (no shared memory)
// ============================================================
__global__ void dot_shuffle(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_id = blockIdx.x;
    if (room_id >= num_rooms) return;
    
    int tid = threadIdx.x;
    
    // Each thread processes dim/num_threads elements
    int elements = (dim + blockDim.x - 1) / blockDim.x;
    float sum = 0.0f;
    for (int i = 0; i < elements; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < dim) {
            sum += __half2float(weights[room_id * dim + idx]) * __half2float(input[idx]);
        }
    }
    
    // Warp shuffle reduction — no shared memory, no __syncthreads
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread of first warp writes result
    if (tid == 0) {
        float dot = sum;
        output[room_id] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
    }
}

// ============================================================
// Kernel 3: Warp shuffle + block reduction (for > 32 threads)
// ============================================================
__global__ void dot_shuffle_block(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_id = blockIdx.x;
    if (room_id >= num_rooms) return;
    
    int tid = threadIdx.x;
    int elements = (dim + blockDim.x - 1) / blockDim.x;
    float sum = 0.0f;
    for (int i = 0; i < elements; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < dim) {
            sum += __half2float(weights[room_id * dim + idx]) * __half2float(input[idx]);
        }
    }
    
    // Warp-level reduction
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Block-level: first thread of each warp
    extern __shared__ float warp_sums[];
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float block_sum = (lane < blockDim.x / warpSize) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            float dot = block_sum;
            output[room_id] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
        }
    }
}

// ============================================================
// Kernel 4: Multi-room with warp shuffle (4 rooms/block)
// ============================================================
__global__ void dot_shuffle_multi4(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_base = blockIdx.x * 4;
    int room_id = threadIdx.x % 4;  // 0-3 within block
    int dim_idx = threadIdx.x / 4;  // thread within room
    
    if (room_base + room_id >= num_rooms) return;
    if (dim_idx * 8 >= dim) return;  // Each thread handles 8 elements
    
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8 && (dim_idx * 8 + i) < dim; i++) {
        int d = dim_idx * 8 + i;
        sum += __half2float(weights[(room_base + room_id) * dim + d]) * __half2float(input[d]);
    }
    
    // Warp shuffle: 32 threads per warp, but they're interleaved (4 rooms × 8 threads)
    // We need per-room reduction. Use __shfl_down_sync with stride 4.
    // Thread layout: [R0T0, R1T0, R2T0, R3T0, R0T1, R1T1, ...] 
    // Lane = threadIdx.x, stride = 4
    for (int offset = 32 / 2; offset >= 4; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, sum, offset);
        // Only add if the source is from the same room
        if ((threadIdx.x + offset) / 4 == threadIdx.x / 4) {
            sum += other;
        }
    }
    
    // Last thread per room (lane 31, 27, 23, 19) — nope, this is wrong.
    // Let me use a different approach: separate warps per room.
    
    // Actually, use __shfl_down_sync with stride 4 and mask appropriately
    // Simpler: just use shared memory for this interleaved layout
    // The shuffle approach works best with contiguous room threads.
    
    // Fallback to shared memory for correctness
    // (This kernel is measuring the multi-room benefit, not shuffle)
    extern __shared__ float sdata[];
    int room_local = room_id;
    int room_tid = dim_idx;  // Thread index within room (0-7 for 8 elements/thread = 64 total per room)
    
    // Actually let me just use the warp approach differently.
    // Reassign: threads 0-31 handle room 0, threads 32-63 handle room 1, etc.
    // But we only have 128 threads total (32 per room × 4 rooms).
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction within each room's 32 threads
    // Room r starts at thread r*32
    int base = room_id * 32 + dim_idx;
    for (int s = 4; s > 0; s >>= 1) {  // 32 threads per room, reduce to 1
        if (dim_idx < s) {
            sdata[room_id * 32 + dim_idx] += sdata[room_id * 32 + dim_idx + s];
        }
        __syncthreads();
    }
    
    if (dim_idx == 0) {
        float dot = sdata[room_id * 32];
        output[room_base + room_id] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
    }
}

// ============================================================
// Kernel 5: Contiguous warp layout, 4 rooms/block (proper shuffle)
// ============================================================
__global__ void dot_shuffle_contig4(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    // Block has 128 threads: 4 rooms × 32 threads/room
    // Threads 0-31: room 0, threads 32-63: room 1, etc.
    int room_base = blockIdx.x * 4;
    int room_idx = threadIdx.x / 32;  // Which room (0-3)
    int lane = threadIdx.x % 32;      // Lane within warp (0-31)
    
    if (room_base + room_idx >= num_rooms) return;
    
    // Each thread handles dim/32 = 8 elements
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane * 8 + i;
        if (d < dim) {
            sum += __half2float(weights[(room_base + room_idx) * dim + d]) * __half2float(input[d]);
        }
    }
    
    // Warp shuffle reduction (32 threads per room = exactly 1 warp)
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Lane 0 of each warp writes result
    if (lane == 0) {
        float dot = sum;
        output[room_base + room_idx] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
    }
}

// ============================================================
// Kernel 6: Contiguous warp, 8 rooms/block
// ============================================================
__global__ void dot_shuffle_contig8(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    // 256 threads: 8 rooms × 32 threads/room = 2 warps per block
    // Actually, Orin max 16 blocks/SM. 256 threads = good occupancy.
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane * 8 + i;
        if (d < dim) {
            sum += __half2float(weights[(room_base + room_idx) * dim + d]) * __half2float(input[d]);
        }
    }
    
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        float dot = sum;
        output[room_base + room_idx] = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
    }
}

// Benchmark helper
template<typename Kernel>
float bench(const char* name, Kernel kernel, dim3 grid, dim3 block, int shmem,
            cudaStream_t stream, const half* w, const half* in, float* out, int rooms, int dim) {
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        kernel<<<grid, block, shmem, stream>>>(w, in, out, rooms, dim);
    }
    cudaStreamSynchronize(stream);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        kernel<<<grid, block, shmem, stream>>>(w, in, out, rooms, dim);
    }
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    
    float total_us = std::chrono::duration<float, std::micro>(end - start).count();
    float us_per_iter = total_us / ITERS;
    float room_qps = rooms / (us_per_iter / 1e6f);
    
    printf("  %-30s | %8.3f us | %12.0f room-qps\n", name, us_per_iter, room_qps);
    return us_per_iter;
}

int main() {
    printf("=== Suite #29: Warp Shuffle vs Shared Memory Reduction ===\n\n");
    
    const int batch_sizes[] = {1, 4, 16, 64, 256, 1024};
    const int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    // Allocate
    half* d_weights, *d_input;
    float* d_output;
    cudaMalloc(&d_weights, 1024 * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_output, 1024 * sizeof(float));
    
    // Init
    std::vector<half> h_weights(1024 * DIM);
    std::vector<half> h_input(DIM);
    for (int i = 0; i < 1024 * DIM; i++)
        h_weights[i] = __float2half(0.01f * (float)(i % 1000) / 1000.0f);
    for (int i = 0; i < DIM; i++)
        h_input[i] = __float2half(0.5f * sinf((float)i / DIM * 6.2832f));
    
    cudaMemcpy(d_weights, h_weights.data(), 1024 * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    printf("Dim: %d, Warmup: %d, Iters: %d\n\n", DIM, WARMUP, ITERS);
    
    float baseline_shmem[6] = {0};
    
    for (int b = 0; b < num_batches; b++) {
        int rooms = batch_sizes[b];
        printf("--- Batch: %d rooms ---\n", rooms);
        
        int grid_x = rooms;
        
        // K1: Shared memory (32 threads)
        float t1 = bench("K1: shmem (32 threads)",
            dot_shmem, dim3(grid_x), dim3(32), 32 * sizeof(float),
            stream, d_weights, d_input, d_output, rooms, DIM);
        baseline_shmem[b] = t1;
        
        // K1b: Shared memory (256 threads)
        float t1b = bench("K1b: shmem (256 threads)",
            dot_shmem, dim3(grid_x), dim3(256), 256 * sizeof(float),
            stream, d_weights, d_input, d_output, rooms, DIM);
        
        // K2: Warp shuffle (32 threads, no shared mem)
        float t2 = bench("K2: shuffle (32 threads)",
            dot_shuffle, dim3(grid_x), dim3(32), 0,
            stream, d_weights, d_input, d_output, rooms, DIM);
        
        // K3: Warp shuffle + block (256 threads)
        float t3 = bench("K3: shuffle+block (256)",
            dot_shuffle_block, dim3(grid_x), dim3(256), 8 * sizeof(float),
            stream, d_weights, d_input, d_output, rooms, DIM);
        
        // K5: Contiguous warp 4 rooms/block
        int grid5 = (rooms + 3) / 4;
        float t5 = bench("K5: contig4 (128 threads, shuffle)",
            dot_shuffle_contig4, dim3(grid5), dim3(128), 0,
            stream, d_weights, d_input, d_output, rooms, DIM);
        
        // K6: Contiguous warp 8 rooms/block
        int grid6 = (rooms + 7) / 8;
        float t6 = bench("K6: contig8 (256 threads, shuffle)",
            dot_shuffle_contig8, dim3(grid6), dim3(256), 0,
            stream, d_weights, d_input, d_output, rooms, DIM);
        
        printf("\n  Speedups vs shmem(32):\n");
        printf("    shuffle(32):     %.3fx\n", t1 / t2);
        printf("    shuffle+block:   %.3fx\n", t1 / t3);
        printf("    contig4 shuffle: %.3fx\n", t1 / t5);
        printf("    contig8 shuffle: %.3fx\n", t1 / t6);
        printf("\n");
    }
    
    // Summary table
    printf("\n=== Summary ===\n");
    printf("%-8s | %-12s | %-12s | %-12s | %-12s\n",
           "Rooms", "shmem(32)", "shuffle(32)", "contig4", "contig8");
    printf("---------|--------------|--------------|--------------|-------------\n");
    // (Already printed above, this is for reference)
    
    cudaStreamDestroy(stream);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Suite #29 Complete ===\n");
    return 0;
}
