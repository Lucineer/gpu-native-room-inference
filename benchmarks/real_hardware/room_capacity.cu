/**
 * Suite #53: Room Capacity & Memory Scaling
 * 
 * Commercial question: "How many rooms can one Jetson Orin Nano 8GB serve?"
 * This determines deployment sizing for the deckboss fleet.
 * 
 * Tests:
 * 1. Memory capacity: max rooms at dim=256 before OOM
 * 2. Throughput at capacity: does qps hold as rooms approach memory limit
 * 3. Multi-model serving: different rooms with different dimensions
 * 4. Memory-per-room breakdown: weights, output, sparse overhead
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 room_capacity.cu -o room_capacity
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#define DIM 256
#define WARMUP 200
#define ITERS 5000

// V7 production kernel (contig8 general shuffle)
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

// Variable-dim kernel for multi-model serving
__global__ void infer_var_dim(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    const int* __restrict__ dims)  // per-room dimension
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    int room = room_base + room_idx;
    int dim = dims[room];
    
    // Weight offset: sum of all previous room dimensions
    int weight_offset = 0;
    for (int r = 0; r < room; r++) weight_offset += dims[r];
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[weight_offset + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("  CUDA ERROR at %s: %s\n", msg, cudaGetErrorString(err));
    }
}

int main() {
    printf("=== Suite #53: Room Capacity & Memory Scaling ===\n\n");
    
    // ===== Part 1: Memory Info =====
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("--- GPU Memory Status ---\n");
    printf("  Total: %.1f MB\n", total_mem / 1048576.0);
    printf("  Free:  %.1f MB\n", free_mem / 1048576.0);
    printf("  Used:  %.1f MB\n", (total_mem - free_mem) / 1048576.0);
    
    // ===== Part 2: Memory Per Room Breakdown =====
    printf("\n--- Memory Per Room (dim=256) ---\n");
    size_t weights_per_room = DIM * sizeof(half);
    size_t output_per_room = sizeof(float);
    size_t sparse_overhead = sizeof(int) + DIM * (sizeof(int) + sizeof(half)); // worst case CSR
    printf("  Weights (FP16):    %zu bytes (%.1f KB)\n", weights_per_room, weights_per_room / 1024.0);
    printf("  Output (FP32):     %zu bytes\n", output_per_room);
    printf("  Sparse CSR (max):  %zu bytes (%.1f KB)\n", sparse_overhead, sparse_overhead / 1024.0);
    
    // ===== Part 3: Max Rooms Before OOM =====
    printf("\n--- Max Room Capacity Test (dim=256) ---\n");
    printf("  Allocating rooms until OOM...\n");
    
    half *d_weights = nullptr, *d_input = nullptr;
    float *d_output = nullptr;
    
    // Allocate input first (shared, 512B)
    cudaMalloc(&d_input, DIM * sizeof(half));
    
    // Binary search for max rooms
    int lo = 1000, hi = 500000;
    int max_rooms = 0;
    
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        size_t needed = (size_t)mid * DIM * sizeof(half) + (size_t)mid * sizeof(float);
        
        if (d_weights) { cudaFree(d_weights); d_weights = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        
        cudaError_t err = cudaMalloc(&d_weights, needed * 0.99);  // weights
        if (err == cudaSuccess) {
            err = cudaMalloc(&d_output, mid * sizeof(float));  // output
            if (err == cudaSuccess) {
                max_rooms = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        } else {
            hi = mid - 1;
        }
    }
    
    printf("  Max rooms (dim=256): %d\n", max_rooms);
    printf("  Memory used: %.1f MB\n", (float)max_rooms * DIM * sizeof(half) / 1048576.0);
    
    // Check remaining memory
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  Remaining: %.1f MB\n", free_mem / 1048576.0);
    
    // ===== Part 4: Throughput at Various Room Counts =====
    printf("\n--- Throughput vs Room Count (dim=256) ---\n");
    printf("%-12s | %-10s | %-10s | %-12s | %-10s\n",
           "Rooms", "μs", "M qps", "Memory MB", "Efficiency");
    printf("-------------|------------|------------|--------------|------------\n");
    
    int test_counts[] = {64, 256, 1024, 4096, 8192, 16384, 32768, 65536};
    int n_tests = sizeof(test_counts) / sizeof(test_counts[0]);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Init input
    std::vector<half> hinput(DIM);
    for (int i = 0; i < DIM; i++)
        hinput[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    cudaMemcpy(d_input, hinput.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    for (int t = 0; t < n_tests; t++) {
        int rooms = test_counts[t];
        if (rooms > max_rooms) break;
        
        // Realloc
        if (d_weights) { cudaFree(d_weights); d_weights = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        
        size_t wmem = (size_t)rooms * DIM * sizeof(half);
        cudaMalloc(&d_weights, wmem);
        cudaMalloc(&d_output, rooms * sizeof(float));
        
        // Init weights
        std::vector<half> hw(rooms * DIM);
        for (size_t i = 0; i < (size_t)rooms * DIM; i++)
            hw[i] = __float2half(0.01f * (2.0f * (float)rand() / RAND_MAX - 1.0f));
        cudaMemcpy(d_weights, hw.data(), wmem, cudaMemcpyHostToDevice);
        
        dim3 grid((rooms + 7) / 8);
        
        // Benchmark
        cudaEvent_t se, ee;
        cudaEventCreate(&se);
        cudaEventCreate(&ee);
        
        for (int i = 0; i < WARMUP; i++)
            infer_v7<<<grid, 256, 0, stream>>>(d_weights, d_input, d_output, rooms, DIM);
        cudaStreamSynchronize(stream);
        
        cudaEventRecord(se, stream);
        for (int i = 0; i < ITERS; i++)
            infer_v7<<<grid, 256, 0, stream>>>(d_weights, d_input, d_output, rooms, DIM);
        cudaEventRecord(ee, stream);
        cudaStreamSynchronize(stream);
        
        float ms;
        cudaEventElapsedTime(&ms, se, ee);
        float us = ms / ITERS * 1000;
        float m_qps = rooms / us;
        
        cudaEventDestroy(se);
        cudaEventDestroy(ee);
        
        float mem_mb = (float)rooms * DIM * sizeof(half) / 1048576.0;
        float efficiency = m_qps / 161.0f * 100;  // vs peak 161M
        
        printf("%-12d | %8.2f   | %8.1f   | %10.1f   | %7.1f%%\n",
               rooms, us, m_qps, mem_mb, efficiency);
    }
    
    // ===== Part 5: Capacity at Various Dimensions =====
    printf("\n--- Max Rooms by Dimension ---\n");
    printf("%-8s | %-12s | %-12s | %-12s\n", "Dim", "Max Rooms", "Memory MB", "Weight KB");
    printf("---------|--------------|--------------|--------------\n");
    
    int dims[] = {32, 64, 128, 256, 384, 512, 768, 1024};
    int n_dims = sizeof(dims) / sizeof(dims[0]);
    
    for (int d = 0; d < n_dims; d++) {
        int dim = dims[d];
        size_t usable_mem = free_mem + (size_t)max_rooms * DIM * sizeof(half) + max_rooms * sizeof(float);
        usable_mem *= 0.95;  // leave 5% headroom
        size_t room_mem = (size_t)dim * sizeof(half) + sizeof(float);
        size_t max_r = usable_mem / room_mem;
        
        printf("%-8d | %-12zu | %-10.1f   | %-10.1f\n",
               dim, max_r, (float)max_r * dim * sizeof(half) / 1048576.0,
               (float)dim * sizeof(half) / 1024.0);
    }
    
    // ===== Part 6: Multi-Model Serving (mixed dimensions) =====
    printf("\n--- Multi-Model Serving (mixed dimensions) ---\n");
    printf("  Testing fleet scenario: mix of room types\n");
    
    // Simulated fleet: 50% dim=256, 30% dim=128, 20% dim=64
    int fleet_rooms = 4096;
    std::vector<int> room_dims(fleet_rooms);
    size_t total_weights = 0;
    for (int r = 0; r < fleet_rooms; r++) {
        float rnd = (float)rand() / RAND_MAX;
        if (rnd < 0.5) { room_dims[r] = 256; total_weights += 256; }
        else if (rnd < 0.8) { room_dims[r] = 128; total_weights += 128; }
        else { room_dims[r] = 64; total_weights += 64; }
    }
    
    printf("  Fleet: %d rooms (50%% dim256, 30%% dim128, 20%% dim64)\n", fleet_rooms);
    printf("  Total weight elements: %zu (%.1f MB)\n", total_weights, total_weights * sizeof(half) / 1048576.0);
    
    // Allocate and test uniform-dim baseline (all dim=256)
    {
        if (d_weights) cudaFree(d_weights);
        if (d_output) cudaFree(d_output);
        
        cudaMalloc(&d_weights, (size_t)fleet_rooms * DIM * sizeof(half));
        cudaMalloc(&d_output, fleet_rooms * sizeof(float));
        
        dim3 grid((fleet_rooms + 7) / 8);
        
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        for (int i = 0; i < WARMUP; i++)
            infer_v7<<<grid, 256, 0, stream>>>(d_weights, d_input, d_output, fleet_rooms, DIM);
        cudaStreamSynchronize(stream);
        cudaEventRecord(se, stream);
        for (int i = 0; i < ITERS; i++)
            infer_v7<<<grid, 256, 0, stream>>>(d_weights, d_input, d_output, fleet_rooms, DIM);
        cudaEventRecord(ee, stream);
        cudaStreamSynchronize(stream);
        
        float ms; cudaEventElapsedTime(&ms, se, ee);
        float us = ms / ITERS * 1000;
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        printf("  Uniform dim=256: %.2f μs (%.1f M qps)\n", us, fleet_rooms / us);
    }
    
    // Variable-dim kernel (realistic fleet)
    {
        if (d_weights) cudaFree(d_weights);
        if (d_output) cudaFree(d_output);
        
        int *d_dims;
        cudaMalloc(&d_weights, total_weights * sizeof(half));
        cudaMalloc(&d_output, fleet_rooms * sizeof(float));
        cudaMalloc(&d_dims, fleet_rooms * sizeof(int));
        cudaMemcpy(d_dims, room_dims.data(), fleet_rooms * sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 grid((fleet_rooms + 7) / 8);
        
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        for (int i = 0; i < WARMUP; i++)
            infer_var_dim<<<grid, 256, 0, stream>>>(d_weights, d_input, d_output, fleet_rooms, d_dims);
        cudaStreamSynchronize(stream);
        cudaEventRecord(se, stream);
        for (int i = 0; i < ITERS; i++)
            infer_var_dim<<<grid, 256, 0, stream>>>(d_weights, d_input, d_output, fleet_rooms, d_dims);
        cudaEventRecord(ee, stream);
        cudaStreamSynchronize(stream);
        
        float ms; cudaEventElapsedTime(&ms, se, ee);
        float us = ms / ITERS * 1000;
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        printf("  Variable-dim:     %.2f μs (%.1f M qps) — %.2fx vs uniform\n",
               us, fleet_rooms / us, (fleet_rooms / us) / (fleet_rooms / (ms / ITERS * 1000)));
        
        cudaFree(d_dims);
    }
    
    // Memory savings from variable-dim
    float uniform_mem = (float)fleet_rooms * DIM * sizeof(half) / 1048576.0;
    float var_mem = (float)total_weights * sizeof(half) / 1048576.0;
    printf("  Memory savings: %.1f MB → %.1f MB (%.1f%% reduction)\n",
           uniform_mem, var_mem, (1.0 - var_mem / uniform_mem) * 100);
    
    cudaStreamDestroy(stream);
    if (d_weights) cudaFree(d_weights);
    if (d_output) cudaFree(d_output);
    cudaFree(d_input);
    
    printf("\n=== Suite #53 Complete ===\n");
    return 0;
}
