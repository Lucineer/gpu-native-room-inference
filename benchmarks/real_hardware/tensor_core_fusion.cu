/**
 * Tensor Core Fusion for Multi-Room GPU Inference
 * Jetson Orin Nano (sm_87), FP16 tensor cores, CUDA 12.6
 * 
 * Real hardware benchmark — compiled and run on actual Jetson Orin Nano 8GB
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda::wmma;

#define ROOM_SIZE 256
#define WARP_SIZE 32
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * Tensor core fusion kernel (fixed: uses shared memory for WMMA loads)
 */
__global__ void tensor_core_room_fusion(
    const half* __restrict__ room_weights,
    const half* __restrict__ room_input,
    float* __restrict__ room_output,
    const int num_rooms
) {
    // Shared memory for WMMA fragments (aligned to 128 bytes)
    __shared__ half smem_input[TILE_K];
    __shared__ half smem_weights[TILE_K];
    
    fragment<matrix_a, TILE_M, TILE_N, TILE_K, half, row_major> a_frag;
    fragment<matrix_b, TILE_M, TILE_N, TILE_K, half, col_major> b_frag;
    fragment<accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    
    int warp_id = blockIdx.x;
    int lane = threadIdx.x;
    
    if (warp_id >= num_rooms) return;
    
    const half* weights = room_weights + warp_id * ROOM_SIZE;
    
    // Load input into shared memory using first 16 lanes
    if (lane < TILE_K) {
        smem_input[lane] = room_input[lane];
    }
    __syncwarp();
    
    // Load input from shared memory into fragment A
    load_matrix_sync(a_frag, smem_input, TILE_K);
    
    float result = 0.0f;
    
    // Process weights in 16-element chunks via tensor cores
    for (int chunk = 0; chunk < 16; chunk++) {
        int offset = chunk * TILE_K;
        if (offset >= ROOM_SIZE) break;
        
        // Load weights chunk into shared memory
        if (lane < TILE_K) {
            smem_weights[lane] = weights[offset + lane];
        }
        __syncwarp();
        
        // Load from shared memory into fragment B
        load_matrix_sync(b_frag, smem_weights, TILE_K);
        
        // Zero accumulator
        fill_fragment(c_frag, 0.0f);
        
        // TENSOR CORE: 16x16x16 matrix multiply-accumulate
        mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        // Extract and accumulate results
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++) {
            result += c_frag.x[i];
        }
    }
    
    // Lane 0 writes result with GELU activation
    if (lane == 0) {
        float x = result;
        float gelu = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        room_output[warp_id] = gelu;
    }
}

/**
 * Standard warp room baseline
 */
__global__ void standard_warp_room(
    const half* __restrict__ room_weights,
    const half* __restrict__ room_input,
    float* __restrict__ room_output,
    const int num_rooms
) {
    int warp_id = blockIdx.x;
    int lane = threadIdx.x;
    
    if (warp_id >= num_rooms) return;
    
    const half* weights = room_weights + warp_id * ROOM_SIZE;
    
    float sum = 0.0f;
    for (int i = lane; i < ROOM_SIZE; i += WARP_SIZE) {
        sum += __half2float(__hmul(weights[i], room_input[i]));
    }
    
    // Warp shuffle reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        float x = sum;
        room_output[warp_id] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

/**
 * Thread-as-room baseline
 */
__global__ void thread_room(
    const half* __restrict__ room_weights,
    const half* __restrict__ room_input,
    float* __restrict__ room_output,
    const int num_rooms
) {
    int room_id = blockIdx.x;
    if (room_id >= num_rooms) return;
    
    const half* weights = room_weights + room_id * ROOM_SIZE;
    float sum = 0.0f;
    
    for (int i = threadIdx.x; i < ROOM_SIZE; i += blockDim.x) {
        sum += __half2float(__hmul(weights[i], room_input[i]));
    }
    
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) {
        float x = sdata[0];
        room_output[room_id] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

int main(int argc, char** argv) {
    printf("=== Tensor Core Fusion Benchmark ===\n");
    printf("Compiled: CUDA 12.6, sm_87\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute: %d.%d | SMs: %d | Clock: %.0f MHz | Mem: %.0f MB\n",
           prop.major, prop.minor, prop.multiProcessorCount,
           prop.clockRate / 1000.0, prop.totalGlobalMem / 1048576.0);
    printf("Shared Mem/Block: %.0f KB\n\n", prop.sharedMemPerBlock / 1024.0);
    
    const int num_rooms = 6;
    const int ITERS = 100000;
    const size_t weights_bytes = num_rooms * ROOM_SIZE * sizeof(half);
    const size_t input_bytes = ROOM_SIZE * sizeof(half);
    const size_t output_bytes = num_rooms * sizeof(float);
    
    half* h_weights = (half*)malloc(weights_bytes);
    half* h_input = (half*)malloc(input_bytes);
    float* h_tc = (float*)malloc(output_bytes);
    float* h_warp = (float*)malloc(output_bytes);
    float* h_thread = (float*)malloc(output_bytes);
    
    srand(42);
    for (int r = 0; r < num_rooms; r++)
        for (int i = 0; i < ROOM_SIZE; i++)
            h_weights[r * ROOM_SIZE + i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < ROOM_SIZE; i++)
        h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_w, *d_in;
    float *d_tc, *d_warp, *d_thread;
    CUDA_CHECK(cudaMalloc(&d_w, weights_bytes));
    CUDA_CHECK(cudaMalloc(&d_in, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_tc, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_warp, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_thread, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_w, h_weights, weights_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, h_input, input_bytes, cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < 200; i++) {
        tensor_core_room_fusion<<<num_rooms, WARP_SIZE>>>(d_w, d_in, d_tc, num_rooms);
        standard_warp_room<<<num_rooms, WARP_SIZE>>>(d_w, d_in, d_warp, num_rooms);
        thread_room<<<num_rooms, 256>>>(d_w, d_in, d_thread, num_rooms);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Benchmark tensor core fusion
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        tensor_core_room_fusion<<<num_rooms, WARP_SIZE>>>(d_w, d_in, d_tc, num_rooms);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_tc = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_tc, start, stop));
    float lat_tc = ms_tc / ITERS;
    
    // Benchmark standard warp
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        standard_warp_room<<<num_rooms, WARP_SIZE>>>(d_w, d_in, d_warp, num_rooms);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_warp = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_warp, start, stop));
    float lat_warp = ms_warp / ITERS;
    
    // Benchmark thread
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        thread_room<<<num_rooms, 256>>>(d_w, d_in, d_thread, num_rooms);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_thread = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_thread, start, stop));
    float lat_thread = ms_thread / ITERS;
    
    CUDA_CHECK(cudaMemcpy(h_tc, d_tc, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_warp, d_warp, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_thread, d_thread, output_bytes, cudaMemcpyDeviceToHost));
    
    printf("=== REAL HARDWARE RESULTS (%d rooms, %d iterations) ===\n\n", num_rooms, ITERS);
    printf("%-25s %12s %12s %12s\n", "Implementation", "Latency", "Throughput", "vs Thread");
    printf("%-25s %12s %12s %12s\n", "-------------------------", "------------", "------------", "------------");
    printf("%-25s %10.4f ms %10.0f/s %10.2fx\n", "Thread-as-Room", lat_thread, 1000.0/lat_thread, 1.0);
    printf("%-25s %10.4f ms %10.0f/s %10.2fx\n", "Standard Warp", lat_warp, 1000.0/lat_warp, lat_thread/lat_warp);
    printf("%-25s %10.4f ms %10.0f/s %10.2fx\n", "Tensor Core Fusion", lat_tc, 1000.0/lat_tc, lat_thread/lat_tc);
    
    printf("\n=== TENSOR CORE vs WARP ===\n");
    printf("Speedup: %.2fx\n", lat_warp / lat_tc);
    printf("Throughput gain: +%.0f qps\n", (1000.0/lat_tc) - (1000.0/lat_warp));
    
    printf("\n=== PROJECTED vs ACTUAL ===\n");
    printf("Projected tensor core: 0.015 ms, 66,666 qps\n");
    printf("Actual tensor core:    %.4f ms, %.0f qps\n", lat_tc, 1000.0/lat_tc);
    
    printf("\n=== CORRECTNESS ===\n");
    float max_diff = 0;
    for (int i = 0; i < num_rooms; i++) {
        float d = fabsf(h_thread[i] - h_tc[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("Max diff (tc vs thread): %.6f %s\n", max_diff, max_diff < 0.1 ? "✓ PASS" : "✗ FAIL");
    
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_tc));
    CUDA_CHECK(cudaFree(d_warp));
    CUDA_CHECK(cudaFree(d_thread));
    free(h_weights); free(h_input); free(h_tc); free(h_warp); free(h_thread);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
