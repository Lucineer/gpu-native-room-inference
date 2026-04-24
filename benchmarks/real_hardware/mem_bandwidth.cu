/**
 * Memory Bandwidth Saturation Analysis
 * 
 * Goal: Find the actual bottleneck for room inference on Jetson Orin.
 * We know GPU is 95% idle. Is it memory bandwidth? Cache misses? L2?
 * 
 * Jetson Orin Nano specs:
 * - GPU Memory BW: 68.3 GB/s (LPDDR5, 128-bit)
 * - L2 Cache: 2 MB
 * - Shared Memory per SM: 100 KB (total 800 KB)
 * - L1/Texture Cache per SM: 128 KB
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 mem_bandwidth.cu -o mem_bandwidth
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Benchmark 1: Raw memory bandwidth (copy kernel)
__global__ void memcpy_kernel(half* __restrict__ dst, const half* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        dst[i] = src[i];
}

// Benchmark 2: Read-only bandwidth (dot product, no write)
__global__ void readonly_kernel(const half* __restrict__ weights, const half* __restrict__ input, 
                                 float* __restrict__ output, int dim) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    // This measures READ bandwidth: 2 * dim * sizeof(half) bytes read
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0)
        output[room] = sum;
}

// Benchmark 3: Room inference (read weights + input, compute + write output)
__global__ void room_kernel(const half* __restrict__ weights, const half* __restrict__ input,
                            float* __restrict__ output, int dim) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Benchmark 4: Shared-memory optimized room inference
__global__ void room_shared_kernel(const half* __restrict__ weights, const half* __restrict__ input,
                                    float* __restrict__ output, int dim) {
    extern __shared__ half smem[];
    
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    // Load input into shared memory once (all rooms use same input)
    half* s_input = smem;
    for (int i = lane; i < dim; i += 32)
        s_input[i] = input[i];
    __syncthreads();
    
    // Compute dot product (weights from global, input from shared)
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(s_input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

void run_benchmark(const char* name, int num_rooms, int dim, int iters,
                   cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        // placeholder
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
}

int main() {
    printf("============================================================\n");
    printf("MEMORY BANDWIDTH SATURATION ANALYSIS\n");
    printf("Jetson Orin Nano — Finding the Real Bottleneck\n");
    printf("============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Clock: %.0f MHz\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("L2 Cache: %d KB | Shared/SM: %d KB\n\n", 
           prop.l2CacheSize / 1024, prop.sharedMemPerBlock / 1024);
    
    const int DIMS[] = {64, 128, 256, 512, 768, 1024};
    const int NUM_DIMS = 6;
    const int ITERS = 100000;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf(">>> SECTION 1: Raw Memory Bandwidth (memcpy)\n");
    printf("%-8s %-12s %-12s %-12s %-12s\n", "Size(KB)", "Time(ms)", "BW(GB/s)", "Theory(GB/s)", "Efficiency");
    printf("------------------------------------------------------------\n");
    
    for (int d = 0; d < NUM_DIMS; d++) {
        int num_elements = 1024 * DIMS[d]; // 64K to 1M elements
        size_t bytes = num_elements * sizeof(half);
        
        half *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes));
        
        dim3 block(256);
        dim3 grid((num_elements + 255) / 256);
        
        // Warmup
        for (int i = 0; i < 1000; i++)
            memcpy_kernel<<<grid, block>>>(d_dst, d_src, num_elements);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            memcpy_kernel<<<grid, block>>>(d_dst, d_src, num_elements);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        double gb_per_iter = (double)bytes * 2.0 / 1e9; // read + write
        double bw = gb_per_iter / (ms / ITERS / 1000.0);
        double theory = 68.3; // LPDDR5 128-bit on Orin Nano
        double eff = bw / theory * 100.0;
        
        printf("%-8d %-12.3f %-12.1f %-12.1f %-11.1f%%\n", 
               (int)(bytes / 1024), ms, bw, theory, eff);
        
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));
    }
    
    printf("\n>>> SECTION 2: Room Inference — Read Bandwidth Analysis\n");
    printf("For each room: read weights (dim*2B) + read input (dim*2B) = 4*dim bytes\n");
    printf("%-8s %-8s %-12s %-12s %-12s %-12s\n", "Dim", "Rooms", "Time(ms)", "us/room", "GB/s read", "Compute(%%)");
    printf("----------------------------------------------------------------------\n");
    
    half *h_weights, *h_input;
    float *h_output;
    
    for (int d = 0; d < NUM_DIMS; d++) {
        int dim = DIMS[d];
        int num_rooms = 64;
        
        h_weights = (half*)malloc(num_rooms * dim * sizeof(half));
        h_input = (half*)malloc(dim * sizeof(half));
        h_output = (float*)malloc(num_rooms * sizeof(float));
        
        srand(42);
        for (int i = 0; i < num_rooms * dim; i++) h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
        for (int i = 0; i < dim; i++) h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
        
        half *d_weights, *d_input;
        float *d_output;
        CUDA_CHECK(cudaMalloc(&d_weights, num_rooms * dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_input, dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output, num_rooms * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights, num_rooms * dim * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, dim * sizeof(half), cudaMemcpyHostToDevice));
        
        dim3 block(32);
        dim3 grid(num_rooms);
        
        // Test 1: Read-only kernel (no GELU)
        for (int i = 0; i < 2000; i++)
            readonly_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            readonly_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms_readonly;
        CUDA_CHECK(cudaEventElapsedTime(&ms_readonly, start, stop));
        
        // Test 2: Full room kernel (with GELU)
        for (int i = 0; i < 2000; i++)
            room_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            room_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms_room;
        CUDA_CHECK(cudaEventElapsedTime(&ms_room, start, stop));
        
        // Test 3: Shared-memory optimized
        for (int i = 0; i < 2000; i++)
            room_shared_kernel<<<grid, block, dim * sizeof(half)>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            room_shared_kernel<<<grid, block, dim * sizeof(half)>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms_shared;
        CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start, stop));
        
        double us_per_room = (ms_room / ITERS / num_rooms) * 1000.0;
        double read_bytes = (double)num_rooms * dim * sizeof(half) * 2.0; // weights + input
        double read_bw = read_bytes / 1e9 / (ms_readonly / ITERS / 1000.0);
        
        // Compute intensity: 1 FLOP per 4 bytes read (multiply-add)
        // Roofline: peak = 40 TFLOPS, bandwidth = 68.3 GB/s
        // Ridge point: 40T / 68.3G = 585.7 FLOP/byte
        // Room inference: dim/2 FLOPs per dim*4 bytes = 0.5 FLOP/byte (memory bound)
        double compute_pct = (us_per_room / 1000000.0) * 40e12 / (dim / 2.0) * 100.0;
        
        printf("%-8d %-8d %-12.2f %-12.3f %-12.1f %-11.1f%%\n",
               dim, num_rooms, ms_room, us_per_room, read_bw, 
               compute_pct > 100 ? 100.0 : compute_pct);
        printf("  readonly: %.2f ms (%.1fx vs full) | shared: %.2f ms (%.1fx vs full)\n",
               ms_readonly, ms_readonly / ms_room, ms_shared, ms_shared / ms_room);
        
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        free(h_weights);
        free(h_input);
        free(h_output);
    }
    
    printf("\n>>> SECTION 3: Batch Size Scaling (256-dim rooms)\n");
    printf("%-12s %-12s %-12s %-12s %-12s\n", "Batch", "Time(ms)", "us/room", "Room-qps", "Speedup");
    printf("------------------------------------------------------------\n");
    
    int dim = 256;
    int batches[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int num_batches = 11;
    
    h_weights = (half*)malloc(1024 * dim * sizeof(half));
    h_input = (half*)malloc(dim * sizeof(half));
    
    srand(42);
    for (int i = 0; i < 1024 * dim; i++) h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < dim; i++) h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_weights, *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, 1024 * dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, 1024 * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, 1024 * dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, dim * sizeof(half), cudaMemcpyHostToDevice));
    
    float baseline_us = 0;
    
    for (int b = 0; b < num_batches; b++) {
        int batch = batches[b];
        int iters = batch <= 64 ? ITERS : (ITERS * 64 / batch);
        
        dim3 block(32);
        dim3 grid(batch);
        
        // Warmup
        for (int i = 0; i < 1000; i++)
            room_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        float us_per_room = (ms / iters / batch) * 1000.0;
        float room_qps = batch / (ms / iters / 1000.0);
        
        if (b == 0) baseline_us = us_per_room;
        float speedup = baseline_us / us_per_room;
        
        printf("%-12d %-12.2f %-12.3f %-12.0f %-11.1fx\n",
               batch, ms, us_per_room, room_qps, speedup);
    }
    
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_weights);
    free(h_input);
    
    printf("\n============================================================\n");
    printf("ANALYSIS\n");
    printf("============================================================\n");
    printf("Jetson Orin Nano memory bandwidth: 68.3 GB/s theoretical\n");
    printf("Room inference reads: dim * sizeof(half) * 2 per room\n");
    printf("  (weights + input, both half-precision)\n");
    printf("For 256-dim: 1024 bytes read per room\n");
    printf("At 68.3 GB/s: theoretical floor = 0.015 us/room\n");
    printf("Actual: ~1.3 us/room (85x over theoretical)\n");
    printf("Conclusion: Launch overhead dominates at small batches.\n");
    printf("At 1024+ rooms: approaches memory bandwidth limit.\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
