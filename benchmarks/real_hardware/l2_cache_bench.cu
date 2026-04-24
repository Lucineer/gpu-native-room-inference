/**
 * L2 Cache Effectiveness Benchmark
 * 
 * Jetson Orin has 2MB L2 cache. If we can keep room weights in L2,
 * we avoid the 44 GB/s memory bandwidth ceiling entirely.
 * 
 * L2 bandwidth is typically 10-20x higher than DRAM.
 * 
 * This benchmark tests:
 * 1. How many rooms fit in L2 (256-dim weights)
 * 2. Repeated access to same rooms (cache hit scenario)
 * 3. Sequential vs strided access patterns
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 l2_cache_bench.cu -o l2_cache_bench
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void room_inference(const half* __restrict__ weights, const half* __restrict__ input,
                                float* __restrict__ output, int dim) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

// Same computation but with explicit L2 cache hints
__global__ void room_inference_cached(const half* __restrict__ weights, const half* __restrict__ input,
                                       float* __restrict__ output, int dim) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    // Use __ldcs (cached streaming load) for weights
    for (int i = lane; i < dim; i += 32) {
        float w = __half2float(__ldcs(&weights[room * dim + i]));
        sum += w * __half2float(input[i]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

// Persistently cached access (same rooms repeatedly)
__global__ void room_inference_persist(const half* __restrict__ weights, const half* __restrict__ input,
                                        float* __restrict__ output, int dim, int repeat) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int r = 0; r < repeat; r++) {
        for (int i = lane; i < dim; i += 32)
            sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

int main() {
    printf("============================================================\n");
    printf("L2 CACHE EFFECTIVENESS BENCHMARK\n");
    printf("Jetson Orin Nano — 2MB L2 Cache\n");
    printf("============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | L2: %d KB | Shared/SM: %d KB\n\n", 
           prop.name, prop.l2CacheSize / 1024, prop.sharedMemPerBlock / 1024);
    
    // L2 capacity analysis
    int dim = 256;
    size_t bytes_per_room = dim * sizeof(half); // 512 bytes per room weights
    int rooms_in_1mb = (1024 * 1024) / bytes_per_room;
    int rooms_in_2mb = (2 * 1024 * 1024) / bytes_per_room;
    
    printf("Weight data per room (256-dim FP16): %zu bytes\n", bytes_per_room);
    printf("Rooms fitting in 1MB L2: %d\n", rooms_in_1mb);
    printf("Rooms fitting in 2MB L2: %d\n", rooms_in_2mb);
    printf("Rooms fitting in 512KB L2: %d\n", (512 * 1024) / bytes_per_room);
    printf("Note: L2 is shared with all GPU workloads\n\n");
    
    const int MAX_ROOMS = 2048;
    const int ITERS = 100000;
    
    half *h_weights = (half*)malloc(MAX_ROOMS * dim * sizeof(half));
    half *h_input = (half*)malloc(dim * sizeof(half));
    
    srand(42);
    for (int i = 0; i < MAX_ROOMS * dim; i++)
        h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < dim; i++)
        h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_weights, *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, MAX_ROOMS * dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, MAX_ROOMS * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, MAX_ROOMS * dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, dim * sizeof(half), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test 1: Varying batch sizes (measure L2 effectiveness)
    printf(">>> Test 1: Batch Size vs L2 Hit Rate\n");
    printf("%-12s %-12s %-12s %-12s %-12s\n", "Batch", "us/room", "Room-qps", "Data in L2?", "Notes");
    printf("----------------------------------------------------------------\n");
    
    int batch_sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int num_sizes = 9;
    
    for (int b = 0; b < num_sizes; b++) {
        int batch = batch_sizes[b];
        int iters = batch <= 64 ? ITERS : (ITERS * 64 / batch);
        
        dim3 block(32);
        dim3 grid(batch);
        
        // Warmup
        for (int i = 0; i < 2000; i++)
            room_inference<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_inference<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        float us_per_room = (ms / iters / batch) * 1000.0;
        float room_qps = batch / (ms / iters / 1000.0);
        size_t total_data = (size_t)batch * bytes_per_room;
        int l2_pct = (int)((double)total_data / (2.0 * 1024 * 1024) * 100);
        
        const char* note = l2_pct <= 50 ? "L2 likely" : (l2_pct <= 100 ? "L2 partial" : "DRAM");
        
        printf("%-12d %-12.3f %-12.0f %-12s %d%% of L2\n",
               batch, us_per_room, room_qps, note, l2_pct);
    }
    
    // Test 2: Repeated access (same rooms, measuring cache warming)
    printf("\n>>> Test 2: Cache Warming — Repeated Access to Same Rooms\n");
    printf("Running same 16 rooms multiple times per kernel launch\n");
    printf("%-12s %-12s %-12s %-12s\n", "Repeats", "us/room", "Room-qps", "Speedup");
    printf("------------------------------------------------------------\n");
    
    int warm_rooms = 16;
    int repeats[] = {1, 2, 4, 8, 16, 32};
    int num_repeats = 6;
    
    for (int r = 0; r < num_repeats; r++) {
        int rep = repeats[r];
        dim3 block(32);
        dim3 grid(warm_rooms);
        
        for (int i = 0; i < 2000; i++)
            room_inference_persist<<<grid, block>>>(d_weights, d_input, d_output, dim, rep);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            room_inference_persist<<<grid, block>>>(d_weights, d_input, d_output, dim, rep);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        float total_rooms = (float)ITERS * warm_rooms * rep;
        float us_per_room = ms / total_rooms * 1000.0;
        float room_qps = total_rooms / (ms / 1000.0);
        
        printf("%-12d %-12.3f %-12.0f %-11.1fx\n", 
               rep, us_per_room, room_qps, (float)rep / us_per_room / (1.0 / 7.0));
    }
    
    // Test 3: Cached streaming load hint
    printf("\n>>> Test 3: __ldcs (cached streaming) vs Regular Load\n");
    printf("%-12s %-12s %-12s %-12s\n", "Batch", "Regular", "Cached", "Speedup");
    printf("------------------------------------------------------------\n");
    
    for (int b = 0; b < 5; b++) {
        int batch = batch_sizes[b];
        int iters = batch <= 64 ? ITERS : (ITERS * 64 / batch);
        
        dim3 block(32);
        dim3 grid(batch);
        
        // Regular
        for (int i = 0; i < 2000; i++)
            room_inference<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_inference<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms_reg;
        CUDA_CHECK(cudaEventElapsedTime(&ms_reg, start, stop));
        
        // Cached streaming
        for (int i = 0; i < 2000; i++)
            room_inference_cached<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_inference_cached<<<grid, block>>>(d_weights, d_input, d_output, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms_cach;
        CUDA_CHECK(cudaEventElapsedTime(&ms_cach, start, stop));
        
        printf("%-12d %-12.3f %-12.3f %-11.2fx\n",
               batch, ms_reg / iters * 1000.0, ms_cach / iters * 1000.0, ms_reg / ms_cach);
    }
    
    printf("\n============================================================\n");
    printf("ANALYSIS\n");
    printf("============================================================\n");
    printf("2MB L2 cache fits ~4000 rooms of 256-dim FP16 weights.\n");
    printf("At batch <= 256 rooms: all weights likely in L2.\n");
    printf("At batch > 2048 rooms: weights exceed L2, DRAM access needed.\n");
    printf("Cache warming: repeated access shows diminishing returns\n");
    printf("(L2 already warm from first access in same kernel).\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_weights);
    free(h_input);
    
    return 0;
}
