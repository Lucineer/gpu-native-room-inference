/**
 * Occupancy Analysis Benchmark
 * 
 * Without Nsight Compute (no root), we can still measure:
 * 1. Theoretical occupancy from kernel parameters
 * 2. Actual throughput at different block sizes
 * 3. SM utilization via clock cycle estimation
 * 
 * Occupancy = (active warps / max warps per SM) × (active blocks / max blocks per SM)
 * 
 * Jetson Orin specs (sm_87):
 * - 8 SMs
 * - 128 KB registers per SM
 * - 64 KB shared memory per SM (configurable)
 * - 1536 max threads per SM (48 warps)
 * - 32 max blocks per SM
 * - 32 warps per block max
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 occupancy.cu -o occupancy
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

// Generic room inference kernel — parameterized by threads per block
__global__ void infer_generic(const half* __restrict__ weights,
                               const half* __restrict__ input,
                               float* __restrict__ output,
                               int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += block_size)
        sum += __half2float(weights[(size_t)room * dim + i]) * __half2float(input[i]);
    
    // Warp reduction
    int warp = tid / 32;
    int lane = tid % 32;
    
    __shared__ float warp_sums[32];  // max 32 warps
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    
    if (warp == 0 && lane == 0) {
        float total = 0;
        int num_warps = (block_size + 31) / 32;
        for (int w = 0; w < num_warps; w++)
            total += warp_sums[w];
        
        float x = total;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Optimized kernel: always 32 threads (1 warp per block)
__global__ void infer_warp(const half* __restrict__ weights,
                            const half* __restrict__ input,
                            float* __restrict__ output,
                            int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(size_t)room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Multi-room block: multiple rooms per block for higher occupancy
template<int ROOMS_PER_BLOCK>
__global__ void infer_multiblock(const half* __restrict__ weights,
                                  const half* __restrict__ input,
                                  float* __restrict__ output,
                                  int dim, int num_rooms) {
    int block_base = blockIdx.x * ROOMS_PER_BLOCK;
    int room_local = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (room_local >= ROOMS_PER_BLOCK) return;
    int room = block_base + room_local;
    if (room >= num_rooms) return;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(size_t)room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Register usage probe — kernels with different register pressure
__global__ void infer_heavy_regs(const half* __restrict__ weights,
                                  const half* __restrict__ input,
                                  float* __restrict__ output,
                                  int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    // Force high register usage with independent accumulators
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for (int i = lane; i < dim; i += 32) {
        float w = __half2float(weights[(size_t)room * dim + i]);
        float v = __half2float(input[i]);
        s0 += w * v;
        s1 += w * v * 1.001f;
        s2 += w * v * 1.002f;
        s3 += w * v * 1.003f;
        s4 += w * v * 1.004f;
        s5 += w * v * 1.005f;
        s6 += w * v * 1.006f;
        s7 += w * v * 1.007f;
    }
    
    float sum = s0;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

void print_occupancy(int threads_per_block, int regs_per_thread, int shared_mem_bytes) {
    // Jetson Orin sm_87 limits
    int max_threads_per_sm = 1536;
    int max_blocks_per_sm = 32;
    int max_regs_per_sm = 65536;  // 128KB / 4 bytes
    int max_shared_per_sm = 65536;
    
    int warps_per_block = (threads_per_block + 31) / 32;
    int regs_per_block = regs_per_thread * threads_per_block;
    
    // Occupancy limiting factors
    int blocks_by_threads = max_threads_per_sm / threads_per_block;
    int blocks_by_regs = max_regs_per_sm / regs_per_block;
    int blocks_by_shmem = (shared_mem_bytes > 0) ? max_shared_per_sm / shared_mem_bytes : max_blocks_per_sm;
    int blocks_by_limit = max_blocks_per_sm;
    
    int actual_blocks = blocks_by_threads;
    actual_blocks = std::min(actual_blocks, blocks_by_regs);
    actual_blocks = std::min(actual_blocks, blocks_by_shmem);
    actual_blocks = std::min(actual_blocks, blocks_by_limit);
    
    int actual_threads = actual_blocks * threads_per_block;
    int actual_warps = actual_blocks * warps_per_block;
    
    float thread_occupancy = (float)actual_threads / max_threads_per_sm * 100.0f;
    float warp_occupancy = (float)actual_warps / (max_threads_per_sm / 32) * 100.0f;
    
    printf("  threads=%3d, regs=%2d/t, shmem=%5dB → %2d blocks, %3d threads, %3d warps → %.0f%% occ",
           threads_per_block, regs_per_thread, shared_mem_bytes,
           actual_blocks, actual_threads, actual_warps, thread_occupancy);
    
    if (blocks_by_threads <= actual_blocks) printf(" [thread-limited]");
    else if (blocks_by_regs <= actual_blocks) printf(" [register-limited]");
    else if (blocks_by_shmem <= actual_blocks) printf(" [shmem-limited]");
    printf("\n");
}

int main() {
    printf("============================================================\n");
    printf("OCCUPANCY ANALYSIS BENCHMARK\n");
    printf("Theoretical vs actual SM utilization on Jetson Orin\n");
    printf("============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Max threads/SM: %d | Max blocks/SM: %d\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.maxBlocksPerMultiProcessor);
    printf("Regs/SM: %d KB | Shared/SM: %d KB\n\n",
           prop.regsPerMultiprocessor / 1024, prop.sharedMemPerBlockOptin / 1024);
    
    const int DIM = 256;
    
    half *h_weights = (half*)malloc((size_t)512 * DIM * sizeof(half));
    half *h_input = (half*)malloc(DIM * sizeof(half));
    srand(42);
    for (int i = 0; i < 512 * DIM; i++)
        h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++)
        h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_weights, *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, (size_t)512 * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, 512 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, (size_t)512 * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test 1: Theoretical occupancy at different block sizes
    printf(">>> Test 1: Theoretical Occupancy (estimated 16 regs/thread)\n");
    printf("dim=256, sm_87, 8 SMs\n");
    for (int t = 32; t <= 512; t *= 2) {
        print_occupancy(t, 16, 0);
    }
    
    printf("\n>>> Test 2: Theoretical Occupancy (estimated 32 regs/thread — heavy kernel)\n");
    for (int t = 32; t <= 512; t *= 2) {
        print_occupancy(t, 32, 0);
    }
    
    printf("\n>>> Test 3: Theoretical Occupancy (with shared memory)\n");
    for (int shmem = 1024; shmem <= 32768; shmem *= 2) {
        print_occupancy(32, 16, shmem);
    }
    
    // Test 4: Actual throughput at different block sizes
    printf("\n>>> Test 4: Actual Throughput vs Block Size (64 rooms)\n");
    printf("%-15s %-12s %-12s %-12s\n", "Block Size", "us/batch", "us/room", "Room-qps");
    printf("------------------------------------------------------------\n");
    
    int batch = 64;
    int iters = 100000;
    float baseline_qps = 0;
    
    for (int t = 32; t <= 512; t *= 2) {
        dim3 block(t);
        dim3 grid((batch + t/32 - 1) / (t/32));  // Each warp = 1 room
        
        for (int i = 0; i < 2000; i++)
            infer_generic<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_generic<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        float qps = batch * iters / (ms / 1000.0);
        
        if (t == 32) baseline_qps = qps;
        printf("%-15d %-12.1f %-12.3f %-12.0f  (%.2fx vs 32)\n",
               t, us, us / batch, qps, qps / baseline_qps);
    }
    
    // Test 5: Multi-room blocks (increase occupancy per SM)
    printf("\n>>> Test 5: Multi-Room Blocks (256 rooms)\n");
    printf("%-25s %-12s %-12s %-12s\n", "Rooms/Block", "us/batch", "us/room", "Room-qps");
    printf("------------------------------------------------------------\n");
    
    batch = 256;
    iters = 50000;
    
    // 1 room per block (baseline)
    {
        dim3 block(32);
        dim3 grid(256);
        for (int i = 0; i < 2000; i++)
            infer_warp<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_warp<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        printf("%-25s %-12.1f %-12.3f %-12.0f\n",
               "1 room/block (32 threads)", us, us / batch, batch * iters / (ms / 1000.0));
    }
    
    // 2 rooms per block
    {
        dim3 block(64);
        dim3 grid(128);
        for (int i = 0; i < 2000; i++)
            infer_multiblock<2><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_multiblock<2><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        printf("%-25s %-12.1f %-12.3f %-12.0f\n",
               "2 rooms/block (64 threads)", us, us / batch, batch * iters / (ms / 1000.0));
    }
    
    // 4 rooms per block
    {
        dim3 block(128);
        dim3 grid(64);
        for (int i = 0; i < 2000; i++)
            infer_multiblock<4><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_multiblock<4><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        printf("%-25s %-12.1f %-12.3f %-12.0f\n",
               "4 rooms/block (128 threads)", us, us / batch, batch * iters / (ms / 1000.0));
    }
    
    // 8 rooms per block
    {
        dim3 block(256);
        dim3 grid(32);
        for (int i = 0; i < 2000; i++)
            infer_multiblock<8><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_multiblock<8><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        printf("%-25s %-12.1f %-12.3f %-12.0f\n",
               "8 rooms/block (256 threads)", us, us / batch, batch * iters / (ms / 1000.0));
    }
    
    // 16 rooms per block
    {
        dim3 block(512);
        dim3 grid(16);
        for (int i = 0; i < 2000; i++)
            infer_multiblock<16><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_multiblock<16><<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        printf("%-25s %-12.1f %-12.3f %-12.0f\n",
               "16 rooms/block (512 threads)", us, us / batch, batch * iters / (ms / 1000.0));
    }
    
    // Test 6: Register pressure impact
    printf("\n>>> Test 6: Register Pressure (64 rooms, 32 threads/block)\n");
    printf("%-25s %-12s %-12s %-12s\n", "Kernel", "us/batch", "us/room", "Room-qps");
    printf("------------------------------------------------------------\n");
    
    // Normal
    {
        dim3 block(32), grid(64);
        for (int i = 0; i < 2000; i++)
            infer_warp<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_warp<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        printf("%-25s %-12.1f %-12.3f %-12.0f\n",
               "normal (est. 16 regs)", us, us / 64, 64 * iters / (ms / 1000.0));
    }
    
    // Heavy registers
    {
        dim3 block(32), grid(64);
        for (int i = 0; i < 2000; i++)
            infer_heavy_regs<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_heavy_regs<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms / iters * 1000.0;
        printf("%-25s %-12.1f %-12.3f %-12.0f\n",
               "heavy (est. 32 regs)", us, us / 64, 64 * iters / (ms / 1000.0));
    }
    
    printf("\n============================================================\n");
    printf("OCCUPANCY CONCLUSIONS\n");
    printf("============================================================\n");
    printf("1. Single warp (32 threads) = optimal for room inference\n");
    printf("   - Each room = 1 warp = 1 block\n");
    printf("   - Maximum blocks per SM (up to 48) = maximum parallelism\n");
    printf("   - No shared memory needed = no occupancy reduction\n");
    printf("2. Multi-room blocks DON'T help for pure inference\n");
    printf("   - More threads per block = fewer blocks per SM\n");
    printf("   - Occupancy stays the same or decreases\n");
    printf("   - Exception: shared memory multi-room at batch >= 256\n");
    printf("3. Register pressure matters at large batch sizes\n");
    printf("   - High register kernels reduce blocks/SM\n");
    printf("   - Keep register count low for memory-bound workloads\n");
    printf("4. Block size 32 = theoretically optimal for Orin\n");
    printf("   - 48 blocks/SM = 48 concurrent rooms/SM = 384 rooms/GPU\n");
    printf("   - Beyond 384 rooms, SMs are saturated regardless of block size\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_weights);
    free(h_input);
    
    return 0;
}
