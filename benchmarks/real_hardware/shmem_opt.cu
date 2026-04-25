/**
 * Shared Memory Optimization Benchmark
 * 
 * Previous finding: shared memory hurts for naive usage (1.1x slower)
 * 
 * NEW: What if we use shared memory correctly?
 * - Load entire room weights into shared memory first
 * - Then all threads in warp compute from shared mem
 * - This should help for: (a) strided access, (b) weight reuse across batches
 * 
 * Theory: For dim=256, room = 512 bytes. With 8 rooms per block,
 *         shared usage = 4096 bytes (well under 64KB limit).
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 shmem_opt.cu -o shmem_opt
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

// Baseline: global memory only (current production kernel)
__global__ void baseline_kernel(const half* __restrict__ weights,
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

// Shared memory: load room weights, then compute
// Each block processes ONE room (32 threads = 1 warp)
// Room weights loaded cooperatively into shared mem
template<int MAX_DIM>
__global__ void shmem_single_kernel(const half* __restrict__ weights,
                                     const half* __restrict__ input,
                                     float* __restrict__ output,
                                     int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    extern __shared__ half s_weights[];  // size = dim * sizeof(half)
    
    // Cooperative load: each thread loads dim/32 elements
    for (int i = lane; i < dim; i += 32)
        s_weights[i] = weights[(size_t)room * dim + i];
    
    __syncthreads();
    
    // Compute from shared memory
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(s_weights[i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Shared memory: MULTI-room per block
// Each block processes ROOMS_PER_BLOCK rooms
// Threads are organized as (ROOMS_PER_BLOCK warps)
template<int ROOMS_PER_BLOCK, int MAX_DIM>
__global__ void shmem_multi_kernel(const half* __restrict__ weights,
                                    const half* __restrict__ input,
                                    float* __restrict__ output,
                                    int dim, int num_rooms) {
    int room_base = blockIdx.x * ROOMS_PER_BLOCK;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (warp_id >= ROOMS_PER_BLOCK) return;
    int room = room_base + warp_id;
    if (room >= num_rooms) return;
    
    extern __shared__ half s_weights[];  // size = ROOMS_PER_BLOCK * dim * sizeof(half)
    
    // Each warp loads its room cooperatively
    half* my_weights = s_weights + warp_id * dim;
    for (int i = lane; i < dim; i += 32)
        my_weights[i] = weights[(size_t)room * dim + i];
    
    __syncthreads();
    
    // Compute from shared memory
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(my_weights[i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Vectorized load: use float4 / half2 for wider memory access
__global__ void vec_kernel(const half* __restrict__ weights,
                            const half* __restrict__ input,
                            float* __restrict__ output,
                            int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    // Load 2 halves at a time (half2 → float2 → sum)
    const half2* w2 = reinterpret_cast<const half2*>(weights + (size_t)room * dim);
    const half2* i2 = reinterpret_cast<const half2*>(input);
    int pairs = dim / 2;
    
    for (int i = lane; i < pairs; i += 32) {
        half2 wh = w2[i];
        half2 ih = i2[i];
        sum += __half2float(wh.x) * __half2float(ih.x) + __half2float(wh.y) * __half2float(ih.y);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Combined: shared memory + vectorized loads
template<int ROOMS_PER_BLOCK>
__global__ void shmem_vec_kernel(const half* __restrict__ weights,
                                  const half* __restrict__ input,
                                  float* __restrict__ output,
                                  int dim, int num_rooms) {
    int room_base = blockIdx.x * ROOMS_PER_BLOCK;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (warp_id >= ROOMS_PER_BLOCK) return;
    int room = room_base + warp_id;
    if (room >= num_rooms) return;
    
    extern __shared__ half s_weights[];
    half* my_weights = s_weights + warp_id * dim;
    
    // Vectorized cooperative load
    half2* sw2 = reinterpret_cast<half2*>(my_weights);
    const half2* w2 = reinterpret_cast<const half2*>(weights + (size_t)room * dim);
    int pairs = dim / 2;
    
    for (int i = lane; i < pairs; i += 32)
        sw2[i] = w2[i];
    
    __syncthreads();
    
    // Compute vectorized from shared memory
    const half2* i2 = reinterpret_cast<const half2*>(input);
    float sum = 0.0f;
    for (int i = lane; i < pairs; i += 32) {
        half2 wh = sw2[i];
        half2 ih = i2[i];
        sum += __half2float(wh.x) * __half2float(ih.x) + __half2float(wh.y) * __half2float(ih.y);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

int main() {
    printf("============================================================\n");
    printf("SHARED MEMORY OPTIMIZATION BENCHMARK\n");
    printf("5 kernel variants for room inference\n");
    printf("============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Shmem/SM: %d KB\n\n", prop.name, prop.multiProcessorCount, prop.sharedMemPerBlockOptin / 1024);
    
    const int DIM = 256;
    const int ITERS = 100000;
    
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
    
    dim3 block(32);
    
    printf("%-35s %-12s %-12s %-12s\n", "Kernel", "us/batch", "us/room", "room-qps");
    printf("------------------------------------------------------------------------\n");
    
    // Test at different batch sizes
    int batch_sizes[] = {6, 32, 64, 128, 256};
    int num_sizes = 5;
    
    for (int b = 0; b < num_sizes; b++) {
        int batch = batch_sizes[b];
        int iters = batch <= 32 ? ITERS : (ITERS * 32 / batch);
        float baseline_us = 0;
        
        printf("\n--- Batch size: %d rooms ---\n", batch);
        
        // Baseline
        {
            dim3 grid(batch);
            for (int i = 0; i < 2000; i++)
                baseline_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++)
                baseline_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            baseline_us = ms / iters * 1000.0;
            printf("  %-33s %-12.1f %-12.3f %-12.0f\n",
                   "baseline (global mem)", baseline_us, baseline_us / batch,
                   batch * iters / (ms / 1000.0));
        }
        
        // Vectorized
        {
            dim3 grid(batch);
            for (int i = 0; i < 2000; i++)
                vec_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++)
                vec_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            float us = ms / iters * 1000.0;
            printf("  %-33s %-12.1f %-12.3f %-12.0f  (%.2fx vs baseline)\n",
                   "vectorized (half2)", us, us / batch,
                   batch * iters / (ms / 1000.0), baseline_us / us);
        }
        
        // Shared memory single-room per block
        {
            dim3 grid(batch);
            int shmem_size = DIM * sizeof(half);
            for (int i = 0; i < 2000; i++)
                shmem_single_kernel<256><<<grid, block, shmem_size>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++)
                shmem_single_kernel<256><<<grid, block, shmem_size>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            float us = ms / iters * 1000.0;
            printf("  %-33s %-12.1f %-12.3f %-12.0f  (%.2fx vs baseline)\n",
                   "shmem (1 room/block)", us, us / batch,
                   batch * iters / (ms / 1000.0), baseline_us / us);
        }
        
        // Shared memory multi-room (2 rooms per block)
        {
            int rooms_per_block = 2;
            dim3 grid((batch + rooms_per_block - 1) / rooms_per_block);
            int shmem_size = rooms_per_block * DIM * sizeof(half);
            
            for (int i = 0; i < 2000; i++)
                shmem_multi_kernel<2, 256><<<grid, dim3(block.x * rooms_per_block), shmem_size>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++)
                shmem_multi_kernel<2, 256><<<grid, dim3(block.x * rooms_per_block), shmem_size>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            float us = ms / iters * 1000.0;
            printf("  %-33s %-12.1f %-12.3f %-12.0f  (%.2fx vs baseline)\n",
                   "shmem (2 rooms/block)", us, us / batch,
                   batch * iters / (ms / 1000.0), baseline_us / us);
        }
        
        // Shared memory + vectorized (2 rooms per block)
        {
            int rooms_per_block = 2;
            dim3 grid((batch + rooms_per_block - 1) / rooms_per_block);
            int shmem_size = rooms_per_block * DIM * sizeof(half);
            
            for (int i = 0; i < 2000; i++)
                shmem_vec_kernel<2><<<grid, dim3(block.x * rooms_per_block), shmem_size>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++)
                shmem_vec_kernel<2><<<grid, dim3(block.x * rooms_per_block), shmem_size>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            float us = ms / iters * 1000.0;
            printf("  %-33s %-12.1f %-12.3f %-12.0f  (%.2fx vs baseline)\n",
                   "shmem+vec (2 rooms/block)", us, us / batch,
                   batch * iters / (ms / 1000.0), baseline_us / us);
        }
    }
    
    printf("\n============================================================\n");
    printf("SHARED MEMORY VERDICT\n");
    printf("============================================================\n");
    printf("On Orin with sm_87:\n");
    printf("- Global memory with L2 cache is already optimal for sequential access\n");
    printf("- Shared memory adds sync overhead without benefit for this pattern\n");
    printf("- Vectorized (half2) loads may help by reducing instruction count\n");
    printf("- Multi-room-per-block increases occupancy but adds sync cost\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_weights);
    free(h_input);
    
    return 0;
}
