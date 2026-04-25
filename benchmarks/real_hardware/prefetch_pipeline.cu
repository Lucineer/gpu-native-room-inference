/**
 * Double-Buffered Prefetch Pipeline (#25)
 * 
 * Novel idea: overlap weight loading with computation using double buffering.
 * While GPU computes batch N, DMA engine prefetches batch N+1 weights.
 * 
 * On Jetson: CPU and GPU share memory, so DMA is CPU→GPU cache transfer.
 * But with pinned memory + async prefetch, we can overlap:
 *   Compute batch N  ||  Prefetch batch N+1 weights
 * 
 * Architecture:
 * - 2 weight buffers (ping-pong)
 * - 2 streams: compute stream + prefetch stream  
 * - CUDA events for synchronization
 * - Input buffer reused across batches
 * 
 * Test scenarios:
 * 1. No prefetch (baseline)
 * 2. Single prefetch (prefetch next batch)
 * 3. Full pipeline (double-buffered, continuous)
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 prefetch_pipeline.cu -o prefetch_pipeline
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

__global__ void infer_batch(const half* __restrict__ weights,
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
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Multi-room fused variant (4 rooms/block, half2)
template<int RPB>
__global__ void infer_batch_opt(const half* __restrict__ weights,
                                 const half* __restrict__ input,
                                 float* __restrict__ output,
                                 int dim, int num_rooms) {
    int block_base = blockIdx.x * RPB;
    int room_local = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_local >= RPB) return;
    int room = block_base + room_local;
    if (room >= num_rooms) return;
    
    int pairs = dim / 2;
    const half2* w = reinterpret_cast<const half2*>(weights + (size_t)room * dim);
    const half2* inp = reinterpret_cast<const half2*>(input);
    
    float sum = 0.0f;
    for (int i = lane; i < pairs; i += 32) {
        half2 wp = w[i], ip = inp[i];
        half2 p = __hmul2(wp, ip);
        sum += __half2float(p.x) + __half2float(p.y);
    }
    
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

int main() {
    printf("============================================================\n");
    printf("DOUBLE-BUFFERED PREFETCH PIPELINE (#25)\n");
    printf("Overlap weight loading with computation\n");
    printf("============================================================\n\n");
    
    const int DIM = 256;
    const int ROOMS_PER_BATCH = 64;
    const int TOTAL_ROOMS = 512;
    const int NUM_BATCHES = TOTAL_ROOMS / ROOMS_PER_BATCH;
    
    // Allocate 2 weight buffers (ping-pong)
    half *d_w[2], *d_input;
    float *d_out;
    size_t w_size = (size_t)ROOMS_PER_BATCH * DIM * sizeof(half);
    
    CUDA_CHECK(cudaMalloc(&d_w[0], w_size));
    CUDA_CHECK(cudaMalloc(&d_w[1], w_size));
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, ROOMS_PER_BATCH * sizeof(float)));
    
    // Generate data
    half *h_w = (half*)malloc(w_size);
    half *h_in = (half*)malloc(DIM * sizeof(half));
    srand(42);
    for (int i = 0; i < ROOMS_PER_BATCH * DIM; i++)
        h_w[i] = __float2half((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++)
        h_in[i] = __float2half((float)rand()/RAND_MAX - 0.5f);
    
    CUDA_CHECK(cudaMemcpy(d_w[0], h_w, w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_in, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    // Generate different weight sets for each batch
    half **h_batches = (half**)malloc(NUM_BATCHES * sizeof(half*));
    for (int b = 0; b < NUM_BATCHES; b++) {
        h_batches[b] = (half*)malloc(w_size);
        for (int i = 0; i < ROOMS_PER_BATCH * DIM; i++)
            h_batches[b][i] = __float2half((float)rand()/RAND_MAX - 0.5f);
    }
    
    cudaStream_t compute_stream, prefetch_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking));
    
    cudaEvent_t compute_done[2], prefetch_done[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaEventCreate(&compute_done[i]));
        CUDA_CHECK(cudaEventCreate(&prefetch_done[i]));
    }
    
    cudaEvent_t total_start, total_stop;
    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    
    dim3 block32(32);
    int iters = 5000;
    
    // ===== Test 1: No Prefetch (baseline) =====
    printf(">>> Test 1: Sequential Processing (no prefetch)\n");
    printf("Each batch: copy weights H2D → compute → copy results D2H\n");
    {
        CUDA_CHECK(cudaEventRecord(total_start));
        for (int iter = 0; iter < iters; iter++) {
            for (int b = 0; b < NUM_BATCHES; b++) {
                CUDA_CHECK(cudaMemcpyAsync(d_w[0], h_batches[b], w_size, cudaMemcpyHostToDevice, compute_stream));
                infer_batch<<<dim3(ROOMS_PER_BATCH), block32, 0, compute_stream>>>(d_w[0], d_input, d_out, DIM, ROOMS_PER_BATCH);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventRecord(total_stop));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, total_start, total_stop));
        float us_per_batch = ms / (iters * NUM_BATCHES) * 1000;
        float total_rooms = iters * TOTAL_ROOMS;
        printf("  Total: %.0f ms for %d iterations x %d batches\n", ms, iters, NUM_BATCHES);
        printf("  Per batch: %.1f us | Per room: %.3f us | Throughput: %.1f M room-qps\n",
               us_per_batch, us_per_batch/ROOMS_PER_BATCH, total_rooms/(ms/1000.0)/1e6);
    }
    
    // ===== Test 2: Prefetch Next Batch =====
    printf("\n>>> Test 2: Single Prefetch (prefetch batch N+1 while computing N)\n");
    {
        CUDA_CHECK(cudaEventRecord(total_start));
        for (int iter = 0; iter < iters; iter++) {
            // Prefetch batch 0
            CUDA_CHECK(cudaMemcpyAsync(d_w[0], h_batches[0], w_size, cudaMemcpyHostToDevice, prefetch_stream));
            CUDA_CHECK(cudaEventRecord(prefetch_done[0], prefetch_stream));
            
            for (int b = 0; b < NUM_BATCHES; b++) {
                int next_b = (b + 1) % NUM_BATCHES;
                int cur_buf = b % 2;
                int next_buf = (b + 1) % 2;
                
                // Wait for current batch weights
                CUDA_CHECK(cudaStreamWaitEvent(compute_stream, prefetch_done[cur_buf], 0));
                
                // Compute current batch
                infer_batch<<<dim3(ROOMS_PER_BATCH), block32, 0, compute_stream>>>(d_w[cur_buf], d_input, d_out, DIM, ROOMS_PER_BATCH);
                
                // Start prefetching next batch (async, on separate stream)
                CUDA_CHECK(cudaMemcpyAsync(d_w[next_buf], h_batches[next_b], w_size, cudaMemcpyHostToDevice, prefetch_stream));
                CUDA_CHECK(cudaEventRecord(prefetch_done[next_buf], prefetch_stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventRecord(total_stop));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, total_start, total_stop));
        float us_per_batch = ms / (iters * NUM_BATCHES) * 1000;
        float total_rooms = iters * TOTAL_ROOMS;
        printf("  Per batch: %.1f us | Per room: %.3f us | Throughput: %.1f M room-qps\n",
               us_per_batch, us_per_batch/ROOMS_PER_BATCH, total_rooms/(ms/1000.0)/1e6);
    }
    
    // ===== Test 3: Optimized Kernel + Prefetch =====
    printf("\n>>> Test 3: Optimized Kernel (4 rooms/block) + Prefetch\n");
    {
        CUDA_CHECK(cudaEventRecord(total_start));
        for (int iter = 0; iter < iters; iter++) {
            CUDA_CHECK(cudaMemcpyAsync(d_w[0], h_batches[0], w_size, cudaMemcpyHostToDevice, prefetch_stream));
            CUDA_CHECK(cudaEventRecord(prefetch_done[0], prefetch_stream));
            
            for (int b = 0; b < NUM_BATCHES; b++) {
                int next_b = (b + 1) % NUM_BATCHES;
                int cur_buf = b % 2;
                int next_buf = (b + 1) % 2;
                
                CUDA_CHECK(cudaStreamWaitEvent(compute_stream, prefetch_done[cur_buf], 0));
                infer_batch_opt<4><<<dim3(ROOMS_PER_BATCH/4), dim3(128), 0, compute_stream>>>(d_w[cur_buf], d_input, d_out, DIM, ROOMS_PER_BATCH);
                CUDA_CHECK(cudaMemcpyAsync(d_w[next_buf], h_batches[next_b], w_size, cudaMemcpyHostToDevice, prefetch_stream));
                CUDA_CHECK(cudaEventRecord(prefetch_done[next_buf], prefetch_stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventRecord(total_stop));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, total_start, total_stop));
        float us_per_batch = ms / (iters * NUM_BATCHES) * 1000;
        float total_rooms = iters * TOTAL_ROOMS;
        printf("  Per batch: %.1f us | Per room: %.3f us | Throughput: %.1f M room-qps\n",
               us_per_batch, us_per_batch/ROOMS_PER_BATCH, total_rooms/(ms/1000.0)/1e6);
    }
    
    // ===== Test 4: No H2D (weights already on GPU) =====
    printf("\n>>> Test 4: Weights Pre-loaded (no H2D transfer)\n");
    {
        // Pre-load all batches
        half *d_all_weights;
        CUDA_CHECK(cudaMalloc(&d_all_weights, (size_t)TOTAL_ROOMS * DIM * sizeof(half)));
        for (int b = 0; b < NUM_BATCHES; b++)
            CUDA_CHECK(cudaMemcpy(d_all_weights + (size_t)b * ROOMS_PER_BATCH * DIM, h_batches[b], w_size, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(total_start));
        for (int iter = 0; iter < iters; iter++) {
            for (int b = 0; b < NUM_BATCHES; b++) {
                half* batch_w = d_all_weights + (size_t)b * ROOMS_PER_BATCH * DIM;
                infer_batch<<<dim3(ROOMS_PER_BATCH), block32, 0, compute_stream>>>(batch_w, d_input, d_out, DIM, ROOMS_PER_BATCH);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventRecord(total_stop));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, total_start, total_stop));
        float us_per_batch = ms / (iters * NUM_BATCHES) * 1000;
        float total_rooms = iters * TOTAL_ROOMS;
        printf("  Per batch: %.1f us | Per room: %.3f us | Throughput: %.1f M room-qps\n",
               us_per_batch, us_per_batch/ROOMS_PER_BATCH, total_rooms/(ms/1000.0)/1e6);
        printf("  (This is the theoretical maximum — all weights in GPU memory)\n");
        
        CUDA_CHECK(cudaFree(d_all_weights));
    }
    
    printf("\n============================================================\n");
    printf("PREFETCH PIPELINE VERDICT\n");
    printf("============================================================\n");
    printf("On Jetson (unified memory), H2D is a cache operation.\n");
    printf("Prefetch helps when:\n");
    printf("1. Weights don't fit in GPU L2 cache (256KB on Orin)\n");
    printf("2. Sequential batches touch different weight regions\n");
    printf("3. DMA engine can overlap with compute\n");
    printf("Prefetch doesn't help when:\n");
    printf("1. All weights fit in L2 (small batch, same weights)\n");
    printf("2. Unified memory already caches everything\n");
    printf("3. Compute is so fast that prefetch can't keep up\n");
    printf("\nFor deckboss: keep hot room weights in L2.\n");
    printf("Prefetch cold rooms when weight-swap occurs.\n");
    
    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(compute_done[i]);
        cudaEventDestroy(prefetch_done[i]);
    }
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(prefetch_stream);
    CUDA_CHECK(cudaFree(d_w[0]));
    CUDA_CHECK(cudaFree(d_w[1]));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out));
    free(h_w); free(h_in);
    for (int b = 0; b < NUM_BATCHES; b++) free(h_batches[b]);
    free(h_batches);
    
    return 0;
}
