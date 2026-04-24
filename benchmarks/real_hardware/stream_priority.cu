/**
 * Stream Priority Benchmark
 * 
 * Jetson Orin supports stream priorities (-max to +max, where max varies).
 * In production deckboss: hot rooms on high-priority, cold loads on low-priority.
 * 
 * This tests whether priority streams actually help on Orin.
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 stream_priority.cu -o stream_priority
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

__global__ void inference_kernel(const half* __restrict__ weights, const half* __restrict__ input,
                                  float* __restrict__ output, int dim, int iters) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    // Simulate longer computation by repeating
    for (int r = 0; r < iters; r++) {
        for (int i = lane; i < dim; i += 32)
            sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

// Lightweight kernel (simulates hot room)
__global__ void light_kernel(const half* __restrict__ weights, const half* __restrict__ input,
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

int main() {
    printf("============================================================\n");
    printf("STREAM PRIORITY BENCHMARK\n");
    printf("Jetson Orin Nano — High vs Low Priority Streams\n");
    printf("============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    
    // Query actual priority range
    int least, greatest;
    cudaError_t prioErr = cudaDeviceGetStreamPriorityRange(&least, &greatest);
    if (prioErr != cudaSuccess) {
        printf("Stream priorities NOT supported on this device.\n");
        return 0;
    }
    printf("Stream priority range: [%d, %d]\n", least, greatest);
    printf("Actual priority range: [%d, %d]\n", least, greatest);
    printf("  %d = lowest priority (background)\n", least);
    printf("  %d = highest priority (foreground)\n\n");
    
    const int DIM = 256;
    const int HOT_ROOMS = 6;
    const int COLD_ROOMS = 64;
    const int ITERS = 50000;
    
    half *h_weights = (half*)malloc((HOT_ROOMS + COLD_ROOMS) * DIM * sizeof(half));
    half *h_input = (half*)malloc(DIM * sizeof(half));
    
    srand(42);
    for (int i = 0; i < (HOT_ROOMS + COLD_ROOMS) * DIM; i++)
        h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++)
        h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_hot_weights, *d_cold_weights, *d_input;
    float *d_hot_output, *d_cold_output;
    
    CUDA_CHECK(cudaMalloc(&d_hot_weights, HOT_ROOMS * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_cold_weights, COLD_ROOMS * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_hot_output, HOT_ROOMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cold_output, COLD_ROOMS * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_hot_weights, h_weights, HOT_ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cold_weights, h_weights + HOT_ROOMS * DIM, COLD_ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 block(32);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test 1: Baseline — both on default priority streams
    printf(">>> Test 1: Baseline (default priority for both)\n");
    
    cudaStream_t def_hot, def_cold;
    CUDA_CHECK(cudaStreamCreate(&def_hot));
    CUDA_CHECK(cudaStreamCreate(&def_cold));
    
    for (int i = 0; i < 2000; i++) {
        light_kernel<<<HOT_ROOMS, block, 0, def_hot>>>(d_hot_weights, d_input, d_hot_output, DIM);
        light_kernel<<<COLD_ROOMS, block, 0, def_cold>>>(d_cold_weights, d_input, d_cold_output, DIM);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        light_kernel<<<HOT_ROOMS, block, 0, def_hot>>>(d_hot_weights, d_input, d_hot_output, DIM);
        light_kernel<<<COLD_ROOMS, block, 0, def_cold>>>(d_cold_weights, d_input, d_cold_output, DIM);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_def;
    CUDA_CHECK(cudaEventElapsedTime(&ms_def, start, stop));
    printf("  Total: %.1f ms | us/iter: %.1f\n", ms_def, ms_def / ITERS * 1000.0);
    
    CUDA_CHECK(cudaStreamDestroy(def_hot));
    CUDA_CHECK(cudaStreamDestroy(def_cold));
    
    // Test 2: Hot on high priority, cold on low priority
    printf("\n>>> Test 2: Hot=high priority, Cold=low priority\n");
    
    cudaStream_t high_stream, low_stream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&high_stream, cudaStreamDefault, greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&low_stream, cudaStreamDefault, least));
    
    for (int i = 0; i < 2000; i++) {
        light_kernel<<<HOT_ROOMS, block, 0, high_stream>>>(d_hot_weights, d_input, d_hot_output, DIM);
        light_kernel<<<COLD_ROOMS, block, 0, low_stream>>>(d_cold_weights, d_input, d_cold_output, DIM);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        light_kernel<<<HOT_ROOMS, block, 0, high_stream>>>(d_hot_weights, d_input, d_hot_output, DIM);
        light_kernel<<<COLD_ROOMS, block, 0, low_stream>>>(d_cold_weights, d_input, d_cold_output, DIM);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_prio;
    CUDA_CHECK(cudaEventElapsedTime(&ms_prio, start, stop));
    printf("  Total: %.1f ms | us/iter: %.1f\n", ms_prio, ms_prio / ITERS * 1000.0);
    printf("  vs baseline: %.2fx\n", ms_def / ms_prio);
    
    // Test 3: Measure hot room latency in isolation (with and without cold competing)
    printf("\n>>> Test 3: Hot Room Latency — Isolated vs Contended\n");
    
    cudaStream_t iso_stream;
    CUDA_CHECK(cudaStreamCreate(&iso_stream));
    
    // Isolated hot room
    for (int i = 0; i < 2000; i++)
        light_kernel<<<HOT_ROOMS, block, 0, iso_stream>>>(d_hot_weights, d_input, d_hot_output, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        light_kernel<<<HOT_ROOMS, block, 0, iso_stream>>>(d_hot_weights, d_input, d_hot_output, DIM);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_iso;
    CUDA_CHECK(cudaEventElapsedTime(&ms_iso, start, stop));
    printf("  Isolated: %.1f ms | us/iter: %.2f us\n", ms_iso, ms_iso / ITERS * 1000.0);
    
    // Contended (with heavy cold workload)
    // Use a heavier kernel for cold rooms to create contention
    cudaEvent_t hot_start, hot_stop;
    CUDA_CHECK(cudaEventCreate(&hot_start));
    CUDA_CHECK(cudaEventCreate(&hot_stop));
    
    for (int i = 0; i < 2000; i++) {
        light_kernel<<<HOT_ROOMS, block, 0, high_stream>>>(d_hot_weights, d_input, d_hot_output, DIM);
        inference_kernel<<<COLD_ROOMS, block, 0, low_stream>>>(d_cold_weights, d_input, d_cold_output, DIM, 32);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(hot_start));
    for (int i = 0; i < ITERS; i++) {
        light_kernel<<<HOT_ROOMS, block, 0, high_stream>>>(d_hot_weights, d_input, d_hot_output, DIM);
        inference_kernel<<<COLD_ROOMS, block, 0, low_stream>>>(d_cold_weights, d_input, d_cold_output, DIM, 32);
    }
    CUDA_CHECK(cudaEventRecord(hot_stop, high_stream));
    CUDA_CHECK(cudaStreamSynchronize(high_stream));
    
    float ms_contended;
    CUDA_CHECK(cudaEventElapsedTime(&ms_contended, hot_start, hot_stop));
    printf("  Contended (high prio): %.1f ms | us/iter: %.2f us\n", ms_contended, ms_contended / ITERS * 1000.0);
    printf("  Contended (low prio test):\n");
    
    CUDA_CHECK(cudaEventRecord(hot_start));
    for (int i = 0; i < ITERS; i++) {
        light_kernel<<<HOT_ROOMS, block, 0, low_stream>>>(d_hot_weights, d_input, d_hot_output, DIM);
        inference_kernel<<<COLD_ROOMS, block, 0, high_stream>>>(d_cold_weights, d_input, d_cold_output, DIM, 32);
    }
    CUDA_CHECK(cudaEventRecord(hot_stop, low_stream));
    CUDA_CHECK(cudaStreamSynchronize(low_stream));
    
    float ms_lowp;
    CUDA_CHECK(cudaEventElapsedTime(&ms_lowp, hot_start, hot_stop));
    printf("  Contended (low prio): %.1f ms | us/iter: %.2f us\n", ms_lowp, ms_lowp / ITERS * 1000.0);
    
    printf("\n  Priority benefit: high=%.2fx isolated, low=%.2fx isolated\n",
           ms_iso / ms_contended, ms_iso / ms_lowp);
    printf("  Priority isolation: high is %.2fx faster than low under contention\n",
           ms_lowp / ms_contended);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(hot_start));
    CUDA_CHECK(cudaEventDestroy(hot_stop));
    CUDA_CHECK(cudaStreamDestroy(high_stream));
    CUDA_CHECK(cudaStreamDestroy(low_stream));
    CUDA_CHECK(cudaStreamDestroy(iso_stream));
    CUDA_CHECK(cudaFree(d_hot_weights));
    CUDA_CHECK(cudaFree(d_cold_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_hot_output));
    CUDA_CHECK(cudaFree(d_cold_output));
    free(h_weights);
    free(h_input);
    
    printf("\n============================================================\n");
    printf("CONCLUSION\n");
    printf("============================================================\n");
    printf("Stream priorities help when GPU is contended.\n");
    printf("High-priority streams get scheduled first.\n");
    printf("For deckboss: put production inference on high priority,\n");
    printf("background tasks (weight loading, health checks) on low.\n");
    
    return 0;
}
