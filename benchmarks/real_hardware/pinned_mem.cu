/**
 * Pinned Memory Transfer Benchmark
 * 
 * Problem: D2H memcpy adds ~25μs to API-level inference.
 * With pageable memory, the copy goes through a staging buffer.
 * With pinned memory, DMA transfers directly to host.
 * 
 * Theory: pinned memory should cut D2H by 2-5x on Orin.
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 pinned_mem.cu -o pinned_mem
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

__global__ void infer_kernel(const half* __restrict__ weights,
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

int main() {
    printf("============================================================\n");
    printf("PINNED MEMORY TRANSFER BENCHMARK\n");
    printf("D2H latency: pageable vs pinned vs device-only\n");
    printf("============================================================\n\n");
    
    const int DIM = 256;
    const int ITERS = 50000;
    
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
    
    // Pinned memory allocation
    float *h_output_pageable = (float*)malloc(512 * sizeof(float));
    float *h_output_pinned;
    CUDA_CHECK(cudaMallocHost(&h_output_pinned, 512 * sizeof(float)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    dim3 block(32);
    
    int batch_sizes[] = {1, 6, 32, 64, 128, 256};
    int num_sizes = 6;
    
    printf("%-8s %-12s %-12s %-12s %-12s\n", 
           "Rooms", "GPU-only", "Pageable", "Pinned", "Speedup");
    printf("------------------------------------------------------------\n");
    
    for (int b = 0; b < num_sizes; b++) {
        int batch = batch_sizes[b];
        int iters = batch <= 6 ? ITERS : (ITERS * 6 / batch);
        dim3 grid(batch);
        
        // GPU-only (no D2H)
        for (int i = 0; i < 2000; i++)
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_gpu;
        CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, start, stop));
        float us_gpu = ms_gpu / iters * 1000.0;
        
        // Pageable D2H
        for (int i = 0; i < 2000; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpy(h_output_pageable, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpy(h_output_pageable, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_page;
        CUDA_CHECK(cudaEventElapsedTime(&ms_page, start, stop));
        float us_page = ms_page / iters * 1000.0;
        
        // Pinned D2H
        for (int i = 0; i < 2000; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpy(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpy(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_pin;
        CUDA_CHECK(cudaEventElapsedTime(&ms_pin, start, stop));
        float us_pin = ms_pin / iters * 1000.0;
        
        printf("%-8d %-12.1f %-12.1f %-12.1f %-12.2fx\n",
               batch, us_gpu, us_page, us_pin, us_page / us_pin);
    }
    
    // Test 2: Async pinned D2H with stream
    printf("\n--- Async Pinned D2H (overlapping compute + copy) ---\n");
    printf("%-8s %-12s %-12s %-12s %-12s\n",
           "Rooms", "Sync", "Async", "Overlap", "Speedup");
    printf("------------------------------------------------------------\n");
    
    for (int b = 0; b < num_sizes; b++) {
        int batch = batch_sizes[b];
        int iters = batch <= 6 ? ITERS : (ITERS * 6 / batch);
        dim3 grid(batch);
        
        // Sync pinned
        for (int i = 0; i < 2000; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpy(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpy(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_sync;
        CUDA_CHECK(cudaEventElapsedTime(&ms_sync, start, stop));
        
        // Async pinned with stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        for (int i = 0; i < 2000; i++) {
            infer_kernel<<<grid, block, 0, stream>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpyAsync(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            infer_kernel<<<grid, block, 0, stream>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpyAsync(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms_async;
        CUDA_CHECK(cudaEventElapsedTime(&ms_async, start, stop));
        
        CUDA_CHECK(cudaStreamDestroy(stream));
        
        printf("%-8d %-12.1f %-12.1f %-12.1f %-12.2fx\n",
               batch, 
               ms_sync / iters * 1000.0,
               ms_async / iters * 1000.0,
               (ms_sync - ms_async) / iters * 1000.0,
               ms_sync / ms_async);
    }
    
    // Test 3: Zero-copy (mapped memory)
    printf("\n--- Zero-Copy (cudaHostAllocMapped) ---\n");
    float *h_output_zerocopy;
    float *d_output_mapped;
    CUDA_CHECK(cudaHostAlloc(&h_output_zerocopy, 512 * sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_output_mapped, h_output_zerocopy, 0));
    
    printf("%-8s %-12s %-12s %-12s\n",
           "Rooms", "Standard", "Zero-copy", "Ratio");
    printf("------------------------------------------------------------\n");
    
    for (int b = 0; b < num_sizes; b++) {
        int batch = batch_sizes[b];
        int iters = batch <= 6 ? ITERS : (ITERS * 6 / batch);
        dim3 grid(batch);
        
        // Standard D2H to pinned
        for (int i = 0; i < 2000; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpyAsync(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost, 0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, batch);
            CUDA_CHECK(cudaMemcpyAsync(h_output_pinned, d_output, batch * sizeof(float), cudaMemcpyDeviceToHost, 0));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_std;
        CUDA_CHECK(cudaEventElapsedTime(&ms_std, start, stop));
        
        // Zero-copy (write directly to mapped memory)
        for (int i = 0; i < 2000; i++)
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output_mapped, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            infer_kernel<<<grid, block>>>(d_weights, d_input, d_output_mapped, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_zc;
        CUDA_CHECK(cudaEventElapsedTime(&ms_zc, start, stop));
        
        printf("%-8d %-12.1f %-12.1f %-12.2fx\n",
               batch,
               ms_std / iters * 1000.0,
               ms_zc / iters * 1000.0,
               ms_std / ms_zc);
    }
    
    // Test 4: Pinned input (H2D)
    printf("\n--- Pinned Input (H2D direction) ---\n");
    float *h_input_pageable = (float*)malloc(DIM * sizeof(float));
    float *h_input_pinned;
    CUDA_CHECK(cudaMallocHost(&h_input_pinned, DIM * sizeof(float)));
    float *d_input_pinned;
    CUDA_CHECK(cudaMalloc(&d_input_pinned, DIM * sizeof(float)));
    
    // Init data
    for (int i = 0; i < DIM; i++) {
        h_input_pageable[i] = (float)rand() / RAND_MAX;
        h_input_pinned[i] = h_input_pageable[i];
    }
    
    printf("Pageable H2D: ");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10000; i++)
        CUDA_CHECK(cudaMemcpy(d_input_pinned, h_input_pageable, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("%.2f us/copy\n", ms / 10000 * 1000.0);
    
    printf("Pinned H2D:   ");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10000; i++)
        CUDA_CHECK(cudaMemcpy(d_input_pinned, h_input_pinned, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("%.2f us/copy\n", ms / 10000 * 1000.0);
    
    printf("Async pinned: ");
    cudaStream_t s2;
    CUDA_CHECK(cudaStreamCreate(&s2));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10000; i++)
        CUDA_CHECK(cudaMemcpyAsync(d_input_pinned, h_input_pinned, DIM * sizeof(float), cudaMemcpyHostToDevice, s2));
    CUDA_CHECK(cudaStreamSynchronize(s2));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("%.2f us/copy\n", ms / 10000 * 1000.0);
    CUDA_CHECK(cudaStreamDestroy(s2));
    
    // Memory usage
    printf("\n--- Memory Usage ---\n");
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.0f MB free / %.0f MB total\n", free_mem / (1024*1024.0), total_mem / (1024*1024.0));
    
    printf("\n============================================================\n");
    printf("PINNED MEMORY VERDICT\n");
    printf("============================================================\n");
    printf("On Jetson Orin (unified memory, no discrete VRAM):\n");
    printf("- Pinned memory may have LESS advantage than discrete GPUs\n");
    printf("- Unified memory already shares physical RAM\n");
    printf("- Zero-copy eliminates explicit D2H copy entirely\n");
    printf("- Async pinned enables compute/copy overlap\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_output_pinned));
    CUDA_CHECK(cudaFreeHost(h_input_pinned));
    CUDA_CHECK(cudaFree(d_input_pinned));
    CUDA_CHECK(cudaFreeHost(h_output_zerocopy));
    free(h_weights);
    free(h_input);
    free(h_output_pageable);
    free(h_input_pageable);
    
    return 0;
}
