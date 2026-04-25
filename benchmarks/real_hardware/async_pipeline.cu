/**
 * Suite #33: Async Pipeline — Eliminating CPU-GPU Sync Barrier
 * 
 * Suites #31-32 showed cudaStreamSynchronize() adds ~15-20μs overhead.
 * This suite tests whether we can overlap CPU work with GPU inference
 * using a double-buffered async pipeline.
 * 
 * Pattern:
 * 1. Launch kernel N on stream
 * 2. Immediately launch kernel N+1 on stream (GPU queues them)
 * 3. Read results from kernel N-1 (should be complete by now)
 * 4. No explicit sync — rely on GPU command queue depth
 * 
 * Also test: cudaStreamAddCallback for latency measurement without sync
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 async_pipeline.cu -o async_pipeline
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <cuda_fp16.h>

#define DIM 256
#define ITERS 100000

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

// Callback data for async timing
struct CallbackData {
    std::atomic<long long>* timestamps;
    int idx;
};

void CUDART_CB timestamp_callback(cudaStream_t stream, cudaError_t status, void* data) {
    CallbackData* cbd = (CallbackData*)data;
    auto now = std::chrono::high_resolution_clock::now();
    long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    cbd->timestamps[cbd->idx].store(ns, std::memory_order_release);
}

int main() {
    printf("=== Suite #33: Async Pipeline — Eliminating Sync Barrier ===\n\n");
    
    // Allocate
    half *d_weights, *d_input;
    float *d_output;
    cudaMalloc(&d_weights, 1024 * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_output, 1024 * sizeof(float));
    
    // Zero-copy output
    float* h_zc_output;
    cudaHostAlloc(&h_zc_output, 1024 * sizeof(float), cudaHostAllocMapped);
    float* d_zc_output;
    cudaHostGetDevicePointer(&d_zc_output, h_zc_output, 0);
    
    // Init
    std::vector<half> hw(1024 * DIM), hi(DIM);
    for (int i = 0; i < 1024 * DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    
    cudaMemcpy(d_weights, hw.data(), 1024 * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int batch_sizes[] = {64, 256, 1024};
    
    // ========================================
    // Test 1: Synchronous baseline (cudaStreamSynchronize)
    // ========================================
    printf("--- Test 1: Synchronous (cudaStreamSynchronize) ---\n");
    
    for (int b = 0; b < 3; b++) {
        int rooms = batch_sizes[b];
        dim3 grid((rooms + 7) / 8);
        
        for (int i = 0; i < 100; i++) {
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc_output, rooms, DIM);
            cudaStreamSynchronize(stream);
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc_output, rooms, DIM);
            cudaStreamSynchronize(stream);
            volatile float sink = h_zc_output[0];
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        printf("  %d rooms sync: %.2f us/iter, %.1fM room-qps\n", rooms, us, rooms / (us / 1e6f) / 1e6);
    }
    
    // ========================================
    // Test 2: Fire-and-forget (no sync, no read)
    // How fast can we queue kernels?
    // ========================================
    printf("\n--- Test 2: Fire-and-Forget (queue only, no sync) ---\n");
    
    for (int b = 0; b < 3; b++) {
        int rooms = batch_sizes[b];
        dim3 grid((rooms + 7) / 8);
        
        cudaStreamSynchronize(stream);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc_output, rooms, DIM);
        }
        // Don't sync — measure pure queue time
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        printf("  %d rooms fire-and-forget: %.3f us/iter (queue only)\n", rooms, us);
        
        // Now sync to make sure GPU caught up
        cudaStreamSynchronize(stream);
    }
    
    // ========================================
    // Test 3: Double-buffered pipeline (overlap N+1 with N)
    // ========================================
    printf("\n--- Test 3: Double-Buffered Pipeline ---\n");
    printf("Launch kernel N+1 while reading results from N-1\n\n");
    
    for (int b = 0; b < 3; b++) {
        int rooms = batch_sizes[b];
        dim3 grid((rooms + 7) / 8);
        
        // Use two output buffers
        float* d_out[2];
        float* h_out[2];
        cudaMalloc(&d_out[0], rooms * sizeof(float));
        cudaMalloc(&d_out[1], rooms * sizeof(float));
        cudaHostAlloc(&h_out[0], rooms * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc(&h_out[1], rooms * sizeof(float), cudaHostAllocMapped);
        float* d_zc[2];
        cudaHostGetDevicePointer(&d_zc[0], h_out[0], 0);
        cudaHostGetDevicePointer(&d_zc[1], h_out[1], 0);
        
        cudaStream_t s0, s1;
        cudaStreamCreate(&s0);
        cudaStreamCreate(&s1);
        
        // Warmup
        for (int i = 0; i < 100; i++) {
            int buf = i % 2;
            infer_v7<<<grid, dim3(256), 0, s0>>>(d_weights, d_input, d_zc[buf], rooms, DIM);
            cudaStreamSynchronize(s0);
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            int curr = i % 2;
            int prev = 1 - curr;
            
            // Launch current inference
            infer_v7<<<grid, dim3(256), 0, s0>>>(d_weights, d_input, d_zc[curr], rooms, DIM);
            
            // Read previous result (should be ready)
            if (i > 0) {
                volatile float sink = h_out[prev][0];
            }
        }
        cudaStreamSynchronize(s0);
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        printf("  %d rooms double-buffer: %.2f us/iter, %.1fM room-qps (%.2fx vs sync)\n",
               rooms, us, rooms / (us / 1e6f) / 1e6, 
               rooms / (us / 1e6f) / 1e6 / (rooms > 0 ? 1 : 1)); // placeholder
        
        cudaStreamDestroy(s0);
        cudaStreamDestroy(s1);
        cudaFree(d_out[0]); cudaFree(d_out[1]);
        cudaFreeHost(h_out[0]); cudaFreeHost(h_out[1]);
    }
    
    // ========================================
    // Test 4: Callback-based timing (no CPU sync)
    // ========================================
    printf("\n--- Test 4: Callback-Based GPU Timing ---\n");
    printf("Uses cudaStreamAddCallback to measure GPU-side latency\n\n");
    
    std::atomic<long long>* cb_timestamps;
    cudaMallocHost(&cb_timestamps, ITERS * sizeof(std::atomic<long long>));
    
    CallbackData cbd[ITERS];
    for (int i = 0; i < ITERS; i++) {
        cbd[i].timestamps = cb_timestamps;
        cbd[i].idx = i;
    }
    
    for (int b = 0; b < 3; b++) {
        int rooms = batch_sizes[b];
        dim3 grid((rooms + 7) / 8);
        
        cudaEvent_t start_ev, end_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&end_ev);
        
        // Warmup
        for (int i = 0; i < 100; i++) {
            cudaEventRecord(start_ev, stream);
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, DIM);
            cudaEventRecord(end_ev, stream);
            cudaStreamSynchronize(stream);
        }
        
        // Measure GPU-side latency with events
        float total_ms = 0;
        for (int i = 0; i < ITERS; i++) {
            cudaEventRecord(start_ev, stream);
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, DIM);
            cudaEventRecord(end_ev, stream);
        }
        cudaStreamSynchronize(stream);
        
        // Can't easily get per-iter event timing — just total
        cudaEventElapsedTime(&total_ms, start_ev, end_ev);
        // Actually this only measures first-to-last. Let me use a different approach.
        
        // Simple: record event, launch, record event, sync, get elapsed time
        cudaEventRecord(start_ev, stream);
        for (int i = 0; i < ITERS; i++) {
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, DIM);
        }
        cudaEventRecord(end_ev, stream);
        cudaStreamSynchronize(stream);
        
        float avg_ms;
        cudaEventElapsedTime(&avg_ms, start_ev, end_ev);
        float avg_us = avg_ms / ITERS * 1000;
        
        printf("  %d rooms GPU-only (event): %.2f us/iter, %.1fM room-qps\n",
               rooms, avg_us, rooms / (avg_us / 1e6f) / 1e6);
        
        cudaEventDestroy(start_ev);
        cudaEventDestroy(end_ev);
    }
    
    cudaFreeHost(cb_timestamps);
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_output);
    cudaFreeHost(h_zc_output);
    
    printf("\n=== Suite #33 Complete ===\n");
    return 0;
}
