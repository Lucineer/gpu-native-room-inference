/**
 * Suite #34: GPU Command Queue Depth — How Much Can We Overlap?
 * 
 * Suite #33 showed double-buffering reaches 104M room-qps by eliminating sync.
 * Suite #33 showed fire-and-forget queue time is 6.75μs.
 * 
 * Question: How deep can we fill the GPU command queue before it backs up?
 * If we launch 100 kernels without syncing, when does the queue start slowing down?
 * 
 * This measures the GPU's "ingestion capacity" — how fast it can accept new work.
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 queue_depth.cu -o queue_depth
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_fp16.h>

#define DIM 256

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

int main() {
    printf("=== Suite #34: GPU Command Queue Depth ===\n\n");
    
    // Allocate
    half *d_weights, *d_input;
    float *d_output;
    cudaMalloc(&d_weights, 1024 * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_output, 1024 * sizeof(float));
    
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
    
    int rooms = 256;
    dim3 grid((rooms + 7) / 8);
    
    // ========================================
    // Test 1: Queue depth — launch N kernels, then sync all
    // Measure: total time / N = average per-kernel time
    // If queue backs up, later kernels will be slower
    // ========================================
    printf("--- Test 1: Queue Depth Scaling ---\n");
    printf("Launch N kernels back-to-back, then sync. Average per-kernel time.\n\n");
    
    int depths[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    printf("%-8s | %-14s | %-14s | %-14s | %-14s\n",
           "Depth", "Total (ms)", "Per-kernel (us)", "Room-qps", "vs depth=1");
    printf("---------|----------------|----------------|----------------|----------------\n");
    
    float baseline_us = 0;
    for (int d = 0; d < 11; d++) {
        int depth = depths[d];
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < depth; j++)
                infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, DIM);
            cudaStreamSynchronize(stream);
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < depth; j++)
                infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_output, rooms, DIM);
            cudaStreamSynchronize(stream);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        float total_us = std::chrono::duration<float, std::micro>(t1 - t0).count() / 100;
        float per_kernel_us = total_us / depth;
        float qps = rooms / (per_kernel_us / 1e6f);
        
        if (d == 0) baseline_us = per_kernel_us;
        
        printf("%-8d | %12.2f   | %12.2f   | %12.0f   | %.3fx\n",
               depth, total_us / 1000.0f, per_kernel_us, qps, baseline_us / per_kernel_us);
    }
    
    // ========================================
    // Test 2: Continuous pipeline with N buffers
    // ========================================
    printf("\n--- Test 2: N-Buffer Pipeline Throughput ---\n");
    printf("How many buffers before we saturate the GPU?\n\n");
    
    printf("%-8s | %-14s | %-14s | %-14s | %-14s\n",
           "Buffers", "Time (ms)", "Per-iter (us)", "Room-qps", "vs 2-buf");
    printf("---------|----------------|----------------|----------------|----------------\n");
    
    float two_buf_us = 0;
    int num_buffers_list[] = {2, 4, 8, 16, 32, 64};
    
    for (int nb = 0; nb < 6; nb++) {
        int nbuf = num_buffers_list[nb];
        int iters = 50000 / nbuf;  // Keep total iterations reasonable
        
        // Allocate buffers
        std::vector<float*> d_buf(nbuf), h_buf(nbuf), d_zc(nbuf);
        for (int i = 0; i < nbuf; i++) {
            cudaMalloc(&d_buf[i], rooms * sizeof(float));
            cudaHostAlloc(&h_buf[i], rooms * sizeof(float), cudaHostAllocMapped);
            cudaHostGetDevicePointer(&d_zc[i], h_buf[i], 0);
        }
        
        // Warmup
        for (int i = 0; i < nbuf * 2; i++) {
            int buf = i % nbuf;
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc[buf], rooms, DIM);
        }
        cudaStreamSynchronize(stream);
        
        // Benchmark: N-buffer pipeline
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters * nbuf; i++) {
            int buf = i % nbuf;
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc[buf], rooms, DIM);
            // Read previous buffer
            if (i >= nbuf) {
                int prev = (i - nbuf) % nbuf;
                volatile float sink = h_buf[prev][0];
            }
        }
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        float total_us = std::chrono::duration<float, std::micro>(t1 - t0).count();
        float per_iter_us = total_us / (iters * nbuf);
        float qps = rooms / (per_iter_us / 1e6f);
        
        if (nb == 0) two_buf_us = per_iter_us;
        
        printf("%-8d | %12.2f   | %12.2f   | %12.0f   | %.3fx\n",
               nbuf, total_us / 1000.0f, per_iter_us, qps, two_buf_us / per_iter_us);
        
        // Cleanup
        for (int i = 0; i < nbuf; i++) {
            cudaFree(d_buf[i]);
            cudaFreeHost(h_buf[i]);
        }
    }
    
    // ========================================
    // Test 3: Multiple streams — independent pipelines
    // ========================================
    printf("\n--- Test 3: Multiple Independent Pipelines ---\n");
    printf("Each stream runs its own double-buffer pipeline.\n\n");
    
    printf("%-8s | %-14s | %-14s | %-14s | %-14s\n",
           "Streams", "Total (ms)", "Per-iter (us)", "Room-qps", "vs 1 stream");
    printf("---------|----------------|----------------|----------------|----------------\n");
    
    float one_stream_us = 0;
    int stream_counts[] = {1, 2, 4, 8};
    int iters_per_stream = 20000;
    
    for (int sc = 0; sc < 4; sc++) {
        int nstreams = stream_counts[sc];
        
        std::vector<cudaStream_t> streams(nstreams);
        std::vector<float*> d_out(nstreams * 2), h_out(nstreams * 2), d_zc(nstreams * 2);
        for (int s = 0; s < nstreams; s++) {
            cudaStreamCreate(&streams[s]);
            for (int b = 0; b < 2; b++) {
                cudaMalloc(&d_out[s * 2 + b], rooms * sizeof(float));
                cudaHostAlloc(&h_out[s * 2 + b], rooms * sizeof(float), cudaHostAllocMapped);
                cudaHostGetDevicePointer(&d_zc[s * 2 + b], h_out[s * 2 + b], 0);
            }
        }
        
        // Warmup
        for (int i = 0; i < 100; i++) {
            for (int s = 0; s < nstreams; s++) {
                int buf = i % 2;
                infer_v7<<<grid, dim3(256), 0, streams[s]>>>(d_weights, d_input, d_zc[s * 2 + buf], rooms, DIM);
            }
            for (int s = 0; s < nstreams; s++) cudaStreamSynchronize(streams[s]);
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters_per_stream; i++) {
            for (int s = 0; s < nstreams; s++) {
                int curr = i % 2;
                int prev = 1 - curr;
                infer_v7<<<grid, dim3(256), 0, streams[s]>>>(d_weights, d_input, d_zc[s * 2 + curr], rooms, DIM);
                if (i > 0) volatile float sink = h_out[s * 2 + prev][0];
            }
        }
        for (int s = 0; s < nstreams; s++) cudaStreamSynchronize(streams[s]);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        float total_us = std::chrono::duration<float, std::micro>(t1 - t0).count();
        float per_iter_us = total_us / iters_per_stream;
        float total_qps = nstreams * rooms / (per_iter_us / 1e6f);
        
        if (sc == 0) one_stream_us = per_iter_us;
        
        printf("%-8d | %12.2f   | %12.2f   | %12.0f   | %.3fx\n",
               nstreams, total_us / 1000.0f, per_iter_us, total_qps, one_stream_us / per_iter_us);
        
        // Cleanup
        for (int s = 0; s < nstreams; s++) {
            cudaStreamDestroy(streams[s]);
            for (int b = 0; b < 2; b++) {
                cudaFree(d_out[s * 2 + b]);
                cudaFreeHost(h_out[s * 2 + b]);
            }
        }
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_output);
    
    printf("\n=== Suite #34 Complete ===\n");
    return 0;
}
