/**
 * Suite #32: CUDA Graphs for Complete Inference Pipeline
 * 
 * Suite #5 showed CUDA Graphs give 1.34× for single kernel.
 * Suite #7 showed CUDA Graphs + Streams conflict.
 * Suite #31 showed launch+sync overhead is 20μs.
 * 
 * Can CUDA Graphs capture the ENTIRE pipeline (weight load → infer → read)
 * to eliminate the ~20μs overhead?
 * 
 * Test: CUDA Graph with embedded cudaMemcpy + kernel + output read
 * vs sequential (non-graph) pipeline
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 graph_pipeline.cu -o graph_pipeline
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
#define WARMUP 200
#define ITERS 10000

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
    printf("=== Suite #32: CUDA Graphs for Complete Pipeline ===\n\n");
    
    int batch_sizes[] = {1, 4, 16, 64, 256, 1024};
    
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
    
    printf("%-8s | %-14s | %-14s | %-8s | %-14s\n",
           "Rooms", "Sequential(us)", "Graph(us)", "Speedup", "Room-qps(graph)");
    printf("---------|----------------|----------------|----------|----------------\n");
    
    for (int b = 0; b < 6; b++) {
        int rooms = batch_sizes[b];
        dim3 grid((rooms + 7) / 8);
        dim3 block(256);
        
        // ---- Sequential: cudaMemcpy + kernel + streamSync + cpuRead ----
        std::vector<float> seq_lat(WARMUP + ITERS);
        for (int i = 0; i < WARMUP + ITERS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            infer_v7<<<grid, block, 0, stream>>>(d_weights, d_input, d_zc_output, rooms, DIM);
            cudaStreamSynchronize(stream);
            volatile float sink = h_zc_output[0];
            auto t1 = std::chrono::high_resolution_clock::now();
            seq_lat[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
        }
        std::sort(seq_lat.begin(), seq_lat.end());
        float seq_p50 = seq_lat[WARMUP + ITERS / 2];
        float seq_p99 = seq_lat[WARMUP + (int)(ITERS * 0.99)];
        
        // ---- CUDA Graph: capture the same operations ----
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        
        // Capture
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        infer_v7<<<grid, block, 0, stream>>>(d_weights, d_input, d_zc_output, rooms, DIM);
        cudaStreamEndCapture(stream, &graph);
        
        // Instantiate
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        
        // Warmup graph
        for (int i = 0; i < WARMUP; i++) {
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
            volatile float sink = h_zc_output[0];
        }
        
        // Benchmark graph
        std::vector<float> graph_lat(ITERS);
        for (int i = 0; i < ITERS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
            volatile float sink = h_zc_output[0];
            auto t1 = std::chrono::high_resolution_clock::now();
            graph_lat[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
        }
        
        cudaStreamSynchronize(stream);
        
        std::sort(graph_lat.begin(), graph_lat.end());
        float graph_p50 = graph_lat[ITERS / 2];
        float graph_p99 = graph_lat[(int)(ITERS * 0.99)];
        
        float graph_qps = rooms / (graph_p50 / 1e6f);
        float speedup = seq_p50 / graph_p50;
        
        printf("%-8d | %12.2f   | %12.2f   | %.3fx   | %12.0f\n",
               rooms, seq_p50, graph_p50, speedup, graph_qps);
        
        printf("         (p99: %10.2f   | %10.2f   | p99 ratio: %.3fx)\n",
               seq_p99, graph_p99, seq_p99 / graph_p99);
        
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }
    
    // ---- Test: Graph with 4 streams (known to conflict from suite #7) ----
    printf("\n--- Graph + 4 Streams (Suite #7 revisit) ---\n");
    
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) cudaStreamCreate(&streams[i]);
    
    int rooms = 256;
    dim3 grid((rooms + 7) / 8);
    
    // Sequential 4 streams (no graph)
    {
        for (int i = 0; i < WARMUP; i++) {
            for (int s = 0; s < 4; s++)
                infer_v7<<<grid, dim3(256), 0, streams[s]>>>(d_weights, d_input, d_zc_output + s * rooms, rooms, DIM);
            for (int s = 0; s < 4; s++) cudaStreamSynchronize(streams[s]);
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            for (int s = 0; s < 4; s++)
                infer_v7<<<grid, dim3(256), 0, streams[s]>>>(d_weights, d_input, d_zc_output + s * rooms, rooms, DIM);
            for (int s = 0; s < 4; s++) cudaStreamSynchronize(streams[s]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        printf("  4 streams (no graph): %.2f us for 4×256 rooms = %.1fM room-qps\n",
               us, 4 * rooms / (us / 1e6f) / 1e6);
    }
    
    // Single graph, 4×256 rooms in one launch
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        int total = 1024;
        dim3 g((total + 7) / 8);
        
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        infer_v7<<<g, dim3(256), 0, stream>>>(d_weights, d_input, d_zc_output, total, DIM);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        
        for (int i = 0; i < WARMUP; i++) {
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        printf("  Graph (1024 rooms, 1 launch): %.2f us = %.1fM room-qps\n",
               us, total / (us / 1e6f) / 1e6);
        
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }
    
    // Cleanup
    for (int i = 0; i < 4; i++) cudaStreamDestroy(streams[i]);
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_output);
    cudaFreeHost(h_zc_output);
    
    printf("\n=== Suite #32 Complete ===\n");
    return 0;
}
