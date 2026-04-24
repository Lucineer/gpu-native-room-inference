/**
 * CUDA Graphs Benchmark — Eliminating Launch Overhead on Jetson
 * 
 * Proves that CUDA Graphs can eliminate the ~35μs kernel launch overhead
 * that dominates edge GPU inference on the Orin Nano.
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 cuda_graphs_bench.cu -o cuda_graphs_bench
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

// Simple room inference kernel: 256-element dot product + GELU
__global__ void room_inference_kernel(
    const half* weights,
    const half* input,
    float* output,
    int dim
) {
    int room_id = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room_id * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room_id] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Two-kernel pipeline: inference + copy result back
__global__ void scale_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scale;
}

int main() {
    printf("========================================\n");
    printf("CUDA Graphs Benchmark\n");
    printf("Jetson Orin Nano, sm_87, CUDA 12.6\n");
    printf("========================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Clock: %.0f MHz\n\n", 
           prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0);
    
    const int DIM = 256;
    const int NUM_ROOMS = 6;
    const int ITERS = 100000;
    
    // Allocate
    half *d_weights, *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, NUM_ROOMS * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, NUM_ROOMS * sizeof(float)));
    
    // Init
    half *h_weights = (half*)malloc(NUM_ROOMS * DIM * sizeof(half));
    half *h_input = (half*)malloc(DIM * sizeof(half));
    float *h_output = (float*)malloc(NUM_ROOMS * sizeof(float));
    
    srand(42);
    for (int i = 0; i < NUM_ROOMS * DIM; i++) h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++) h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, NUM_ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < 500; i++)
        room_inference_kernel<<<NUM_ROOMS, 32>>>(d_weights, d_input, d_output, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    //=== Test 1: Standard launch (baseline) ===
    printf("=== Test 1: Standard Kernel Launch ===\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        room_inference_kernel<<<NUM_ROOMS, 32>>>(d_weights, d_input, d_output, DIM);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_standard = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_standard, start, stop));
    float lat_standard = ms_standard / ITERS;
    printf("  %d iterations in %.2f ms\n", ITERS, ms_standard);
    printf("  Latency: %.4f ms (%.1f us)\n", lat_standard, lat_standard * 1000);
    printf("  Throughput: %.0f qps\n\n", 1000.0 / lat_standard);
    
    //=== Test 2: CUDA Graph (single kernel) ===
    printf("=== Test 2: CUDA Graph (Single Kernel) ===\n");
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // Capture graph
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t node;
    cudaKernelNodeParams params = {0};
    params.func = (void*)room_inference_kernel;
    params.gridDim = dim3(NUM_ROOMS, 1, 1);
    params.blockDim = dim3(32, 1, 1);
    params.sharedMemBytes = 0;
    int dim_val = DIM;
    int num_rooms_val = NUM_ROOMS;
    void* kernel_args[] = {&d_weights, &d_input, &d_output, &dim_val};
    params.kernelParams = kernel_args;
    params.extra = nullptr;
    
    CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    // Warmup graph
    for (int i = 0; i < 500; i++)
        CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_graph = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_graph, start, stop));
    float lat_graph = ms_graph / ITERS;
    printf("  %d iterations in %.2f ms\n", ITERS, ms_graph);
    printf("  Latency: %.4f ms (%.1f us)\n", lat_graph, lat_graph * 1000);
    printf("  Throughput: %.0f qps\n\n", 1000.0 / lat_graph);
    
    //=== Test 3: CUDA Graph (multi-kernel pipeline) ===
    printf("=== Test 3: CUDA Graph (Inference + Scale Pipeline) ===\n");
    
    cudaGraph_t graph2;
    cudaGraphExec_t graphExec2;
    CUDA_CHECK(cudaGraphCreate(&graph2, 0));
    
    // Node 1: room inference
    cudaKernelNodeParams params1 = {0};
    params1.func = (void*)room_inference_kernel;
    params1.gridDim = dim3(NUM_ROOMS, 1, 1);
    params1.blockDim = dim3(32, 1, 1);
    params1.sharedMemBytes = 0;
    params1.kernelParams = kernel_args;
    params1.extra = nullptr;
    
    cudaGraphNode_t node1;
    CUDA_CHECK(cudaGraphAddKernelNode(&node1, graph2, nullptr, 0, &params1));
    
    // Node 2: scale (depends on node1)
    cudaKernelNodeParams params2 = {0};
    params2.func = (void*)scale_kernel;
    params2.gridDim = dim3(1, 1, 1);
    params2.blockDim = dim3(NUM_ROOMS, 1, 1);
    params2.sharedMemBytes = 0;
    float scale = 2.0f;
    void* scale_args[] = {&d_output, &num_rooms_val, &scale};
    params2.kernelParams = scale_args;
    params2.extra = nullptr;
    
    cudaGraphNode_t node2;
    cudaGraphNode_t dependencies[] = {node1};
    CUDA_CHECK(cudaGraphAddKernelNode(&node2, graph2, dependencies, 1, &params2));
    
    CUDA_CHECK(cudaGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
    
    for (int i = 0; i < 500; i++)
        CUDA_CHECK(cudaGraphLaunch(graphExec2, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        CUDA_CHECK(cudaGraphLaunch(graphExec2, 0));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_pipeline = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pipeline, start, stop));
    float lat_pipeline = ms_pipeline / ITERS;
    printf("  %d iterations in %.2f ms\n", ITERS, ms_pipeline);
    printf("  Latency: %.4f ms (%.1f us)\n", lat_pipeline, lat_pipeline * 1000);
    printf("  Throughput: %.0f qps\n\n", 1000.0 / lat_pipeline);
    
    //=== Test 4: Standard launch (same pipeline, no graph) ===
    printf("=== Test 4: Standard Launch (Same Pipeline, No Graph) ===\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        room_inference_kernel<<<NUM_ROOMS, 32>>>(d_weights, d_input, d_output, DIM);
        scale_kernel<<<1, NUM_ROOMS>>>(d_output, NUM_ROOMS, 2.0f);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_std_pipe = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_std_pipe, start, stop));
    float lat_std_pipe = ms_std_pipe / ITERS;
    printf("  %d iterations in %.2f ms\n", ITERS, ms_std_pipe);
    printf("  Latency: %.4f ms (%.1f us)\n", lat_std_pipe, lat_std_pipe * 1000);
    printf("  Throughput: %.0f qps\n\n", 1000.0 / lat_std_pipe);
    
    //=== Summary ===
    printf("========================================\n");
    printf("SUMMARY\n");
    printf("========================================\n");
    printf("%-35s %12s %12s %10s\n", "Test", "Latency(us)", "QPS", "vs Standard");
    printf("%-35s %12s %12s %10s\n", "-----------------------------------", "------------", "------------", "----------");
    printf("%-35s %10.1f us %12.0f %10s\n", "Standard (single kernel)", lat_standard * 1000, 1000.0 / lat_standard, "1.00x");
    printf("%-35s %10.1f us %12.0f %10.2fx\n", "CUDA Graph (single kernel)", lat_graph * 1000, 1000.0 / lat_graph, lat_standard / lat_graph);
    printf("%-35s %10.1f us %12.0f %10.2fx\n", "CUDA Graph (pipeline)", lat_pipeline * 1000, 1000.0 / lat_pipeline, lat_standard / lat_pipeline);
    printf("%-35s %10.1f us %12.0f %10.2fx\n", "Standard (pipeline)", lat_std_pipe * 1000, 1000.0 / lat_std_pipe, lat_standard / lat_std_pipe);
    
    printf("\n=== KEY FINDING ===\n");
    float graph_speedup = lat_standard / lat_graph;
    printf("CUDA Graph speedup: %.2fx\n", graph_speedup);
    printf("Launch overhead eliminated: ~%.1f us\n", (lat_standard - lat_graph) * 1000);
    
    float pipeline_graph_vs_std = lat_std_pipe / lat_pipeline;
    printf("Pipeline graph vs standard pipeline: %.2fx\n", pipeline_graph_vs_std);
    
    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec2));
    CUDA_CHECK(cudaGraphDestroy(graph2));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_weights);
    free(h_input);
    free(h_output);
    
    printf("\nDone.\n");
    return 0;
}
