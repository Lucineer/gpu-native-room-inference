/**
 * Ultimate Edge Inference Benchmark
 * 
 * Combines all discovered optimizations:
 * 1. CUDA Graphs (eliminate launch overhead)
 * 2. 4 CUDA Streams (overlap CPU dispatch with GPU compute)
 * 3. Batched room inference (maximize GPU utilization)
 * 
 * This is the theoretical maximum throughput for deckboss on Jetson Orin.
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 ultimate_bench.cu -o ultimate_bench
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

void run_test(const char* name, int num_rooms, int num_iters, bool use_streams, 
              bool use_graph, int num_streams, const half* d_weights, 
              const half* d_input, float* d_output, int dim) {
    
    printf("\n=== %s ===\n", name);
    printf("  Rooms: %d | Iters: %d | Streams: %s (%d) | Graph: %s\n",
           num_rooms, num_iters, 
           use_streams ? "yes" : "no", use_streams ? num_streams : 0,
           use_graph ? "yes" : "no");
    
    dim3 grid(num_rooms, 1, 1);
    dim3 block(32, 1, 1);
    int dim_val = dim;
    
    cudaStream_t streams[8];
    if (use_streams) {
        for (int i = 0; i < num_streams; i++)
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Warmup
    for (int i = 0; i < 500; i++) {
        if (use_streams)
            room_inference_kernel<<<grid, block, 0, streams[i % num_streams]>>>(
                d_weights, d_input, d_output, dim_val);
        else
            room_inference_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim_val);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create graph if needed
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    if (use_graph) {
        if (use_streams) {
            // Graph with streams: create per-stream graphs
            CUDA_CHECK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
            room_inference_kernel<<<grid, block, 0, streams[0]>>>(d_weights, d_input, d_output, dim_val);
            CUDA_CHECK(cudaStreamEndCapture(streams[0], &graph));
        } else {
            CUDA_CHECK(cudaGraphCreate(&graph, 0));
            cudaKernelNodeParams params = {0};
            params.func = (void*)room_inference_kernel;
            params.gridDim = grid;
            params.blockDim = block;
            params.sharedMemBytes = 0;
            void* args[] = {&d_weights, &d_input, &d_output, &dim_val};
            params.kernelParams = args;
            params.extra = nullptr;
            cudaGraphNode_t node;
            CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
        }
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        
        // Warmup graph
        for (int i = 0; i < 500; i++) {
            if (use_streams)
                CUDA_CHECK(cudaGraphLaunch(graphExec, streams[0]));
            else
                CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < num_iters; i++) {
        if (use_graph) {
            if (use_streams)
                CUDA_CHECK(cudaGraphLaunch(graphExec, streams[i % num_streams]));
            else
                CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
        } else {
            if (use_streams)
                room_inference_kernel<<<grid, block, 0, streams[i % num_streams]>>>(
                    d_weights, d_input, d_output, dim_val);
            else
                room_inference_kernel<<<grid, block>>>(d_weights, d_input, d_output, dim_val);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    float per_iter = ms / num_iters;
    float per_room = per_iter / num_rooms;
    float total_rooms = (float)num_iters * num_rooms;
    float room_qps = total_rooms / (ms / 1000.0);
    
    printf("  Total: %.2f ms (%d iters x %d rooms = %d room-inferences)\n", 
           ms, num_iters, num_rooms, (int)total_rooms);
    printf("  Per-iter: %.1f us | Per-room: %.3f us\n", per_iter * 1000, per_room * 1000);
    printf("  Room QPS: %.0f | Iter QPS: %.0f\n", room_qps, 1000.0 / per_iter);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    if (use_graph) {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    if (use_streams) {
        for (int i = 0; i < num_streams; i++)
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

int main() {
    printf("============================================================\n");
    printf("ULTIMATE EDGE INFERENCE BENCHMARK\n");
    printf("All Optimizations Combined — Jetson Orin Nano\n");
    printf("============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Clock: %.0f MHz\n\n", 
           prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0);
    
    const int DIM = 256;
    const int ITERS = 50000;
    
    half *d_weights, *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, 64 * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, 64 * sizeof(float)));
    
    half *h_weights = (half*)malloc(64 * DIM * sizeof(half));
    half *h_input = (half*)malloc(DIM * sizeof(half));
    
    srand(42);
    for (int i = 0; i < 64 * DIM; i++) h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++) h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, 64 * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run all configurations for 6 rooms (production deckboss)
    printf(">>> 6 Rooms (production deckboss)\n");
    run_test("Baseline (1 stream, no graph)", 6, ITERS, false, false, 0, d_weights, d_input, d_output, DIM);
    run_test("4 Streams", 6, ITERS, true, false, 4, d_weights, d_input, d_output, DIM);
    run_test("CUDA Graph", 6, ITERS, false, true, 0, d_weights, d_input, d_output, DIM);
    run_test("CUDA Graph + 4 Streams", 6, ITERS, true, true, 4, d_weights, d_input, d_output, DIM);
    
    // Run for 64 rooms (fleet scale)
    printf("\n>>> 64 Rooms (fleet scale)\n");
    run_test("Baseline (1 stream, no graph)", 64, ITERS, false, false, 0, d_weights, d_input, d_output, DIM);
    run_test("4 Streams", 64, ITERS, true, false, 4, d_weights, d_input, d_output, DIM);
    run_test("CUDA Graph", 64, ITERS, false, true, 0, d_weights, d_input, d_output, DIM);
    run_test("CUDA Graph + 4 Streams", 64, ITERS, true, true, 4, d_weights, d_input, d_output, DIM);
    
    // Massive batch (4096 rooms)
    printf("\n>>> 4096 Rooms (maximum throughput)\n");
    run_test("Baseline", 4096, ITERS/10, false, false, 0, d_weights, d_input, d_output, DIM);
    run_test("4 Streams", 4096, ITERS/10, true, false, 4, d_weights, d_input, d_output, DIM);
    
    printf("\n============================================================\n");
    printf("CONCLUSION\n");
    printf("============================================================\n");
    printf("The optimal deckboss configuration:\n");
    printf("  - 4 CUDA streams (saturates Orin pipeline)\n");
    printf("  - CUDA Graphs (eliminate launch overhead)\n");
    printf("  - Batch as many rooms as possible per launch\n");
    printf("Expected: 300K+ room-qps for production, 80M+ for large batch\n");
    
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_weights);
    free(h_input);
    
    return 0;
}
