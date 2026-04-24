/**
 * DECKBOSS ROOM INFERENCE — Production Kernel
 * Jetson Orin Nano | sm_87 | CUDA 12.6
 * 
 * Warp shuffle-based room inference kernel.
 * Each warp processes one room's output (16 neurons from 256-weight dot products).
 * 
 * Architecture:
 *   Input:  shared input vector (256 FP16)
 *   Weights: per-room weight matrices (16×256 FP16)  
 *   Output: per-room activation vectors (16 FP32)
 *   Rooms: up to 12 concurrent (warp per room)
 * 
 * Performance target: <0.1ms for 12 rooms (deckboss spec)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define MAX_ROOMS 12
#define NEURONS 16
#define INPUT_SIZE 256

__global__ void room_inference(
    const half* __restrict__ weights,  // [rooms][NEURONS][INPUT_SIZE]
    const half* __restrict__ input,    // [INPUT_SIZE] shared across rooms
    float* __restrict__ output,        // [rooms][NEURONS]
    const int num_rooms
) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    if (room >= num_rooms) return;
    
    const half* w = weights + room * NEURONS * INPUT_SIZE;
    
    // Each warp processes all NEURONS output values sequentially
    // For each output neuron: dot product of weight row with input
    for (int n = 0; n < NEURONS; n++) {
        float sum = 0.0f;
        // Cooperative: 32 lanes, each handles 8 elements (256/32=8)
        #pragma unroll
        for (int k = lane; k < INPUT_SIZE; k += 32)
            sum += __half2float(__hmul(w[n * INPUT_SIZE + k], input[k]));
        
        // Warp reduction
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, s);
        
        // GELU activation
        if (lane == 0) {
            float x = sum;
            float gelu = x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            output[room * NEURONS + n] = gelu;
        }
    }
}

// Baseline: thread-based (one block per room)
__global__ void room_inference_thread(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    const int num_rooms
) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    
    const half* w = weights + room * NEURONS * INPUT_SIZE;
    int tid = threadIdx.x;
    
    for (int n = tid; n < NEURONS; n += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < INPUT_SIZE; k++)
            sum += __half2float(__hmul(w[n * INPUT_SIZE + k], input[k]));
        float x = sum;
        output[room * NEURONS + n] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  DECKBOSS ROOM INFERENCE — Production Benchmark          ║\n");
    printf("║  Jetson Orin Nano | sm_87 | CUDA 12.6                   ║\n");
    printf("║                                                          ║\n");
    printf("║  12 rooms × 16 neurons × 256 weights = 49,152 FLOPs     ║\n");
    printf("║  FP16 weights, FP32 accumulation, GELU activation        ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | sm_%d%d | %d SMs | %.0f MHz | %.0f MB\n\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           p.clockRate/1000.0, p.totalGlobalMem/1048576.0);
    
    const int ITERS = 100000;
    
    // Test with increasing room counts
    int room_counts[] = {1, 3, 6, 9, 12};
    int num_counts = sizeof(room_counts)/sizeof(room_counts[0]);
    
    // Allocate for max rooms
    const size_t w_bytes = MAX_ROOMS * NEURONS * INPUT_SIZE * sizeof(half);
    const size_t in_bytes = INPUT_SIZE * sizeof(half);
    const size_t out_bytes = MAX_ROOMS * NEURONS * sizeof(float);
    
    half *hw = (half*)malloc(w_bytes);
    half *hi = (half*)malloc(in_bytes);
    float *h1 = (float*)malloc(out_bytes);
    float *h2 = (float*)malloc(out_bytes);
    
    srand(42);
    for (size_t i = 0; i < MAX_ROOMS*NEURONS*INPUT_SIZE; i++)
        hw[i] = __float2half((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < INPUT_SIZE; i++)
        hi[i] = __float2half((float)rand()/RAND_MAX - 0.5f);
    
    half *dw,*di; float *d1,*d2;
    cudaMalloc(&dw,w_bytes); cudaMalloc(&di,in_bytes);
    cudaMalloc(&d1,out_bytes); cudaMalloc(&d2,out_bytes);
    cudaMemcpy(dw,hw,w_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(di,hi,in_bytes,cudaMemcpyHostToDevice);
    
    cudaEvent_t es,ee; cudaEventCreate(&es); cudaEventCreate(&ee);
    float ms;
    
    printf("┌────────┬────────────────┬────────────┬────────────┬──────────┐\n");
    printf("│ Rooms  │ Implementation │   Latency  │ Throughput │  GFLOPS  │\n");
    printf("├────────┼────────────────┼────────────┼────────────┼──────────┤\n");
    
    for (int rc = 0; rc < num_counts; rc++) {
        int nr = room_counts[rc];
        float flops = (float)nr * NEURONS * INPUT_SIZE * 2;
        
        // Warmup
        for (int i=0;i<200;i++) {
            room_inference<<<nr,32>>>(dw,di,d1,nr);
            room_inference_thread<<<nr,256>>>(dw,di,d2,nr);
        }
        cudaDeviceSynchronize();
        
        // Warp
        cudaEventRecord(es);
        for (int i=0;i<ITERS;i++) room_inference<<<nr,32>>>(dw,di,d1,nr);
        cudaEventRecord(ee); cudaEventSynchronize(ee);
        cudaEventElapsedTime(&ms,es,ee); float lw=ms/ITERS;
        
        // Thread
        cudaEventRecord(es);
        for (int i=0;i<ITERS;i++) room_inference_thread<<<nr,256>>>(dw,di,d2,nr);
        cudaEventRecord(ee); cudaEventSynchronize(ee);
        cudaEventElapsedTime(&ms,es,ee); float lt=ms/ITERS;
        
        printf("│ %4d   │ Warp (32 thr)   │ %8.4f ms │ %8.0f/s │ %7.2f  │\n",
               nr, lw, 1000/lw, flops/(lw/1000)/1e9);
        printf("│ %4d   │ Thread (256 thr) │ %8.4f ms │ %8.0f/s │ %7.2f  │\n",
               nr, lt, 1000/lt, flops/(lt/1000)/1e9);
        printf("│        │ Warp speedup:  %17.2fx              │\n", lt/lw);
    }
    
    printf("└────────┴────────────────┴────────────┴────────────┴──────────┘\n");
    
    // Verify correctness for 12 rooms
    cudaMemcpy(h1,d1,out_bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h2,d2,out_bytes,cudaMemcpyDeviceToHost);
    printf("\n=== Correctness (12 rooms × 16 neurons) ===\n");
    float maxabs = 0;
    int ok = 1;
    for (int r = 0; r < 12; r++) {
        for (int n = 0; n < NEURONS; n++) {
            float d = fabsf(h1[r*NEURONS+n] - h2[r*NEURONS+n]);
            if (d > maxabs) maxabs = d;
            if (isnan(h1[r*NEURONS+n])) ok = 0;
        }
    }
    printf("Warp vs Thread: max abs diff = %.2e, all finite = %s\n", maxabs, ok?"✓":"✗");
    
    printf("\n=== DECKBOSS TARGETS ===\n");
    printf("Target: 12 rooms in <0.1ms\n");
    // Run 12-room benchmark one more time for the exact number
    for (int i=0;i<200;i++) room_inference<<<12,32>>>(dw,di,d1,12);
    cudaDeviceSynchronize();
    cudaEventRecord(es);
    for (int i=0;i<ITERS;i++) room_inference<<<12,32>>>(dw,di,d1,12);
    cudaEventRecord(ee);
        cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms,es,ee);
    float l12 = ms/ITERS;
    const char* result = (l12 < 0.1f) ? "MEETS TARGET" : "above target"; printf("Actual: 12 rooms in %.4f ms -- %s\n", l12, result);
    printf("Memory per room: %zu bytes (weights) + %zu bytes (output)\n",
           NEURONS*INPUT_SIZE*sizeof(half), NEURONS*sizeof(float));
    printf("Total for 12 rooms: %zu KB weights + %zu bytes output\n",
           12*NEURONS*INPUT_SIZE*sizeof(half)/1024, 12*NEURONS*sizeof(float));
    
    cudaFree(dw);cudaFree(di);cudaFree(d1);cudaFree(d2);
    free(hw);free(hi);free(h1);free(h2);
    cudaEventDestroy(es);cudaEventDestroy(ee);
    return 0;
}
