/**
 * Quantization Impact Benchmark
 * 
 * Tests FP16 vs INT8 vs INT4 (packed) room inference.
 * Hypothesis: Memory bandwidth is the bottleneck → quantization should
 * proportionally improve throughput at large batch sizes.
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 quant_bench.cu -o quant_bench
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// FP16 baseline (current production)
__global__ void room_fp16(const half* __restrict__ weights, const half* __restrict__ input,
                          float* __restrict__ output, int dim) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0)
        output[room] = sum;
}

// INT8 quantized (weights and input as int8, dequantize on-the-fly)
__global__ void room_int8(const int8_t* __restrict__ weights, 
                          const int8_t* __restrict__ input,
                          float* __restrict__ output,
                          float w_scale, float w_zero,
                          float i_scale, float i_zero,
                          int dim) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float w = (float)weights[room * dim + i] * w_scale + w_zero;
        float x = (float)input[i] * i_scale + i_zero;
        sum += w * x;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0)
        output[room] = sum;
}

// INT8 with shared memory dequantization
__global__ void room_int8_shared(const int8_t* __restrict__ weights,
                                  const int8_t* __restrict__ input,
                                  float* __restrict__ output,
                                  float w_scale, float w_zero,
                                  float i_scale, float i_zero,
                                  int dim) {
    extern __shared__ float s_input[];
    
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    // Dequantize input to shared memory (reused across all rooms)
    for (int i = lane; i < dim; i += 32)
        s_input[i] = (float)input[i] * i_scale + i_zero;
    __syncthreads();
    
    // Compute with weights dequantized inline
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float w = (float)weights[room * dim + i] * w_scale + w_zero;
        sum += w * s_input[i];
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0)
        output[room] = sum;
}

// INT4 packed (2 values per byte)
__global__ void room_int4(const uint8_t* __restrict__ weights,
                           const uint8_t* __restrict__ input,
                           float* __restrict__ output,
                           float w_scale, float w_zero,
                           float i_scale, float i_zero,
                           int dim) {
    int room = blockIdx.x;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        // Unpack INT4: high nibble = first value, low nibble = second
        int byte_idx = i / 2;
        int w_val = (i & 1) ? (weights[room * (dim/2) + byte_idx] & 0x0F) 
                            : ((weights[room * (dim/2) + byte_idx] >> 4) & 0x0F);
        int i_val = (i & 1) ? (input[byte_idx] & 0x0F) 
                            : ((input[byte_idx] >> 4) & 0x0F);
        // Sign-extend from 4-bit (values 0-15 → -8 to 7)
        if (w_val > 7) w_val -= 16;
        if (i_val > 7) i_val -= 16;
        
        float w = (float)w_val * w_scale + w_zero;
        float x = (float)i_val * i_scale + i_zero;
        sum += w * x;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0)
        output[room] = sum;
}

void run_test(const char* name, int num_rooms, int dim, int iters, float& us_per_room) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        // Placeholder
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("============================================================\n");
    printf("QUANTIZATION IMPACT BENCHMARK\n");
    printf("FP16 vs INT8 vs INT4 — Jetson Orin Nano\n");
    printf("============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Clock: %.0f MHz\n\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0);
    
    const int DIM = 256;
    const int MAX_ROOMS = 1024;
    const int ITERS = 100000;
    
    // Allocate and initialize FP16 data
    half *h_weights_fp16 = (half*)malloc(MAX_ROOMS * DIM * sizeof(half));
    half *h_input_fp16 = (half*)malloc(DIM * sizeof(half));
    
    srand(42);
    for (int i = 0; i < MAX_ROOMS * DIM; i++)
        h_weights_fp16[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++)
        h_input_fp16[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    // Quantize to INT8 (symmetric)
    int8_t *h_weights_i8 = (int8_t*)malloc(MAX_ROOMS * DIM);
    int8_t *h_input_i8 = (int8_t*)malloc(DIM);
    
    float w_scale = 0.0039f; // ~1/256
    float w_zero = 0.0f;
    float i_scale = 0.0039f;
    float i_zero = 0.0f;
    
    for (int i = 0; i < MAX_ROOMS * DIM; i++) {
        float v = __half2float(h_weights_fp16[i]);
        int q = (int)(v / w_scale + 0.5f);
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        h_weights_i8[i] = (int8_t)q;
    }
    for (int i = 0; i < DIM; i++) {
        float v = __half2float(h_input_fp16[i]);
        int q = (int)(v / i_scale + 0.5f);
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        h_input_i8[i] = (int8_t)q;
    }
    
    // Quantize to INT4 packed
    uint8_t *h_weights_i4 = (uint8_t*)calloc(MAX_ROOMS * (DIM / 2), 1);
    uint8_t *h_input_i4 = (uint8_t*)calloc(DIM / 2, 1);
    
    float i4_scale = 0.0625f; // ~1/16
    
    for (int i = 0; i < MAX_ROOMS * DIM; i++) {
        float v = __half2float(h_weights_fp16[i]);
        int q = (int)(v / i4_scale + 8.5f);
        q = q > 15 ? 15 : (q < 0 ? 0 : q);
        int byte_idx = i / 2;
        if (i & 1)
            h_weights_i4[MAX_ROOMS * (DIM/2) + byte_idx - MAX_ROOMS * (DIM/2)] |= (q & 0x0F); // wrong
    }
    // Simpler INT4 packing
    for (int i = 0; i < MAX_ROOMS * DIM; i += 2) {
        float v0 = __half2float(h_weights_fp16[i]);
        float v1 = __half2float(h_weights_fp16[i+1]);
        int q0 = (int)(v0 / i4_scale + 8.5f); q0 = q0 > 15 ? 15 : (q0 < 0 ? 0 : q0);
        int q1 = (int)(v1 / i4_scale + 8.5f); q1 = q1 > 15 ? 15 : (q1 < 0 ? 0 : q1);
        int room = i / DIM;
        int elem = (i % DIM) / 2;
        h_weights_i4[room * (DIM/2) + elem] = (uint8_t)((q0 << 4) | q1);
    }
    for (int i = 0; i < DIM; i += 2) {
        float v0 = __half2float(h_input_fp16[i]);
        float v1 = __half2float(h_input_fp16[i+1]);
        int q0 = (int)(v0 / i4_scale + 8.5f); q0 = q0 > 15 ? 15 : (q0 < 0 ? 0 : q0);
        int q1 = (int)(v1 / i4_scale + 8.5f); q1 = q1 > 15 ? 15 : (q1 < 0 ? 0 : q1);
        h_input_i4[i/2] = (uint8_t)((q0 << 4) | q1);
    }
    
    // GPU allocations
    half *d_weights_fp16, *d_input_fp16;
    int8_t *d_weights_i8, *d_input_i8;
    uint8_t *d_weights_i4, *d_input_i4;
    float *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_weights_fp16, MAX_ROOMS * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input_fp16, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weights_i8, MAX_ROOMS * DIM));
    CUDA_CHECK(cudaMalloc(&d_input_i8, DIM));
    CUDA_CHECK(cudaMalloc(&d_weights_i4, MAX_ROOMS * (DIM / 2)));
    CUDA_CHECK(cudaMalloc(&d_input_i4, DIM / 2));
    CUDA_CHECK(cudaMalloc(&d_output, MAX_ROOMS * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_weights_fp16, h_weights_fp16, MAX_ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_fp16, h_input_fp16, DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights_i8, h_weights_i8, MAX_ROOMS * DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_i8, h_input_i8, DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights_i4, h_weights_i4, MAX_ROOMS * (DIM / 2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_i4, h_input_i4, DIM / 2, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int batches[] = {1, 4, 16, 64, 256, 1024};
    int num_batches = 6;
    
    printf(">>> Batch Size Scaling by Precision (256-dim rooms)\n\n");
    
    for (int b = 0; b < num_batches; b++) {
        int batch = batches[b];
        int iters = batch <= 64 ? ITERS : (ITERS * 64 / batch);
        
        dim3 block(32);
        dim3 grid(batch);
        
        printf("--- %d rooms ---\n", batch);
        
        // FP16
        for (int i = 0; i < 2000; i++)
            room_fp16<<<grid, block>>>(d_weights_fp16, d_input_fp16, d_output, DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_fp16<<<grid, block>>>(d_weights_fp16, d_input_fp16, d_output, DIM);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_fp16; CUDA_CHECK(cudaEventElapsedTime(&ms_fp16, start, stop));
        float us_fp16 = (ms_fp16 / iters / batch) * 1000.0;
        float qps_fp16 = batch / (ms_fp16 / iters / 1000.0);
        
        // INT8
        for (int i = 0; i < 2000; i++)
            room_int8<<<grid, block>>>(d_weights_i8, d_input_i8, d_output,
                                        w_scale, w_zero, i_scale, i_zero, DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_int8<<<grid, block>>>(d_weights_i8, d_input_i8, d_output,
                                        w_scale, w_zero, i_scale, i_zero, DIM);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_i8; CUDA_CHECK(cudaEventElapsedTime(&ms_i8, start, stop));
        float us_i8 = (ms_i8 / iters / batch) * 1000.0;
        float qps_i8 = batch / (ms_i8 / iters / 1000.0);
        
        // INT8 + shared memory
        for (int i = 0; i < 2000; i++)
            room_int8_shared<<<grid, block, DIM * sizeof(float)>>>(d_weights_i8, d_input_i8, d_output,
                                                                      w_scale, w_zero, i_scale, i_zero, DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_int8_shared<<<grid, block, DIM * sizeof(float)>>>(d_weights_i8, d_input_i8, d_output,
                                                                      w_scale, w_zero, i_scale, i_zero, DIM);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_i8s; CUDA_CHECK(cudaEventElapsedTime(&ms_i8s, start, stop));
        float us_i8s = (ms_i8s / iters / batch) * 1000.0;
        float qps_i8s = batch / (ms_i8s / iters / 1000.0);
        
        // INT4
        for (int i = 0; i < 2000; i++)
            room_int4<<<grid, block>>>(d_weights_i4, d_input_i4, d_output,
                                        i4_scale, 0.0f, i4_scale, 0.0f, DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            room_int4<<<grid, block>>>(d_weights_i4, d_input_i4, d_output,
                                        i4_scale, 0.0f, i4_scale, 0.0f, DIM);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_i4; CUDA_CHECK(cudaEventElapsedTime(&ms_i4, start, stop));
        float us_i4 = (ms_i4 / iters / batch) * 1000.0;
        float qps_i4 = batch / (ms_i4 / iters / 1000.0);
        
        printf("  %-8s %-12s %-12s %-12s %-12s\n", "Prec", "us/room", "Room-qps", "vs FP16", "Memory");
        printf("  %-8s %-12.3f %-12.0f %-12s %-12s\n", "FP16", us_fp16, qps_fp16, "1.00x", "2 bytes/elem");
        printf("  %-8s %-12.3f %-12.0f %-12.2fx %-12s\n", "INT8", us_i8, qps_i8, us_fp16/us_i8, "1 byte/elem");
        printf("  %-8s %-12.3f %-12.0f %-12.2fx %-12s\n", "INT8+sm", us_i8s, qps_i8s, us_fp16/us_i8s, "1B+smem");
        printf("  %-8s %-12.3f %-12.0f %-12.2fx %-12s\n", "INT4", us_i4, qps_i4, us_fp16/us_i4, "0.5 bytes/el");
        printf("\n");
    }
    
    // Memory capacity comparison
    printf("============================================================\n");
    printf("MEMORY CAPACITY COMPARISON\n");
    printf("============================================================\n\n");
    
    float gpu_budget_gb = 6.0f; // usable GPU memory
    float per_room_fp16 = (float)DIM * 2 * 2.0f / (1024*1024); // weights + input, 2 bytes each
    float per_room_i8 = (float)DIM * 2 * 1.0f / (1024*1024);
    float per_room_i4 = (float)(DIM/2) * 2 * 0.5f / (1024*1024); // packed
    
    printf("Per-room memory (256-dim):\n");
    printf("  FP16: %.3f MB | INT8: %.3f MB | INT4: %.3f MB\n", 
           per_room_fp16, per_room_i8, per_room_i4);
    printf("Rooms in 6GB budget:\n");
    printf("  FP16: %d | INT8: %d | INT4: %d\n",
           (int)(gpu_budget_gb * 1024 / per_room_fp16),
           (int)(gpu_budget_gb * 1024 / per_room_i8),
           (int)(gpu_budget_gb * 1024 / per_room_i4));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_weights_fp16));
    CUDA_CHECK(cudaFree(d_input_fp16));
    CUDA_CHECK(cudaFree(d_weights_i8));
    CUDA_CHECK(cudaFree(d_input_i8));
    CUDA_CHECK(cudaFree(d_weights_i4));
    CUDA_CHECK(cudaFree(d_input_i4));
    CUDA_CHECK(cudaFree(d_output));
    
    free(h_weights_fp16);
    free(h_input_fp16);
    free(h_weights_i8);
    free(h_input_i8);
    free(h_weights_i4);
    free(h_input_i4);
    
    return 0;
}
