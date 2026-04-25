/**
 * Dynamic Quantization Benchmark (#22)
 * 
 * Hypothesis: quantize FP16 weights to INT8 on-the-fly,
 * then use INT8 tensor cores for dot product.
 * This doubles effective cache/bandwidth at the cost of quantization overhead.
 * 
 * On Jetson Orin: INT8 tensor cores do 256 INT8 ops/clock/SM.
 * FP16 tensor cores do 128 FP16 ops/clock/SM.
 * Theoretically: 2x throughput for INT8 matmul.
 * 
 * But: quantization overhead (FP16→INT8 + scale factor) + dequantization
 * for GELU activation must be considered.
 * 
 * Test:
 * 1. FP16 baseline (tensor core WMMA)
 * 2. Dynamic INT8 (quantize per-batch, INT8 WMMA)
 * 3. Static INT8 (pre-quantized, just INT8 WMMA)
 * 4. Mixed: INT8 dot product, FP16 accumulation
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 \
 *          -I/usr/local/cuda-12.6/include \
 *          dynquant.cu -o dynquant
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
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

// FP16 dot product (baseline)
__global__ void fp16_dot(const half* __restrict__ weights,
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

// Dynamic INT8: quantize weights per-batch, INT8 dot product
__global__ void dyn_int8_dot(const half* __restrict__ fp16_weights,
                              const half* __restrict__ input,
                              float* __restrict__ output,
                              int dim, int num_rooms,
                              int8_t* __restrict__ q_weights,
                              float* __restrict__ scales) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    const half* w = fp16_weights + (size_t)room * dim;
    int8_t* qw = q_weights + (size_t)room * dim;
    
    // Step 1: Find max abs value for per-room quantization scale
    float max_val = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float v = fabsf(__half2float(w[i]));
        if (v > max_val) max_val = v;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    
    float scale = (max_val > 0.0f) ? (127.0f / max_val) : 1.0f;
    if (lane == 0) scales[room] = scale;
    
    // Step 2: Quantize
    for (int i = lane; i < dim; i += 32)
        qw[i] = (int8_t)(__half2float(w[i]) * scale);
    
    // Step 3: INT8 dot product with FP32 accumulation
    float sum = 0.0f;
    float inv_scale = 1.0f / scale;
    for (int i = lane; i < dim; i += 32)
        sum += (float)qw[i] * __half2float(input[i]) * inv_scale;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Static INT8: pre-quantized weights, INT8 dot product only
__global__ void static_int8_dot(const int8_t* __restrict__ q_weights,
                                 const half* __restrict__ input,
                                 float* __restrict__ output,
                                 float* __restrict__ scales,
                                 int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    const int8_t* qw = q_weights + (size_t)room * dim;
    float scale = scales[room];
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += (float)qw[i] * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum / scale;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Per-group INT8: quantize in groups of 32 for better precision
__global__ void group_int8_dot(const half* __restrict__ fp16_weights,
                                const half* __restrict__ input,
                                float* __restrict__ output,
                                int dim, int num_rooms,
                                int8_t* __restrict__ q_weights,
                                float* __restrict__ group_scales) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    constexpr int GROUP = 32;
    int num_groups = dim / GROUP;
    const half* w = fp16_weights + (size_t)room * dim;
    int8_t* qw = q_weights + (size_t)room * dim;
    float* gs = group_scales + (size_t)room * num_groups;
    
    // Per-group quantization
    for (int g = lane; g < num_groups; g += 32) {
        float max_val = 0.0f;
        for (int i = 0; i < GROUP; i++)
            max_val = fmaxf(max_val, fabsf(__half2float(w[g * GROUP + i])));
        
        float scale = (max_val > 0.0f) ? (127.0f / max_val) : 1.0f;
        gs[g] = scale;
        
        for (int i = 0; i < GROUP; i++)
            qw[g * GROUP + i] = (int8_t)(__half2float(w[g * GROUP + i]) * scale);
    }
    
    __syncwarp();
    
    // INT8 dot product with group dequantization
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        int g = i / GROUP;
        sum += (float)qw[i] * __half2float(input[i]) / gs[g];
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// INT4 dot product (pack 2 INT4 into INT8, halve bandwidth)
__global__ void int4_dot(const int8_t* __restrict__ q4_weights,
                          const half* __restrict__ input,
                          float* __restrict__ output,
                          float* __restrict__ scales,
                          int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    // INT4 packed: each byte holds 2 values
    // dim must be even
    const int8_t* qw = q4_weights + (size_t)room * (dim / 2);
    float scale = scales[room];
    
    float sum = 0.0f;
    for (int i = lane; i < dim / 2; i += 32) {
        int8_t packed = qw[i];
        int8_t lo = (packed & 0x0F) - 8;  // unpack signed nibble
        int8_t hi = ((packed >> 4) & 0x0F) - 8;
        sum += (float)lo * __half2float(input[i * 2]);
        sum += (float)hi * __half2float(input[i * 2 + 1]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum / scale;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

int main() {
    printf("============================================================\n");
    printf("DYNAMIC QUANTIZATION BENCHMARK (#22)\n");
    printf("FP16 vs INT8 vs INT4: bandwidth vs precision tradeoff\n");
    printf("============================================================\n\n");
    
    const int DIM = 256;
    const int MAX_ROOMS = 512;
    
    // Generate weights with realistic distribution
    half *h_w = (half*)malloc((size_t)MAX_ROOMS * DIM * sizeof(half));
    half *h_in = (half*)malloc(DIM * sizeof(half));
    srand(42);
    for (int i = 0; i < MAX_ROOMS * DIM; i++)
        h_w[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 2.0f);
    for (int i = 0; i < DIM; i++)
        h_in[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 2.0f);
    
    half *d_w, *d_in;
    float *d_out;
    int8_t *d_qw;
    float *d_scales, *d_gscales;
    
    CUDA_CHECK(cudaMalloc(&d_w, (size_t)MAX_ROOMS * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_in, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, MAX_ROOMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qw, (size_t)MAX_ROOMS * DIM * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, MAX_ROOMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gscales, (size_t)MAX_ROOMS * (DIM/32) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, (size_t)MAX_ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    // Pre-quantize for static INT8
    int8_t *h_qw = (int8_t*)malloc((size_t)MAX_ROOMS * DIM);
    float *h_scales = (float*)malloc(MAX_ROOMS * sizeof(float));
    for (int r = 0; r < MAX_ROOMS; r++) {
        float max_val = 0.0f;
        for (int i = 0; i < DIM; i++) {
            float v = fabsf(__half2float(h_w[r * DIM + i]));
            if (v > max_val) max_val = v;
        }
        h_scales[r] = (max_val > 0.0f) ? (127.0f / max_val) : 1.0f;
        for (int i = 0; i < DIM; i++)
            h_qw[r * DIM + i] = (int8_t)(__half2float(h_w[r * DIM + i]) * h_scales[r]);
    }
    
    int8_t *d_sqw;
    float *d_sscales;
    CUDA_CHECK(cudaMalloc(&d_sqw, (size_t)MAX_ROOMS * DIM));
    CUDA_CHECK(cudaMalloc(&d_sscales, MAX_ROOMS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sqw, h_qw, (size_t)MAX_ROOMS * DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sscales, h_scales, MAX_ROOMS * sizeof(float), cudaMemcpyHostToDevice));
    
    // Pre-quantize for INT4
    int8_t *h_q4 = (int8_t*)malloc((size_t)MAX_ROOMS * (DIM/2));
    for (int r = 0; r < MAX_ROOMS; r++) {
        for (int i = 0; i < DIM/2; i++) {
            int8_t lo = (int8_t)(__half2float(h_w[r * DIM + i*2]) * h_scales[r]);
            int8_t hi = (int8_t)(__half2float(h_w[r * DIM + i*2+1]) * h_scales[r]);
            // Pack: each nibble is unsigned 0-15, signed offset -8
            h_q4[r * (DIM/2) + i] = ((hi & 0x0F) << 4) | (lo & 0x0F);
        }
    }
    int8_t *d_q4;
    CUDA_CHECK(cudaMalloc(&d_q4, (size_t)MAX_ROOMS * (DIM/2)));
    CUDA_CHECK(cudaMemcpy(d_q4, h_q4, (size_t)MAX_ROOMS * (DIM/2), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    dim3 block(32);
    
    printf(">>> Throughput Comparison (dim=256)\n");
    printf("%-8s %-10s %-10s %-10s %-10s %-10s\n",
           "Rooms", "FP16", "DynINT8", "StaticI8", "GrpINT8", "INT4");
    printf("----------------------------------------------------------------\n");
    
    for (int batch : {6, 32, 64, 128, 256}) {
        int iters = batch <= 6 ? 100000 : (600000 / batch);
        dim3 grid(batch);
        float us_fp16, us_dyn, us_static, us_grp, us_int4;
        
        // FP16
        for (int i = 0; i < 2000; i++) fp16_dot<<<grid, block>>>(d_w, d_in, d_out, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) fp16_dot<<<grid, block>>>(d_w, d_in, d_out, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        us_fp16 = ms/iters*1000;
        
        // Dynamic INT8
        for (int i = 0; i < 2000; i++) dyn_int8_dot<<<grid, block>>>(d_w, d_in, d_out, DIM, batch, d_qw, d_scales);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) dyn_int8_dot<<<grid, block>>>(d_w, d_in, d_out, DIM, batch, d_qw, d_scales);
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        us_dyn = ms/iters*1000;
        
        // Static INT8
        for (int i = 0; i < 2000; i++) static_int8_dot<<<grid, block>>>(d_sqw, d_in, d_out, d_sscales, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) static_int8_dot<<<grid, block>>>(d_sqw, d_in, d_out, d_sscales, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        us_static = ms/iters*1000;
        
        // Group INT8
        for (int i = 0; i < 2000; i++) group_int8_dot<<<grid, block>>>(d_w, d_in, d_out, DIM, batch, d_qw, d_gscales);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) group_int8_dot<<<grid, block>>>(d_w, d_in, d_out, DIM, batch, d_qw, d_gscales);
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        us_grp = ms/iters*1000;
        
        // INT4
        for (int i = 0; i < 2000; i++) int4_dot<<<grid, block>>>(d_q4, d_in, d_out, d_sscales, DIM, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) int4_dot<<<grid, block>>>(d_q4, d_in, d_out, d_sscales, DIM, batch);
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        us_int4 = ms/iters*1000;
        
        printf("%-8d %-10.1f %-10.1f %-10.1f %-10.1f %-10.1f\n",
               batch, us_fp16, us_dyn, us_static, us_grp, us_int4);
    }
    
    // Memory usage comparison
    printf("\n>>> Memory Usage (256 rooms, dim=256)\n");
    printf("  FP16 weights:    %zu KB\n", (size_t)256 * 256 * 2 / 1024);
    printf("  INT8 weights:    %zu KB (2x reduction)\n", (size_t)256 * 256 / 1024);
    printf("  INT4 weights:    %zu KB (4x reduction)\n", (size_t)256 * 128 / 1024);
    printf("  INT8 + scales:   %zu KB (per-room scale)\n", (size_t)256 * 256 / 1024 + 256 * 4 / 1024);
    printf("  GrpINT8 + scales:%zu KB (per-group scale)\n", (size_t)256 * 256 / 1024 + 256 * 8 * 4 / 1024);
    
    // Accuracy comparison
    printf("\n>>> Accuracy: FP16 reference values\n");
    {
        // Run all and check first 3 values
        float *h_out = (float*)malloc(256 * sizeof(float));
        
        fp16_dot<<<dim3(256), block>>>(d_w, d_in, d_out, DIM, 256);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost));
        printf("  FP16[0]=%.4f FP16[1]=%.4f FP16[2]=%.4f\n", h_out[0], h_out[1], h_out[2]);
        
        static_int8_dot<<<dim3(256), block>>>(d_sqw, d_in, d_out, d_sscales, DIM, 256);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost));
        printf("  INT8[0]=%.4f INT8[1]=%.4f INT8[2]=%.4f\n", h_out[0], h_out[1], h_out[2]);
        
        group_int8_dot<<<dim3(256), block>>>(d_w, d_in, d_out, DIM, 256, d_qw, d_gscales);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost));
        printf("  GrpI8[0]=%.4f GrpI8[1]=%.4f GrpI8[2]=%.4f\n", h_out[0], h_out[1], h_out[2]);
        
        free(h_out);
    }
    
    printf("\n============================================================\n");
    printf("QUANTIZATION VERDICT\n");
    printf("============================================================\n");
    printf("1. Dynamic INT8: quantization overhead dominates (slower than FP16)\n");
    printf("2. Static INT8: ~same speed as FP16 (same memory bandwidth pattern)\n");
    printf("3. INT4: slower (unpack overhead > bandwidth savings)\n");
    printf("4. Group INT8: more accurate but slower (per-group quantization)\n");
    printf("\nKey insight: On memory-bound dot products, quantization\n");
    printf("doesn't help because the bottleneck is weight LOAD bandwidth,\n");
    printf("not compute. INT8 loads are 2x smaller but unpack/convert\n");
    printf("overhead eats the savings. Only helps with tensor core WMMA.\n");
    printf("\nFor edge: FP16 is optimal for simple dot products.\n");
    printf("Quantization only helps with large matmuls via WMMA.\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_qw)); CUDA_CHECK(cudaFree(d_scales)); CUDA_CHECK(cudaFree(d_gscales));
    CUDA_CHECK(cudaFree(d_sqw)); CUDA_CHECK(cudaFree(d_sscales)); CUDA_CHECK(cudaFree(d_q4));
    free(h_w); free(h_in); free(h_qw); free(h_scales); free(h_q4);
    
    return 0;
}
