/**
 * Suite #49: Quantized Multi-Layer Fusion
 * 
 * Suite #22: FP16 wins for single-layer (quantization overhead > savings)
 * Suite #47: OI=1.0 always memory-bound
 * Suite #48: Fusion increases OI to 2-16
 * 
 * Hypothesis: At high OI (many fused layers), quantization reduces memory traffic
 * enough to offset dequant overhead. At OI=16 (8 layers), weight reads = 4MB.
 * INT8 halves weight reads → 2MB. Dequant overhead is fixed per element.
 * 
 * Test: FP16 vs INT8 vs INT4 at 1, 2, 4, 8 fused layers
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 quant_fusion.cu -o quant_fusion
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#define DIM 256
#define WARMUP 500
#define ITERS 5000

// FP16 baseline: N-layer fused
__global__ void infer_fp16_nlayer(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim, int num_layers)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    float val = 0.0f;
    long long layer_stride = (long long)num_rooms * dim;
    
    for (int layer = 0; layer < num_layers; layer++) {
        const half* w = weights + layer * layer_stride;
        float sum = 0.0f;
        for (int i = lane; i < dim; i += 32) {
            float in_val = (layer == 0) ? __half2float(input[i]) : val;
            sum += __half2float(w[room * dim + i]) * in_val;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum = __shfl_sync(0xffffffff, sum, 0);
        
        if (layer < num_layers - 1) {
            float c = 0.7978845608f, a = 0.044715f;
            val = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
        } else {
            val = sum;
        }
    }
    if (lane == 0) output[room] = val;
}

// INT8 fused: weights stored as int8, dequantized inline
// Format: [scale_f16][weights_i8...] per room per layer
// Each room: 2 bytes scale + dim bytes weights
__global__ void infer_int8_nlayer(
    const int8_t* __restrict__ qweights,  // [num_layers][num_rooms][1 + dim]
    const float* __restrict__ scales,      // [num_layers][num_rooms] — pre-extracted scales
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim, int num_layers)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    float val = 0.0f;
    
    for (int layer = 0; layer < num_layers; layer++) {
        const int8_t* qw = qweights + (long long)layer * num_rooms * dim + room * dim;
        float scale = scales[(long long)layer * num_rooms + room];
        float sum = 0.0f;
        
        for (int i = lane; i < dim; i += 32) {
            float in_val = (layer == 0) ? __half2float(input[i]) : val;
            sum += (float)qw[i] * scale * in_val;
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum = __shfl_sync(0xffffffff, sum, 0);
        
        if (layer < num_layers - 1) {
            float c = 0.7978845608f, a = 0.044715f;
            val = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
        } else {
            val = sum;
        }
    }
    if (lane == 0) output[room] = val;
}

// INT4 fused: two weights per byte
__global__ void infer_int4_nlayer(
    const int8_t* __restrict__ qweights,  // packed: 2 weights per byte
    const float* __restrict__ scales,
    const float* __restrict__ zeros,       // zero points
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim, int num_layers)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    float val = 0.0f;
    
    for (int layer = 0; layer < num_layers; layer++) {
        const int8_t* qw = qweights + (long long)layer * num_rooms * (dim / 2) + room * (dim / 2);
        float scale = scales[(long long)layer * num_rooms + room];
        float zero = zeros[(long long)layer * num_rooms + room];
        float sum = 0.0f;
        
        for (int i = lane; i < dim; i += 32) {
            int8_t packed = qw[i / 2];
            float w_val;
            if (i % 2 == 0)
                w_val = (float)((packed >> 4) & 0xF) * scale + zero;
            else
                w_val = (float)(packed & 0xF) * scale + zero;
            
            float in_val = (layer == 0) ? __half2float(input[i]) : val;
            sum += w_val * in_val;
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum = __shfl_sync(0xffffffff, sum, 0);
        
        if (layer < num_layers - 1) {
            float c = 0.7978845608f, a = 0.044715f;
            val = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
        } else {
            val = sum;
        }
    }
    if (lane == 0) output[room] = val;
}

int main() {
    printf("=== Suite #49: Quantized Multi-Layer Fusion ===\n\n");
    
    int rooms = 1024;
    int max_layers = 8;
    int layer_counts[] = {1, 2, 4, 6, 8};
    int num_lc = sizeof(layer_counts) / sizeof(layer_counts[0]);
    
    // Allocate
    half *d_fp16_weights, *d_input;
    int8_t *d_int8_weights, *d_int4_weights;
    float *d_scales, *d_zeros, *d_out;
    
    size_t fp16_bytes = (size_t)max_layers * rooms * DIM * sizeof(half);
    size_t int8_bytes = (size_t)max_layers * rooms * DIM * sizeof(int8_t);
    size_t int4_bytes = (size_t)max_layers * rooms * (DIM / 2) * sizeof(int8_t);
    size_t scale_bytes = (size_t)max_layers * rooms * sizeof(float);
    
    cudaMalloc(&d_fp16_weights, fp16_bytes);
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_int8_weights, int8_bytes);
    cudaMalloc(&d_int4_weights, int4_bytes);
    cudaMalloc(&d_scales, scale_bytes);
    cudaMalloc(&d_zeros, scale_bytes);
    cudaMalloc(&d_out, rooms * sizeof(float));
    
    // Generate data
    std::vector<half> hw_fp16(fp16_bytes / sizeof(half));
    std::vector<int8_t> hw_int8(int8_bytes);
    std::vector<int8_t> hw_int4(int4_bytes);
    std::vector<float> h_scales(scale_bytes);
    std::vector<float> h_zeros(scale_bytes);
    std::vector<half> hi(DIM);
    
    for (size_t i = 0; i < fp16_bytes / sizeof(half); i++)
        hw_fp16[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    
    // Quantize to INT8: symmetric, per-room-per-layer scale
    for (int l = 0; l < max_layers; l++) {
        for (int r = 0; r < rooms; r++) {
            float max_val = 0.0f;
            for (int d = 0; d < DIM; d++) {
                float v = fabsf(__half2float(hw_fp16[((size_t)l * rooms + r) * DIM + d]));
                if (v > max_val) max_val = v;
            }
            float scale = max_val / 127.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            h_scales[(size_t)l * rooms + r] = scale;
            h_zeros[(size_t)l * rooms + r] = 0.0f;
            
            for (int d = 0; d < DIM; d++) {
                float v = __half2float(hw_fp16[((size_t)l * rooms + r) * DIM + d]);
                hw_int8[((size_t)l * rooms + r) * DIM + d] = (int8_t)(roundf(v / scale));
            }
            
            // INT4: pack two weights per byte
            for (int d = 0; d < DIM / 2; d++) {
                float v0 = __half2float(hw_fp16[((size_t)l * rooms + r) * DIM + d * 2]);
                float v1 = __half2float(hw_fp16[((size_t)l * rooms + r) * DIM + d * 2 + 1]);
                int8_t q0 = (int8_t)std::max(-8, std::min(7, (int)roundf(v0 / scale)));
                int8_t q1 = (int8_t)std::max(-8, std::min(7, (int)roundf(v1 / scale)));
                hw_int4[((size_t)l * rooms + r) * (DIM / 2) + d] = (int8_t)((q0 << 4) | (q1 & 0xF));
            }
        }
    }
    
    cudaMemcpy(d_fp16_weights, hw_fp16.data(), fp16_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_int8_weights, hw_int8.data(), int8_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_int4_weights, hw_int4.data(), int4_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_zeros, h_zeros.data(), scale_bytes, cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    dim3 grid((rooms + 7) / 8);
    
    printf("--- FP16 vs INT8 vs INT4 Across Layer Counts (1024 rooms, dim=256) ---\n");
    printf("%-6s | %-10s | %-10s | %-6s | %-10s | %-6s | %-10s | %-6s\n",
           "Layers", "FP16(us)", "INT8(us)", "Ratio", "INT4(us)", "Ratio", "Mem Saved", "OI");
    printf("-------|------------|------------|--------|------------|--------|------------|------\n");
    
    for (int lc = 0; lc < num_lc; lc++) {
        int L = layer_counts[lc];
        float fp16_us, int8_us, int4_us;
        
        // FP16
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_fp16_nlayer<<<grid,256,0,stream>>>(d_fp16_weights,d_input,d_out,rooms,DIM,L);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_fp16_nlayer<<<grid,256,0,stream>>>(d_fp16_weights,d_input,d_out,rooms,DIM,L);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); fp16_us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // INT8
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_int8_nlayer<<<grid,256,0,stream>>>(d_int8_weights,d_scales,d_input,d_out,rooms,DIM,L);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_int8_nlayer<<<grid,256,0,stream>>>(d_int8_weights,d_scales,d_input,d_out,rooms,DIM,L);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); int8_us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // INT4
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_int4_nlayer<<<grid,256,0,stream>>>(d_int4_weights,d_scales,d_zeros,d_input,d_out,rooms,DIM,L);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_int4_nlayer<<<grid,256,0,stream>>>(d_int4_weights,d_scales,d_zeros,d_input,d_out,rooms,DIM,L);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); int4_us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        float fp16_mem = (float)L * rooms * DIM * 2;  // bytes
        float int8_mem = (float)L * rooms * DIM * 1 + L * rooms * 4;  // int8 weights + float scales
        float int4_mem = (float)L * rooms * (DIM / 2) * 1 + L * rooms * 8;  // int4 weights + float scales + zeros
        float mem_saved_int8 = (1.0f - int8_mem / fp16_mem) * 100.0f;
        float mem_saved_int4 = (1.0f - int4_mem / fp16_mem) * 100.0f;
        float oi = 2.0f * L;
        
        printf("%-6d | %8.2f   | %8.2f   | %.3fx  | %8.2f   | %.3fx  | %5.1f%%/%4.1f%% | %.0f\n",
               L, fp16_us, int8_us, fp16_us/int8_us, int4_us, fp16_us/int4_us,
               mem_saved_int8, mem_saved_int4, oi);
    }
    
    // ===== Correctness =====
    printf("\n--- Correctness (2 layers, first 4 rooms) ---\n");
    {
        int L = 2;
        infer_fp16_nlayer<<<grid, 256, 0, stream>>>(d_fp16_weights, d_input, d_out, rooms, DIM, L);
        std::vector<float> h_out(rooms);
        cudaMemcpy(h_out.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        infer_int8_nlayer<<<grid, 256, 0, stream>>>(d_int8_weights, d_scales, d_input, d_out, rooms, DIM, L);
        std::vector<float> h_out_i8(rooms);
        cudaMemcpy(h_out_i8.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        infer_int4_nlayer<<<grid, 256, 0, stream>>>(d_int4_weights, d_scales, d_zeros, d_input, d_out, rooms, DIM, L);
        std::vector<float> h_out_i4(rooms);
        cudaMemcpy(h_out_i4.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int r = 0; r < 4; r++) {
            float err8 = fabsf(h_out[r] - h_out_i8[r]) / (fabsf(h_out[r]) + 1e-10f) * 100.0f;
            float err4 = fabsf(h_out[r] - h_out_i4[r]) / (fabsf(h_out[r]) + 1e-10f) * 100.0f;
            printf("  Room %d: FP16=%.6f  INT8=%.6f (%.2f%% err)  INT4=%.6f (%.2f%% err)\n",
                   r, h_out[r], h_out_i8[r], err8, h_out_i4[r], err4);
        }
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_fp16_weights); cudaFree(d_input); cudaFree(d_int8_weights);
    cudaFree(d_int4_weights); cudaFree(d_scales); cudaFree(d_zeros); cudaFree(d_out);
    
    printf("\n=== Suite #49 Complete ===\n");
    return 0;
}
