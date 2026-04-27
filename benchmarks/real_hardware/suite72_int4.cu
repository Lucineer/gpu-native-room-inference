/**
 * Suite #72: INT4 Quantization — 2-bit Packed Weights
 * 
 * CUTTING EDGE: INT4 is the frontier of inference quantization (2025-2026).
 * GPT-4, Claude, Llama all serve with INT4 weights. NVIDIA's TensorRT-LLM 
 * defaults to INT4 for LLM inference.
 * 
 * INT4 packs two 4-bit values into one byte:
 * - 2× less memory than INT8
 * - 2× better bandwidth utilization (Jetson is bandwidth-bound!)
 * - 4× less than FP32, 2× less than FP16
 * 
 * Trade-off: INT4 has wider quantization noise (~6-8% avg error vs 4% for INT8).
 * For ranking workloads where exact values don't matter, this is acceptable.
 * 
 * Packing scheme: byte = (high_nibble << 4) | low_nibble
 * Each nibble holds a value in [-8, 7] (signed 4-bit).
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#define DIM 256
#define ITERS 10000
#define WARMUP 500

// INT8 baseline kernel (same as production)
__global__ void __launch_bounds__(256, 8) int8_infer(
    const signed char* __restrict__ weights,
    const signed char* __restrict__ input,
    const float* __restrict__ scales,
    float iscale,
    float* __restrict__ output,
    int dim,
    int num_rooms
) {
    int gi = blockIdx.x * 8 + threadIdx.x / 32;
    if (gi >= num_rooms) return;
    int lane = threadIdx.x % 32;
    
    int sum = 0;
    #pragma unroll 8
    for (int i = lane; i < dim; i += 32)
        sum += (int)weights[(size_t)gi * dim + i] * (int)input[i];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) output[gi] = (float)sum * scales[gi] * iscale;
}

// INT4 kernel — unpack 2 values per byte on-the-fly
__global__ void __launch_bounds__(256, 8) int4_infer(
    const unsigned char* __restrict__ weights,  // [num_rooms * dim/2] packed INT4
    const signed char* __restrict__ input,       // [dim] INT8 input
    const float* __restrict__ scales,            // [num_rooms] weight scales
    float iscale,
    float* __restrict__ output,
    int dim,
    int num_rooms
) {
    int gi = blockIdx.x * 8 + threadIdx.x / 32;
    if (gi >= num_rooms) return;
    int lane = threadIdx.x % 32;
    
    int sum = 0;
    int packed_dim = dim / 2;
    
    #pragma unroll 4
    for (int i = lane; i < packed_dim; i += 32) {
        unsigned char packed = weights[(size_t)gi * packed_dim + i];
        // Unpack: high nibble = signed 4-bit, low nibble = signed 4-bit
        // Sign extend: if bit 3 is set, value is negative
        int hi = (int)(packed >> 4);
        int lo = (int)(packed & 0x0F);
        // Sign extend from 4 bits
        if (hi >= 8) hi -= 16;
        if (lo >= 8) lo -= 16;
        int idx = i * 2;
        sum += hi * (int)input[idx] + lo * (int)input[idx + 1];
    }
    
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) output[gi] = (float)sum * scales[gi] * iscale;
}

// INT4 kernel with sign-magnitude encoding (alternative)
// Uses 1 sign bit + 3 magnitude bits = [-7, 7] range
__global__ void __launch_bounds__(256, 8) int4_signmag_infer(
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input,
    const float* __restrict__ scales,
    float iscale,
    float* __restrict__ output,
    int dim,
    int num_rooms
) {
    int gi = blockIdx.x * 8 + threadIdx.x / 32;
    if (gi >= num_rooms) return;
    int lane = threadIdx.x % 32;
    
    int sum = 0;
    int packed_dim = dim / 2;
    
    #pragma unroll 4
    for (int i = lane; i < packed_dim; i += 32) {
        unsigned char packed = weights[(size_t)gi * packed_dim + i];
        // Sign-magnitude decode: high nibble
        int hi_mag = (int)((packed >> 4) & 0x07);  // 3 magnitude bits
        int hi_sign = (packed >> 7) & 1;            // 1 sign bit
        int hi = hi_sign ? -hi_mag : hi_mag;
        
        // Low nibble
        int lo_mag = (int)(packed & 0x07);
        int lo_sign = (packed >> 3) & 1;
        int lo = lo_sign ? -lo_mag : lo_mag;
        
        int idx = i * 2;
        sum += hi * (int)input[idx] + lo * (int)input[idx + 1];
    }
    
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) output[gi] = (float)sum * scales[gi] * iscale;
}

// Host-side INT4 quantization (two's complement)
static void quantize_i4_twc(const float* data, unsigned char* out, int len, float* out_scale) {
    float abs_max = 0;
    for (int i = 0; i < len; i++) {
        float v = data[i] < 0 ? -data[i] : data[i];
        if (v > abs_max) abs_max = v;
    }
    float scale = abs_max > 0 ? abs_max / 8.0f : 1.0f;  // INT4 max = 7 or -8
    float inv = abs_max > 0 ? 8.0f / abs_max : 0;
    
    for (int i = 0; i < len; i += 2) {
        int v0 = (int)(data[i] * inv);
        int v1 = (int)(data[i + 1] * inv);
        if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
        if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8;
        // Pack: high nibble = v0, low nibble = v1
        unsigned char hi = (unsigned char)(v0 & 0x0F);
        unsigned char lo = (unsigned char)(v1 & 0x0F);
        out[i / 2] = (hi << 4) | lo;
    }
    if (out_scale) *out_scale = scale;
}

// Host-side INT4 quantization (sign-magnitude)
static void quantize_i4_signmag(const float* data, unsigned char* out, int len, float* out_scale) {
    float abs_max = 0;
    for (int i = 0; i < len; i++) {
        float v = data[i] < 0 ? -data[i] : data[i];
        if (v > abs_max) abs_max = v;
    }
    float scale = abs_max > 0 ? abs_max / 7.0f : 1.0f;  // sign-mag max = 7
    float inv = abs_max > 0 ? 7.0f / abs_max : 0;
    
    for (int i = 0; i < len; i += 2) {
        // High nibble: sign(1) + magnitude(3)
        float a0 = data[i] < 0 ? -data[i] : data[i];
        float a1 = data[i+1] < 0 ? -data[i+1] : data[i+1];
        int m0 = (int)(a0 * inv); if (m0 > 7) m0 = 7;
        int m1 = (int)(a1 * inv); if (m1 > 7) m1 = 7;
        int s0 = data[i] < 0 ? 1 : 0;
        int s1 = data[i+1] < 0 ? 1 : 0;
        unsigned char hi = (unsigned char)((s0 << 3) | m0);
        unsigned char lo = (unsigned char)((s1 << 3) | m1);
        out[i / 2] = (hi << 4) | lo;
    }
    if (out_scale) *out_scale = scale;
}

int main() {
    printf("=== Suite #72: INT4 Quantization — 2-bit Packed Weights ===\n\n");
    
    int R = 4096;
    
    // Generate FP32 data
    std::vector<float> hw((size_t)R * DIM), hi(DIM);
    srand(42);
    for (size_t i = 0; i < (size_t)R * DIM; i++)
        hw[i] = 0.5f * (2.0f * (float)rand() / RAND_MAX - 1.0f);
    for (int i = 0; i < DIM; i++)
        hi[i] = cosf((float)i / DIM * 6.2832f);
    
    // INT8 quantization (baseline)
    std::vector<signed char> wq8((size_t)R * DIM), iq8(DIM);
    std::vector<float> ws8(R);
    float iscale8;
    {
        float imax = 0;
        for (int i = 0; i < DIM; i++) { float v = fabsf(hi[i]); if (v > imax) imax = v; }
        iscale8 = imax / 127.0f;
        float iinv = imax > 0 ? 127.0f / imax : 0;
        for (int i = 0; i < DIM; i++) {
            float v = hi[i] * iinv;
            iq8[i] = (signed char)(v > 127 ? 127 : (v < -127 ? -127 : (int)v));
        }
        for (int r = 0; r < R; r++) {
            float wmax = 0;
            for (int d = 0; d < DIM; d++) {
                float v = fabsf(hw[r * DIM + d]);
                if (v > wmax) wmax = v;
            }
            ws8[r] = wmax / 127.0f;
            float inv = wmax > 0 ? 127.0f / wmax : 0;
            for (int d = 0; d < DIM; d++) {
                float v = hw[r * DIM + d] * inv;
                wq8[r * DIM + d] = (signed char)(v > 127 ? 127 : (v < -127 ? -127 : (int)v));
            }
        }
    }
    
    // INT4 quantization (two's complement)
    std::vector<unsigned char> wq4((size_t)R * DIM / 2);
    std::vector<float> ws4(R);
    for (int r = 0; r < R; r++) {
        quantize_i4_twc(hw.data() + r * DIM, wq4.data() + r * DIM / 2, DIM, &ws4[r]);
    }
    
    // GPU memory
    signed char *dw8, *di8;
    float *ds8, *dout;
    cudaMalloc(&dw8, (size_t)R * DIM);
    cudaMalloc(&di8, DIM);
    cudaMalloc(&ds8, R * sizeof(float));
    cudaMalloc(&dout, R * sizeof(float));
    cudaMemcpy(dw8, wq8.data(), (size_t)R * DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(di8, iq8.data(), DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(ds8, ws8.data(), R * sizeof(float), cudaMemcpyHostToDevice);
    
    unsigned char *dw4;
    float *ds4;
    cudaMalloc(&dw4, (size_t)R * DIM / 2);
    cudaMalloc(&ds4, R * sizeof(float));
    cudaMemcpy(dw4, wq4.data(), (size_t)R * DIM / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ds4, ws4.data(), R * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    
    // ===== Memory footprint comparison =====
    printf("--- Memory Footprint (4096 rooms, dim=256) ---\n");
    printf("  FP32: %d bytes (%.1f KB)\n", R * DIM * 4, (float)R * DIM * 4 / 1024);
    printf("  FP16: %d bytes (%.1f KB)\n", R * DIM * 2, (float)R * DIM * 2 / 1024);
    printf("  INT8: %d bytes (%.1f KB)\n", R * DIM, (float)R * DIM / 1024);
    printf("  INT4: %d bytes (%.1f KB)\n", R * DIM / 2, (float)R * DIM / 2 / 1024);
    printf("  INT4 compression: %.1fx vs FP32, %.1fx vs INT8\n",
           (float)(R * DIM * 4) / (R * DIM / 2), (float)(R * DIM) / (R * DIM / 2));
    
    // ===== Correctness check =====
    printf("\n--- Correctness (first 5 rooms) ---\n");
    {
        // Run INT8
        int n = 5;
        dim3 g(1);
        int8_infer<<<g, block>>>(dw8, di8, ds8, iscale8, dout, DIM, n);
        cudaDeviceSynchronize();
        float results8[5];
        cudaMemcpy(results8, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Run INT4
        int4_infer<<<g, block>>>(dw4, di8, ds4, iscale8, dout, DIM, n);
        cudaDeviceSynchronize();
        float results4[5];
        cudaMemcpy(results4, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Reference
        for (int t = 0; t < n; t++) {
            float ref = 0;
            for (int d = 0; d < DIM; d++) ref += hw[t * DIM + d] * hi[d];
            float err8 = fabsf(results8[t] - ref) / (fabsf(ref) + 1e-10f);
            float err4 = fabsf(results4[t] - ref) / (fabsf(ref) + 1e-10f);
            printf("  Room %d: ref=%.4f  INT8=%.4f (%.2f%%)  INT4=%.4f (%.2f%%)\n",
                   t, ref, results8[t], err8 * 100, results4[t], err4 * 100);
        }
    }
    
    // ===== Performance sweep =====
    printf("\n--- Throughput Sweep ---\n");
    printf("%8s  %10s  %12s  %10s  %12s  %8s\n",
           "Rooms", "INT8 us", "INT8 M qps", "INT4 us", "INT4 M qps", "Speedup");
    printf("%8s  %10s  %12s  %10s  %12s  %8s\n",
           "------", "-------", "----------", "-------", "----------", "-------");
    
    int batches[] = {64, 256, 1024, 4096};
    int nb = sizeof(batches) / sizeof(batches[0]);
    
    for (int b = 0; b < nb; b++) {
        int n = batches[b];
        dim3 g((n + 7) / 8);
        
        // INT8
        for (int w = 0; w < WARMUP; w++)
            int8_infer<<<g, block>>>(dw8, di8, ds8, iscale8, dout, DIM, n);
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; i++)
            int8_infer<<<g, block>>>(dw8, di8, ds8, iscale8, dout, DIM, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        float i8_us = ms / ITERS * 1000;
        cudaEventDestroy(s); cudaEventDestroy(e);
        
        // INT4 two's complement
        for (int w = 0; w < WARMUP; w++)
            int4_infer<<<g, block>>>(dw4, di8, ds4, iscale8, dout, DIM, n);
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; i++)
            int4_infer<<<g, block>>>(dw4, di8, ds4, iscale8, dout, DIM, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        float i4_us = ms / ITERS * 1000;
        cudaEventDestroy(s); cudaEventDestroy(e);
        
        printf("%8d  %8.2f    %8.1f    %8.2f    %8.1f    %6.2fx\n",
               n, i8_us, (float)n / i8_us * 1000, i4_us, (float)n / i4_us * 1000, i8_us / i4_us);
    }
    
    // ===== INT4 sign-magnitude comparison =====
    printf("\n--- INT4 Encoding: Two's Complement vs Sign-Magnitude ---\n");
    {
        int n = 4096;
        dim3 g((n + 7) / 8);
        
        // Sign-mag quantize
        std::vector<unsigned char> wq4sm((size_t)R * DIM / 2);
        std::vector<float> ws4sm(R);
        for (int r = 0; r < R; r++)
            quantize_i4_signmag(hw.data() + r * DIM, wq4sm.data() + r * DIM / 2, DIM, &ws4sm[r]);
        
        unsigned char *dw4sm;
        float *ds4sm;
        cudaMalloc(&dw4sm, (size_t)R * DIM / 2);
        cudaMalloc(&ds4sm, R * sizeof(float));
        cudaMemcpy(dw4sm, wq4sm.data(), (size_t)R * DIM / 2, cudaMemcpyHostToDevice);
        cudaMemcpy(ds4sm, ws4sm.data(), R * sizeof(float), cudaMemcpyHostToDevice);
        
        for (int w = 0; w < WARMUP; w++)
            int4_signmag_infer<<<g, block>>>(dw4sm, di8, ds4sm, iscale8, dout, DIM, n);
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; i++)
            int4_signmag_infer<<<g, block>>>(dw4sm, di8, ds4sm, iscale8, dout, DIM, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        float sm_us = ms / ITERS * 1000;
        cudaEventDestroy(s); cudaEventDestroy(e);
        
        // TWC already measured above at 4096
        printf("  Two's complement: (from above)\n");
        printf("  Sign-magnitude:  %.2f us, %.1f M qps\n", sm_us, (float)n / sm_us * 1000);
        
        cudaFree(dw4sm); cudaFree(ds4sm);
    }
    
    cudaFree(dw8); cudaFree(di8); cudaFree(ds8); cudaFree(dout);
    cudaFree(dw4); cudaFree(ds4);
    
    printf("\n=== Suite #72 Complete ===\n");
    return 0;
}
