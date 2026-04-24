/**
 * Shared Memory Optimization for Warp Room Inference
 * Jetson Orin Nano | sm_87
 * 
 * Current kernel loads input[0..255] from global memory 16x (once per neuron)
 * Optimization: load into shared memory once, all 16 neurons read from shared
 * Expected: ~10-15% speedup from reduced global memory traffic
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>


#define NEURONS 16
#define INPUT_SIZE 256

// v0: Current production kernel (global memory, no shared)
__global__ void warp_global(const half* w, const half* in, float* out, int nr) {
    int rid = blockIdx.x, lane = threadIdx.x;
    if (rid >= nr) return;
    const half* ww = w + rid * NEURONS * INPUT_SIZE;
    for (int n = 0; n < NEURONS; n++) {
        float s = 0.0f;
        for (int k = lane; k < INPUT_SIZE; k += 32)
            s += __half2float(__hmul(ww[n * INPUT_SIZE + k], in[k]));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) {
            float x = s;
            out[rid * NEURONS + n] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
    }
}

// v1: Shared memory input (load once, read 16x)
__global__ void warp_shared_in(const half* w, const half* in, float* out, int nr) {
    __shared__ half s_in[INPUT_SIZE];
    int rid = blockIdx.x, lane = threadIdx.x;
    if (rid >= nr) return;
    const half* ww = w + rid * NEURONS * INPUT_SIZE;
    
    // Each warp loads all 256 input elements (32 threads x 8 elements each)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = lane + i * 32;
        if (idx < INPUT_SIZE) s_in[idx] = in[idx];
    }
    
    // Also load weights into shared memory
    __shared__ half s_w[NEURONS * INPUT_SIZE];
    #pragma unroll
    for (int i = 0; i < NEURONS * 8; i++) {
        int idx = lane + i * 32;
        if (idx < NEURONS * INPUT_SIZE) s_w[idx] = ww[idx];
    }
    __syncwarp();
    
    for (int n = 0; n < NEURONS; n++) {
        float s = 0.0f;
        #pragma unroll
        for (int k = lane; k < INPUT_SIZE; k += 32)
            s += __half2float(__hmul(s_w[n * INPUT_SIZE + k], s_in[k]));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) {
            float x = s;
            out[rid * NEURONS + n] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
    }
}

// v2: Shared memory, vectorized loads (half2 = 2 elements per load)
__global__ void warp_shared_vec(const half* w, const half* in, float* out, int nr) {
    __shared__ half2 s_in[INPUT_SIZE / 2];
    __shared__ half2 s_w[NEURONS * INPUT_SIZE / 2];
    int rid = blockIdx.x, lane = threadIdx.x;
    if (rid >= nr) return;
    const half2* ww = reinterpret_cast<const half2*>(w + rid * NEURONS * INPUT_SIZE);
    const half2* ii = reinterpret_cast<const half2*>(in);
    
    // Load input as half2 (128 elements, 32 threads x 4)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane + i * 32;
        if (idx < INPUT_SIZE / 2) s_in[idx] = ii[idx];
    }
    // Load all weights as half2
    #pragma unroll
    for (int i = 0; i < NEURONS * 4; i++) {
        int idx = lane + i * 32;
        if (idx < NEURONS * INPUT_SIZE / 2) s_w[idx] = ww[idx];
    }
    __syncwarp();
    
    for (int n = 0; n < NEURONS; n++) {
        float s = 0.0f;
        #pragma unroll
        for (int k = lane; k < INPUT_SIZE / 2; k += 32) {
            half2 wv = s_w[n * (INPUT_SIZE / 2) + k];
            half2 iv = s_in[k];
            // Manual half2 multiply-add
            s += __half2float(__hmul(wv.x, iv.x)) + __half2float(__hmul(wv.y, iv.y));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) {
            float x = s;
            out[rid * NEURONS + n] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
    }
}

// v3: Full shared memory + register blocking (8 elements per thread in registers)
__global__ void warp_reg_block(const half* w, const half* in, float* out, int nr) {
    __shared__ half s_w[NEURONS * INPUT_SIZE];
    int rid = blockIdx.x, lane = threadIdx.x;
    if (rid >= nr) return;
    const half* ww = w + rid * NEURONS * INPUT_SIZE;
    
    // Load weights into shared
    #pragma unroll
    for (int i = 0; i < NEURONS * 8; i++) {
        int idx = lane + i * 32;
        if (idx < NEURONS * INPUT_SIZE) s_w[idx] = ww[idx];
    }
    
    // Each thread loads 8 input elements into registers
    half r_in[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) r_in[i] = in[lane + i * 32];
    __syncwarp();
    
    for (int n = 0; n < NEURONS; n++) {
        float s = 0.0f;
        const half* wn = s_w + n * INPUT_SIZE;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            half wv = wn[lane + i * 32];
            s += __half2float(__hmul(wv, r_in[i]));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) {
            float x = s;
            out[rid * NEURONS + n] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
    }
}

int main() {
    printf("=== Shared Memory Optimization Sweep ===\n\n");
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | sm_%d%d | shared/kB: %d\n\n", p.name, p.major, p.minor, p.sharedMemPerBlock / 1024);
    
    const int NR = 12, ITERS = 50000;
    size_t wb = NR * NEURONS * INPUT_SIZE * sizeof(half);
    half *hw = (half*)malloc(wb), *hi = (half*)malloc(INPUT_SIZE * sizeof(half));
    float *h0 = (float*)malloc(NR * NEURONS * sizeof(float));
    float *h1 = (float*)malloc(NR * NEURONS * sizeof(float));
    float *h2 = (float*)malloc(NR * NEURONS * sizeof(float));
    float *h3 = (float*)malloc(NR * NEURONS * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < NR * NEURONS * INPUT_SIZE; i++) hw[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < INPUT_SIZE; i++) hi[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *dw, *di; float *d0, *d1, *d2, *d3;
    cudaMalloc(&dw, wb); cudaMalloc(&di, INPUT_SIZE * sizeof(half));
    cudaMalloc(&d0, NR*NEURONS*sizeof(float)); cudaMalloc(&d1, NR*NEURONS*sizeof(float));
    cudaMalloc(&d2, NR*NEURONS*sizeof(float)); cudaMalloc(&d3, NR*NEURONS*sizeof(float));
    cudaMemcpy(dw, hw, wb, cudaMemcpyHostToDevice);
    cudaMemcpy(di, hi, INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < 200; i++) {
        warp_global<<<NR,32>>>(dw,di,d0,NR);
        warp_shared_in<<<NR,32>>>(dw,di,d1,NR);
        warp_shared_vec<<<NR,32>>>(dw,di,d2,NR);
        warp_reg_block<<<NR,32>>>(dw,di,d3,NR);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);
    float ms, flops = (float)NR * NEURONS * INPUT_SIZE * 2;
    float l0, l1, l2, l3;
    
    auto bench = [&](auto kern, int iters) -> float {
        cudaEventRecord(es);
        for (int i = 0; i < iters; i++) kern<<<NR,32>>>(dw,di,d0,NR);
        cudaEventRecord(ee); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, es, ee);
        return ms / iters;
    };
    
    l0 = bench(warp_global, ITERS);
    l1 = bench(warp_shared_in, ITERS);
    l2 = bench(warp_shared_vec, ITERS);
    l3 = bench(warp_reg_block, ITERS);
    
    // Correctness
    warp_global<<<NR,32>>>(dw,di,d0,NR);
    warp_shared_in<<<NR,32>>>(dw,di,d1,NR);
    warp_shared_vec<<<NR,32>>>(dw,di,d2,NR);
    warp_reg_block<<<NR,32>>>(dw,di,d3,NR);
    cudaDeviceSynchronize();
    cudaMemcpy(h0,d0,NR*NEURONS*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h1,d1,NR*NEURONS*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h2,d2,NR*NEURONS*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h3,d3,NR*NEURONS*sizeof(float),cudaMemcpyDeviceToHost);
    
    float maxd1=0, maxd2=0, maxd3=0;
    for (int i = 0; i < NR*NEURONS; i++) {
        maxd1 = fmaxf(maxd1, fabsf(h0[i]-h1[i]));
        maxd2 = fmaxf(maxd2, fabsf(h0[i]-h2[i]));
        maxd3 = fmaxf(maxd3, fabsf(h0[i]-h3[i]));
    }
    
    printf("12 rooms x 16 neurons x 256 weights, GELU, %d iters\n\n", ITERS);
    printf("%-35s %10s %10s %10s %10s\n", "Variant", "Latency", "QPS", "Speedup", "MaxDiff");
    printf("%-35s %10s %10s %10s %10s\n", "-----------------------------------", "----------", "----------", "----------", "----------");
    printf("%-35s %8.4f ms %8.0f/s %8s %10.6f\n", "v0: Global mem (baseline)", l0, 1000/l0, "-", 0.0f);
    printf("%-35s %8.4f ms %8.0f/s %8.2fx %10.6f\n", "v1: Shared input + shared weights", l1, 1000/l1, l0/l1, maxd1);
    printf("%-35s %8.4f ms %8.0f/s %8.2fx %10.6f\n", "v2: Shared + half2 vectorized", l2, 1000/l2, l0/l2, maxd2);
    printf("%-35s %8.4f ms %8.0f/s %8.2fx %10.6f\n", "v3: Shared weights + reg input", l3, 1000/l3, l0/l3, maxd3);
    
    float best = fminf(fminf(l1,l2),l3);
    printf("\nBest: %.4f ms (%.2fx over baseline)\n", best, l0/best);
    printf("GFLOPS: %.2f\n", flops/(best/1000)/1e9);
    
    cudaFree(dw);cudaFree(di);cudaFree(d0);cudaFree(d1);cudaFree(d2);cudaFree(d3);
    free(hw);free(hi);free(h0);free(h1);free(h2);free(h3);
    cudaEventDestroy(es);cudaEventDestroy(ee);
    return 0;
}
