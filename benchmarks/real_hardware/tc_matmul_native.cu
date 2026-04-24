#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda::wmma;

// Warp matmul baseline
__global__ void warp_matmul(const half* A, const half* B, float* C, int K) {
    int lane = threadIdx.x;
    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 16; n++) {
            float s = 0.0f;
            for (int k = lane; k < K; k += 32)
                s += __half2float(__hmul(A[m*K+k], B[k*16+n]));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                s += __shfl_down_sync(0xffffffff, s, o);
            if (lane == 0) C[m*16+n] = s;
        }
    }
}

// TC matmul using store_matrix_sync for full 16x16 extraction
__global__ void tc_matmul_native(const half* A, const half* B, float* C, int K) {
    __shared__ alignas(128) half sa[16*16], sb[16*16];
    __shared__ float sc[16*16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,row_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    int lane = threadIdx.x;
    
    fill_fragment(cf, 0.0f);
    
    // For each K-chunk of 16: load tiles, accumulate
    for (int kk = 0; kk < K; kk += 16) {
        // Load A tile: 16 rows x 16 cols
        for (int i = lane; i < 256; i += 32)
            sa[i] = A[(i / 16) * K + kk + (i % 16)];
        // Load B tile: 16 rows x 16 cols  
        for (int i = lane; i < 256; i += 32)
            sb[i] = B[(kk + i / 16) * 16 + (i % 16)];
        __syncwarp();
        
        load_matrix_sync(af, sa, 16);
        load_matrix_sync(bf, sb, 16);
        mma_sync(cf, af, bf, cf);
    }
    
    // Extract full 16x16 result using store_matrix_sync!
    store_matrix_sync(sc, cf, 16, mem_row_major);
    
    // Write to output
    for (int i = lane; i < 256; i += 32)
        C[i] = sc[i];
}

int main() {
    printf("=== NATIVE TC MATMUL (store_matrix_sync) ===\n\n");
    
    int Ks[] = {16, 32, 64, 128, 256};
    
    for (int ki = 0; ki < 5; ki++) {
        int K = Ks[ki];
        size_t sA = 16 * K * sizeof(half);
        size_t sB = K * 16 * sizeof(half);
        size_t sC = 256 * sizeof(float);
        int ITERS = 10000;
        
        half *hA = (half*)malloc(sA), *hB = (half*)malloc(sB);
        float *hCw = (float*)malloc(sC), *hCt = (float*)malloc(sC);
        srand(42);
        for (size_t i = 0; i < (size_t)16*K; i++) hA[i] = __float2half((float)rand()/RAND_MAX-0.5f);
        for (size_t i = 0; i < (size_t)K*16; i++) hB[i] = __float2half((float)rand()/RAND_MAX-0.5f);
        
        half *dA, *dB; float *dCw, *dCt;
        cudaMalloc(&dA, sA); cudaMalloc(&dB, sB);
        cudaMalloc(&dCw, sC); cudaMalloc(&dCt, sC);
        cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);
        
        for (int i = 0; i < 50; i++) {
            warp_matmul<<<1,32>>>(dA,dB,dCw,K);
            tc_matmul_native<<<1,32>>>(dA,dB,dCt,K);
        }
        cudaDeviceSynchronize();
        
        cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);
        float ms, lw, lt;
        
        cudaEventRecord(es);
        for (int i = 0; i < ITERS; i++) warp_matmul<<<1,32>>>(dA,dB,dCw,K);
        cudaEventRecord(ee); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, es, ee); lw = ms / ITERS;
        
        cudaEventRecord(es);
        for (int i = 0; i < ITERS; i++) tc_matmul_native<<<1,32>>>(dA,dB,dCt,K);
        cudaEventRecord(ee); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, es, ee); lt = ms / ITERS;
        
        // Correctness
        warp_matmul<<<1,32>>>(dA,dB,dCw,K);
        tc_matmul_native<<<1,32>>>(dA,dB,dCt,K);
        cudaDeviceSynchronize();
        cudaMemcpy(hCw, dCw, sC, cudaMemcpyDeviceToHost);
        cudaMemcpy(hCt, dCt, sC, cudaMemcpyDeviceToHost);
        float maxd = 0;
        for (int i = 0; i < 256; i++) maxd = fmaxf(maxd, fabsf(hCw[i]-hCt[i]));
        
        float flops = 2.0f * 16.0f * K * 16;
        printf("16x%-3d x %-3dx16: warp=%.4fms  tc=%.4fms  ratio=%.2fx  diff=%.6f  GFLOPS: w=%.1f t=%.1f  %s\n",
               K, K, lw, lt, lw/lt, maxd, flops/(lw/1000)/1e9, flops/(lt/1000)/1e9,
               lw/lt > 1.0 ? "TC WINS!" : "");
        
        cudaFree(dA);cudaFree(dB);cudaFree(dCw);cudaFree(dCt);
        free(hA);free(hB);free(hCw);free(hCt);
        cudaEventDestroy(es);cudaEventDestroy(ee);
    }
    return 0;
}
