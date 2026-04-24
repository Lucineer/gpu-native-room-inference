#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda::wmma;

// Warp matmul: C = A * B where A is MxK, B is KxN
__global__ void warp_matmul(const half* A, const half* B, float* C, int M, int K, int N) {
    int lane = threadIdx.x;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = lane; k < K; k += 32)
                s += __half2float(__hmul(A[m*K+k], B[k*N+n]));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                s += __shfl_down_sync(0xffffffff, s, o);
            if (lane == 0) C[m*N+n] = s;
        }
    }
}

// TC matmul: native 16x16x16 WMMA blocks
__global__ void tc_matmul(const half* A, const half* B, float* C, int M, int K, int N) {
    // For M,N,K all multiples of 16, single warp does one 16x16x16 tile
    // A: M rows x K cols, B: K rows x N cols
    __shared__ alignas(128) half sa[16*16], sb[16*16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    int lane = threadIdx.x;
    
    // For simplicity: M=N=16, K is multiple of 16
    fill_fragment(cf, 0.0f);
    
    for (int kk = 0; kk < K; kk += 16) {
        // Load 16x16 tile of A (16 rows, 16 cols starting at kk)
        for (int r = lane; r < 256; r += 32) {
            int row = r / 16, col = r % 16;
            sa[row * 16 + col] = A[row * K + kk + col];
        }
        // Load 16x16 tile of B (16 rows starting at kk, 16 cols)
        for (int r = lane; r < 256; r += 32) {
            int row = r / 16, col = r % 16;
            sb[row * 16 + col] = B[(kk + row) * N + col];
        }
        __syncwarp();
        
        load_matrix_sync(af, sa, 16);
        load_matrix_sync(bf, sb, 16);
        mma_sync(cf, af, bf, cf);
    }
    
    // Store result — 16x16 output, 32 lanes x 8 elements each
    // We need to extract the full matrix
    // For now, just sum all fragment values per lane to verify correctness
    // Actually, let's store via shared memory
    __shared__ float sc[16*16];
    for (int i = 0; i < 8; i++)
        sc[lane * 8 + i] = cf.x[i];
    __syncwarp();
    
    // Each lane has 8 elements. The 256 values = full 16x16 matrix.
    // Copy to output
    if (lane < 16) {
        for (int n = 0; n < 16; n++) {
            // We need to figure out the mapping...
            // From fragment_map test: row 0 is in lanes 0-3
            // Let's just store raw and compare
            C[lane * 16 + n] = sc[lane * 16 + n]; // This won't be right but let's see
        }
    }
}

// Simpler TC approach: use 16 WMMA dot products (one per output element)
// C[m][n] = sum_k A[m][k] * B[k][n]
__global__ void tc_matmul_dotprod(const half* A, const half* B, float* C, int M, int K, int N) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    int lane = threadIdx.x;
    
    // Compute one row of output: C[m][n] for all n in [0..N-1], fixed m
    // Actually for matmul the input vectors are different for each output element
    // Let's do: for each output row m, for each output col n, dot product of A[m,:] and B[:,n]
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int c = 0; c < K/16; c++) {
                int koff = c * 16;
                if (lane < 16) {
                    sa[lane] = A[m * K + koff + lane];
                    sb[lane] = B[(koff + lane) * N + n];
                }
                __syncwarp();
                load_matrix_sync(af, sa, 16);
                load_matrix_sync(bf, sb, 16);
                fill_fragment(cf, 0.0f);
                mma_sync(cf, af, bf, cf);
                if (lane == 0) sum += cf.x[0];
            }
            if (lane == 0) C[m * N + n] = sum;
        }
    }
}

int main() {
    printf("=== TC MATMUL: Where Tensor Cores WIN ===\n\n");
    
    // Test: 16xK x Kx16 for various K values
    int Ks[] = {16, 32, 64, 128, 256};
    int M = 16, N = 16;
    
    for (int ki = 0; ki < 5; ki++) {
        int K = Ks[ki];
        size_t sA = (size_t)M * K * sizeof(half);
        size_t sB = (size_t)K * N * sizeof(half);
        size_t sC = (size_t)M * N * sizeof(float);
        int ITERS = 10000;
        
        half *hA = (half*)malloc(sA), *hB = (half*)malloc(sB);
        float *hCw = (float*)malloc(sC), *hCt = (float*)malloc(sC);
        srand(42);
        for (size_t i = 0; i < (size_t)M*K; i++) hA[i] = __float2half((float)rand()/RAND_MAX-0.5f);
        for (size_t i = 0; i < (size_t)K*N; i++) hB[i] = __float2half((float)rand()/RAND_MAX-0.5f);
        
        half *dA, *dB; float *dCw, *dCt;
        cudaMalloc(&dA, sA); cudaMalloc(&dB, sB);
        cudaMalloc(&dCw, sC); cudaMalloc(&dCt, sC);
        cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);
        
        for (int i = 0; i < 50; i++) {
            warp_matmul<<<1,32>>>(dA,dB,dCw,M,K,N);
            tc_matmul_dotprod<<<1,32>>>(dA,dB,dCt,M,K,N);
        }
        cudaDeviceSynchronize();
        
        cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);
        float ms, lw, lt;
        
        cudaEventRecord(es);
        for (int i = 0; i < ITERS; i++) warp_matmul<<<1,32>>>(dA,dB,dCw,M,K,N);
        cudaEventRecord(ee); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, es, ee); lw = ms / ITERS;
        
        cudaEventRecord(es);
        for (int i = 0; i < ITERS; i++) tc_matmul_dotprod<<<1,32>>>(dA,dB,dCt,M,K,N);
        cudaEventRecord(ee); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, es, ee); lt = ms / ITERS;
        
        // Correctness
        warp_matmul<<<1,32>>>(dA,dB,dCw,M,K,N);
        tc_matmul_dotprod<<<1,32>>>(dA,dB,dCt,M,K,N);
        cudaDeviceSynchronize();
        cudaMemcpy(hCw, dCw, sC, cudaMemcpyDeviceToHost);
        cudaMemcpy(hCt, dCt, sC, cudaMemcpyDeviceToHost);
        float maxd = 0;
        for (size_t i = 0; i < (size_t)M*N; i++) maxd = fmaxf(maxd, fabsf(hCw[i]-hCt[i]));
        
        float flops = 2.0f * M * K * N;
        printf("16x%d x %dx16: warp=%.4fms tc=%.4fms ratio=%.2fx diff=%.4f GFLOPS: w=%.1f t=%.1f\n",
               K, lw, lt, lw/lt, maxd, flops/(lw/1000)/1e9, flops/(lt/1000)/1e9);
        
        cudaFree(dA);cudaFree(dB);cudaFree(dCw);cudaFree(dCt);
        free(hA);free(hB);free(hCw);free(hCt);
        cudaEventDestroy(es);cudaEventDestroy(ee);
    }
    return 0;
}
