#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda::wmma;

#define INPUT_SIZE 256

__global__ void warp_mv(const half* w, const half* in, float* out, int nn) {
    int lane = threadIdx.x;
    const half* ww = w;
    for (int n = 0; n < nn; n++) {
        float s = 0.0f;
        for (int k = lane; k < INPUT_SIZE; k += 32)
            s += __half2float(__hmul(ww[n * INPUT_SIZE + k], in[k]));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) out[n] = s;
    }
}

__global__ void tc_mv(const half* w, const half* in, float* out, int nn) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    int lane = threadIdx.x;
    const half* ww = w;
    for (int n = 0; n < nn; n++) {
        float sum = 0.0f;
        for (int c = 0; c < 16; c++) {
            int koff = c * 16;
            if (lane < 16) { sa[lane] = in[koff+lane]; sb[lane] = ww[n*INPUT_SIZE+koff+lane]; }
            __syncwarp();
            load_matrix_sync(af, sa, 16);
            load_matrix_sync(bf, sb, 16);
            fill_fragment(cf, 0.0f);
            mma_sync(cf, af, bf, cf);
            if (lane == 0) sum += cf.x[0];
        }
        if (lane == 0) out[n] = sum;
    }
}

int main() {
    printf("=== SCALE SWEEP ===\n\n");
    int neurons[] = {16, 32, 64, 128};
    
    for (int ni = 0; ni < 4; ni++) {
        int nn = neurons[ni];
        size_t wb = (size_t)nn * INPUT_SIZE * sizeof(half);
        int ITERS = 10000;
        
        half *hw = (half*)malloc(wb);
        half *hi = (half*)malloc(INPUT_SIZE * sizeof(half));
        float *ho = (float*)calloc(nn, sizeof(float));
        srand(42);
        for (size_t i = 0; i < (size_t)nn*INPUT_SIZE; i++) hw[i] = __float2half((float)rand()/RAND_MAX-0.5f);
        for (int i = 0; i < INPUT_SIZE; i++) hi[i] = __float2half((float)rand()/RAND_MAX-0.5f);
        
        half *dw, *di; float *d1, *d2;
        cudaError_t e;
        e = cudaMalloc(&dw, wb); if(e!=cudaSuccess){printf("skip %d: OOM\n",nn); free(hw);free(hi);free(ho); continue;}
        cudaMalloc(&di, INPUT_SIZE*sizeof(half));
        cudaMalloc(&d1, (size_t)nn*sizeof(float));
        cudaMalloc(&d2, (size_t)nn*sizeof(float));
        cudaMemcpy(dw, hw, wb, cudaMemcpyHostToDevice);
        cudaMemcpy(di, hi, INPUT_SIZE*sizeof(half), cudaMemcpyHostToDevice);
        
        for (int i = 0; i < 50; i++) {
            warp_mv<<<1,32>>>(dw,di,d1,nn);
            tc_mv<<<1,32>>>(dw,di,d2,nn);
        }
        cudaDeviceSynchronize();
        
        cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);
        float ms, lw, lt;
        
        cudaEventRecord(es);
        for (int i = 0; i < ITERS; i++) warp_mv<<<1,32>>>(dw,di,d1,nn);
        cudaEventRecord(ee); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, es, ee); lw = ms / ITERS;
        
        cudaEventRecord(es);
        for (int i = 0; i < ITERS; i++) tc_mv<<<1,32>>>(dw,di,d2,nn);
        cudaEventRecord(ee); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, es, ee); lt = ms / ITERS;
        
        float flops = (float)nn * INPUT_SIZE * 2;
        printf("neurons=%-3d  warp=%.4fms  tc=%.4fms  ratio=%.2fx  GFLOPS: w=%.1f t=%.1f\n", 
               nn, lw, lt, lw/lt, flops/(lw/1000)/1e9, flops/(lt/1000)/1e9);
        
        cudaFree(dw); cudaFree(di); cudaFree(d1); cudaFree(d2);
        free(hw); free(hi); free(ho);
        cudaEventDestroy(es); cudaEventDestroy(ee);
    }
    return 0;
}
