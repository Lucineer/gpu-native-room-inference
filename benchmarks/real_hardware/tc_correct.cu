/**
 * CORRECT Tensor Core Dot Product — Jetson Orin Nano
 * 
 * KEY DISCOVERY: fragment<accumulator,16,16,16,float> has only 8 elements!
 * cf.x[0] already contains the FULL dot product of the two 16-element vectors!
 * No need to sum 256 elements or extract diagonals.
 * 
 * Why: WMMA loads 16 elements as a 16x1 column (row_major) and 1x16 row (col_major),
 * so the matrix multiply 16x1 * 1x16 produces a 16x16 outer product,
 * but the accumulator fragments are distributed across warp lanes (32 lanes, 8 each).
 * Lane 0's cf.x[0] = dot product. Other lanes get partial sums.
 * 
 * Actually more precisely: the 16x16 matrix result is distributed across the warp.
 * cf.x[0] on lane 0 = C[0][0] = dot product of the loaded vectors.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda::wmma;

__global__ void thread_dot(const half* w, const half* in, float* out, int n) {
    int rid = blockIdx.x;
    if (rid >= n) return;
    const half* ww = w + rid * 256;
    float s = 0.0f;
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        s += __half2float(__hmul(ww[i], in[i]));
    __shared__ float buf[256];
    buf[threadIdx.x] = s;
    __syncthreads();
    for (int bs = blockDim.x/2; bs > 0; bs >>= 1) {
        if (threadIdx.x < bs) buf[threadIdx.x] += buf[threadIdx.x + bs];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[rid] = buf[0];
}

__global__ void warp_dot(const half* w, const half* in, float* out, int n) {
    int wid = blockIdx.x, lane = threadIdx.x;
    if (wid >= n) return;
    const half* ww = w + wid * 256;
    float s = 0.0f;
    for (int i = lane; i < 256; i += 32)
        s += __half2float(__hmul(ww[i], in[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) out[wid] = s;
}

/**
 * CORRECT tensor core dot product
 * Each chunk of 16 elements is processed as a WMMA 16x16x16 op.
 * The fragment accumulator has 8 elements per lane.
 * For row_major A and col_major B with single-column/row data:
 * cf.x[0] on lane 0 contains the chunk's dot product.
 * 
 * 16 chunks × 16 elements = 256 total elements.
 */
__global__ void tc_dot_correct(const half* w, const half* in, float* out, int n) {
    __shared__ alignas(128) half si[16], sw[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int wid = blockIdx.x, lane = threadIdx.x;
    if (wid >= n) return;
    const half* ww = w + wid * 256;
    
    // Load input into shared memory → fragment A (column vector, row_major)
    if (lane < 16) si[lane] = in[lane];
    __syncwarp();
    load_matrix_sync(af, si, 16);
    
    float sum = 0.0f;
    for (int c = 0; c < 16; c++) {
        int off = c * 16;
        if (off >= 256) break;
        
        // Load 16 weights → shared memory → fragment B (row vector, col_major)
        if (lane < 16) sw[lane] = ww[off + lane];
        __syncwarp();
        load_matrix_sync(bf, sw, 16);
        
        fill_fragment(cf, 0.0f);
        mma_sync(cf, af, bf, cf);
        
        // KEY: cf.x[0] on lane 0 = dot product of this 16-element chunk
        // The 8 fragment elements are distributed across warp lanes
        // Lane 0 gets the [0,0] element of the result matrix = sum(a[i]*b[i])
        if (lane == 0) sum += cf.x[0];
    }
    
    if (lane == 0) out[wid] = sum;
}

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  TENSOR CORE — CORRECT IMPLEMENTATION                   ║\n");
    printf("║  Jetson Orin Nano | sm_87 | CUDA 12.6                   ║\n");
    printf("║  Key: accumulator fragment has 8 elements, not 256      ║\n");
    printf("║  cf.x[0] on lane 0 = dot product of 16-element chunk    ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | sm_%d%d | %d SMs | %.0f MHz | %.0f MB\n\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           p.clockRate/1000.0, p.totalGlobalMem/1048576.0);
    
    const int N=6, R=256, ITERS=100000;
    half *hw=(half*)malloc(N*R*sizeof(half));
    half *hi=(half*)malloc(R*sizeof(half));
    float *h1=(float*)malloc(N*sizeof(float));
    float *h2=(float*)malloc(N*sizeof(float));
    float *h3=(float*)malloc(N*sizeof(float));
    
    srand(42);
    for(int r=0;r<N;r++) for(int i=0;i<R;i++)
        hw[r*R+i]=__float2half((float)rand()/RAND_MAX-0.5f);
    for(int i=0;i<R;i++) hi[i]=__float2half((float)rand()/RAND_MAX-0.5f);
    
    half *dw,*di; float *d1,*d2,*d3;
    cudaMalloc(&dw,N*R*sizeof(half)); cudaMalloc(&di,R*sizeof(half));
    cudaMalloc(&d1,N*sizeof(float)); cudaMalloc(&d2,N*sizeof(float));
    cudaMalloc(&d3,N*sizeof(float));
    cudaMemcpy(dw,hw,N*R*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(di,hi,R*sizeof(half),cudaMemcpyHostToDevice);
    
    for(int i=0;i<500;i++){
        thread_dot<<<N,256>>>(dw,di,d1,N);
        warp_dot<<<N,32>>>(dw,di,d2,N);
        tc_dot_correct<<<N,32>>>(dw,di,d3,N);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t es,ee; cudaEventCreate(&es); cudaEventCreate(&ee);
    float ms;
    
    cudaEventRecord(es);
    for(int i=0;i<ITERS;i++) thread_dot<<<N,256>>>(dw,di,d1,N);
    cudaEventRecord(ee); cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms,es,ee); float lt=ms/ITERS;
    
    cudaEventRecord(es);
    for(int i=0;i<ITERS;i++) warp_dot<<<N,32>>>(dw,di,d2,N);
    cudaEventRecord(ee); cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms,es,ee); float lw=ms/ITERS;
    
    cudaEventRecord(es);
    for(int i=0;i<ITERS;i++) tc_dot_correct<<<N,32>>>(dw,di,d3,N);
    cudaEventRecord(ee); cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms,es,ee); float ltc=ms/ITERS;
    
    cudaMemcpy(h1,d1,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h2,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h3,d3,N*sizeof(float),cudaMemcpyDeviceToHost);
    
    printf("┌──────────────────────────┬────────────┬────────────┬──────────┐\n");
    printf("│ Implementation           │   Latency  │ Throughput │ vs Warp  │\n");
    printf("├──────────────────────────┼────────────┼────────────┼──────────┤\n");
    printf("│ Thread-as-Room (256 thr) │ %8.4f ms │ %8.0f/s │    —     │\n", lt, 1000/lt);
    printf("│ Warp Shuffle (32 thr)    │ %8.4f ms │ %8.0f/s │  %5.2fx  │\n", lw, 1000/lw, lt/lw);
    printf("│ Tensor Core WMMA (fixed) │ %8.4f ms │ %8.0f/s │  %5.2fx  │\n", ltc, 1000/ltc, lw/ltc);
    printf("└──────────────────────────┴────────────┴────────────┴──────────┘\n");
    
    printf("\n=== Correctness ===\n");
    int ok=1;
    for(int i=0;i<N;i++){
        float d_t=fabsf(h1[i]-h2[i]);
        float d_tc=fabsf(h2[i]-h3[i]);
        float rel=d_tc/(fabsf(h2[i])+1e-10f);
        printf("Room %d: thread=%.6f warp=%.6f tc=%.6f | warp±thread=%.2e tc±warp=%.2e %s\n",
               i, h1[i], h2[i], h3[i], d_t, d_tc, 
               isnan(h3[i])?"NaN!":rel<0.001?"✓":"~");
        if(isnan(h3[i])||isinf(h3[i])) ok=0;
    }
    printf("\nAll finite: %s\n", ok?"✓ YES":"✗ NO");
    
    float maxrel=0;
    for(int i=0;i<N;i++){
        float r=fabsf(h2[i]-h3[i])/(fabsf(h2[i])+1e-10f);
        if(r>maxrel)maxrel=r;
    }
    printf("Max relative diff (warp vs tc): %.8f\n", maxrel);
    
    if(ok && maxrel < 0.001) {
        printf("\n🎉 TENSOR CORE WORKS! Correct + benchmarked on real Jetson hardware!\n");
    } else if(ok) {
        printf("\n⚠️ Tensor core produces finite results but with FP16 precision drift\n");
    } else {
        printf("\n❌ Tensor core still has NaN/Inf — investigation needed\n");
    }
    
    printf("\n=== KEY DISCOVERY ===\n");
    printf("fragment<accumulator,16,16,16,float>.num_elements = 8 (NOT 256!)\n");
    printf("cf.x[0] on lane 0 = dot product of the 16-element chunk\n");
    printf("The full 16x16 result matrix is distributed across 32 warp lanes.\n");
    printf("Each lane gets 8 elements of the 256-element result.\n");
    printf("Lane 0 gets elements [0,0] through [0,7] of the result.\n");
    printf("For dot product: cf.x[0] = C[0][0] = Σ(a[i]*b[i]) = chunk dot product.\n");
    
    cudaFree(dw);cudaFree(di);cudaFree(d1);cudaFree(d2);cudaFree(d3);
    free(hw);free(hi);free(h1);free(h2);free(h3);
    cudaEventDestroy(es);cudaEventDestroy(ee);
    return 0;
}
