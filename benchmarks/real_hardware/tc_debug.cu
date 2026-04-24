#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cmath>

using namespace nvcuda::wmma;

__global__ void tc_test_known(float* result) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    
    // Known values: a = [1,1,1,...,1] (16 ones), b = [2,2,2,...,2] (16 twos)
    // Dot product = 16 * 1 * 2 = 32
    if (lane < 16) { sa[lane] = __float2half(1.0f); sb[lane] = __float2half(2.0f); }
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);
    
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    if (lane == 0) {
        result[0] = cf.x[0];
        result[1] = cf.x[1];
        result[2] = cf.x[2];
        result[3] = cf.x[3];
        result[4] = cf.x[4];
        result[5] = cf.x[5];
        result[6] = cf.x[6];
        result[7] = cf.x[7];
        result[8] = (float)cf.num_elements;
    }
}

__global__ void tc_test_known2(float* result) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    
    // a = [1,2,3,...,16], b = [1,1,1,...,1]
    // Dot product = 1+2+3+...+16 = 136
    if (lane < 16) { sa[lane] = __float2half(lane + 1.0f); sb[lane] = __float2half(1.0f); }
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);
    
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    if (lane == 0) {
        result[0] = cf.x[0];
        result[1] = cf.x[1];
        result[2] = cf.x[2];
        result[3] = cf.x[3];
        result[4] = cf.x[4];
        result[5] = cf.x[5];
        result[6] = cf.x[6];
        result[7] = cf.x[7];
    }
}

__global__ void tc_test_known3(float* result) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    
    // a = [1,2,...,16], b = [16,15,...,1] (reversed)
    // Dot product = 1*16 + 2*15 + 3*14 + ... + 16*1
    // = 16 + 30 + 42 + 52 + 60 + 66 + 70 + 72 + 72 + 70 + 66 + 60 + 52 + 42 + 30 + 16
    // = 816
    if (lane < 16) {
        sa[lane] = __float2half(lane + 1.0f);
        sb[lane] = __float2half(16.0f - lane);
    }
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);
    
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    if (lane == 0) {
        result[0] = cf.x[0];
        result[1] = cf.x[1];
        result[2] = cf.x[2];
        result[3] = cf.x[3];
        result[4] = cf.x[4];
        result[5] = cf.x[5];
        result[6] = cf.x[6];
        result[7] = cf.x[7];
    }
}

// Test: load A once, then do 2 chunks with different B
__global__ void tc_test_multichunk(float* result) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    
    // a = [1,2,...,16] (fixed)
    // chunk 0: b = [1,1,...,1], dot = 136
    // chunk 1: b = [2,2,...,2], dot = 272
    // total = 408
    
    if (lane < 16) sa[lane] = __float2half(lane + 1.0f);
    __syncwarp();
    load_matrix_sync(af, sa, 16);
    
    float total = 0.0f;
    
    // Chunk 0: b = [1,1,...,1]
    if (lane < 16) sb[lane] = __float2half(1.0f);
    __syncwarp();
    load_matrix_sync(bf, sb, 16);
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    if (lane == 0) { result[0] = cf.x[0]; total += cf.x[0]; }
    
    // Chunk 1: b = [2,2,...,2]
    if (lane < 16) sb[lane] = __float2half(2.0f);
    __syncwarp();
    load_matrix_sync(bf, sb, 16);
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    if (lane == 0) { result[1] = cf.x[0]; total += cf.x[0]; }
    
    if (lane == 0) result[2] = total;
}

int main() {
    printf("=== WMMA Known Value Debug ===\n\n");
    
    float *d_r, h_r[16];
    cudaMalloc(&d_r, 16*sizeof(float));
    
    // Test 1: ones * twos = 32
    tc_test_known<<<1,32>>>(d_r);
    cudaDeviceSynchronize();
    cudaMemcpy(h_r, d_r, 16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 1: a=[1]*16, b=[2]*16\n");
    printf("  Expected dot product: 32.0\n");
    printf("  cf.x[0..7]: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
           h_r[0],h_r[1],h_r[2],h_r[3],h_r[4],h_r[5],h_r[6],h_r[7]);
    printf("  num_elements: %.0f\n", h_r[8]);
    printf("  cf.x[0] == 32? %s\n\n", fabsf(h_r[0]-32.0f)<0.01f?"YES":"NO");
    
    // Test 2: [1..16] * [1]*16 = 136
    tc_test_known2<<<1,32>>>(d_r);
    cudaDeviceSynchronize();
    cudaMemcpy(h_r, d_r, 16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 2: a=[1..16], b=[1]*16\n");
    printf("  Expected: 136.0\n");
    printf("  cf.x[0..7]: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
           h_r[0],h_r[1],h_r[2],h_r[3],h_r[4],h_r[5],h_r[6],h_r[7]);
    printf("  cf.x[0] == 136? %s\n\n", fabsf(h_r[0]-136.0f)<0.01f?"YES":"NO");
    
    // Test 3: [1..16] * [16..1] = 816
    tc_test_known3<<<1,32>>>(d_r);
    cudaDeviceSynchronize();
    cudaMemcpy(h_r, d_r, 16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 3: a=[1..16], b=[16..1]\n");
    printf("  Expected: 816.0\n");
    printf("  cf.x[0..7]: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
           h_r[0],h_r[1],h_r[2],h_r[3],h_r[4],h_r[5],h_r[6],h_r[7]);
    printf("  cf.x[0] == 816? %s\n\n", fabsf(h_r[0]-816.0f)<0.01f?"YES":"NO");
    
    // Test 4: Multi-chunk with reused fragment A
    tc_test_multichunk<<<1,32>>>(d_r);
    cudaDeviceSynchronize();
    cudaMemcpy(h_r, d_r, 16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 4: Multi-chunk (fragment A reused)\n");
    printf("  Chunk 0: a=[1..16], b=[1]*16, expected=136.0, got=%.1f %s\n",
           h_r[0], fabsf(h_r[0]-136.0f)<0.01f?"✓":"✗");
    printf("  Chunk 1: a=[1..16], b=[2]*16, expected=272.0, got=%.1f %s\n",
           h_r[1], fabsf(h_r[1]-272.0f)<0.01f?"✓":"✗");
    printf("  Total:   expected=408.0, got=%.1f %s\n",
           h_r[2], fabsf(h_r[2]-408.0f)<0.01f?"✓":"✗");
    
    cudaFree(d_r);
    return 0;
}
