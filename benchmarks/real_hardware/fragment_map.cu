/**
 * WMMA Fragment Layout Map — Complete Discovery
 * Jetson Orin Nano | sm_87 | CUDA 12.6
 * 
 * Goal: understand exactly which element of the 16x16 result matrix
 * each lane's cf.x[i] corresponds to.
 * 
 * Method: Use known input where only ONE element is non-zero,
 * then check which lane/offset has the non-zero result.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda::wmma;

#define NUM_LANES 32
#define FRAG_SIZE 8
#define MATRIX_DIM 16

// Each lane writes its 8 fragment values
__global__ void map_fragments(float* out) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,row_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    
    // Test: A[0][0] = 1.0, all others = 0
    // B[i][j] = 1.0 for all i,j
    // Result: C[0][j] = 1.0 for all j, C[i][j] = 0 for i > 0
    if (lane < 16) { sa[lane] = __float2half(0.0f); sb[lane] = __float2half(1.0f); }
    __syncwarp();
    if (lane == 0) sa[0] = __float2half(1.0f);  // Only A[0][0] = 1
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    // Each lane writes its 8 values
    for (int i = 0; i < FRAG_SIZE; i++) {
        out[lane * FRAG_SIZE + i] = cf.x[i];
    }
}

// Test 2: A[1][0] = 1.0 only → C[1][j] should be 1.0
__global__ void map_test2(float* out) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,row_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    if (lane < 16) { sa[lane] = __float2half(0.0f); sb[lane] = __float2half(1.0f); }
    __syncwarp();
    if (lane == 0) sa[1] = __float2half(1.0f);  // A[1][0] = 1
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    for (int i = 0; i < FRAG_SIZE; i++)
        out[lane * FRAG_SIZE + i] = cf.x[i];
}

// Test 3: Full identity: A[i][i] = i+1, B = identity
// Result: C[i][j] = A[i][j] (diagonal values)
__global__ void map_test3(float* out) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,row_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    if (lane < 16) {
        sa[lane] = __float2half(0.0f);  // Zero
        sb[lane] = __float2half(0.0f);
    }
    __syncwarp();
    // Fill first column of A with 1,2,...,16 and first row of B with 1,2,...,16
    // A = column vector [1..16], B = row vector [1..16]
    // Result = outer product: C[i][j] = (i+1)*(j+1)
    if (lane < 16) {
        sa[lane] = __float2half(lane + 1.0f);
        sb[lane] = __float2half(lane + 1.0f);
    }
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    for (int i = 0; i < FRAG_SIZE; i++)
        out[lane * FRAG_SIZE + i] = cf.x[i];
}

// Test 4: Row-major B — each row of B loaded separately
// A = [1..16] (column), B row-major: row 0 = [1,2,...,16]
// This tests how row_major B is interpreted by WMMA
__global__ void map_test4(float* out) {
    __shared__ alignas(128) half sa[16], sb[16*16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,row_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    
    // A = column [1,2,...,16] (row_major loads first column? or first row?)
    if (lane < 16) sa[lane] = __float2half(lane + 1.0f);
    
    // B = identity matrix (row-major: B[i][i] = 1)
    // With leading dimension 16
    for (int idx = lane; idx < 256; idx += 32) {
        int r = idx / 16, c = idx % 16;
        sb[r * 16 + c] = __float2half(r == c ? 1.0f : 0.0f);
    }
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);  // ldm = 16 (stride between rows)
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    for (int i = 0; i < FRAG_SIZE; i++)
        out[lane * FRAG_SIZE + i] = cf.x[i];
}

void print_map(const char* name, float* data, int size) {
    printf("\n=== %s ===\n", name);
    printf("Lane  Frag values                                    Non-zero?\n");
    for (int l = 0; l < NUM_LANES; l++) {
        printf("L%02d   ", l);
        int nonzero = 0;
        for (int i = 0; i < FRAG_SIZE; i++) {
            float v = data[l * FRAG_SIZE + i];
            printf("%8.2f", v);
            if (fabsf(v) > 0.001) nonzero++;
        }
        printf("  %s\n", nonzero > 0 ? "← HAS DATA" : "");
    }
    
    // Summary: which (lane, offset) pairs have non-zero values
    printf("Non-zero entries: ");
    for (int l = 0; l < NUM_LANES; l++) {
        for (int i = 0; i < FRAG_SIZE; i++) {
            float v = data[l * FRAG_SIZE + i];
            if (fabsf(v) > 0.001)
                printf("[L%d:%d]=%.1f ", l, i, v);
        }
    }
    printf("\n");
}

int main() {
    printf("=== WMMA Fragment Layout Discovery ===\n\n");
    
    float *d_out, h_out[256];
    cudaMalloc(&d_out, 256 * sizeof(float));
    
    // Test 1: A[0][0]=1, B=all 1s → C[0][j]=1 for all j
    map_fragments<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost);
    print_map("Test 1: A[0][0]=1, B=ones → expect C[0][j]=1 only", h_out, 256);
    
    // Test 2: A[1][0]=1, B=ones → C[1][j]=1 for all j
    map_test2<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost);
    print_map("Test 2: A[1][0]=1, B=ones → expect C[1][j]=1 only", h_out, 256);
    
    // Test 3: outer product [1..16]×[1..16] → C[i][j]=(i+1)(j+1)
    map_test3<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost);
    print_map("Test 3: A=[1..16], B=[1..16] → C[i][j]=(i+1)(j+1)", h_out, 256);
    
    // Test 4: A=[1..16], B=identity → C = A (column preserved)
    map_test4<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost);
    print_map("Test 4: A=[1..16], B=identity → C should have A values", h_out, 256);
    
    cudaFree(d_out);
    return 0;
}
