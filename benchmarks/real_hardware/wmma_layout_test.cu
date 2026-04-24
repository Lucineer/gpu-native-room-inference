/**
 * WMMA Fragment Layout Investigation
 * Jetson Orin Nano, sm_87, CUDA 12.6
 * 
 * Goal: understand how fragment<matrix_a/b, 16,16,16> and
 * fragment<accumulator, 16,16,16> store their elements in .x[]
 * 
 * Theory: For 16x16 matrix multiply C = A * B:
 * - A is row_major: A[0..15] maps to rows 0-15, column 0
 * - B is col_major: B[0..15] maps to row 0, columns 0-15
 * - Result C[i][j] = A[i][0] * B[0][j] (outer product of two vectors)
 * - For dot product: sum of diagonal C[i][i] = sum(A[i]*B[i])
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda::wmma;

// Kernel that prints fragment layout info
__global__ void inspect_wmma(float* out_diag, float* out_all, float* out_sizes) {
    __shared__ alignas(128) half sa[16];
    __shared__ alignas(128) half sb[16];
    
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    
    int lane = threadIdx.x;
    
    // Set up: identity-like test
    // A = [1,2,3,...,16] (row vector, row_major means column 0)
    // B = [1,2,3,...,16] (column vector, col_major means row 0)
    if (lane < 16) { sa[lane] = __float2half(lane + 1.0f); sb[lane] = __float2half(lane + 1.0f); }
    __syncwarp();
    
    load_matrix_sync(af, sa, 16);
    load_matrix_sync(bf, sb, 16);
    
    fill_fragment(cf, 0.0f);
    mma_sync(cf, af, bf, cf);
    
    // Write out all 256 accumulator elements
    if (lane == 0) {
        out_sizes[0] = (float)af.num_elements;
        out_sizes[1] = (float)bf.num_elements;
        out_sizes[2] = (float)cf.num_elements;
        
        // Sum diagonal: C[i][i]
        // Need to figure out correct indexing
        // Try 1: row-major cf.x[row*16+col]
        float diag1 = 0, all1 = 0;
        for (int i = 0; i < 256; i++) {
            all1 += cf.x[i];
            // Print first 32 elements for inspection
        }
        out_all[0] = all1;
        
        // Print first 32 values for debugging
        // We'll copy all 256 to output
        for (int i = 0; i < 256; i++) {
            out_diag[i] = cf.x[i];
        }
    }
}

int main() {
    printf("=== WMMA Fragment Layout Investigation ===\n\n");
    
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | sm_%d%d\n\n", p.name, p.major, p.minor);
    
    float *d_diag, *d_all, *d_sizes;
    float *h_diag = (float*)calloc(256, sizeof(float));
    float *h_all = (float*)calloc(1, sizeof(float));
    float *h_sizes = (float*)calloc(3, sizeof(float));
    
    cudaMalloc(&d_diag, 256*sizeof(float));
    cudaMalloc(&d_all, sizeof(float));
    cudaMalloc(&d_sizes, 3*sizeof(float));
    
    inspect_wmma<<<1, 32>>>(d_diag, d_all, d_sizes);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_diag, d_diag, 256*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all, d_all, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sizes, d_sizes, 3*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Fragment sizes: A=%d B=%d C=%d\n", (int)h_sizes[0], (int)h_sizes[1], (int)h_sizes[2]);
    printf("Sum of ALL 256 elements: %.2f\n", h_all[0]);
    
    // Expected: A=[1..16], B=[1..16], outer product C[i][j] = (i+1)*(j+1)
    // Diagonal sum: sum(i^2 for i in 1..16) = 16*17*33/6 = 1496
    // All elements sum: sum(i*j for i,j in 1..16) = (16*17/2)^2 = 18496
    
    printf("\nExpected values:\n");
    printf("  Outer product A*B where A=B=[1..16]\n");
    printf("  C[i][j] = (i+1)*(j+1)\n");
    printf("  Sum of ALL elements: 18496.0\n");
    printf("  Sum of diagonal: 1496.0\n");
    printf("  Sum of row 0: 136.0\n");
    printf("  Sum of col 0: 136.0\n");
    
    printf("\nActual ALL sum: %.2f\n", h_all[0]);
    printf("Match: %s\n", fabsf(h_all[0] - 18496.0f) < 1.0f ? "✓ PERFECT" : "✗ WRONG");
    
    printf("\n--- First 48 fragment values (row-major interpretation) ---\n");
    for (int row = 0; row < 3; row++) {
        printf("Row %2d: ", row);
        for (int col = 0; col < 16; col++) {
            int idx = row * 16 + col;
            if (idx < 256)
                printf("%8.1f", h_diag[idx]);
        }
        printf("\n");
    }
    
    printf("\n--- Checking specific positions ---\n");
    // If row-major: C[0][0]=1, C[0][1]=2, C[1][0]=2, C[15][15]=256
    printf("cf.x[0]  (C[0][0]? expected 1.0):  %.1f\n", h_diag[0]);
    printf("cf.x[1]  (C[0][1]? expected 2.0):  %.1f\n", h_diag[1]);
    printf("cf.x[16] (C[1][0]? expected 2.0):  %.1f\n", h_diag[16]);
    printf("cf.x[17] (C[1][1]? expected 4.0):  %.1f\n", h_diag[17]);
    printf("cf.x[255](C[15][15]? expected 256.0): %.1f\n", h_diag[255]);
    
    // Diagonal sum with row-major indexing
    float diag_sum = 0;
    for (int i = 0; i < 16; i++) diag_sum += h_diag[i*16+i];
    printf("\nDiagonal sum (row-major idx): %.2f (expected 1496.0)\n", diag_sum);
    
    // Maybe it's column-major?
    float diag_sum_col = 0;
    for (int i = 0; i < 16; i++) diag_sum_col += h_diag[i + i*16]; // same thing for square
    printf("Diagonal sum (col-major idx): %.2f (expected 1496.0)\n", diag_sum_col);
    
    // Try column-major: cf.x[col*16+row]
    diag_sum = 0;
    for (int i = 0; i < 16; i++) diag_sum += h_diag[i + i*16]; // col*16+row
    printf("Diagonal sum (transposed idx): %.2f\n", diag_sum);
    
    cudaFree(d_diag); cudaFree(d_all); cudaFree(d_sizes);
    free(h_diag); free(h_all); free(h_sizes);
    return 0;
}
