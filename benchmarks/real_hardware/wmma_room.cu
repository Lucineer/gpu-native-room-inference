/**
 * Suite #37: Tensor Core WMMA for Room Inference
 * 
 * The V7 kernel uses regular FP16 arithmetic (__half2float multiply).
 * The Jetson Orin has Tensor Cores that can do 16×16×16 FP16 matmul in one instruction.
 * 
 * Can WMMA accelerate room dot products?
 * 
 * Room inference: weight[room][dim] · input[dim] = scalar
 * This is a matrix-vector multiply: W[num_rooms × dim] × x[dim × 1] = y[num_rooms × 1]
 * 
 * WMMA works on 16×16 tiles. For dim=256, that's 16 tiles along dim.
 * Each warp computes one 16×16 output tile.
 * 
 * Challenge: WMMA requires 16×16 alignment and specific memory layout.
 * The Orin's tensor cores do FP16×FP16→FP32 (or FP16×FP16→FP16).
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 wmma_room.cu -o wmma_room
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <mma.h>

#define DIM 256
#define WARMUP 200
#define ITERS 10000

// V7 baseline: regular FP16 dot product
__global__ void infer_v7(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(room_base + room_idx) * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room_base + room_idx] = sum;
}

// WMMA-based: use tensor cores for the matmul
// W[num_rooms × dim] × x[dim × 1] = y[num_rooms × 1]
// Each warp handles a 16×16 tile of W, producing 16 outputs
// Need dim to be multiple of 16 (256 is fine)
__global__ void infer_wmma(
    const half* __restrict__ weights,  // [num_rooms][dim], row-major
    const half* __restrict__ input,    // [dim], treated as [1][dim]
    float* __restrict__ output,
    int num_rooms, int dim)
{
    // Each block handles 16×16 rooms
    int room_start = blockIdx.x * 16;
    int room_in_block = threadIdx.y;  // 0-15: which row within the 16×16 tile
    int lane = threadIdx.x;           // 0-31: thread within warp
    
    if (room_start + room_in_block >= num_rooms) return;
    
    // WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
    
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    // Process 16 columns at a time (k dimension)
    for (int k = 0; k < dim; k += 16) {
        // Load A: W[room_start + row][k:k+16]
        nvcuda::wmma::load_matrix_sync(a_frag, weights + (room_start + room_in_block) * dim + k, dim);
        
        // Load B: input[k:k+16] — same for all rows (broadcast)
        // We need input laid out as [16][16] for WMMA. But input is just [dim].
        // Trick: use col_major for B, so input[k:k+16] loads correctly
        // Actually for matrix-vector: B is 16×1, not 16×16.
        // WMMA requires both to be 16×16. So we need to pad input to 16×16.
        // Alternative: use wmma::load_matrix_sync with a stride that matches
        // 
        // For a matrix-vector product W (M×K) × x (K×1) = y (M×1):
        // We can reshape x into a 16×16 matrix where each row is the same.
        // Or better: process K in groups of 16, accumulate.
        // 
        // The B matrix for each k-step is 16×16, but we want 16×1.
        // If we set stride=0 for B loading, it'll repeat the same 16 values.
        // Actually wmma::load_matrix_sync doesn't support stride for matrix_b easily.
        //
        // Simpler: treat this as W[M×K] × I[K×K_diagonal] where I is diagonal.
        // That doesn't work either.
        //
        // Cleanest approach: pre-pad input into 16×dim/16 matrix, each row identical.
        // Or: just load 16 elements and manually create the B fragment.
        
        // Manual B fragment fill (input is the same for all rooms)
        // B is 16×16 row-major. We want B[k'][j] = input[k*16 + j] for the dot product.
        // But actually we're doing a matmul, so each row of B contributes.
        // For W (16×K) × x (K×1): we want sum over k of W[r][k] * x[k]
        // In WMMA terms: A (16×16) × B (16×16) → C (16×16)
        // We need B to be diagonal: B[j][j] = x[k*16+j], B[j][i] = 0 for i≠j
        // This wastes 15/16 of the tensor core capacity.
        
        // Actually, the correct way: treat the vector x as a K×1 matrix.
        // WMMA requires 16×16. So pad x to 16×16 by repeating.
        // B[j][i] = x[k*16 + i] for all j (broadcast across rows)
        // Then C[r][j] = sum_i A[r][i] * B[j][i] = sum_i W[r][k*16+i] * x[k*16+i]
        // All 16 rows of C will have the same value (partial sum).
        // Take C[r][0] as the result.
        
        nvcuda::wmma::load_matrix_sync(b_frag, input + k, 16);  // stride=16 means row-major, but input is contiguous
        // Actually stride should be 16 to load input[k:k+16] as a 16-wide row
        // But input is 1D, so stride=16 loads elements k, k+1, ..., k+15 as one row
        // and for subsequent rows it would go to k+16, k+17, ... which is wrong
        // We need all rows of B to be the same.
        // 
        // Best approach: just use the shared memory trick.
        // Or: manually fill B fragment.
        
        // Manual fill: B is row-major 16×16, all rows identical to input[k:k+16]
        for (int j = 0; j < 16; j++) {
            // B fragment stores 16x16 = 256 elements
            // In row-major: element (j, i) is at j*16 + i
            // fragment.x[j*16 + i] = input[k + i]  (for all j)
            for (int i = 0; i < 16; i++) {
                b_frag.x[j * 16 + i] = input[k + i];
            }
        }
        
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Extract result: C[r][0] (all columns have the same value due to B broadcast)
    // Actually no — since all rows of B are the same, C[r][j] = sum_i W[r][i]*x[i] for all j
    // So C[r][0] = C[r][1] = ... = C[r][15] = dot product
    if (lane == 0) {
        output[room_start + room_in_block] = c_frag.x[0];
    }
}

// Simpler WMMA: each warp does one room
// W[1][256] × x[256] = y[1]
// Use 16×16 WMMA tiles: 16 tiles along K dimension
__global__ void infer_wmma_simple(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;  // 8 warps per block = 8 rooms per block
    int room_idx = threadIdx.x / 32;  // which warp
    int lane = threadIdx.x % 32;
    
    if (room_base + room_idx >= num_rooms) return;
    
    int room = room_base + room_idx;
    
    // Each warp computes dot product of weights[room][:256] with input[:256]
    // Using WMMA: treat weight as 1×256 matrix, input as 256×1 vector
    // Or: weight as 16×16 tiles (1 row, padded), input as 16×16 tiles
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
    
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < dim; k += 16) {
        // Load 16 elements of weight (row 0 of a 16×16 tile)
        // We need 16×16, but weight row is only 16 elements wide per k-step
        // Padding: remaining 15 rows are zeros
        // This wastes 15/16 of tensor core capacity
        
        nvcuda::wmma::load_matrix_sync(a_frag, weights + room * dim + k, dim);
        
        // For B: input[k:k+16], broadcast to all 16 rows
        for (int j = 0; j < 16; j++)
            for (int i = 0; i < 16; i++)
                b_frag.x[j * 16 + i] = input[k + i];
        
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    if (lane == 0) {
        // All 16 elements of C[0][:] are the same (dot product)
        output[room] = c_frag.x[0];
    }
}

int main() {
    printf("=== Suite #37: Tensor Core WMMA for Room Inference ===\n\n");
    
    int batch_sizes[] = {16, 64, 256, 1024};
    
    half *d_weights, *d_input;
    float *d_out_v7, *d_out_wmma, *d_out_wmma_simple;
    cudaMalloc(&d_weights, 1024 * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_out_v7, 1024 * sizeof(float));
    cudaMalloc(&d_out_wmma, 1024 * sizeof(float));
    cudaMalloc(&d_out_wmma_simple, 1024 * sizeof(float));
    
    std::vector<half> hw(1024 * DIM), hi(DIM);
    for (int i = 0; i < 1024 * DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    
    cudaMemcpy(d_weights, hw.data(), 1024 * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    printf("%-8s | %-14s | %-14s | %-8s | %-14s | %-8s\n",
           "Rooms", "V7 (us)", "WMMA (us)", "V7/WMMA", "WMMA-simple", "V7/Simple");
    printf("---------|----------------|----------------|----------|----------------|----------\n");
    
    for (int b = 0; b < 4; b++) {
        int rooms = batch_sizes[b];
        dim3 gv7((rooms + 7) / 8);
        
        // V7 baseline
        for (int i = 0; i < WARMUP; i++)
            infer_v7<<<gv7, dim3(256), 0, stream>>>(d_weights, d_input, d_out_v7, rooms, DIM);
        cudaStreamSynchronize(stream);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_v7<<<gv7, dim3(256), 0, stream>>>(d_weights, d_input, d_out_v7, rooms, DIM);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        float v7_us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        
        // WMMA (16×16 block)
        dim3 gw((rooms + 15) / 16);
        for (int i = 0; i < WARMUP; i++)
            infer_wmma<<<gw, dim3(32, 16, 1), 0, stream>>>(d_weights, d_input, d_out_wmma, rooms, DIM);
        cudaStreamSynchronize(stream);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_wmma<<<gw, dim3(32, 16, 1), 0, stream>>>(d_weights, d_input, d_out_wmma, rooms, DIM);
        cudaStreamSynchronize(stream);
        t1 = std::chrono::high_resolution_clock::now();
        float wmma_us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        
        // WMMA simple (1 warp per room)
        for (int i = 0; i < WARMUP; i++)
            infer_wmma_simple<<<gv7, dim3(256), 0, stream>>>(d_weights, d_input, d_out_wmma_simple, rooms, DIM);
        cudaStreamSynchronize(stream);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_wmma_simple<<<gv7, dim3(256), 0, stream>>>(d_weights, d_input, d_out_wmma_simple, rooms, DIM);
        cudaStreamSynchronize(stream);
        t1 = std::chrono::high_resolution_clock::now();
        float ws_us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        
        printf("%-8d | %12.2f   | %12.2f   | %.3fx   | %12.2f   | %.3fx\n",
               rooms, v7_us, wmma_us, v7_us / wmma_us, ws_us, v7_us / ws_us);
    }
    
    // Verify correctness
    printf("\n--- Correctness Check ---\n");
    {
        int rooms = 16;
        dim3 gv7(2);
        
        infer_v7<<<gv7, dim3(256), 0, stream>>>(d_weights, d_input, d_out_v7, rooms, DIM);
        infer_wmma_simple<<<gv7, dim3(256), 0, stream>>>(d_weights, d_input, d_out_wmma_simple, rooms, DIM);
        cudaStreamSynchronize(stream);
        
        std::vector<float> ov(rooms), ow(rooms);
        cudaMemcpy(ov.data(), d_out_v7, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(ow.data(), d_out_wmma_simple, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_err = 0;
        for (int i = 0; i < rooms; i++)
            max_err = std::max(max_err, fabsf(ov[i] - ow[i]));
        
        printf("  V7 vs WMMA-simple max error: %.6f\n", max_err);
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_weights); cudaFree(d_input);
    cudaFree(d_out_v7); cudaFree(d_out_wmma); cudaFree(d_out_wmma_simple);
    
    printf("\n=== Suite #37 Complete ===\n");
    return 0;
}
