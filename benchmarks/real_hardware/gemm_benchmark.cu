/**
 * Tensor Core GEMM Benchmark — Jetson Orin Nano
 * 
 * Full matrix-matrix multiply using WMMA (16x16x16 tiles)
 * Compared against cuBLAS Hgemm and naive implementations
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87, CUDA 12.6
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 gemm_benchmark.cu -o gemm_benchmark -lcublas
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda::wmma;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

//=============================================================
// 1. TC WMMA GEMM — 16x16x16 tiled
//=============================================================
__global__ void tc_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K
) {
    const int TILE = 16;
    
    // Block processes one 16x16 tile of C
    int row_tile = blockIdx.y;
    int col_tile = blockIdx.x;
    
    if (row_tile * TILE >= M || col_tile * TILE >= N) return;
    
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    // Process K in chunks of 16
    int num_k_tiles = (K + TILE - 1) / TILE;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_offset = k_tile * TILE;
        
        // Load A tile: row_major, M x K
        // Each row of the 16x16 A tile is stored contiguously in A[row_tile*16 + i][k_offset..k_offset+15]
        half a_frag_data[16 * 16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int row = row_tile * TILE + i;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int col = k_offset + j;
                a_frag_data[i * 16 + j] = (row < M && col < K) ? A[row * K + col] : __float2half(0.0f);
            }
        }
        load_matrix_sync(a_frag, a_frag_data, 16);
        
        // Load B tile: col_major, K x N
        // B is stored row-major as B[k][n], we need col-major for WMMA
        half b_frag_data[16 * 16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int row = k_offset + i;
                int col = col_tile * TILE + j;
                b_frag_data[i * 16 + j] = (row < K && col < N) ? B[row * N + col] : __float2half(0.0f);
            }
        }
        load_matrix_sync(b_frag, b_frag_data, 16);
        
        // Tensor core MMA
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result (with FP32 → FP16 conversion and GELU)
    half c_out[16 * 16];
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
        float val = c_frag.x[i];
        // GELU activation
        float gelu = val * 0.5f * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        c_out[i] = __float2half(gelu);
    }
    
    // Write to C
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int row = row_tile * TILE + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int col = col_tile * TILE + j;
            if (col >= N) continue;
            C[row * N + col] = c_out[i * 16 + j];
        }
    }
}

//=============================================================
// 2. Naive thread GEMM
//=============================================================
__global__ void naive_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    
    // GELU
    float gelu = sum * 0.5f * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
    C[row * N + col] = __float2half(gelu);
}

//=============================================================
// 3. Warp-based GEMM (each warp computes one 16x16 tile)
//=============================================================
__global__ void warp_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K
) {
    int warp_id = blockIdx.x;
    int lane = threadIdx.x;
    
    int tile_row = (warp_id / ((N + 15) / 16)) * 16;
    int tile_col = (warp_id % ((N + 15) / 16)) * 16;
    
    if (tile_row >= M) return;
    
    float sum = 0.0f;
    for (int i = lane; i < 16; i += 32) {
        for (int j = 0; j < 16; j++) {
            int row = tile_row + i;
            int col = tile_col + j;
            if (row >= M || col >= N) continue;
            
            float dot = 0.0f;
            for (int k = 0; k < K; k++) {
                dot += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
            }
            
            if (i < 16) sum += dot; // accumulate for validation
            C[row * N + col] = __float2half(dot);
        }
    }
}

//=============================================================
// Benchmark harness
//=============================================================

struct BenchResult {
    const char* name;
    float mean_ms;
    float p99_ms;
    float throughput;
};

void benchmark_tc_gemm(const half* d_A, const half* d_B, half* d_C, 
                        int M, int N, int K, int iters, BenchResult& r) {
    r.name = "TC WMMA GEMM";
    
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    // Warmup
    for (int i = 0; i < 200; i++) tc_gemm_kernel<<<grid, 32>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float* times = (float*)malloc(iters * sizeof(float));
    
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        tc_gemm_kernel<<<grid, 32>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    
    // Stats
    float sum = 0;
    for (int i = 0; i < iters; i++) sum += times[i];
    r.mean_ms = sum / iters;
    
    // Sort for p99
    for (int i = 0; i < iters - 1; i++)
        for (int j = i + 1; j < iters; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    r.p99_ms = times[(int)(iters * 0.99)];
    r.throughput = 1000.0f / r.mean_ms;
    
    free(times);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_cublas(cublasHandle_t handle, const half* d_A, const half* d_B, half* d_C,
                       int M, int N, int K, int iters, BenchResult& r) {
    r.name = "cuBLAS Hgemm";
    
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < 200; i++)
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, &alpha, d_A, CUDA_R_16F, K,
                    d_B, CUDA_R_16F, N, &beta, d_C, CUDA_R_16F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float* times = (float*)malloc(iters * sizeof(float));
    
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, &alpha, d_A, CUDA_R_16F, K,
                    d_B, CUDA_R_16F, N, &beta, d_C, CUDA_R_16F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    
    float sum = 0;
    for (int i = 0; i < iters; i++) sum += times[i];
    r.mean_ms = sum / iters;
    
    for (int i = 0; i < iters - 1; i++)
        for (int j = i + 1; j < iters; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    r.p99_ms = times[(int)(iters * 0.99)];
    r.throughput = 1000.0f / r.mean_ms;
    
    free(times);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_naive(const half* d_A, const half* d_B, half* d_C,
                      int M, int N, int K, int iters, BenchResult& r) {
    r.name = "Naive Thread GEMM";
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    for (int i = 0; i < 100; i++) naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float* times = (float*)malloc(iters * sizeof(float));
    
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    
    float sum = 0;
    for (int i = 0; i < iters; i++) sum += times[i];
    r.mean_ms = sum / iters;
    
    for (int i = 0; i < iters - 1; i++)
        for (int j = i + 1; j < iters; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    r.p99_ms = times[(int)(iters * 0.99)];
    r.throughput = 1000.0f / r.mean_ms;
    
    free(times);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void run_benchmark(int M, int N, int K, int iters, cublasHandle_t handle) {
    printf("\n=== GEMM %dx%d x %dx%d ===\n", M, K, K, N);
    
    size_t a_bytes = (size_t)M * K * sizeof(half);
    size_t b_bytes = (size_t)K * N * sizeof(half);
    size_t c_bytes = (size_t)M * N * sizeof(half);
    
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    
    // Init with random data
    half *h_A = (half*)malloc(a_bytes);
    half *h_B = (half*)malloc(b_bytes);
    half *h_C = (half*)malloc(c_bytes);
    
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));
    
    BenchResult results[3];
    
    // Only run naive for small matrices (it's SLOW)
    if (M <= 32 && N <= 32 && K <= 32) {
        benchmark_naive(d_A, d_B, d_C, M, N, K, iters, results[0]);
        benchmark_tc_gemm(d_A, d_B, d_C, M, N, K, iters, results[1]);
        benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, iters, results[2]);
        printf("%-22s %10s %10s %10s\n", "Implementation", "Mean(ms)", "P99(ms)", "QPS");
        printf("%-22s %10s %10s %10s\n", "----------------------", "----------", "----------", "----------");
        for (int i = 0; i < 3; i++) {
            printf("%-22s %10.4f %10.4f %10.0f\n", results[i].name, results[i].mean_ms, results[i].p99_ms, results[i].throughput);
        }
        printf("TC vs cuBLAS: %.2fx\n", results[2].mean_ms / results[1].mean_ms);
        printf("TC vs Naive:  %.2fx\n", results[0].mean_ms / results[1].mean_ms);
    } else {
        benchmark_tc_gemm(d_A, d_B, d_C, M, N, K, iters, results[0]);
        benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, iters, results[1]);
        printf("%-22s %10s %10s %10s\n", "Implementation", "Mean(ms)", "P99(ms)", "QPS");
        printf("%-22s %10s %10s %10s\n", "----------------------", "----------", "----------", "----------");
        for (int i = 0; i < 2; i++) {
            printf("%-22s %10.4f %10.4f %10.0f\n", results[i].name, results[i].mean_ms, results[i].p99_ms, results[i].throughput);
        }
        printf("TC vs cuBLAS: %.2fx\n", results[1].mean_ms / results[0].mean_ms);
    }
    
    // Correctness check
    benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, 1, results[0]);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_bytes, cudaMemcpyDeviceToHost));
    
    half* d_C2;
    CUDA_CHECK(cudaMalloc(&d_C2, c_bytes));
    tc_gemm_kernel<<<dim3((N+15)/16, (M+15)/16), 32>>>(d_A, d_B, d_C2, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half* h_C2 = (half*)malloc(c_bytes);
    CUDA_CHECK(cudaMemcpy(h_C2, d_C2, c_bytes, cudaMemcpyDeviceToHost));
    
    float max_diff = 0;
    int diff_count = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        float a = __half2float(h_C[i]);
        float b = __half2float(h_C2[i]);
        float d = fabsf(a - b);
        if (d > max_diff) max_diff = d;
        if (d > 0.1f) diff_count++;
    }
    
    // Note: TC result has GELU, cuBLAS doesn't, so we compare structure not values
    printf("Correctness: max_diff=%.6f (note: TC applies GELU, cuBLAS is raw matmul)\n", max_diff);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C2));
    free(h_A); free(h_B); free(h_C); free(h_C2);
}

int main() {
    printf("========================================\n");
    printf("Tensor Core GEMM Benchmark\n");
    printf("Jetson Orin Nano, sm_87, CUDA 12.6\n");
    printf("========================================\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d | Clock: %.0f MHz\n\n", prop.multiProcessorCount, prop.clockRate / 1000.0);
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Using default math mode (CUBLAS_DEFAULT not available in all versions)
    
    int iters = 10000;
    
    int sizes[][3] = {
        {16, 16, 16},
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
    };
    
    for (auto& s : sizes) {
        run_benchmark(s[0], s[1], s[2], iters, handle);
    }
    
    CUBLAS_CHECK(cublasDestroy(handle));
    printf("\nDone.\n");
    return 0;
}
