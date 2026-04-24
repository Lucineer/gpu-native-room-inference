/**
 * Tensor Core GEMM Benchmark v2 — Jetson Orin Nano
 * 
 * Uses shared memory for WMMA loads (local memory causes launch failure on sm_87)
 * Compared against cuBLAS Hgemm
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87, CUDA 12.6
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 gemm_benchmark_v2.cu -o gemm_benchmark_v2 -lcublas
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
// TC WMMA GEMM — shared memory tiled
// Each block: 1 warp, 16x16 tile of C
//=============================================================
__global__ void tc_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K
) {
    // Shared memory for loading A and B tiles
    __shared__ half s_A[16 * 16];  // 16x16 tile of A (row-major)
    __shared__ half s_B[16 * 16];  // 16x16 tile of B (col-major layout)
    
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    int tile_row = blockIdx.y * 16;
    int tile_col = blockIdx.x * 16;
    int lane = threadIdx.x;
    
    if (tile_row >= M || tile_col >= N) return;
    
    fill_fragment(c_frag, 0.0f);
    
    int num_k_tiles = (K + 15) / 16;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_offset = k_tile * 16;
        
        // Load A tile into shared memory
        // A is M x K row-major: A[row * K + col]
        // We need 16 rows starting at tile_row, 16 cols starting at k_offset
        // Lane i loads row (lane/16), col (lane%16) — but we have 32 lanes for 16x16
        if (lane < 16) {
            for (int r = lane; r < 16; r += 16) {
                for (int c = 0; c < 16; c++) {
                    int row = tile_row + r;
                    int col = k_offset + c;
                    s_A[r * 16 + c] = (row < M && col < K) ? A[row * K + col] : __float2half(0.0f);
                }
            }
        }
        
        // Load B tile into shared memory (in col-major order for WMMA)
        // B is K x N row-major: B[k * N + n]
        // We need 16 rows at k_offset, 16 cols at tile_col
        if (lane >= 16) {
            for (int r = 0; r < 16; r++) {
                for (int c = lane - 16; c < 16; c += 16) {
                    int row = k_offset + r;
                    int col = tile_col + c;
                    // Store in col-major layout: s_B[row + col*16]
                    s_B[r + c * 16] = (row < K && col < N) ? B[row * N + col] : __float2half(0.0f);
                }
            }
        }
        
        __syncwarp();
        
        // Load fragments from shared memory
        load_matrix_sync(a_frag, s_A, 16);   // row-major, leading dim 16
        load_matrix_sync(b_frag, s_B, 16);   // col-major, leading dim 16
        
        // Tensor core MMA
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result using shared memory (to avoid local memory)
    __shared__ float s_C[16 * 16];
    store_matrix_sync(s_C, c_frag, 16, mem_row_major);
    
    // Write to global memory with GELU activation
    __syncwarp();
    if (lane < 16) {
        for (int r = lane; r < 16; r += 16) {
            for (int c = 0; c < 16; c++) {
                int row = tile_row + r;
                int col = tile_col + c;
                if (row < M && col < N) {
                    float val = s_C[r * 16 + c];
                    float gelu = val * 0.5f * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
                    C[row * N + col] = __float2half(gelu);
                }
            }
        }
    }
}

//=============================================================
// Naive thread GEMM
//=============================================================
__global__ void naive_gemm_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    const int M, const int N, const int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    
    float gelu = sum * 0.5f * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
    C[row * N + col] = __float2half(gelu);
}

//=============================================================
// Benchmark helpers
//=============================================================
struct BenchResult {
    const char* name;
    float mean_ms;
    float p99_ms;
    float throughput;
};

void compute_stats(float* times, int n, BenchResult& r) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += times[i];
    r.mean_ms = sum / n;
    
    // Simple sort for percentile
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    r.p99_ms = times[(int)(n * 0.99)];
    r.throughput = 10000.0f / r.mean_ms;  // ops per second (1 GEMM = 1 op)
}

void benchmark_tc(const half* d_A, const half* d_B, half* d_C,
                  int M, int N, int K, int iters, BenchResult& r) {
    r.name = "TC WMMA GEMM";
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
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
    compute_stats(times, iters, r);
    free(times);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_cublas(cublasHandle_t handle, const half* d_A, const half* d_B, half* d_C,
                      int M, int N, int K, int iters, BenchResult& r) {
    r.name = "cuBLAS Hgemm";
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    
    for (int i = 0; i < 200; i++)
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, &alpha, d_A, CUDA_R_16F, K,
                    d_B, CUDA_R_16F, N, &beta, d_C, CUDA_R_16F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    compute_stats(times, iters, r);
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
    compute_stats(times, iters, r);
    free(times);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void run_benchmark(int M, int N, int K, int iters, cublasHandle_t handle) {
    printf("\n=== GEMM %dx%d x %dx%d (%d total mults) ===\n", M, K, K, N, M * N * K);
    
    size_t a_bytes = (size_t)M * K * sizeof(half);
    size_t b_bytes = (size_t)K * N * sizeof(half);
    size_t c_bytes = (size_t)M * N * sizeof(half);
    
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    
    half *h_A = (half*)malloc(a_bytes);
    half *h_B = (half*)malloc(b_bytes);
    
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));
    
    printf("%-22s %10s %10s %10s\n", "Implementation", "Mean(ms)", "P99(ms)", "QPS");
    printf("%-22s %10s %10s %10s\n", "----------------------", "----------", "----------", "----------");
    
    int n_impls = 0;
    BenchResult results[3];
    
    // TC always
    benchmark_tc(d_A, d_B, d_C, M, N, K, iters, results[n_impls++]);
    
    // cuBLAS always
    benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, iters, results[n_impls++]);
    
    // Naive only for small
    bool ran_naive = false;
    if (M <= 32 && N <= 32) {
        benchmark_naive(d_A, d_B, d_C, M, N, K, iters, results[n_impls++]);
        ran_naive = true;
    }
    
    for (int i = 0; i < n_impls; i++)
        printf("%-22s %10.4f %10.4f %10.0f\n", results[i].name, results[i].mean_ms, results[i].p99_ms, results[i].throughput);
    
    printf("\n  TC vs cuBLAS: %.2fx\n", results[1].mean_ms / results[0].mean_ms);
    if (ran_naive)
        printf("  TC vs Naive:  %.2fx\n", results[2].mean_ms / results[0].mean_ms);
    
    // Compute FLOPS for cuBLAS (reference)
    double flops = 2.0 * M * N * K;  // multiply-add = 2 FLOPs
    double cuBLAS_gflops = (flops / (results[1].mean_ms * 1e-3)) / 1e9;
    double tc_gflops = (flops / (results[0].mean_ms * 1e-3)) / 1e9;
    printf("  cuBLAS: %.1f GFLOPS | TC: %.1f GFLOPS\n", cuBLAS_gflops, tc_gflops);
    printf("  Orin peak FP16 TC: ~40 TFLOPS (theoretical)\n");
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
}

int main() {
    printf("========================================\n");
    printf("TC GEMM Benchmark v2 (shared mem)\n");
    printf("Jetson Orin Nano, sm_87, CUDA 12.6\n");
    printf("========================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Clock: %.0f MHz\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("Shared mem/block: %.0f KB\n\n", prop.sharedMemPerBlock / 1024.0);
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
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
    
    // Non-square test
    run_benchmark(64, 256, 128, iters, handle);
    run_benchmark(256, 64, 128, iters, handle);
    
    CUBLAS_CHECK(cublasDestroy(handle));
    printf("\n=== BENCHMARK COMPLETE ===\n");
    return 0;
}
