/**
 * Suite #52: Sparse Kernel — Skip Zero Weights
 * 
 * Suite #51 showed the dense V7 kernel ignores sparsity (1.00x).
 * A sparse kernel that skips zero weights should be faster for
 * high-sparsity models (pruned networks, MoE sparse experts).
 * 
 * Approach: compact non-zero weights into a CSR-like format.
 * Each room has: [count, indices[count], values[count]]
 * Kernel iterates only over non-zero weights.
 * 
 * Questions:
 * 1. At what sparsity level does sparse kernel beat dense?
 * 2. What's the overhead of the index lookup?
 * 3. Can we use warp-cooperative sparse iteration?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 sparse_kernel.cu -o sparse_kernel
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#define DIM 256
#define WARMUP 500
#define ITERS 10000

// Dense V7 baseline
__global__ void infer_dense(
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
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room_base + room_idx] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

// Sparse kernel: CSR format
// sparse_data[room] = [nnz, idx0, val0, idx1, val1, ...]
// Each room's sparse data is packed: 1 int (nnz) + nnz * (1 int + 1 half)
struct SparseEntry { int idx; half val; };

__global__ void infer_sparse(
    const SparseEntry* __restrict__ sparse_data,  // [num_rooms * max_nnz]
    const int* __restrict__ nnz_per_room,         // [num_rooms]
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    int nnz = nnz_per_room[room];
    const SparseEntry* entries = sparse_data + room * DIM;  // max alloc per room
    
    float sum = 0.0f;
    // Each thread processes a subset of non-zero entries
    for (int i = lane; i < nnz; i += 32) {
        float w = __half2float(entries[i].val);
        float x = __half2float(input[entries[i].idx]);
        sum += w * x;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

// Sparse kernel v2: warp-cooperative, 32 threads share the nnz iteration
__global__ void infer_sparse_coop(
    const SparseEntry* __restrict__ sparse_data,
    const int* __restrict__ nnz_per_room,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    int nnz = nnz_per_room[room];
    const SparseEntry* entries = sparse_data + room * DIM;
    
    float sum = 0.0f;
    // Warp-cooperative: each thread handles entries[i*32 + lane]
    for (int i = 0; i * 32 + lane < nnz; i++) {
        int idx = i * 32 + lane;
        float w = __half2float(entries[idx].val);
        float x = __half2float(input[entries[idx].idx]);
        sum += w * x;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

// Sparse kernel v3: sorted indices, vectorized input access
// If indices are sorted, adjacent threads may access nearby input elements
// (better L2 cache behavior)
__global__ void infer_sparse_sorted(
    const int* __restrict__ indices,    // [num_rooms * max_nnz]
    const half* __restrict__ values,    // [num_rooms * max_nnz]
    const int* __restrict__ nnz_per_room,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    int nnz = nnz_per_room[room];
    int base = room * DIM;
    
    float sum = 0.0f;
    for (int i = lane; i < nnz; i += 32) {
        float w = __half2float(values[base + i]);
        float x = __half2float(input[indices[base + i]]);
        sum += w * x;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

int main() {
    printf("=== Suite #52: Sparse Kernel — Skip Zero Weights ===\n\n");
    
    int rooms = 1024;
    
    // Allocate dense
    half *d_dense_w, *d_input;
    float *d_out;
    cudaMalloc(&d_dense_w, rooms * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_out, rooms * sizeof(float));
    
    // Allocate sparse
    SparseEntry *d_sparse_data;
    int *d_nnz;
    int *d_indices;
    half *d_values;
    cudaMalloc(&d_sparse_data, rooms * DIM * sizeof(SparseEntry));
    cudaMalloc(&d_nnz, rooms * sizeof(int));
    cudaMalloc(&d_indices, rooms * DIM * sizeof(int));
    cudaMalloc(&d_values, rooms * DIM * sizeof(half));
    
    // Init input
    std::vector<half> hi(DIM);
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    dim3 grid((rooms + 7) / 8);
    
    // ===== Test at various sparsity levels =====
    printf("--- Sparse vs Dense Across Sparsity Levels (1024 rooms, dim=256) ---\n");
    printf("%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n",
           "Sparsity", "Dense", "Sparse1", "Ratio", "Coop", "Ratio", "Sorted", "Ratio");
    printf("----------|----------|----------|----------|----------|----------|----------|----------\n");
    
    float sparsity_levels[] = {0.0f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f};
    unsigned int seed = 42;
    
    for (int s = 0; s < 7; s++) {
        float sparsity = sparsity_levels[s];
        
        // Generate weights with given sparsity
        std::vector<half> hw(rooms * DIM);
        std::vector<SparseEntry> sparse(rooms * DIM);
        std::vector<int> nnz_arr(rooms);
        std::vector<int> idx_arr(rooms * DIM);
        std::vector<half> val_arr(rooms * DIM);
        
        for (int r = 0; r < rooms; r++) {
            int count = 0;
            for (int d = 0; d < DIM; d++) {
                float v = 0.0f;
                if ((float)rand_r(&seed) / RAND_MAX >= sparsity) {
                    v = 0.01f * (2.0f * (float)rand_r(&seed) / RAND_MAX - 1.0f);
                }
                int idx = r * DIM + d;
                hw[idx] = __float2half(v);
                
                if (fabsf(v) > 1e-10f) {
                    sparse[idx] = {(int)d, __float2half(v)};
                    idx_arr[idx] = d;
                    val_arr[idx] = __float2half(v);
                    count++;
                }
            }
            nnz_arr[r] = count;
        }
        
        cudaMemcpy(d_dense_w, hw.data(), rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sparse_data, sparse.data(), rooms * DIM * sizeof(SparseEntry), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nnz, nnz_arr.data(), rooms * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, idx_arr.data(), rooms * DIM * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, val_arr.data(), rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
        
        float dense_us, sparse_us, coop_us, sorted_us;
        
        // Dense
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_dense<<<grid,256,0,stream>>>(d_dense_w,d_input,d_out,rooms,DIM);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_dense<<<grid,256,0,stream>>>(d_dense_w,d_input,d_out,rooms,DIM);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); dense_us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // Sparse v1
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_sparse<<<grid,256,0,stream>>>(d_sparse_data,d_nnz,d_input,d_out,rooms);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_sparse<<<grid,256,0,stream>>>(d_sparse_data,d_nnz,d_input,d_out,rooms);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); sparse_us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // Sparse cooperative
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_sparse_coop<<<grid,256,0,stream>>>(d_sparse_data,d_nnz,d_input,d_out,rooms);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_sparse_coop<<<grid,256,0,stream>>>(d_sparse_data,d_nnz,d_input,d_out,rooms);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); coop_us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        // Sparse sorted (SoA)
        { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
          for (int i=0;i<WARMUP;i++) infer_sparse_sorted<<<grid,256,0,stream>>>(d_indices,d_values,d_nnz,d_input,d_out,rooms);
          cudaStreamSynchronize(stream); cudaEventRecord(s,stream);
          for (int i=0;i<ITERS;i++) infer_sparse_sorted<<<grid,256,0,stream>>>(d_indices,d_values,d_nnz,d_input,d_out,rooms);
          cudaEventRecord(e,stream); cudaStreamSynchronize(stream);
          float ms; cudaEventElapsedTime(&ms,s,e); sorted_us=ms/ITERS*1000;
          cudaEventDestroy(s); cudaEventDestroy(e); }
        
        printf("%-10.0f%% | %6.2f   | %6.2f   | %.3fx  | %6.2f   | %.3fx  | %6.2f   | %.3fx\n",
               sparsity * 100, dense_us, sparse_us, dense_us/sparse_us,
               coop_us, dense_us/coop_us, sorted_us, dense_us/sorted_us);
    }
    
    // ===== Memory usage comparison =====
    printf("\n--- Memory Usage Comparison ---\n");
    printf("Dense:  %d rooms × %d dim × 2 bytes = %.1f KB\n", rooms, DIM, (float)rooms * DIM * 2 / 1024);
    for (int s = 0; s < 7; s++) {
        float sp = sparsity_levels[s];
        int nnz_avg = (int)(DIM * (1.0f - sp));
        float sparse_mem = (float)rooms * nnz_avg * (sizeof(int) + sizeof(half)) + rooms * sizeof(int);
        float dense_mem = (float)rooms * DIM * sizeof(half);
        printf("Sparse %3.0f%%: ~%d nnz/room, %.1f KB (%.1fx vs dense)\n",
               sp * 100, nnz_avg, sparse_mem / 1024, dense_mem / sparse_mem);
    }
    
    // ===== Correctness =====
    printf("\n--- Correctness Check (50%% sparse, first 4 rooms) ---\n");
    {
        // Generate 50% sparse
        seed = 123;
        std::vector<half> hw_c(rooms * DIM);
        std::vector<SparseEntry> sp_c(rooms * DIM);
        std::vector<int> nnz_c(rooms);
        for (int r = 0; r < rooms; r++) {
            int cnt = 0;
            for (int d = 0; d < DIM; d++) {
                float v = 0.0f;
                if ((float)rand_r(&seed) / RAND_MAX >= 0.5f)
                    v = 0.01f * (2.0f * (float)rand_r(&seed) / RAND_MAX - 1.0f);
                hw_c[r * DIM + d] = __float2half(v);
                if (fabsf(v) > 1e-10f) { sp_c[r * DIM + cnt] = {d, __float2half(v)}; cnt++; }
            }
            nnz_c[r] = cnt;
        }
        
        cudaMemcpy(d_dense_w, hw_c.data(), rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sparse_data, sp_c.data(), rooms * DIM * sizeof(SparseEntry), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nnz, nnz_c.data(), rooms * sizeof(int), cudaMemcpyHostToDevice);
        
        infer_dense<<<1, 256, 0, stream>>>(d_dense_w, d_input, d_out, rooms, DIM);
        std::vector<float> out_dense(rooms);
        cudaMemcpy(out_dense.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        infer_sparse<<<1, 256, 0, stream>>>(d_sparse_data, d_nnz, d_input, d_out, rooms);
        std::vector<float> out_sparse(rooms);
        cudaMemcpy(out_sparse.data(), d_out, rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int r = 0; r < 4; r++) {
            float err = fabsf(out_dense[r] - out_sparse[r]);
            printf("  Room %d: dense=%.8f, sparse=%.8f, diff=%.2e\n",
                   r, out_dense[r], out_sparse[r], err);
        }
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_dense_w); cudaFree(d_input); cudaFree(d_out);
    cudaFree(d_sparse_data); cudaFree(d_nnz);
    cudaFree(d_indices); cudaFree(d_values);
    
    printf("\n=== Suite #52 Complete ===\n");
    return 0;
}
