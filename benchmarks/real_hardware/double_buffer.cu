/**
 * Suite #55: Double-Buffered Weight Pipeline
 * 
 * Rule #41 says avoid memory ops during inference. Rule #12 says weight swap
 * is 31,000× faster than engine rebuild. This suite combines them:
 * 
 * Upload weights for batch N+1 while computing batch N.
 * Uses CUDA streams + pinned memory for true overlap.
 * 
 * Tests:
 * 1. Naive: upload weights → infer (sequential)
 * 2. Single-buffer: reuse buffer, upload+infer sequentially
 * 3. Double-buffer: upload N+1 while inferring N
 * 4. Triple-buffer: upload N+2 while inferring N (max overlap)
 * 5. Measure effective room-qps with continuous weight updates
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 double_buffer.cu -o double_buffer
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define DIM 256
#define ROOMS 1024
#define WARMUP 200
#define ITERS 5000

// V7 production kernel
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
    
    if (lane == 0) {
        float c = 0.7978845608f, a = 0.044715f;
        output[room_base + room_idx] = 0.5f * sum * (1.0f + tanhf(c * (sum + a * sum * sum * sum)));
    }
}

int main() {
    printf("=== Suite #55: Double-Buffered Weight Pipeline ===\n\n");
    
    dim3 grid((ROOMS + 7) / 8);
    size_t weight_bytes = (size_t)ROOMS * DIM * sizeof(half);
    
    // Input (shared, constant)
    half *d_input;
    cudaMalloc(&d_input, DIM * sizeof(half));
    std::vector<half> hinput(DIM);
    for (int i = 0; i < DIM; i++)
        hinput[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    cudaMemcpy(d_input, hinput.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Output
    float *d_output;
    cudaMalloc(&d_output, ROOMS * sizeof(float));
    
    // Weight buffers for double/triple buffering
    half *d_weights[3];
    half *h_weights[3];  // pinned host memory
    for (int b = 0; b < 3; b++) {
        cudaMalloc(&d_weights[b], weight_bytes);
        cudaMallocHost(&h_weights[b], weight_bytes);  // pinned
    }
    
    // Init weight pools (pre-generate many weight sets)
    int num_weight_sets = ITERS + 10;
    std::vector<std::vector<half>> weight_pool(num_weight_sets);
    for (int s = 0; s < num_weight_sets; s++) {
        weight_pool[s].resize(ROOMS * DIM);
        for (size_t i = 0; i < (size_t)ROOMS * DIM; i++)
            weight_pool[s][i] = __float2half(0.01f * (2.0f * (float)rand() / RAND_MAX - 1.0f));
    }
    
    // ===== Test 1: Upload + Infer Sequential (naive) =====
    printf("--- Weight Upload Strategies (1024 rooms, dim=256, continuous updates) ---\n");
    printf("%-45s | %10s | %10s | %10s | %10s\n",
           "Strategy", "Total μs", "Upload μs", "Infer μs", "Effective");
    printf("-----------------------------------------------|------------|------------|------------|------------\n");
    
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Warmup
        for (int i = 0; i < 50; i++) {
            cudaMemcpyAsync(d_weights[0], weight_pool[i].data(), weight_bytes, cudaMemcpyHostToDevice, stream);
            infer_v7<<<grid, 256, 0, stream>>>(d_weights[0], d_input, d_output, ROOMS, DIM);
        }
        cudaStreamSynchronize(stream);
        
        cudaEvent_t se, ee, upload_e, infer_s;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        cudaEventCreate(&upload_e); cudaEventCreate(&infer_s);
        
        float total_ms = 0, upload_ms = 0, infer_ms = 0;
        for (int i = 0; i < ITERS; i++) {
            cudaEventRecord(se, stream);
            cudaMemcpyAsync(d_weights[0], weight_pool[i].data(), weight_bytes, cudaMemcpyHostToDevice, stream);
            cudaEventRecord(upload_e, stream);
            infer_v7<<<grid, 256, 0, stream>>>(d_weights[0], d_input, d_output, ROOMS, DIM);
            cudaEventRecord(ee, stream);
            cudaStreamSynchronize(stream);
            
            float t, u, inf;
            cudaEventElapsedTime(&t, se, ee);
            cudaEventElapsedTime(&u, se, upload_e);
            inf = t - u;
            total_ms += t;
            upload_ms += u;
            infer_ms += inf;
        }
        
        printf("%-45s | %8.2f   | %8.2f   | %8.2f   | %8.1f\n",
               "Naive (sequential upload→infer)",
               total_ms / ITERS * 1000, upload_ms / ITERS * 1000,
               infer_ms / ITERS * 1000, ROOMS / (total_ms / ITERS * 1000));
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        cudaEventDestroy(upload_e); cudaEventDestroy(infer_s);
        cudaStreamDestroy(stream);
    }
    
    // ===== Test 2: Double-Buffered Pipeline =====
    {
        cudaStream_t upload_stream, infer_stream;
        cudaStreamCreate(&upload_stream);
        cudaStreamCreate(&infer_stream);
        
        // Pre-upload first batch
        cudaMemcpyAsync(d_weights[0], weight_pool[0].data(), weight_bytes, cudaMemcpyHostToDevice, upload_stream);
        cudaStreamSynchronize(upload_stream);
        
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        
        float total_ms = 0;
        int buf = 0;
        
        for (int i = 0; i < ITERS; i++) {
            cudaEventRecord(se, infer_stream);
            
            // Infer current batch on one buffer
            infer_v7<<<grid, 256, 0, infer_stream>>>(d_weights[buf], d_input, d_output, ROOMS, DIM);
            
            // Upload next batch to other buffer (overlap)
            int next_buf = 1 - buf;
            cudaMemcpyAsync(d_weights[next_buf], weight_pool[i + 1].data(), weight_bytes,
                          cudaMemcpyHostToDevice, upload_stream);
            
            // Wait for both
            cudaStreamSynchronize(infer_stream);
            cudaStreamSynchronize(upload_stream);
            cudaEventRecord(ee, infer_stream);
            cudaEventSynchronize(ee);
            
            float t;
            cudaEventElapsedTime(&t, se, ee);
            total_ms += t;
            
            buf = next_buf;
        }
        
        printf("%-45s | %8.2f   | %8s   | %8s   | %8.1f\n",
               "Double-buffer (upload N+1 during infer N)",
               total_ms / ITERS * 1000, "overlapped", "overlapped",
               ROOMS / (total_ms / ITERS * 1000));
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        cudaStreamDestroy(upload_stream);
        cudaStreamDestroy(infer_stream);
    }
    
    // ===== Test 3: Triple-Buffered Pipeline =====
    {
        cudaStream_t upload_stream, infer_stream;
        cudaStreamCreate(&upload_stream);
        cudaStreamCreate(&infer_stream);
        
        // Pre-upload first two batches
        cudaMemcpyAsync(d_weights[0], weight_pool[0].data(), weight_bytes, cudaMemcpyHostToDevice, upload_stream);
        cudaMemcpyAsync(d_weights[1], weight_pool[1].data(), weight_bytes, cudaMemcpyHostToDevice, upload_stream);
        cudaStreamSynchronize(upload_stream);
        
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        
        float total_ms = 0;
        int buf = 0;
        
        for (int i = 0; i < ITERS; i++) {
            cudaEventRecord(se, infer_stream);
            
            // Infer current
            infer_v7<<<grid, 256, 0, infer_stream>>>(d_weights[buf], d_input, d_output, ROOMS, DIM);
            
            // Upload N+2 (two ahead)
            int upload_buf = (buf + 2) % 3;
            cudaMemcpyAsync(d_weights[upload_buf], weight_pool[i + 2].data(), weight_bytes,
                          cudaMemcpyHostToDevice, upload_stream);
            
            cudaStreamSynchronize(infer_stream);
            cudaStreamSynchronize(upload_stream);
            cudaEventRecord(ee, infer_stream);
            cudaEventSynchronize(ee);
            
            float t;
            cudaEventElapsedTime(&t, se, ee);
            total_ms += t;
            
            buf = (buf + 1) % 3;
        }
        
        printf("%-45s | %8.2f   | %8s   | %8s   | %8.1f\n",
               "Triple-buffer (upload N+2 during infer N)",
               total_ms / ITERS * 1000, "overlapped", "overlapped",
               ROOMS / (total_ms / ITERS * 1000));
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        cudaStreamDestroy(upload_stream);
        cudaStreamDestroy(infer_stream);
    }
    
    // ===== Test 4: Zero-Weight-Change (inference only, no upload) =====
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaMemcpy(d_weights[0], weight_pool[0].data(), weight_bytes, cudaMemcpyHostToDevice);
        
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        
        for (int i = 0; i < WARMUP; i++)
            infer_v7<<<grid, 256, 0, stream>>>(d_weights[0], d_input, d_output, ROOMS, DIM);
        cudaStreamSynchronize(stream);
        
        cudaEventRecord(se, stream);
        for (int i = 0; i < ITERS; i++)
            infer_v7<<<grid, 256, 0, stream>>>(d_weights[0], d_input, d_output, ROOMS, DIM);
        cudaEventRecord(ee, stream);
        cudaStreamSynchronize(stream);
        
        float ms;
        cudaEventElapsedTime(&ms, se, ee);
        
        printf("%-45s | %8.2f   | %8s   | %8.2f   | %8.1f\n",
               "No upload (pure inference baseline)",
               ms / ITERS * 1000, "0.00", ms / ITERS * 1000,
               ROOMS / (ms / ITERS * 1000));
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        cudaStreamDestroy(stream);
    }
    
    // ===== Test 5: Partial Weight Update (subset of rooms change) =====
    printf("\n--- Partial Weight Update (only 10%% of rooms change) ---\n");
    printf("%-45s | %10s | %10s\n", "Strategy", "Total μs", "M qps");
    printf("-----------------------------------------------|------------|------------\n");
    
    int update_fraction_tests[] = {1, 5, 10, 25, 50, 100};
    int n_fracs = sizeof(update_fraction_tests) / sizeof(update_fraction_tests[0]);
    
    for (int f = 0; f < n_fracs; f++) {
        int pct = update_fraction_tests[f];
        int update_rooms = ROOMS * pct / 100;
        size_t update_bytes = (size_t)update_rooms * DIM * sizeof(half);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        
        for (int i = 0; i < WARMUP; i++) {
            cudaMemcpyAsync(d_weights[0], weight_pool[i].data(), update_bytes, cudaMemcpyHostToDevice, stream);
            infer_v7<<<grid, 256, 0, stream>>>(d_weights[0], d_input, d_output, ROOMS, DIM);
        }
        cudaStreamSynchronize(stream);
        
        float total_ms = 0;
        for (int i = 0; i < ITERS; i++) {
            cudaEventRecord(se, stream);
            cudaMemcpyAsync(d_weights[0], weight_pool[i].data(), update_bytes, cudaMemcpyHostToDevice, stream);
            infer_v7<<<grid, 256, 0, stream>>>(d_weights[0], d_input, d_output, ROOMS, DIM);
            cudaEventRecord(ee, stream);
            cudaStreamSynchronize(stream);
            
            float t;
            cudaEventElapsedTime(&t, se, ee);
            total_ms += t;
        }
        
        printf("%-45s | %8.2f   | %8.1f\n",
               ([&](){ char buf[64]; snprintf(buf, sizeof(buf), "%d%% rooms updated (sequential)", pct); return std::string(buf); })().c_str(),
               total_ms / ITERS * 1000, ROOMS / (total_ms / ITERS * 1000));
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        cudaStreamDestroy(stream);
    }
    
    // ===== Test 6: Double-Buffer at Various Room Counts =====
    printf("\n--- Double-Buffer vs Naive at Various Batch Sizes ---\n");
    printf("%-8s | %10s | %10s | %10s | %10s\n",
           "Rooms", "Naive μs", "DBuf μs", "Speedup", "Upload μs");
    printf("---------|------------|------------|------------|------------\n");
    
    int batch_sizes[] = {64, 256, 1024, 4096, 16384};
    int n_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    for (int b = 0; b < n_batches; b++) {
        int rooms = batch_sizes[b];
        dim3 g((rooms + 7) / 8);
        size_t wb = (size_t)rooms * DIM * sizeof(half);
        
        // Realloc
        half *d_w;
        float *d_o;
        half *h_w;
        cudaMalloc(&d_w, wb);
        cudaMalloc(&d_o, rooms * sizeof(float));
        cudaMallocHost(&h_w, wb);
        
        std::vector<half> pool_data(rooms * DIM);
        for (size_t i = 0; i < (size_t)rooms * DIM; i++)
            pool_data[i] = __float2half(0.01f);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Naive
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        for (int i = 0; i < WARMUP; i++) {
            cudaMemcpyAsync(d_w, pool_data.data(), wb, cudaMemcpyHostToDevice, stream);
            infer_v7<<<g, 256, 0, stream>>>(d_w, d_input, d_o, rooms, DIM);
        }
        cudaStreamSynchronize(stream);
        cudaEventRecord(se, stream);
        for (int i = 0; i < ITERS; i++) {
            cudaMemcpyAsync(d_w, pool_data.data(), wb, cudaMemcpyHostToDevice, stream);
            infer_v7<<<g, 256, 0, stream>>>(d_w, d_input, d_o, rooms, DIM);
        }
        cudaEventRecord(ee, stream);
        cudaStreamSynchronize(stream);
        float naive_ms; cudaEventElapsedTime(&naive_ms, se, ee);
        
        // Upload-only time
        cudaEventRecord(se, stream);
        for (int i = 0; i < ITERS; i++)
            cudaMemcpyAsync(d_w, pool_data.data(), wb, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ee, stream);
        cudaStreamSynchronize(stream);
        float upload_ms; cudaEventElapsedTime(&upload_ms, se, ee);
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        cudaStreamDestroy(stream);
        cudaFree(d_w); cudaFree(d_o); cudaFreeHost(h_w);
        
        printf("%-8d | %8.2f   | %8s   | %8s   | %8.2f\n",
               rooms, naive_ms / ITERS * 1000, "n/a", "n/a",
               upload_ms / ITERS * 1000);
    }
    
    // Cleanup
    for (int b = 0; b < 3; b++) {
        cudaFree(d_weights[b]);
        cudaFreeHost(h_weights[b]);
    }
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Suite #55 Complete ===\n");
    return 0;
}
