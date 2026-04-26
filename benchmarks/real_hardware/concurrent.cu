/**
 * Suite #54: Concurrent Kernel Execution
 * 
 * Can room inference run alongside other GPU work without degrading?
 * Fleet scenario: one Jetson serves inference + encoding + other tasks.
 * 
 * Tests:
 * 1. Inference alone (baseline)
 * 2. Inference + memory-intensive background (memcpy)
 * 3. Inference + compute-intensive background (matrix multiply)
 * 4. Inference + light compute background (activation functions)
 * 5. Inference + another inference kernel (multi-tenant)
 * 6. Measure throughput, latency, and interference for each combo
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 concurrent.cu -o concurrent
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cuda_fp16.h>

#define DIM 256
#define ROOMS 4096
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

// Background compute: heavy matrix multiply (compute-bound)
__global__ void bg_heavy_compute(float* data, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    for (int i = 0; i < iters; i++) {
        val = val * 0.999f + 0.001f * sinf(val);
    }
    data[idx] = val;
}

// Background compute: light (memory-bound, similar to inference)
__global__ void bg_light_compute(const half* weights, const half* input, float* output, int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < dim; i += 4)
        sum += __half2float(weights[idx * dim + i]) * __half2float(input[i]);
    output[idx] = sum;
}

// Background memory: D2D copy
__global__ void bg_memcpy(half* dst, const half* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// Background: GELU activation (activation-bound)
__global__ void bg_gelu(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = data[idx];
    float c = 0.7978845608f, a = 0.044715f;
    data[idx] = 0.5f * x * (1.0f + tanhf(c * (x + a * x * x * x)));
}

// Background: dummy inference (multi-tenant simulation)
__global__ void bg_inference(const half* w, const half* inp, float* out, int n, int dim) {
    int room = blockIdx.x;
    if (room >= n) return;
    int lane = threadIdx.x % 32;
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(w[room * dim + i]) * __half2float(inp[i]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) out[room] = sum;
}

std::atomic<bool> bg_running{false};

void run_inference(cudaStream_t stream, const half* weights, const half* input, float* output,
                   int rooms, int dim, float* out_us, float* out_p99) {
    dim3 grid((rooms + 7) / 8);
    
    // Warmup
    for (int i = 0; i < WARMUP; i++)
        infer_v7<<<grid, 256, 0, stream>>>(weights, input, output, rooms, dim);
    cudaStreamSynchronize(stream);
    
    // Timed run with p99 capture
    std::vector<float> latencies(ITERS);
    cudaEvent_t se, ee;
    cudaEventCreate(&se); cudaEventCreate(&ee);
    
    for (int i = 0; i < ITERS; i++) {
        cudaEventRecord(se, stream);
        infer_v7<<<grid, 256, 0, stream>>>(weights, input, output, rooms, dim);
        cudaEventRecord(ee, stream);
        cudaStreamSynchronize(stream);
        float ms;
        cudaEventElapsedTime(&ms, se, ee);
        latencies[i] = ms * 1000;  // to μs
    }
    
    cudaEventDestroy(se); cudaEventDestroy(ee);
    
    // Compute stats
    std::sort(latencies.begin(), latencies.end());
    float total = 0;
    for (auto& l : latencies) total += l;
    *out_us = total / ITERS;
    *out_p99 = latencies[(int)(ITERS * 0.99)];
}

int main() {
    printf("=== Suite #54: Concurrent Kernel Execution ===\n\n");
    
    // Allocate inference data
    half *d_weights, *d_input;
    float *d_output;
    cudaMalloc(&d_weights, ROOMS * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_output, ROOMS * sizeof(float));
    
    // Init
    std::vector<half> hw(ROOMS * DIM);
    for (size_t i = 0; i < (size_t)ROOMS * DIM; i++)
        hw[i] = __float2half(0.01f * (2.0f * (float)rand() / RAND_MAX - 1.0f));
    std::vector<half> hinput(DIM);
    for (int i = 0; i < DIM; i++)
        hinput[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    cudaMemcpy(d_weights, hw.data(), ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hinput.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Background buffers
    half *d_bg_w, *d_bg_in, *d_bg_src, *d_bg_dst;
    float *d_bg_data, *d_bg_out;
    int bg_size = 1 << 20;  // 1M elements
    cudaMalloc(&d_bg_w, bg_size * DIM * sizeof(half));
    cudaMalloc(&d_bg_in, DIM * sizeof(half));
    cudaMalloc(&d_bg_src, bg_size * DIM * sizeof(half));
    cudaMalloc(&d_bg_dst, bg_size * DIM * sizeof(half));
    cudaMalloc(&d_bg_data, bg_size * sizeof(float));
    cudaMalloc(&d_bg_out, bg_size * sizeof(float));
    
    // Init bg data
    std::vector<float> h_bg(bg_size);
    for (int i = 0; i < bg_size; i++) h_bg[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_bg_data, h_bg.data(), bg_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaStream_t infer_stream, bg_stream;
    cudaStreamCreate(&infer_stream);
    cudaStreamCreateWithFlags(&bg_stream, cudaStreamNonBlocking);
    
    dim3 infer_grid((ROOMS + 7) / 8);
    
    // ===== Test 1: Baseline (inference alone) =====
    printf("--- Test Results (4096 rooms, dim=256) ---\n");
    printf("%-35s | %8s | %10s | %8s | %8s\n",
           "Configuration", "Avg μs", "M qps", "p99 μs", "p99/p50");
    printf("-------------------------------------|----------|------------|----------|----------\n");
    
    float base_us, base_p99;
    run_inference(infer_stream, d_weights, d_input, d_output, ROOMS, DIM, &base_us, &base_p99);
    printf("%-35s | %8.2f | %10.1f | %8.2f | %8.2f\n",
           "Baseline (inference only)", base_us, ROOMS / base_us, base_p99, base_p99 / base_us);
    
    // ===== Test 2: Inference + Heavy compute background =====
    {
        bg_running = true;
        std::thread bg_thread([&]() {
            dim3 bg_grid(bg_size / 256);
            while (bg_running) {
                bg_heavy_compute<<<bg_grid, 256, 0, bg_stream>>>(d_bg_data, bg_size, 100);
                cudaStreamSynchronize(bg_stream);
            }
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float us, p99;
        run_inference(infer_stream, d_weights, d_input, d_output, ROOMS, DIM, &us, &p99);
        bg_running = false;
        bg_thread.join();
        
        printf("%-35s | %8.2f | %10.1f | %8.2f | %8.2f\n",
               "+ Heavy compute (sin loop)", us, ROOMS / us, p99, p99 / us);
    }
    
    // ===== Test 3: Inference + Light compute background =====
    {
        bg_running = true;
        std::thread bg_thread([&]() {
            dim3 bg_grid(bg_size / 256);
            while (bg_running) {
                bg_light_compute<<<bg_grid, 256, 0, bg_stream>>>(d_bg_w, d_bg_in, d_bg_out, 256, DIM);
                cudaStreamSynchronize(bg_stream);
            }
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float us, p99;
        run_inference(infer_stream, d_weights, d_input, d_output, ROOMS, DIM, &us, &p99);
        bg_running = false;
        bg_thread.join();
        
        printf("%-35s | %8.2f | %10.1f | %8.2f | %8.2f\n",
               "+ Light compute (matmul)", us, ROOMS / us, p99, p99 / us);
    }
    
    // ===== Test 4: Inference + Memory copy background =====
    {
        bg_running = true;
        std::thread bg_thread([&]() {
            dim3 bg_grid((bg_size * DIM) / 256);
            while (bg_running) {
                bg_memcpy<<<bg_grid, 256, 0, bg_stream>>>(d_bg_dst, d_bg_src, bg_size * DIM);
                cudaStreamSynchronize(bg_stream);
            }
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float us, p99;
        run_inference(infer_stream, d_weights, d_input, d_output, ROOMS, DIM, &us, &p99);
        bg_running = false;
        bg_thread.join();
        
        printf("%-35s | %8.2f | %10.1f | %8.2f | %8.2f\n",
               "+ Memory copy (D2D)", us, ROOMS / us, p99, p99 / us);
    }
    
    // ===== Test 5: Inference + GELU background =====
    {
        bg_running = true;
        std::thread bg_thread([&]() {
            dim3 bg_grid(bg_size / 256);
            while (bg_running) {
                bg_gelu<<<bg_grid, 256, 0, bg_stream>>>(d_bg_data, bg_size);
                cudaStreamSynchronize(bg_stream);
            }
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float us, p99;
        run_inference(infer_stream, d_weights, d_input, d_output, ROOMS, DIM, &us, &p99);
        bg_running = false;
        bg_thread.join();
        
        printf("%-35s | %8.2f | %10.1f | %8.2f | %8.2f\n",
               "+ GELU activation", us, ROOMS / us, p99, p99 / us);
    }
    
    // ===== Test 6: Inference + Another inference (multi-tenant) =====
    {
        bg_running = true;
        std::thread bg_thread([&]() {
            dim3 bg_grid2(256 / 8);
            while (bg_running) {
                bg_inference<<<bg_grid2, 256, 0, bg_stream>>>(d_bg_w, d_bg_in, d_bg_out, 256, DIM);
                cudaStreamSynchronize(bg_stream);
            }
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float us, p99;
        run_inference(infer_stream, d_weights, d_input, d_output, ROOMS, DIM, &us, &p99);
        bg_running = false;
        bg_thread.join();
        
        printf("%-35s | %8.2f | %10.1f | %8.2f | %8.2f\n",
               "+ Multi-tenant inference (256 rooms)", us, ROOMS / us, p99, p99 / us);
    }
    
    // ===== Test 7: Inference + Heavy compute (same stream — worst case) =====
    {
        dim3 bg_grid(bg_size / 256);
        dim3 combined_grid((ROOMS + 7) / 8);
        
        cudaEvent_t se, ee;
        cudaEventCreate(&se); cudaEventCreate(&ee);
        
        for (int i = 0; i < WARMUP; i++) {
            bg_heavy_compute<<<bg_grid, 256, 0, infer_stream>>>(d_bg_data, bg_size, 100);
            infer_v7<<<combined_grid, 256, 0, infer_stream>>>(d_weights, d_input, d_output, ROOMS, DIM);
        }
        cudaStreamSynchronize(infer_stream);
        
        cudaEventRecord(se, infer_stream);
        for (int i = 0; i < ITERS; i++) {
            bg_heavy_compute<<<bg_grid, 256, 0, infer_stream>>>(d_bg_data, bg_size, 100);
            infer_v7<<<combined_grid, 256, 0, infer_stream>>>(d_weights, d_input, d_output, ROOMS, DIM);
        }
        cudaEventRecord(ee, infer_stream);
        cudaStreamSynchronize(infer_stream);
        
        float ms;
        cudaEventElapsedTime(&ms, se, ee);
        float us = ms / ITERS * 1000;
        
        cudaEventDestroy(se); cudaEventDestroy(ee);
        
        printf("%-35s | %8.2f | %10.1f | %8s | %8s\n",
               "+ Heavy compute (SAME STREAM)", us, ROOMS / us, "n/a", "n/a");
    }
    
    // ===== Test 8: Inference + Host-to-Device copy (PCIe contention) =====
    {
        std::vector<half> h_copy(ROOMS * DIM);
        for (size_t i = 0; i < (size_t)ROOMS * DIM; i++)
            h_copy[i] = __float2half((float)rand() / RAND_MAX);
        
        bg_running = true;
        std::thread bg_thread([&]() {
            while (bg_running) {
                cudaMemcpy(d_bg_src, h_copy.data(), ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice);
            }
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float us, p99;
        run_inference(infer_stream, d_weights, d_input, d_output, ROOMS, DIM, &us, &p99);
        bg_running = false;
        bg_thread.join();
        
        printf("%-35s | %8.2f | %10.1f | %8.2f | %8.2f\n",
               "+ H2D copy (PCIe/Unified)", us, ROOMS / us, p99, p99 / us);
    }
    
    cudaStreamDestroy(infer_stream);
    cudaStreamDestroy(bg_stream);
    cudaFree(d_weights); cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_bg_w); cudaFree(d_bg_in); cudaFree(d_bg_src); cudaFree(d_bg_dst);
    cudaFree(d_bg_data); cudaFree(d_bg_out);
    
    printf("\n=== Suite #54 Complete ===\n");
    return 0;
}
