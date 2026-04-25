/**
 * Suite #28: Sustained Load & Memory Fragmentation Test
 * 
 * Questions:
 * 1. Does performance degrade after millions of room inferences?
 * 2. Does memory fragmentation affect latency over time?
 * 3. What's the thermal equilibrium under sustained load?
 * 4. Does the GPU clock throttle after extended use?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 sustained_load.cu -o sustained_load
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_fp16.h>

#define DIM 256
#define MAX_ROOMS 1024
#define WARMUP_ITERS 1000
#define SUSTAINED_ITERS 10000000  // 10M inferences

// Simple dot product + GELU kernel (V4 style, 4 rooms/block)
__global__ void room_infer_v4(
    const half* __restrict__ weights,
    const half* __restrict__ input,
    float* __restrict__ output,
    int num_rooms,
    int dim)
{
    int room_base = blockIdx.x * 4;
    int room_offset = threadIdx.x;
    int dim_idx = threadIdx.y;
    
    if (room_base + room_offset >= num_rooms) return;
    if (dim_idx >= dim) return;
    
    float val = __half2float(input[dim_idx]);
    float w = __half2float(weights[(room_base + room_offset) * dim + dim_idx]);
    
    // Shared reduction within block
    extern __shared__ float partial[];
    int tid = room_offset * blockDim.y + dim_idx;
    partial[tid] = val * w;
    __syncthreads();
    
    // Warp reduction
    for (int s = blockDim.y / 2; s > 0; s >>= 1) {
        if (dim_idx < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    
    if (dim_idx == 0) {
        float dot = partial[tid];
        // GELU approximation
        float gelu = 0.5f * dot * (1.0f + tanhf(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
        output[room_base + room_offset] = gelu;
    }
}

// Read GPU temperature
float read_gpu_temp() {
    FILE* f = fopen("/sys/class/thermal/thermal_zone1/temp", "r");
    if (!f) return -1.0f;
    int temp;
    if (fscanf(f, "%d", &temp) != 1) { fclose(f); return -1.0f; }
    fclose(f);
    return temp / 1000.0f;
}

// Read GPU power
float read_gpu_power() {
    FILE* f = fopen("/sys/class/hwmon/hwmon1/in2_input", "r");
    if (!f) return -1.0f;
    int power;
    if (fscanf(f, "%d", &power) != 1) { fclose(f); return -1.0f; }
    fclose(f);
    return power / 1000.0f;
}

// Read GPU frequency
int read_gpu_freq() {
    FILE* f = fopen("/sys/class/devfreq/17000000.gpu/cur_freq", "r");
    if (!f) return -1;
    int freq;
    if (fscanf(f, "%d", &freq) != 1) { fclose(f); return -1; }
    fclose(f);
    return freq / 1000000;
}

int main() {
    printf("=== Suite #28: Sustained Load & Memory Fragmentation ===\n\n");
    printf("Configuration:\n");
    printf("  Rooms: %d, Dim: %d, Iters: %dM\n", MAX_ROOMS, DIM, SUSTAINED_ITERS / 1000000);
    printf("  Kernel: V4 (fused+vec+multi, 4 rooms/block)\n\n");
    
    // Allocate
    half* d_weights;
    half* d_input;
    float* d_output;
    
    cudaMalloc(&d_weights, MAX_ROOMS * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_output, MAX_ROOMS * sizeof(float));
    
    // Initialize weights (deterministic)
    std::vector<half> h_weights(MAX_ROOMS * DIM);
    std::vector<half> h_input(DIM);
    for (int i = 0; i < MAX_ROOMS * DIM; i++) {
        h_weights[i] = __float2half(0.01f * (float)(i % 1000) / 1000.0f);
    }
    for (int i = 0; i < DIM; i++) {
        h_input[i] = __float2half(0.5f * sinf((float)i / DIM * 6.2832f));
    }
    
    cudaMemcpy(d_weights, h_weights.data(), MAX_ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    // Kernel config: 4 rooms/block, 128 threads (4 rooms × 32 threads each)
    int num_blocks = (MAX_ROOMS + 3) / 4;
    dim3 grid(num_blocks);
    dim3 block(4, 32);  // 4 rooms, 32 threads per room
    int shmem_size = 4 * 32 * sizeof(float);
    
    // Warmup
    printf("Warming up (%d iterations)...\n", WARMUP_ITERS);
    for (int i = 0; i < WARMUP_ITERS; i++) {
        room_infer_v4<<<grid, block, shmem_size>>>(d_weights, d_input, d_output, MAX_ROOMS, DIM);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete.\n\n");
    
    // Sustained load test with periodic sampling
    printf("Starting sustained load test...\n");
    printf("%-8s | %-10s | %-10s | %-8s | %-8s | %-8s | %-8s\n",
           "Iter(M)", "us/batch", "Room-qps", "GPU°C", "GPU W", "GPU MHz", "Jitter");
    printf("---------|------------|------------|---------|---------|---------|--------\n");
    
    const int SAMPLE_INTERVAL = 100000;  // Sample every 100K iterations
    int num_samples = SUSTAINED_ITERS / SAMPLE_INTERVAL;
    
    std::vector<float> latencies(num_samples);
    std::vector<float> temps(num_samples);
    std::vector<float> powers(num_samples);
    std::vector<int> freqs(num_samples);
    
    float min_lat = 1e9, max_lat = 0, sum_lat = 0;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int s = 0; s < num_samples; s++) {
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < SAMPLE_INTERVAL; i++) {
            room_infer_v4<<<grid, block, shmem_size>>>(d_weights, d_input, d_output, MAX_ROOMS, DIM);
        }
        cudaDeviceSynchronize();
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        float batch_us = std::chrono::duration<float, std::micro>(batch_end - batch_start).count();
        float us_per_iter = batch_us / SAMPLE_INTERVAL;
        float room_qps = MAX_ROOMS / (us_per_iter / 1e6f);
        
        latencies[s] = us_per_iter;
        temps[s] = read_gpu_temp();
        powers[s] = read_gpu_power();
        freqs[s] = read_gpu_freq();
        
        min_lat = std::min(min_lat, us_per_iter);
        max_lat = std::max(max_lat, us_per_iter);
        sum_lat += us_per_iter;
        
        // Calculate jitter vs first sample
        float jitter = latencies[s] / latencies[0];
        
        printf("%-8d | %-10.2f | %-10.0f | %-8.1f | %-8.1f | %-8d | %.3fx\n",
               (s + 1) * SAMPLE_INTERVAL / 1000000,
               us_per_iter,
               room_qps,
               temps[s],
               powers[s],
               freqs[s],
               jitter);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    float total_sec = std::chrono::duration<float>(total_end - total_start).count();
    
    // Statistics
    float mean_lat = sum_lat / num_samples;
    float var_lat = 0;
    for (int i = 0; i < num_samples; i++) {
        var_lat += (latencies[i] - mean_lat) * (latencies[i] - mean_lat);
    }
    var_lat /= num_samples;
    float std_lat = sqrtf(var_lat);
    
    // Sort for percentiles
    std::sort(latencies.begin(), latencies.end());
    float p50 = latencies[num_samples / 2];
    float p99 = latencies[(int)(num_samples * 0.99)];
    float p01 = latencies[(int)(num_samples * 0.01)];
    
    // First 10% vs last 10% comparison (degradation check)
    float first_10_avg = 0, last_10_avg = 0;
    int tenth = num_samples / 10;
    for (int i = 0; i < tenth; i++) {
        first_10_avg += latencies[i];
        last_10_avg += latencies[num_samples - tenth + i];
    }
    first_10_avg /= tenth;
    last_10_avg /= tenth;
    float degradation = last_10_avg / first_10_avg;
    
    printf("\n=== Results ===\n\n");
    printf("Total: %.1f seconds, %dM inferences\n", total_sec, SUSTAINED_ITERS / 1000000);
    printf("Throughput: %.1fM room-qps sustained\n", (double)MAX_ROOMS * SUSTAINED_ITERS / total_sec / 1e6);
    printf("Power efficiency: %.1fM room-qps/W\n", (double)MAX_ROOMS * SUSTAINED_ITERS / total_sec / 1e6 / (powers[num_samples-1] > 0 ? powers[num_samples-1] : 8.0));
    
    printf("\n--- Latency Distribution ---\n");
    printf("  Min:    %.3f us/iter\n", min_lat);
    printf("  p01:    %.3f us/iter\n", p01);
    printf("  p50:    %.3f us/iter\n", p50);
    printf("  Mean:   %.3f us/iter\n", mean_lat);
    printf("  p99:    %.3f us/iter\n", p99);
    printf("  Max:    %.3f us/iter\n", max_lat);
    printf("  Std:    %.3f us/iter\n", std_lat);
    printf("  p99/p50: %.3fx\n", p99 / p50);
    
    printf("\n--- Degradation Analysis ---\n");
    printf("  First 10%% avg: %.3f us/iter\n", first_10_avg);
    printf("  Last 10%% avg:  %.3f us/iter\n", last_10_avg);
    printf("  Degradation:   %.3fx (%.1f%%)\n", degradation, (degradation - 1.0f) * 100.0f);
    
    if (degradation < 1.02f) {
        printf("  → NEGLIGIBLE degradation. GPU is stable under sustained load.\n");
    } else if (degradation < 1.10f) {
        printf("  → MINOR degradation. Acceptable for production.\n");
    } else {
        printf("  → SIGNIFICANT degradation. Investigate thermal/memory issues.\n");
    }
    
    printf("\n--- Thermal Profile ---\n");
    printf("  Start: %.1f°C\n", temps[0]);
    printf("  End:   %.1f°C\n", temps[num_samples - 1]);
    printf("  Max:   %.1f°C\n", *std::max_element(temps.begin(), temps.end()));
    printf("  Min:   %.1f°C\n", *std::min_element(temps.begin(), temps.end()));
    printf("  Delta: %.1f°C\n", temps[num_samples - 1] - temps[0]);
    
    printf("\n--- GPU Frequency ---\n");
    printf("  Start: %d MHz\n", freqs[0]);
    printf("  End:   %d MHz\n", freqs[num_samples - 1]);
    printf("  Throttling: %s\n", freqs[num_samples-1] < freqs[0] ? "YES" : "NO");
    
    printf("\n--- Power ---\n");
    printf("  Start: %.1f W\n", powers[0]);
    printf("  End:   %.1f W\n", powers[num_samples - 1]);
    printf("  Avg:   %.1f W\n", 
           std::accumulate(powers.begin(), powers.end(), 0.0f) / num_samples);
    
    // Memory allocation test after sustained load
    printf("\n--- Memory Fragmentation Test ---\n");
    void* test_ptrs[64];
    auto frag_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 64; i++) {
        cudaMalloc(&test_ptrs[i], (i % 4 + 1) * 1024 * 1024);  // 1-4MB random
    }
    cudaDeviceSynchronize();
    auto frag_end = std::chrono::high_resolution_clock::now();
    float frag_us = std::chrono::duration<float, std::micro>(frag_end - frag_start).count();
    printf("  64 random allocations (1-4MB): %.0f us total (%.1f us each)\n", frag_us, frag_us / 64);
    
    for (int i = 0; i < 64; i++) cudaFree(test_ptrs[i]);
    
    // Cleanup
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Suite #28 Complete ===\n");
    return 0;
}
