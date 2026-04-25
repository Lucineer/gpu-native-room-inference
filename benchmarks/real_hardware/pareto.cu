/**
 * Suite #43: Pareto Frontier — Latency vs Throughput Tradeoff
 * 
 * The practical question every fleet operator asks:
 * "I need X μs latency. What's the maximum throughput I can get?"
 * 
 * This maps the complete Pareto frontier using the production 
 * double-buffered async pipeline (suite #33) + zero-copy output.
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 pareto.cu -o pareto
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_fp16.h>

#define DIM 256
#define WARMUP 1000
#define ITERS 100000

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

struct Percentiles {
    float p50, p90, p95, p99, p999, max_val;
};

Percentiles compute_pctls(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    int n = v.size();
    return {
        v[n/2],
        v[(int)(n*0.90)],
        v[(int)(n*0.95)],
        v[(int)(n*0.99)],
        v[(int)(n*0.999)],
        v[n-1]
    };
}

int main() {
    printf("=== Suite #43: Pareto Frontier — Latency vs Throughput ===\n\n");
    
    int room_counts[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 
                         384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    int num_sizes = sizeof(room_counts) / sizeof(room_counts[0]);
    
    half *d_weights, *d_input;
    float *d_out[2];
    float *h_out[2];
    float *d_zc[2];
    
    cudaMalloc(&d_weights, 8192 * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    
    for (int b = 0; b < 2; b++) {
        cudaMalloc(&d_out[b], 8192 * sizeof(float));
        cudaHostAlloc(&h_out[b], 8192 * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_zc[b], h_out[b], 0);
    }
    
    std::vector<half> hw(8192 * DIM), hi(DIM);
    for (int i = 0; i < 8192 * DIM; i++)
        hw[i] = __float2half(0.01f * sinf((float)(i % 997) / 997.0f));
    for (int i = 0; i < DIM; i++)
        hi[i] = __float2half(0.5f * cosf((float)i / DIM * 6.2832f));
    
    cudaMemcpy(d_weights, hw.data(), 8192 * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, hi.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // ===== Part 1: Complete Pareto Table =====
    printf("=== Double-Buffered Async Pipeline (Production) ===\n");
    printf("%-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-10s | %-10s\n",
           "Rooms", "p50(μs)", "p90(μs)", "p95(μs)", "p99(μs)", "p999", "qps(M)", "Efficiency");
    printf("---------|----------|----------|----------|----------|----------|------------|------------\n");
    
    struct Result { int rooms; Percentiles p; float qps; float efficiency; };
    std::vector<Result> results;
    
    for (int s = 0; s < num_sizes; s++) {
        int rooms = room_counts[s];
        dim3 grid((rooms + 7) / 8);
        
        std::vector<float> lats(ITERS);
        
        // Warmup
        for (int i = 0; i < 200; i++) {
            int buf = i % 2;
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc[buf], rooms, DIM);
            if (i > 0) volatile float sink = h_out[1-buf][0];
        }
        cudaStreamSynchronize(stream);
        
        // Measure
        for (int i = 0; i < ITERS; i++) {
            int buf = i % 2;
            auto t0 = std::chrono::high_resolution_clock::now();
            infer_v7<<<grid, dim3(256), 0, stream>>>(d_weights, d_input, d_zc[buf], rooms, DIM);
            if (i > 0) volatile float sink = h_out[1-buf][0];
            auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
        }
        cudaStreamSynchronize(stream);
        
        Percentiles p = compute_pctls(lats);
        float qps = rooms / (p.p50 / 1e6f) / 1e6f;
        // Efficiency = qps / theoretical_max (149M from suite #40)
        float efficiency = qps / 149.0f * 100.0f;
        
        printf("%-8d | %6.2f   | %6.2f   | %6.2f   | %6.2f   | %6.2f   | %8.2f   | %6.1f%%\n",
               rooms, p.p50, p.p90, p.p95, p.p99, p.p999, qps, efficiency);
        
        results.push_back({rooms, p, qps, efficiency});
    }
    
    // ===== Part 2: Latency SLA Lookup =====
    printf("\n=== SLA Lookup: Max Throughput at Given Latency ===\n\n");
    
    float sla_targets[] = {10, 20, 50, 100, 200, 500};
    printf("%-12s | %-10s | %-10s | %-10s | %-10s\n",
           "SLA (p99 μs)", "Max Rooms", "qps (M)", "Efficiency", "Config");
    printf("--------------|------------|------------|------------|----------\n");
    
    for (int t = 0; t < 6; t++) {
        float target = sla_targets[t];
        int best_rooms = 1;
        float best_qps = 0;
        
        for (auto& r : results) {
            if (r.p.p99 <= target && r.qps > best_qps) {
                best_qps = r.qps;
                best_rooms = r.rooms;
            }
        }
        
        const char* config = "1 stream";
        if (best_rooms > 1024) config = "1 stream (large batch)";
        if (best_rooms > 4096) config = "1 stream (GPU sat)";
        
        printf("%-12.0f | %8d    | %8.2f   | %8.1f%%   | %s\n",
               target, best_rooms, best_qps, best_qps / 149.0f * 100.0f, config);
    }
    
    // ===== Part 3: Cost Calculator =====
    printf("\n=== Cost Calculator: Jetsons Needed for Target ===\n\n");
    
    struct Target { const char* name; float qps_target; float sla_us; };
    Target targets[] = {
        {"Small fleet (100 rooms)", 10.0f, 50.0f},
        {"Medium fleet (1000 rooms)", 50.0f, 20.0f},
        {"Large fleet (10000 rooms)", 200.0f, 20.0f},
        {"Enterprise (100K rooms)", 1000.0f, 50.0f},
        {"Mega fleet (1M rooms)", 5000.0f, 100.0f},
    };
    
    printf("%-30s | %-10s | %-8s | %-10s | %-10s | %-10s\n",
           "Scenario", "Target(M)", "SLA(μs)", "Jetsons", "Cost($)", "Annual($)");
    printf("--------------------------------|------------|----------|------------|------------|------------\n");
    
    float jetson_qps = 42.5f;  // Practical single-stream qps at 256 rooms
    float jetson_cost = 249.0f;
    
    for (int t = 0; t < 5; t++) {
        float needed = ceil(targets[t].qps_target / jetson_qps);
        float cost = needed * jetson_cost;
        float annual_power = needed * 11.0f * 24 * 365 / 1000.0f;  // kWh
        float power_cost = annual_power * 0.12f;  // $0.12/kWh
        
        printf("%-30s | %8.1f   | %6.0f   | %8.0f   | %8.0f   | %8.0f\n",
               targets[t].name, targets[t].qps_target, targets[t].sla_us,
               needed, cost, power_cost);
    }
    
    cudaStreamDestroy(stream);
    for (int b = 0; b < 2; b++) {
        cudaFree(d_out[b]);
        cudaFreeHost(h_out[b]);
    }
    cudaFree(d_weights); cudaFree(d_input);
    
    printf("\n=== Suite #43 Complete ===\n");
    return 0;
}
