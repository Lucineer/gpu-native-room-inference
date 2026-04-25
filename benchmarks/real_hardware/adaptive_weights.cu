/**
 * Suite #35: LoRA-Style Adaptive Room Weights
 * 
 * Instead of storing full room weights (256 rooms × 256 dims × 2 bytes = 128KB),
 * store a shared base weight (256 dims × 2 bytes = 512 bytes) and small per-room
 * adapters (rank-4 LoRA: 256×4 + 4×256 = 2048 bytes per room, or 512KB total).
 * 
 * Actually that's MORE memory. Let me think differently:
 * 
 * Base weight (256 dims) + per-room scale+shift (2 floats = 8 bytes per room).
 * Effective weight = base * scale + shift. Like RMSNorm but per-room.
 * Memory: 512B + 256×8 = 2.5KB total for 256 rooms. 50× reduction!
 * 
 * Or even simpler: base weight (256 halfs) + per-room bias (1 half = 2 bytes).
 * Memory: 512B + 512B = 1KB. 128× reduction.
 * 
 * Test: how much accuracy do we lose? And is the kernel any faster
 * (since all rooms share the same base weight, better cache utilization)?
 * 
 * Hardware: Jetson Orin Nano 8GB, sm_87
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 adaptive_weights.cu -o adaptive_weights
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cuda_fp16.h>

#define DIM 256
#define WARMUP 200
#define ITERS 10000

// Standard room inference (full per-room weights)
__global__ void infer_full(
    const half* __restrict__ weights,  // [num_rooms][dim]
    const half* __restrict__ input,    // [dim]
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

// Shared base + per-room scale + per-room bias
// effective_weight[r][i] = base[i] * scale[r] + bias[r]
// dot = sum_i (base[i] * scale[r] + bias[r]) * input[i]
//     = scale[r] * sum_i(base[i] * input[i]) + bias[r] * sum_i(input[i])
// Precompute: base_dot = dot(base, input) — shared across ALL rooms!
//            input_sum = sum(input) — also shared!
// Per room: scale[r] * base_dot + bias[r] * input_sum
// This means the dot product is O(1) per room after O(dim) shared computation!
__global__ void infer_adaptive_scale_bias(
    const half* __restrict__ base,      // [dim] — shared base weight
    const half* __restrict__ input,      // [dim]
    const float* __restrict__ scales,    // [num_rooms] — per-room scale
    const float* __restrict__ biases,    // [num_rooms] — per-room bias
    float* __restrict__ output,
    int num_rooms, int dim)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    // Step 1: All threads compute base_dot and input_sum (shared across rooms)
    float base_dot = 0.0f;
    float input_sum = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float bv = __half2float(base[i]);
        float iv = __half2float(input[i]);
        base_dot += bv * iv;
        input_sum += iv;
    }
    
    // Warp reduce both
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        base_dot += __shfl_down_sync(0xffffffff, base_dot, offset);
        input_sum += __shfl_down_sync(0xffffffff, input_sum, offset);
    }
    
    // Step 2: Per-room O(1) computation
    if (lane == 0) {
        output[room] = scales[room] * base_dot + biases[room] * input_sum;
    }
}

// Per-room LoRA rank-4: weight = base + A * B
// A: [num_rooms][dim][rank], B: [num_rooms][rank]
// effective[r][i] = base[i] + sum_k A[r][i][k] * B[r][k]
// dot = sum_i effective[r][i] * input[i]
//     = sum_i base[i]*input[i] + sum_k B[r][k] * sum_i A[r][i][k]*input[i]
//     = base_dot + sum_k B[r][k] * lora_dot[r][k]
// Precompute: base_dot (shared), lora_dot[r][k] (per-room per-rank)
__global__ void infer_lora(
    const half* __restrict__ base,       // [dim]
    const half* __restrict__ input,       // [dim]
    const half* __restrict__ lora_A,      // [num_rooms][rank][dim]
    const half* __restrict__ lora_B,      // [num_rooms][rank]
    float* __restrict__ output,
    int num_rooms, int dim, int rank)
{
    int room_base = blockIdx.x * 8;
    int room_idx = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (room_base + room_idx >= num_rooms) return;
    int room = room_base + room_idx;
    
    // Base dot (shared across rooms in same warp? No, different base values)
    // Actually base is shared — all rooms use the same base
    float base_dot = 0.0f;
    for (int i = lane; i < dim; i += 32)
        base_dot += __half2float(base[i]) * __half2float(input[i]);
    
    // Per-room LoRA dot
    float lora_contribution = 0.0f;
    for (int k = 0; k < rank; k++) {
        float lora_dot = 0.0f;
        for (int i = lane; i < dim; i += 32)
            lora_dot += __half2float(lora_A[(room * rank + k) * dim + i]) * __half2float(input[i]);
        
        // Warp reduce lora_dot
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            lora_dot += __shfl_down_sync(0xffffffff, lora_dot, offset);
        
        if (lane == 0)
            lora_contribution += __half2float(lora_B[room * rank + k]) * lora_dot;
    }
    
    // Reduce base_dot
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        base_dot += __shfl_down_sync(0xffffffff, base_dot, offset);
    
    if (lane == 0) output[room] = base_dot + lora_contribution;
}

// Generate target output from full weights
// Then fit scale+bias to approximate
void fit_scale_bias(const std::vector<float>& full_weights, const std::vector<float>& input,
                    int num_rooms, int dim, std::vector<float>& scales, std::vector<float>& biases) {
    // For each room: target = scale * base_dot + bias * input_sum
    // Where base_dot = dot(base_weights, input), input_sum = sum(input)
    // And target = dot(room_weights, input)
    
    // Compute base weights (average of all room weights)
    std::vector<float> base(dim, 0.0f);
    for (int r = 0; r < num_rooms; r++)
        for (int i = 0; i < dim; i++)
            base[i] += full_weights[r * dim + i] / num_rooms;
    
    float input_sum = 0.0f;
    for (int i = 0; i < dim; i++) input_sum += input[i];
    
    float base_dot = 0.0f;
    for (int i = 0; i < dim; i++) base_dot += base[i] * input[i];
    
    // For each room: target[r] = scale[r] * base_dot + bias[r] * input_sum
    // This is underdetermined (1 equation, 2 unknowns).
    // Use least squares with regularization: minimize ||target - (scale*base_dot + bias*input_sum)||^2
    // Solution: normal equations on the design matrix [base_dot, input_sum]
    
    // Simple approach: fit via 2-variable least squares across all rooms
    float sxx = 0, sxy = 0, sxz = 0, syy = 0, syz = 0;
    std::vector<float> targets(num_rooms);
    for (int r = 0; r < num_rooms; r++) {
        targets[r] = 0;
        for (int i = 0; i < dim; i++)
            targets[r] += full_weights[r * dim + i] * input[i];
        
        // Room-specific base_dot and input_sum would be different
        // But in our model they're shared — this is an approximation
    }
    
    // Actually, let's use a per-room fit with regularization
    // scale[r] * base_dot + bias[r] * input_sum ≈ target[r]
    // Use pseudo-inverse: [base_dot, input_sum] * [scale; bias] = target
    // 2x2 system per room (same for all rooms since base_dot and input_sum are shared):
    // [base_dot^2,  base_dot*input_sum] [scale]   [sum(target * base_dot)]
    // [input_sum*bd, input_sum^2      ] [bias ] = [sum(target * input_sum)]
    
    float bd_is = base_dot * input_sum;
    float bd2 = base_dot * base_dot;
    float is2 = input_sum * input_sum;
    float det = bd2 * is2 - bd_is * bd_is;
    
    if (fabsf(det) > 1e-10f) {
        float rhs_s = 0, rhs_b = 0;
        for (int r = 0; r < num_rooms; r++) {
            rhs_s += targets[r] * base_dot;
            rhs_b += targets[r] * input_sum;
        }
        rhs_s /= num_rooms;
        rhs_b /= num_rooms;
        
        float global_scale = (is2 * rhs_s - bd_is * rhs_b) / det;
        float global_bias = (bd2 * rhs_b - bd_is * rhs_s) / det;
        
        for (int r = 0; r < num_rooms; r++) {
            scales[r] = global_scale;
            biases[r] = global_bias;
        }
    }
}

int main() {
    printf("=== Suite #35: Adaptive Room Weights (Memory Reduction) ===\n\n");
    
    int num_rooms = 256;
    int ranks[] = {1, 2, 4, 8, 16};
    
    // Generate random room weights and input
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    
    // Full weights: 256 rooms, each 256 dims, normally distributed
    std::vector<float> full_weights(num_rooms * DIM);
    for (int i = 0; i < num_rooms * DIM; i++)
        full_weights[i] = nd(rng);
    
    std::vector<float> input(DIM);
    for (int i = 0; i < DIM; i++)
        input[i] = nd(rng);
    
    // Compute base (average weights)
    std::vector<float> base(DIM, 0.0f);
    for (int r = 0; r < num_rooms; r++)
        for (int i = 0; i < DIM; i++)
            base[i] += full_weights[r * DIM + i] / num_rooms;  // BUG: should be DIM not dim
    
    // Fix: compute properly
    for (int i = 0; i < DIM; i++) {
        base[i] = 0;
        for (int r = 0; r < num_rooms; r++)
            base[i] += full_weights[r * DIM + i];
        base[i] /= num_rooms;
    }
    
    // Fit scale+bias
    std::vector<float> scales(num_rooms), biases(num_rooms);
    fit_scale_bias(full_weights, input, num_rooms, DIM, scales, biases);
    
    // Generate LoRA weights: A[r][k][i] = residual[r][i] * random_projection[k]
    // LoRA: effective = base + A @ B
    // residual[r][i] = full_weights[r][i] - base[i]
    
    // Convert to half
    std::vector<half> h_full(num_rooms * DIM), h_base(DIM), h_input(DIM);
    for (int i = 0; i < num_rooms * DIM; i++)
        h_full[i] = __float2half(full_weights[i]);
    for (int i = 0; i < DIM; i++)
        h_base[i] = __float2half(base[i]);
    for (int i = 0; i < DIM; i++)
        h_input[i] = __float2half(input[i]);
    
    // Allocate GPU
    half *d_full, *d_base, *d_input;
    float *d_scales, *d_biases, *d_out_full, *d_out_adaptive, *d_out_lora;
    cudaMalloc(&d_full, num_rooms * DIM * sizeof(half));
    cudaMalloc(&d_base, DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_scales, num_rooms * sizeof(float));
    cudaMalloc(&d_biases, num_rooms * sizeof(float));
    cudaMalloc(&d_out_full, num_rooms * sizeof(float));
    cudaMalloc(&d_out_adaptive, num_rooms * sizeof(float));
    cudaMalloc(&d_out_lora, num_rooms * sizeof(float));
    
    cudaMemcpy(d_full, h_full.data(), num_rooms * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, h_base.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), num_rooms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases.data(), num_rooms * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    dim3 grid((num_rooms + 7) / 8);
    
    // Run full inference
    for (int i = 0; i < WARMUP; i++)
        infer_full<<<grid, dim3(256), 0, stream>>>(d_full, d_input, d_out_full, num_rooms, DIM);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < ITERS; i++)
        infer_full<<<grid, dim3(256), 0, stream>>>(d_full, d_input, d_out_full, num_rooms, DIM);
    cudaStreamSynchronize(stream);
    
    // Run adaptive
    for (int i = 0; i < WARMUP; i++)
        infer_adaptive_scale_bias<<<grid, dim3(256), 0, stream>>>(d_base, d_input, d_scales, d_biases, d_out_adaptive, num_rooms, DIM);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < ITERS; i++)
        infer_adaptive_scale_bias<<<grid, dim3(256), 0, stream>>>(d_base, d_input, d_scales, d_biases, d_out_adaptive, num_rooms, DIM);
    cudaStreamSynchronize(stream);
    
    // Read outputs
    std::vector<float> out_full(num_rooms), out_adaptive(num_rooms);
    cudaMemcpy(out_full.data(), d_out_full, num_rooms * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_adaptive.data(), d_out_adaptive, num_rooms * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute accuracy
    float mae = 0, max_err = 0, rel_err = 0;
    for (int r = 0; r < num_rooms; r++) {
        float err = fabsf(out_full[r] - out_adaptive[r]);
        mae += err / num_rooms;
        max_err = std::max(max_err, err);
        if (fabsf(out_full[r]) > 0.001f)
            rel_err += err / fabsf(out_full[r]) / num_rooms;
    }
    
    printf("--- Scale+Bias Adaptive Weights ---\n");
    printf("Memory: full = %d bytes, adaptive = %d bytes (%.1fx reduction)\n",
           num_rooms * DIM * 2, DIM * 2 + num_rooms * 8,
           (float)(num_rooms * DIM * 2) / (DIM * 2 + num_rooms * 8));
    printf("Accuracy: MAE = %.4f, Max error = %.4f, Relative error = %.2f%%\n\n",
           mae, max_err, rel_err * 100);
    
    // Latency comparison
    {
        for (int i = 0; i < WARMUP; i++)
            infer_full<<<grid, dim3(256), 0, stream>>>(d_full, d_input, d_out_full, num_rooms, DIM);
        cudaStreamSynchronize(stream);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_full<<<grid, dim3(256), 0, stream>>>(d_full, d_input, d_out_full, num_rooms, DIM);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("  Full weights: %.2f us\n", std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS);
    }
    {
        for (int i = 0; i < WARMUP; i++)
            infer_adaptive_scale_bias<<<grid, dim3(256), 0, stream>>>(d_base, d_input, d_scales, d_biases, d_out_adaptive, num_rooms, DIM);
        cudaStreamSynchronize(stream);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_adaptive_scale_bias<<<grid, dim3(256), 0, stream>>>(d_base, d_input, d_scales, d_biases, d_out_adaptive, num_rooms, DIM);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("  Adaptive (scale+bias): %.2f us\n", std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS);
    }
    
    // LoRA benchmarks
    printf("\n--- LoRA Adaptive Weights ---\n");
    
    for (int ri = 0; ri < 5; ri++) {
        int rank = ranks[ri];
        
        // Generate LoRA A and B
        int lora_A_size = num_rooms * rank * DIM;
        int lora_B_size = num_rooms * rank;
        
        std::vector<half> h_lora_A(lora_A_size), h_lora_B(lora_B_size);
        
        for (int r = 0; r < num_rooms; r++) {
            for (int k = 0; k < rank; k++) {
                // B[r][k]: random init
                h_lora_B[r * rank + k] = __float2half(nd(rng) * 0.1f);
                // A[r][k][i]: compute to minimize residual
                float residual_dot = 0;
                for (int i = 0; i < DIM; i++)
                    residual_dot += (full_weights[r * DIM + i] - base[i]) * input[i];
                
                // Distribute residual across rank ranks
                for (int i = 0; i < DIM; i++) {
                    float residual = full_weights[r * DIM + i] - base[i];
                    // Simple: A[r][k][i] = residual / rank (uniform distribution)
                    h_lora_A[(r * rank + k) * DIM + i] = __float2half(residual / rank * nd(rng));
                }
            }
        }
        
        half *d_lora_A, *d_lora_B;
        cudaMalloc(&d_lora_A, lora_A_size * sizeof(half));
        cudaMalloc(&d_lora_B, lora_B_size * sizeof(half));
        cudaMemcpy(d_lora_A, h_lora_A.data(), lora_A_size * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lora_B, h_lora_B.data(), lora_B_size * sizeof(half), cudaMemcpyHostToDevice);
        
        // Run LoRA inference
        for (int i = 0; i < 100; i++)
            infer_lora<<<grid, dim3(256), 0, stream>>>(d_base, d_input, d_lora_A, d_lora_B, d_out_lora, num_rooms, DIM, rank);
        cudaStreamSynchronize(stream);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_lora<<<grid, dim3(256), 0, stream>>>(d_base, d_input, d_lora_A, d_lora_B, d_out_lora, num_rooms, DIM, rank);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        
        // Read and compute accuracy
        std::vector<float> out_lora(num_rooms);
        cudaMemcpy(out_lora.data(), d_out_lora, num_rooms * sizeof(float), cudaMemcpyDeviceToHost);
        
        float lora_mae = 0;
        for (int r = 0; r < num_rooms; r++)
            lora_mae += fabsf(out_full[r] - out_lora[r]) / num_rooms;
        
        int mem = DIM * 2 + num_rooms * rank * (DIM + 1) * 2;
        printf("  Rank %2d: %.2f us, MAE = %.4f, memory = %d bytes (%.1fx reduction)\n",
               rank, us, lora_mae, mem, (float)(num_rooms * DIM * 2) / mem);
        
        cudaFree(d_lora_A);
        cudaFree(d_lora_B);
    }
    
    // Vary rooms for adaptive (fixed dim=256)
    printf("\n--- Adaptive Scaling (Varying Batch Size) ---\n");
    printf("%-8s | %-14s | %-14s | %-8s\n", "Rooms", "Full (us)", "Adaptive (us)", "Speedup");
    printf("---------|----------------|----------------|----------\n");
    
    int batch_sizes[] = {16, 64, 256, 1024};
    for (int b = 0; b < 4; b++) {
        int rooms = batch_sizes[b];
        dim3 g((rooms + 7) / 8);
        
        float t_full, t_adapt;
        
        for (int i = 0; i < WARMUP; i++) {
            infer_full<<<g, dim3(256), 0, stream>>>(d_full, d_input, d_out_full, rooms, DIM);
        }
        cudaStreamSynchronize(stream);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_full<<<g, dim3(256), 0, stream>>>(d_full, d_input, d_out_full, rooms, DIM);
        cudaStreamSynchronize(stream);
        t_full = std::chrono::duration<float, std::micro>(t0.time_since_epoch()).count(); // wrong, fix below
        
        // Let me redo this properly
        cudaStreamSynchronize(stream);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_full<<<g, dim3(256), 0, stream>>>(d_full, d_input, d_out_full, rooms, DIM);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        t_full = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        
        for (int i = 0; i < WARMUP; i++)
            infer_adaptive_scale_bias<<<g, dim3(256), 0, stream>>>(d_base, d_input, d_scales, d_biases, d_out_adaptive, rooms, DIM);
        cudaStreamSynchronize(stream);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            infer_adaptive_scale_bias<<<g, dim3(256), 0, stream>>>(d_base, d_input, d_scales, d_biases, d_out_adaptive, rooms, DIM);
        cudaStreamSynchronize(stream);
        t1 = std::chrono::high_resolution_clock::now();
        t_adapt = std::chrono::duration<float, std::micro>(t1 - t0).count() / ITERS;
        
        printf("%-8d | %12.2f   | %12.2f   | %.3fx\n", rooms, t_full, t_adapt, t_full / t_adapt);
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_full); cudaFree(d_base); cudaFree(d_input);
    cudaFree(d_scales); cudaFree(d_biases);
    cudaFree(d_out_full); cudaFree(d_out_adaptive); cudaFree(d_out_lora);
    
    printf("\n=== Suite #35 Complete ===\n");
    return 0;
}
