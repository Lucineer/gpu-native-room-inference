/**
 * DECKBOSS ROOM PIPELINE — Multi-Room Inference Chain
 * Jetson Orin Nano | sm_87
 * 
 * Simulates the real deckboss workload:
 * 1. Load 12 rooms' weights into GPU memory
 * 2. Run inference on each room (mat-vec + GELU)
 * 3. Select next room based on output scores
 * 4. Chain: room A output → room B input (via projection)
 * 
 * Measures: total latency, per-room latency, room switch cost
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NEURONS 16
#define INPUT_SIZE 256
#define NUM_ROOMS 12

// Half2 vectorized warp mat-vec + GELU (fastest room inference kernel)
__global__ void room_inference_half2(const half2* w, const half2* in, float* out) {
    __shared__ half2 s_in[INPUT_SIZE / 2];
    int lane = threadIdx.x;
    int rid = blockIdx.x;
    const half2* ww = w + rid * NEURONS * (INPUT_SIZE / 2);
    
    // Load input into shared memory (half2 = 2 elements per load)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane + i * 32;
        if (idx < INPUT_SIZE / 2) s_in[idx] = in[idx];
    }
    
    // Load room weights into shared memory
    __shared__ half2 s_w[NEURONS * INPUT_SIZE / 2];
    #pragma unroll
    for (int i = 0; i < NEURONS * 4; i++) {
        int idx = lane + i * 32;
        if (idx < NEURONS * INPUT_SIZE / 2) s_w[idx] = ww[idx];
    }
    __syncwarp();
    
    for (int n = 0; n < NEURONS; n++) {
        float s = 0.0f;
        #pragma unroll
        for (int k = lane; k < INPUT_SIZE / 2; k += 32) {
            half2 wv = s_w[n * (INPUT_SIZE / 2) + k];
            half2 iv = s_in[k];
            s += __half2float(__hmul(wv.x, iv.x)) + __half2float(__hmul(wv.y, iv.y));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) {
            float x = s;
            out[rid * NEURONS + n] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
    }
}

// Room selector: pick room with highest activation sum
__global__ void room_selector(const float* scores, int* selected, int nr) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float best = -1e30f;
    int best_room = 0;
    for (int r = 0; r < nr; r++) {
        float s = 0.0f;
        for (int n = 0; n < NEURONS; n++) s += scores[r * NEURONS + n];
        if (s > best) { best = s; best_room = r; }
    }
    *selected = best_room;
}

// Input projection: map room A's output to room B's input format
// Simple approach: use output neurons as indices into a projection matrix
__global__ void project_output(const float* room_out, half2* next_in, 
                               const half2* proj, int from_room, int to_room) {
    int lane = threadIdx.x;
    // Each room output (16 floats) gets projected to 256 half values
    // via a 16x256 projection matrix stored per-room-pair
    const half2* p = proj + (from_room * NUM_ROOMS + to_room) * NEURONS * (INPUT_SIZE / 2);
    
    for (int k = lane; k < INPUT_SIZE / 2; k += 32) {
        float s = 0.0f;
        #pragma unroll
        for (int n = 0; n < NEURONS; n++) {
            half2 pv = p[n * (INPUT_SIZE / 2) + k];
            s += __half2float(__hmul(pv.x, __float2half(room_out[from_room * NEURONS + n])))
               + __half2float(__hmul(pv.y, __float2half(room_out[from_room * NEURONS + n])));
        }
        next_in[k] = __halves2half2(__float2half(s), __float2half(0.0f));
    }
}

int main() {
    printf("=== DECKBOSS ROOM PIPELINE BENCHMARK ===\n\n");
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | sm_%d%d | %d SMs | %.0f MHz | %.0f MB\n\n", 
           p.name, p.major, p.minor, p.multiProcessorCount, p.clockRate/1000.0, p.totalGlobalMem/1e6);
    
    // Memory layout
    size_t weights_bytes = NUM_ROOMS * NEURONS * INPUT_SIZE * sizeof(half);
    size_t proj_bytes = NUM_ROOMS * NUM_ROOMS * NEURONS * INPUT_SIZE * sizeof(half);
    size_t input_bytes = INPUT_SIZE * sizeof(half);
    size_t output_bytes = NUM_ROOMS * NEURONS * sizeof(float);
    
    printf("Memory layout:\n");
    printf("  Room weights:  %zu KB (%d rooms x %d neurons x %d weights x 2B)\n", 
           weights_bytes/1024, NUM_ROOMS, NEURONS, INPUT_SIZE);
    printf("  Projection:    %zu KB (%d x %d room pairs x %d x %d x 2B)\n",
           proj_bytes/1024, NUM_ROOMS, NUM_ROOMS, NEURONS, INPUT_SIZE);
    printf("  Input buffer:  %zu bytes\n", input_bytes);
    printf("  Output buffer: %zu bytes\n", output_bytes);
    printf("  Total GPU:     %.1f KB\n\n", (weights_bytes + proj_bytes + input_bytes + output_bytes)/1024.0);
    
    // Allocate
    half *h_w = (half*)malloc(weights_bytes);
    half *h_proj = (half*)malloc(proj_bytes);
    half *h_in = (half*)malloc(input_bytes);
    float *h_out = (float*)malloc(output_bytes);
    srand(42);
    for (size_t i = 0; i < weights_bytes/2; i++) h_w[i] = __float2half((float)rand()/RAND_MAX-0.5f);
    for (size_t i = 0; i < proj_bytes/2; i++) h_proj[i] = __float2half((float)rand()/RAND_MAX * 0.1f);
    for (int i = 0; i < INPUT_SIZE; i++) h_in[i] = __float2half((float)rand()/RAND_MAX-0.5f);
    
    half *d_w, *d_in, *d_proj; float *d_out; int *d_sel;
    cudaMalloc(&d_w, weights_bytes);
    cudaMalloc(&d_proj, proj_bytes);
    cudaMalloc(&d_in, input_bytes);
    cudaMalloc(&d_out, output_bytes);
    cudaMalloc(&d_sel, sizeof(int));
    cudaMemcpy(d_w, h_w, weights_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj, h_proj, proj_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, h_in, input_bytes, cudaMemcpyHostToDevice);
    
    int ITERS = 10000;
    
    // Benchmark 1: All 12 rooms in parallel (one kernel launch)
    cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);
    float ms;
    
    for (int i = 0; i < 100; i++)
        room_inference_half2<<<NUM_ROOMS, 32>>>((half2*)d_w, (half2*)d_in, d_out);
    cudaDeviceSynchronize();
    
    cudaEventRecord(es);
    for (int i = 0; i < ITERS; i++)
        room_inference_half2<<<NUM_ROOMS, 32>>>((half2*)d_w, (half2*)d_in, d_out);
    cudaEventRecord(ee); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, es, ee);
    float l_parallel = ms / ITERS;
    
    // Benchmark 2: Single room inference (1 block)
    cudaEventRecord(es);
    for (int i = 0; i < ITERS; i++)
        room_inference_half2<<<1, 32>>>((half2*)d_w, (half2*)d_in, d_out);
    cudaEventRecord(ee); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, es, ee);
    float l_single = ms / ITERS;
    
    // Benchmark 3: Room selection
    cudaEventRecord(es);
    for (int i = 0; i < ITERS; i++)
        room_selector<<<1, 1>>>(d_out, d_sel, NUM_ROOMS);
    cudaEventRecord(ee); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, es, ee);
    float l_select = ms / ITERS;
    
    // Benchmark 4: Input projection (room A → room B)
    half *d_in2;
    cudaMalloc(&d_in2, input_bytes);
    cudaEventRecord(es);
    for (int i = 0; i < ITERS; i++)
        project_output<<<1, 32>>>(d_out, (half2*)d_in2, (half2*)d_proj, 0, 1);
    cudaEventRecord(ee); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, es, ee);
    float l_project = ms / ITERS;
    
    // Benchmark 5: Full pipeline chain (infer → select → project → infer)
    cudaEventRecord(es);
    for (int i = 0; i < ITERS; i++) {
        room_inference_half2<<<NUM_ROOMS, 32>>>((half2*)d_w, (half2*)d_in, d_out);
        room_selector<<<1, 1>>>(d_out, d_sel, NUM_ROOMS);
        // In real pipeline: cudaMemcpy d_sel to host, then project based on selection
    }
    cudaEventRecord(ee); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, es, ee);
    float l_chain = ms / ITERS;
    
    // Benchmark 6: Room switch cost — how fast can we swap weight pointers?
    // On GPU, this is zero (just change blockIdx). On CPU→GPU, it's a memcpy.
    // Test: copy one room's weights (8KB) to a staging buffer
    half *d_stage;
    cudaMalloc(&d_stage, NEURONS * INPUT_SIZE * sizeof(half));
    cudaEventRecord(es);
    for (int i = 0; i < ITERS; i++) {
        int target = i % NUM_ROOMS;
        cudaMemcpy(d_stage, d_w + target * NEURONS * INPUT_SIZE, 
                   NEURONS * INPUT_SIZE * sizeof(half), cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(ee); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, es, ee);
    float l_switch = ms / ITERS;
    
    // Benchmark 7: Multi-hop chain (3 room transitions)
    cudaEventRecord(es);
    for (int i = 0; i < ITERS; i++) {
        // Hop 1: infer all rooms
        room_inference_half2<<<NUM_ROOMS, 32>>>((half2*)d_w, (half2*)d_in, d_out);
        // Hop 2: project output of best room to new input
        project_output<<<1, 32>>>(d_out, (half2*)d_in, (half2*)d_proj, 0, 1);
        // Hop 3: infer again with new input
        room_inference_half2<<<NUM_ROOMS, 32>>>((half2*)d_w, (half2*)d_in, d_out);
        // Hop 4: project again
        project_output<<<1, 32>>>(d_out, (half2*)d_in, (half2*)d_proj, 1, 2);
        // Hop 5: final inference
        room_inference_half2<<<NUM_ROOMS, 32>>>((half2*)d_w, (half2*)d_in, d_out);
    }
    cudaEventRecord(ee); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, es, ee);
    float l_multihop = ms / ITERS;
    
    printf("=== PIPELINE BENCHMARKS (%d iters) ===\n\n", ITERS);
    printf("%-40s %10s %10s %10s\n", "Operation", "Latency", "QPS", "Notes");
    printf("%-40s %10s %10s %10s\n", "----------------------------------------", "----------", "----------", "----------");
    printf("%-40s %8.4f ms %8.0f/s\n", "Single room inference", l_single, 1000/l_single);
    printf("%-40s %8.4f ms %8.0f/s  %d rooms\n", "All 12 rooms parallel", l_parallel, 1000/l_parallel, NUM_ROOMS);
    printf("%-40s %8.4f ms %8.0f/s\n", "Room selection", l_select, 1000/l_select);
    printf("%-40s %8.4f ms %8.0f/s\n", "Input projection (16→256)", l_project, 1000/l_project);
    printf("%-40s %8.4f ms %8.0f/s  infer+select", "Pipeline: infer→select", l_chain, 1000/l_chain);
    printf("%-40s %8.4f ms %8.0f/s  D2D copy 8KB", "Room weight switch", l_switch, 1000/l_switch);
    printf("%-40s %8.4f ms %8.0f/s  3 hops, 5 infers", "Multi-hop chain (3 rooms)", l_multihop, 1000/l_multihop);
    
    printf("\n=== DECKBOSS PRODUCT TARGETS ===\n");
    printf("Target room switch:     < 200 ms  →  actual: %.2f ms  %s\n", l_switch, l_switch < 0.2 ? "PASS ✓" : "FAIL ✗");
    printf("Target inference:       < 1.0 ms   →  actual: %.4f ms  %s\n", l_single, l_single < 1.0 ? "PASS ✓" : "FAIL ✗");
    printf("Target 12 rooms in 8GB: 12 rooms   →  actual: %.1f KB GPU  %s\n", 
           (weights_bytes + proj_bytes)/1024.0, 
           (weights_bytes + proj_bytes) < 8*1024*1024 ? "PASS ✓" : "FAIL ✗");
    printf("Target rooms in 8GB:    12 rooms   →  theoretical max: %d rooms\n", 
           (int)(8*1024*1024.0 / (weights_bytes / NUM_ROOMS)));
    
    float total_per_hop = l_parallel + l_select + l_project;
    printf("\n=== HOPLATENCY ===\n");
    printf("Single hop (infer 12 rooms + select + project): %.4f ms\n", total_per_hop);
    printf("Hops per second: %.0f\n", 1000/total_per_hop);
    printf("3-room chain: %.4f ms (%.0f hops/sec)\n", l_multihop, 3000/l_multihop);
    
    cudaFree(d_w);cudaFree(d_proj);cudaFree(d_in);cudaFree(d_out);cudaFree(d_sel);cudaFree(d_in2);cudaFree(d_stage);
    free(h_w);free(h_proj);free(h_in);free(h_out);
    cudaEventDestroy(es);cudaEventDestroy(ee);
    return 0;
}
