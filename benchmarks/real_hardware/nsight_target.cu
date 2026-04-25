/**
 * Nsight Compute target — simple batched room inference
 * 
 * Profile with: ncu --set full ./nsight_target
 * 
 * This is a minimal reproducer for the production kernel.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void batch_infer(const half* __restrict__ weights,
                             const half* __restrict__ input,
                             float* __restrict__ output,
                             int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(size_t)room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

int main() {
    const int DIM = 256;
    const int NUM_ROOMS = 64;
    const int WARMUP = 1000;
    const int ITERS = 10000;
    
    half *h_weights = (half*)malloc((size_t)NUM_ROOMS * DIM * sizeof(half));
    half *h_input = (half*)malloc(DIM * sizeof(half));
    
    srand(42);
    for (int i = 0; i < NUM_ROOMS * DIM; i++)
        h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++)
        h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_weights, *d_input;
    float *d_output;
    cudaMalloc(&d_weights, (size_t)NUM_ROOMS * DIM * sizeof(half));
    cudaMalloc(&d_input, DIM * sizeof(half));
    cudaMalloc(&d_output, NUM_ROOMS * sizeof(float));
    
    cudaMemcpy(d_weights, h_weights, (size_t)NUM_ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, DIM * sizeof(half), cudaMemcpyHostToDevice);
    
    dim3 block(32);
    dim3 grid(NUM_ROOMS);
    
    // Warmup
    for (int i = 0; i < WARMUP; i++)
        batch_infer<<<grid, block>>>(d_weights, d_input, d_output, DIM, NUM_ROOMS);
    cudaDeviceSynchronize();
    
    printf("Running %d iterations of %d-room batch inference...\n", ITERS, NUM_ROOMS);
    
    for (int i = 0; i < ITERS; i++)
        batch_infer<<<grid, block>>>(d_weights, d_input, d_output, DIM, NUM_ROOMS);
    cudaDeviceSynchronize();
    
    printf("Done.\n");
    
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_weights);
    free(h_input);
    
    return 0;
}
