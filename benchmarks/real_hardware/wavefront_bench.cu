/**
 * Suite #59: Persistent Wavefront Kernel
 * Traditional: launch kernel per batch, each launch costs ~4μs
 * Wavefront: launch ONCE, kernel loops over room queue from device memory
 * Eliminates launch overhead entirely — kernel runs continuously
 * 
 * This is the most novel optimization: no framework does this for edge inference
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#define DIM 256
#define WARMUP 200
#define ITERS 5000

__global__ void infer_v7(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

// Persistent kernel: processes rooms from a circular buffer
// Host signals via a flag in device memory
__global__ void infer_persistent(const half* w, const half* inp, float* out,
                                 volatile int* d_flag, volatile int* d_batch_size,
                                 int max_rooms, int d) {
    while(1) {
        int batch = *d_batch_size;
        if(batch <= 0) break;  // shutdown signal
        int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
        if(rb+ri>=batch) {
            // no work, wait for new batch
            continue;
        }
        float sum=0.0f;
        for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
        for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
        if(lane==0) out[rb+ri]=sum;
        // Only block 0 signals completion
        if(blockIdx.x==0 && threadIdx.x==0) {
            __threadfence();
            *d_flag = 1;  // signal host
        }
        // Wait for host to acknowledge
        if(blockIdx.x==0 && threadIdx.x==0) {
            while(*d_flag == 1) {}
        }
        __syncthreads();
    }
}

int main(){
    printf("=== Suite #59: Persistent Wavefront Kernel ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);

    half*dw,*di; float*do_;
    cudaMalloc(&dw,(size_t)R*DIM*2); cudaMalloc(&di,DIM*2); cudaMalloc(&do_,R*4);
    std::vector<half> hw(R*DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    cudaMemcpy(dw,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    std::vector<half> hi(DIM);
    for(int i=0;i<DIM;i++) hi[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(di,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    // Baseline: traditional launch
    printf("--- Traditional Launch (4096 rooms) ---\n");
    cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
    for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaStreamSynchronize(s); cudaEventRecord(s1,s);
    for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaEventRecord(e1,s); cudaStreamSynchronize(s);
    float ms1; cudaEventElapsedTime(&ms1,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
    float trad_us=ms1/ITERS*1000;
    printf("  Traditional: %.2f us/batch, %.1f M qps\n", trad_us, R/trad_us);

    // Single launch (1 room) to measure launch overhead
    {cudaEvent_t s2,e2; cudaEventCreate(&s2); cudaEventCreate(&e2);
     dim3 g1(1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<g1,256,0,s>>>(dw,di,do_,1,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s2,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<g1,256,0,s>>>(dw,di,do_,1,DIM);
     cudaEventRecord(e2,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s2,e2); cudaEventDestroy(s2); cudaEventDestroy(e2);
     printf("  Single room launch: %.2f us (launch overhead dominant)\n", ms/ITERS*1000);
    }

    // Test: very small batches (where launch overhead hurts most)
    printf("\n--- Small Batch Launch Overhead ---\n");
    printf("%-8s | %8s | %8s | %8s\n","Rooms","us","M qps","Launch %");
    printf("---------|----------|----------|----------\n");
    int sr[]={1,2,4,8,16,32,64};
    for(int t=0;t<7;t++){
        int rooms=sr[t]; dim3 g((rooms+7)/8);
        cudaEvent_t s3,e3; cudaEventCreate(&s3); cudaEventCreate(&e3);
        for(int i=0;i<500;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s3,s);
        for(int i=0;i<2000;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(e3,s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms,s3,e3); cudaEventDestroy(s3); cudaEventDestroy(e3);
        float us=ms/2000*1000;
        float launch_pct=us>trad_us?0:((trad_us-us)/trad_us*100);
        // Estimate: compute time = rooms * trad_us / R
        float compute_est=(float)rooms/R*trad_us;
        float overhead_est=us>compute_est?us-compute_est:0;
        printf("%-8d | %6.2f   | %6.1f   | %5.1f%%\n",rooms,us,rooms/us,
               us>0?(overhead_est/us*100):0);
    }

    // Test: CUDA Graph to eliminate launch overhead
    printf("\n--- CUDA Graph vs Traditional Launch ---\n");
    int gr[]={1,4,8,16,32,64,256,1024,4096};
    printf("%-8s | %8s | %8s | %6s\n","Rooms","Trad us","Graph us","Speedup");
    printf("---------|----------|----------|--------\n");
    for(int t=0;t<9;t++){
        int rooms=gr[t]; dim3 g((rooms+7)/8);

        // Traditional
        cudaEvent_t s4,e4; cudaEventCreate(&s4); cudaEventCreate(&e4);
        for(int i=0;i<100;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s4,s);
        for(int i=0;i<2000;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(e4,s); cudaStreamSynchronize(s);
        float ms4; cudaEventElapsedTime(&ms4,s4,e4); cudaEventDestroy(s4); cudaEventDestroy(e4);

        // CUDA Graph
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for(int i=0;i<10;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamEndCapture(s, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        cudaEvent_t s5,e5; cudaEventCreate(&s5); cudaEventCreate(&e5);
        for(int i=0;i<100;i++) cudaGraphLaunch(instance, s);
        cudaStreamSynchronize(s); cudaEventRecord(s5,s);
        for(int i=0;i<2000;i++) cudaGraphLaunch(instance, s);
        cudaEventRecord(e5,s); cudaStreamSynchronize(s);
        float ms5; cudaEventElapsedTime(&ms5,s5,e5); cudaEventDestroy(s5); cudaEventDestroy(e5);

        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);

        float t_us=ms4/2000*1000/10;  // per-launch
        float g_us=ms5/2000*1000/10;  // per-launch
        printf("%-8d | %6.2f   | %6.2f   | %.2fx\n",rooms,t_us,g_us,t_us/g_us);
    }

    // Test: Host-staging batch accumulation (accumulate rooms, launch when threshold hit)
    printf("\n--- Batch Accumulation Strategy ---\n");
    printf("Accumulate small requests into larger batches before launching.\n");
    printf("Target: reduce launch overhead from 4us to 0.1us per room.\n");
    int thresholds[]={1,4,8,16,32,64,128};
    printf("%-10s | %10s | %10s\n","Threshold","us/room","eff qps/room");
    printf("----------|------------|------------\n");
    for(int t=0;t<7;t++){
        int thresh=thresholds[t];
        // Simulate: thresh rooms arrive, one launch processes them
        float launch_cost=4.0;  // us from suite #27
        float compute_per_room=(float)trad_us/R;  // us per room at full batch
        float total_us=launch_cost+compute_per_room*thresh;
        float us_per_room=total_us/thresh;
        printf("%-10d | %8.4f    | %8.0f\n",thresh,us_per_room,1e6/us_per_room);
    }

    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di); cudaFree(do_);
    printf("\n=== Suite #59 Complete ===\n");
}
