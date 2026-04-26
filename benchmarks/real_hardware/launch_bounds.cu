#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#define DIM 256
#define WARMUP 500
#define ITERS 10000

__global__ void infer_v7(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void __launch_bounds__(256, 2) infer_lb256x2(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void __launch_bounds__(256, 4) infer_lb256x4(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void __launch_bounds__(256, 8) infer_lb256x8(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void __launch_bounds__(128, 4) infer_lb128x4(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*4; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void __launch_bounds__(128, 8) infer_lb128x8(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*4; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

int main(){
    printf("=== Suite #68: Launch Bounds & Compiler Flags ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);

    half*dw,*di; float*do_;
    cudaMalloc(&dw,(size_t)R*DIM*2); cudaMalloc(&di,DIM*2); cudaMalloc(&do_,R*4);
    std::vector<half> hw(R*DIM), hi(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hi[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dw,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(di,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    dim3 g8((R+7)/8), g4((R+3)/4);

    // Baseline
    float base_us;
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     base_us=ms/ITERS*1000;
     printf("  V7 baseline: %.2f us, %.1f M qps\n", base_us, R/base_us);
    }

    // Launch bounds variants
    auto bench=[&](const char* nm, dim3 g, int th, auto k){
        cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
        for(int i=0;i<WARMUP;i++) k<<<g,th,0,s>>>(dw,di,do_,R,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s1,s);
        for(int i=0;i<ITERS;i++) k<<<g,th,0,s>>>(dw,di,do_,R,DIM);
        cudaEventRecord(e1,s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
        float us=ms/ITERS*1000;
        printf("  %-20s %.2f us %.1f M qps (%.2fx)\n", nm, us, R/us, base_us/us);
    };

    printf("\n  Launch Bounds:\n");
    bench("256 threads, 2 blk:", g8, 256, infer_lb256x2);
    bench("256 threads, 4 blk:", g8, 256, infer_lb256x4);
    bench("256 threads, 8 blk:", g8, 256, infer_lb256x8);
    bench("128 threads, 4 blk:", g4, 128, infer_lb128x4);
    bench("128 threads, 8 blk:", g4, 128, infer_lb128x8);

    // Occupancy info
    printf("\n  SM count: ");
    int sms=0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    printf("%d\n", sms);

    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di); cudaFree(do_);
    printf("\n=== Suite #68 Complete ===\n");
}
