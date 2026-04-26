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

// V15: 4 rooms per warp, 16 rooms per block (256 threads = 8 warps * 2 rooms each)
__global__ void infer_v15(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*16; int warp=threadIdx.x/32; int lane=threadIdx.x%32;
    // Each warp processes 2 rooms
    for(int sub=0; sub<2; sub++) {
        int room=rb+warp*2+sub;
        if(room>=n) continue;
        const half* wp=w+room*d;
        float sum=0.0f;
        #pragma unroll 4
        for(int i=lane;i<d;i+=32) sum+=__half2float(wp[i])*__half2float(inp[i]);
        for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
        if(lane==0) out[room]=sum;
    }
}

// V16: 2 rooms per warp, 16 rooms per block
__global__ void infer_v16(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*16; int warp=threadIdx.x/32; int lane=threadIdx.x%32;
    for(int sub=0; sub<2; sub++) {
        int room=rb+warp*2+sub;
        if(room>=n) continue;
        const half* wp=w+room*d;
        float sum=0.0f;
        #pragma unroll 2
        for(int i=lane;i<d;i+=32) sum+=__half2float(wp[i])*__half2float(inp[i]);
        for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
        if(lane==0) out[room]=sum;
    }
}

// V17: 1 room per warp but 4 rooms per block (128 threads)
__global__ void infer_v17(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*4; int warp=threadIdx.x/32; int lane=threadIdx.x%32;
    int room=rb+warp;
    if(room>=n) return;
    const half* wp=w+room*d;
    float sum=0.0f;
    #pragma unroll 4
    for(int i=lane;i<d;i+=32) sum+=__half2float(wp[i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[room]=sum;
}

int main(){
    printf("=== Suite 64d: Correct Multi-Room-Per-Warp ===\n\n");
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

    // V7 baseline
    dim3 g8((R+7)/8);
    float v7_us;
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     v7_us=ms/ITERS*1000;
    }

    // Correctness check
    infer_v7<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
    float* do15; cudaMalloc(&do15,R*4);
    dim3 g16((R+15)/16);
    infer_v15<<<g16,256,0,s>>>(dw,di,do15,R,DIM);
    cudaStreamSynchronize(s);
    std::vector<float> rv7(R), rv15(R);
    cudaMemcpy(rv7.data(),do_,R*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(rv15.data(),do15,R*4,cudaMemcpyDeviceToHost);
    float maxe=0, avge=0;
    for(int i=0;i<R;i++){
        float d=fabsf(rv7[i]-rv15[i]);
        float r=rv7[i]!=0?d/fabsf(rv7[i]):d;
        if(r>maxe) maxe=r; avge+=r;
    }
    avge/=R;
    printf("  V15 correctness: max=%.6f%% avg=%.6f%%\n", maxe*100, avge*100);

    // Benchmarks
    printf("\n--- Multi-Room-Per-Warp Kernels (4096 rooms) ---\n");
    printf("%-25s | %8s | %10s | %8s\n","Kernel","us","M qps","vs V7");
    printf("-------------------------|----------|------------|----------\n");
    printf("%-25s | %6.2f   | %8.1f   | %6.2f\n","V7 (1r/warp,8r/block)",v7_us,R/v7_us,1.0);

    auto bench=[&](const char* nm, dim3 g, int th, auto k){
        cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
        for(int i=0;i<WARMUP;i++) k<<<g,th,0,s>>>(dw,di,do_,R,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s1,s);
        for(int i=0;i<ITERS;i++) k<<<g,th,0,s>>>(dw,di,do_,R,DIM);
        cudaEventRecord(e1,s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
        float us=ms/ITERS*1000;
        printf("%-25s | %6.2f   | %8.1f   | %6.2f\n",nm,us,R/us,v7_us/us);
    };

    bench("V15 (2r/warp,16r,unroll4)", g16, 256, infer_v15);
    bench("V16 (2r/warp,16r,unroll2)", g16, 256, infer_v16);
    dim3 g4((R+3)/4);
    bench("V17 (1r/warp,4r,128t)", g4, 128, infer_v17);

    // Scaling: V15 vs V7
    printf("\n--- V15 vs V7 Scaling ---\n");
    printf("%-10s | %8s | %8s | %6s\n","Rooms","V7 us","V15 us","V15/V7");
    printf("----------|----------|----------|--------\n");
    int rs[]={64,256,1024,4096,8192};
    for(int t=0;t<5;t++){
        int rooms=rs[t];
        dim3 gv((rooms+7)/8), gn((rooms+15)/16);
        cudaEvent_t sa,ea,sb,eb;
        cudaEventCreate(&sa); cudaEventCreate(&ea);
        cudaEventCreate(&sb); cudaEventCreate(&eb);
        for(int i=0;i<WARMUP;i++) infer_v7<<<gv,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sa,s);
        for(int i=0;i<ITERS;i++) infer_v7<<<gv,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(ea,s); cudaStreamSynchronize(s);
        float m1; cudaEventElapsedTime(&m1,sa,ea);
        for(int i=0;i<WARMUP;i++) infer_v15<<<gn,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sb,s);
        for(int i=0;i<ITERS;i++) infer_v15<<<gn,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(eb,s); cudaStreamSynchronize(s);
        float m2; cudaEventElapsedTime(&m2,sb,eb);
        cudaEventDestroy(sa); cudaEventDestroy(ea);
        cudaEventDestroy(sb); cudaEventDestroy(eb);
        printf("%-10d | %6.2f   | %6.2f   | %.2f\n",rooms,
            m1/ITERS*1000, m2/ITERS*1000, (m2/ITERS)/(m1/ITERS));
    }

    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di); cudaFree(do_); cudaFree(do15);
    printf("\n=== Suite 64d Complete ===\n");
}
