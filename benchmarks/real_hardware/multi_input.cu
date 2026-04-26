#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#define DIM 256
#define WARMUP 200
#define ITERS 5000
__global__ void infer_shared(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n)return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}
__global__ void infer_perr(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n)return;
    int room=rb+ri;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[room*d+i])*__half2float(inp[room*d+i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[room]=sum;
}
int main(){
    printf("=== Suite #57: Multiple Input Vectors ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);
    half*dw,*di_s,*di_p; float*do_;
    cudaMalloc(&dw,(size_t)R*DIM*2); cudaMalloc(&di_s,DIM*2);
    cudaMalloc(&di_p,(size_t)R*DIM*2); cudaMalloc(&do_,R*4);
    std::vector<half> hw(R*DIM); for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    cudaMemcpy(dw,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    std::vector<half> hi(DIM); for(int i=0;i<DIM;i++)
        hi[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(di_s,hi.data(),DIM*2,cudaMemcpyHostToDevice);
    std::vector<half> hp(R*DIM); for(size_t i=0;i<(size_t)R*DIM;i++)
        hp[i]=__float2half(0.5f*cosf((float)(i%DIM)/DIM*6.2832f));
    cudaMemcpy(di_p,hp.data(),R*DIM*2,cudaMemcpyHostToDevice);
    printf("--- Shared vs Per-Room Input ---\n");
    printf("%-8s|%-12s|%-12s|%-8s\n","Rooms","Shared us","PerRoom us","Ratio");
    printf("--------|------------|------------|--------\n");
    int tr[]={64,256,1024,4096,8192};
    for(int t=0;t<5;t++){
        int rooms=tr[t]; dim3 g((rooms+7)/8);
        cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
        for(int i=0;i<WARMUP;i++) infer_shared<<<g,256,0,s>>>(dw,di_s,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s1,s);
        for(int i=0;i<ITERS;i++) infer_shared<<<g,256,0,s>>>(dw,di_s,do_,rooms,DIM);
        cudaEventRecord(e1,s); cudaStreamSynchronize(s);
        float ms1; cudaEventElapsedTime(&ms1,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
        cudaEvent_t s2,e2; cudaEventCreate(&s2); cudaEventCreate(&e2);
        for(int i=0;i<WARMUP;i++) infer_perr<<<g,256,0,s>>>(dw,di_p,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s2,s);
        for(int i=0;i<ITERS;i++) infer_perr<<<g,256,0,s>>>(dw,di_p,do_,rooms,DIM);
        cudaEventRecord(e2,s); cudaStreamSynchronize(s);
        float ms2; cudaEventElapsedTime(&ms2,s2,e2); cudaEventDestroy(s2); cudaEventDestroy(e2);
        printf("%-8d|%10.2f  |%10.2f  |%.3fx\n",rooms,ms1/ITERS*1000,ms2/ITERS*1000,ms2/ms1);
    }
    printf("\n--- Fleet Cycle (upload+infer) ---\n");
    printf("%-8s|%-10s|%-10s\n","Rooms","Total us","M qps");
    printf("--------|----------|----------\n");
    int fr[]={64,256,1024,4096};
    for(int t=0;t<4;t++){
        int rooms=fr[t]; dim3 g((rooms+7)/8);
        size_t wb=(size_t)rooms*DIM*2;
        cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
        for(int i=0;i<100;i++){
            cudaMemcpyAsync(dw,hw.data(),wb,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(di_p,hp.data(),wb,cudaMemcpyHostToDevice,s);
            infer_perr<<<g,256,0,s>>>(dw,di_p,do_,rooms,DIM);
        }
        cudaStreamSynchronize(s); cudaEventRecord(s1,s);
        for(int i=0;i<1000;i++){
            cudaMemcpyAsync(dw,hw.data(),wb,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(di_p,hp.data(),wb,cudaMemcpyHostToDevice,s);
            infer_perr<<<g,256,0,s>>>(dw,di_p,do_,rooms,DIM);
        }
        cudaEventRecord(e1,s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
        printf("%-8d|%8.2f   |%8.1f\n",rooms,ms/1000*1000,rooms/(ms/1000*1000));
    }
    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di_s); cudaFree(di_p); cudaFree(do_);
    printf("\n=== Suite #57 Complete ===\n");
}
