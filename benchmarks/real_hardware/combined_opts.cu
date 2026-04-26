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

// V12: 16 rooms/block + FP16 compute + FP32 shuffle reduction
__global__ void infer_v12(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*16; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    const half* wp=w+(rb+ri)*d;
    float sum=0.0f;
    #pragma unroll 4
    for(int i=lane;i<d;i+=32) sum+=__half2float(wp[i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

// V13: 16 rooms/block + FP16 math via __hfma2 (fused multiply-add, 2 elements)
__global__ void infer_v13(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*16; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    const half* wp=w+(rb+ri)*d;
    float2 sum={0.0f,0.0f};
    // Process 2 elements at a time using half2
    const half2* wp2=(const half2*)wp;
    const half2* ip2=(const half2*)inp;
    #pragma unroll 4
    for(int i=lane;i<d/2;i+=16) {
        half2 wv=wp2[i]; half2 iv=ip2[i];
        float2 wv2=__half22float2(wv);
        float2 iv2=__half22float2(iv);
        sum.x+=wv2.x*iv2.x; sum.y+=wv2.y*iv2.y;
    }
    sum.x+=sum.y;
    for(int o=16;o>0;o>>=1) sum.x+=__shfl_down_sync(0xffffffff,sum.x,o);
    if(lane==0) out[rb+ri]=sum.x;
}

// V14: 32 rooms per block (max occupancy)
__global__ void infer_v14(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*32; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    const half* wp=w+(rb+ri)*d;
    float sum=0.0f;
    #pragma unroll 4
    for(int i=lane;i<d;i+=32) sum+=__half2float(wp[i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

int main(){
    printf("=== Suite 64c: V12-V14 Block Size Sweep ===\n\n");
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

    printf("--- Block Size Sweep (4096 rooms) ---\n");
    printf("%-20s | %8s | %10s | %8s\n","Kernel","us","M qps","vs V7");
    printf("----------------------|----------|------------|----------\n");

    float v7_us;
    {dim3 g((R+7)/8); cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     v7_us=ms/ITERS*1000;
     printf("%-20s | %6.2f   | %8.1f   | %6.2f\n","V7 (8r/block)",v7_us,R/v7_us,1.0);
    }

    {dim3 g((R+15)/16); cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v12<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v12<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-20s | %6.2f   | %8.1f   | %6.2f\n","V12 (16r/block)",us,R/us,v7_us/us);
    }

    {dim3 g((R+15)/16); cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v13<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v13<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-20s | %6.2f   | %8.1f   | %6.2f\n","V13 (16r+half2)",us,R/us,v7_us/us);
    }

    {dim3 g((R+31)/32); cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v14<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v14<<<g,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-20s | %6.2f   | %8.1f   | %6.2f\n","V14 (32r/block)",us,R/us,v7_us/us);
    }

    // Correctness check for V12
    printf("\n--- V12 Correctness ---\n");
    dim3 g7((R+7)/8), g12((R+15)/16);
    infer_v7<<<g7,256,0,s>>>(dw,di,do_,R,DIM);
    float* do12; cudaMalloc(&do12,R*4);
    infer_v12<<<g12,256,0,s>>>(dw,di,do12,R,DIM);
    cudaStreamSynchronize(s);
    std::vector<float> rv7(R), rv12(R);
    cudaMemcpy(rv7.data(),do_,R*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(rv12.data(),do12,R*4,cudaMemcpyDeviceToHost);
    float maxe=0, avge=0;
    for(int i=0;i<R;i++){
        float d=fabsf(rv7[i]-rv12[i]);
        float r=rv7[i]!=0?d/fabsf(rv7[i]):d;
        if(r>maxe) maxe=r;
        avge+=r;
    }
    avge/=R;
    printf("  Max error: %.6f%%  Avg error: %.6f%%\n", maxe*100, avge*100);
    cudaFree(do12);

    // V12 scaling
    printf("\n--- V12 Scaling ---\n");
    printf("%-10s | %8s | %8s | %6s\n","Rooms","V7 us","V12 us","V12/V7");
    printf("----------|----------|----------|--------\n");
    int rs[]={256,1024,4096,8192};
    for(int t=0;t<4;t++){
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
        for(int i=0;i<WARMUP;i++) infer_v12<<<gn,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sb,s);
        for(int i=0;i<ITERS;i++) infer_v12<<<gn,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(eb,s); cudaStreamSynchronize(s);
        float m2; cudaEventElapsedTime(&m2,sb,eb);
        cudaEventDestroy(sa); cudaEventDestroy(ea);
        cudaEventDestroy(sb); cudaEventDestroy(eb);
        printf("%-10d | %6.2f   | %6.2f   | %.2f\n",rooms,
            m1/ITERS*1000, m2/ITERS*1000, (m2/ITERS)/(m1/ITERS));
    }

    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di); cudaFree(do_);
    printf("\n=== Suite 64c Complete ===\n");
}
