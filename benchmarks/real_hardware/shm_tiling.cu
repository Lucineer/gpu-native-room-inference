#include <cstdio>
#include <cstdlib>
#include <vector>
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

__global__ void infer_shm8(const half* w, const half* inp, float* out, int n, int d) {
    __shared__ half sw[8][256];
    __shared__ half si[256];
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    for(int i=lane;i<d;i+=32) sw[ri][i]=w[(rb+ri)*d+i];
    if(ri==0) for(int i=lane;i<d;i+=32) si[i]=inp[i];
    __syncthreads();
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(sw[ri][i])*__half2float(si[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

int main(){
    printf("=== Suite #61: Shared Memory Tiling ===\n\n");
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

    dim3 g8((R+7)/8);
    float v7_us, shm_us;

    cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
    for(int i=0;i<WARMUP;i++) infer_v7<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
    cudaStreamSynchronize(s); cudaEventRecord(s1,s);
    for(int i=0;i<ITERS;i++) infer_v7<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
    cudaEventRecord(e1,s); cudaStreamSynchronize(s);
    float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
    v7_us=ms/ITERS*1000;

    cudaEvent_t s2,e2; cudaEventCreate(&s2); cudaEventCreate(&e2);
    for(int i=0;i<WARMUP;i++) infer_shm8<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
    cudaStreamSynchronize(s); cudaEventRecord(s2,s);
    for(int i=0;i<ITERS;i++) infer_shm8<<<g8,256,0,s>>>(dw,di,do_,R,DIM);
    cudaEventRecord(e2,s); cudaStreamSynchronize(s);
    cudaEventElapsedTime(&ms,s2,e2); cudaEventDestroy(s2); cudaEventDestroy(e2);
    shm_us=ms/ITERS*1000;

    printf("V7 (global mem):   %.2f us, %.1f M qps\n", v7_us, R/v7_us);
    printf("SHM 8 rooms/block: %.2f us, %.1f M qps\n", shm_us, R/shm_us);
    printf("SHM/V7 ratio:      %.3f\n", shm_us/v7_us);

    printf("\n--- Scaling ---\n");
    printf("%-8s | %8s | %8s | %6s\n","Rooms","V7 us","SHM us","Ratio");
    printf("---------|----------|----------|--------\n");
    int ts[]={64,256,1024,4096,8192,16384};
    for(int t=0;t<6;t++){
        int rooms=ts[t]; dim3 g((rooms+7)/8);
        cudaEvent_t sa,ea,sb,eb;
        cudaEventCreate(&sa); cudaEventCreate(&ea);
        cudaEventCreate(&sb); cudaEventCreate(&eb);
        for(int i=0;i<100;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sa,s);
        for(int i=0;i<2000;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(ea,s); cudaStreamSynchronize(s);
        float m1; cudaEventElapsedTime(&m1,sa,ea);
        for(int i=0;i<100;i++) infer_shm8<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sb,s);
        for(int i=0;i<2000;i++) infer_shm8<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(eb,s); cudaStreamSynchronize(s);
        float m2; cudaEventElapsedTime(&m2,sb,eb);
        cudaEventDestroy(sa); cudaEventDestroy(ea);
        cudaEventDestroy(sb); cudaEventDestroy(eb);
        printf("%-8d | %6.2f   | %6.2f   | %.2f\n",rooms,
            m1/2000*1000, m2/2000*1000, (m2/2000)/(m1/2000));
    }

    printf("\nShared memory: 8*256*2+256*2 = %d bytes (%.1f%% of 128KB)\n",
        8*256*2+256*2, (float)(8*256*2+256*2)/131072*100);
    printf("V7 reads from L2/global (0 bytes shared)\n");

    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di); cudaFree(do_);
    printf("\n=== Suite #61 Complete ===\n");
}
