/**
 * Suite #58: Bfloat16 vs FP16 vs FP32 for Room Inference
 * Orin Ampere supports BF16 natively via __bfloat16_raw type.
 * BF16 has FP32 dynamic range but FP16 precision.
 * Question: does the wider range tradeoff pay off on real workloads?
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define DIM 256
#define WARMUP 200
#define ITERS 5000

__global__ void infer_fp16(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void infer_bf16(const __nv_bfloat16* w, const __nv_bfloat16* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__bfloat162float(w[(rb+ri)*d+i])*__bfloat162float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void infer_bf16_gelu(const __nv_bfloat16* w, const __nv_bfloat16* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__bfloat162float(w[(rb+ri)*d+i])*__bfloat162float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) {
        float c=0.7978845608f, a=0.044715f;
        out[rb+ri]=0.5f*sum*(1.0f+tanhf(c*(sum+a*sum*sum*sum)));
    }
}

__global__ void infer_fp16_gelu(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) {
        float c=0.7978845608f, a=0.044715f;
        out[rb+ri]=0.5f*sum*(1.0f+tanhf(c*(sum+a*sum*sum*sum)));
    }
}

__global__ void infer_fp32(const float* w, const float* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=w[(rb+ri)*d+i]*inp[i];
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

int main(){
    printf("=== Suite #58: BF16 vs FP16 vs FP32 ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);

    // FP16 data
    half*dw16,*di16; float*do16;
    cudaMalloc(&dw16,(size_t)R*DIM*2); cudaMalloc(&di16,DIM*2); cudaMalloc(&do16,R*4);
    std::vector<half> hw16(R*DIM), hi16(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw16[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hi16[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dw16,hw16.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(di16,hi16.data(),DIM*2,cudaMemcpyHostToDevice);

    // BF16 data
    __nv_bfloat16*dwbf,*dibf; float*dobf;
    cudaMalloc(&dwbf,(size_t)R*DIM*2); cudaMalloc(&dibf,DIM*2); cudaMalloc(&dobf,R*4);
    std::vector<__nv_bfloat16> hwbf(R*DIM), hibf(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hwbf[i]=__float2bfloat16(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hibf[i]=__float2bfloat16(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dwbf,hwbf.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dibf,hibf.data(),DIM*2,cudaMemcpyHostToDevice);

    // FP32 data
    float*dw32,*di32,*do32;
    cudaMalloc(&dw32,(size_t)R*DIM*4); cudaMalloc(&di32,DIM*4); cudaMalloc(&do32,R*4);
    std::vector<float> hw32(R*DIM), hi32(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw32[i]=0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f);
    for(int i=0;i<DIM;i++) hi32[i]=0.5f*cosf((float)i/DIM*6.2832f);
    cudaMemcpy(dw32,hw32.data(),R*DIM*4,cudaMemcpyHostToDevice);
    cudaMemcpy(di32,hi32.data(),DIM*4,cudaMemcpyHostToDevice);

    // Test 1: Throughput comparison
    printf("--- Throughput (4096 rooms, dim=256, dot product only) ---\n");
    printf("%-10s | %8s | %10s | %8s\n","Type","us","M qps","vs FP16");
    printf("----------|----------|------------|----------\n");

    // FP16
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_fp16<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_fp16<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-10s | %6.2f   | %8.1f   | %6.2f\n","FP16",us,R/us,1.0);
    }

    // BF16
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_bf16<<<grid,256,0,s>>>(dwbf,dibf,dobf,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_bf16<<<grid,256,0,s>>>(dwbf,dibf,dobf,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-10s | %6.2f   | %8.1f   | %6.2f\n","BF16",us,R/us,R/us/(R/25.17));
    }

    // FP32
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_fp32<<<grid,256,0,s>>>(dw32,di32,do32,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_fp32<<<grid,256,0,s>>>(dw32,di32,do32,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-10s | %6.2f   | %8.1f   | %6.2f\n","FP32",us,R/us,R/us/(R/25.17));
    }

    // Test 2: With GELU
    printf("\n--- With GELU Activation ---\n");
    printf("%-10s | %8s | %10s | %8s\n","Type","us","M qps","vs FP16");
    printf("----------|----------|------------|----------\n");

    // FP16+GELU
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_fp16_gelu<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_fp16_gelu<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-10s | %6.2f   | %8.1f   | %6.2f\n","FP16+G",us,R/us,1.0);
    }

    // BF16+GELU
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_bf16_gelu<<<grid,256,0,s>>>(dwbf,dibf,dobf,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_bf16_gelu<<<grid,256,0,s>>>(dwbf,dibf,dobf,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-10s | %6.2f   | %8.1f   | %6.2f\n","BF16+G",us,R/us,R/us/(R/25.17));
    }

    // Test 3: Numerical accuracy with large values (where BF16 shines)
    printf("\n--- Large Value Accuracy (weights in [-10,10]) ---\n");
    printf("%-10s | %12s | %12s | %12s\n","Type","Max RelErr","Avg RelErr","vs FP32");
    printf("----------|--------------|--------------|--------------\n");

    // Generate large weights
    std::vector<float> hw_large(R*DIM), hi_l(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++) hw_large[i]=10.0f*(2.0f*(float)rand()/RAND_MAX-1.0f);
    for(int i=0;i<DIM;i++) hi_l[i]=5.0f*cosf((float)i/DIM*6.2832f);

    // FP32 reference
    std::vector<float> ref(R);
    {float*dw_r,*di_r,*do_r;
     cudaMalloc(&dw_r,(size_t)R*DIM*4); cudaMalloc(&di_r,DIM*4); cudaMalloc(&do_r,R*4);
     cudaMemcpy(dw_r,hw_large.data(),R*DIM*4,cudaMemcpyHostToDevice);
     cudaMemcpy(di_r,hi_l.data(),DIM*4,cudaMemcpyHostToDevice);
     infer_fp32<<<grid,256,0,s>>>(dw_r,di_r,do_r,R,DIM);
     cudaMemcpy(ref.data(),do_r,R*4,cudaMemcpyDeviceToHost);
     cudaFree(dw_r); cudaFree(di_r); cudaFree(do_r);
    }

    // FP16 test
    {std::vector<half> hw_h(R*DIM),hi_h(DIM);
     for(size_t i=0;i<(size_t)R*DIM;i++) hw_h[i]=__float2half(hw_large[i]);
     for(int i=0;i<DIM;i++) hi_h[i]=__float2half(hi_l[i]);
     cudaMemcpy(dw16,hw_h.data(),R*DIM*2,cudaMemcpyHostToDevice);
     cudaMemcpy(di16,hi_h.data(),DIM*2,cudaMemcpyHostToDevice);
     infer_fp16<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     std::vector<float> res(R);
     cudaMemcpy(res.data(),do16,R*4,cudaMemcpyDeviceToHost);
     float max_err=0,avg_err=0;
     for(int r=0;r<R;r++){
         float err=ref[r]!=0?fabsf((res[r]-ref[r])/ref[r]):fabsf(res[r]-ref[r]);
         if(err>max_err) max_err=err;
         avg_err+=err;
     }
     avg_err/=R;
     printf("%-10s | %10.4f%% | %10.4f%% | %10s\n","FP16",max_err*100,avg_err*100,"baseline");
    }

    // BF16 test
    {std::vector<__nv_bfloat16> hw_b(R*DIM),hi_b(DIM);
     for(size_t i=0;i<(size_t)R*DIM;i++) hw_b[i]=__float2bfloat16(hw_large[i]);
     for(int i=0;i<DIM;i++) hi_b[i]=__float2bfloat16(hi_l[i]);
     cudaMemcpy(dwbf,hw_b.data(),R*DIM*2,cudaMemcpyHostToDevice);
     cudaMemcpy(dibf,hi_b.data(),DIM*2,cudaMemcpyHostToDevice);
     infer_bf16<<<grid,256,0,s>>>(dwbf,dibf,dobf,R,DIM);
     std::vector<float> res(R);
     cudaMemcpy(res.data(),dobf,R*4,cudaMemcpyDeviceToHost);
     float max_err=0,avg_err=0;
     for(int r=0;r<R;r++){
         float err=ref[r]!=0?fabsf((res[r]-ref[r])/ref[r]):fabsf(res[r]-ref[r]);
         if(err>max_err) max_err=err;
         avg_err+=err;
     }
     avg_err/=R;
     printf("%-10s | %10.4f%% | %10.4f%% | %10s\n","BF16",max_err*100,avg_err*100,"vs FP32");
    }

    // Test 4: Throughput at various batch sizes
    printf("\n--- BF16 vs FP16 at Various Batch Sizes ---\n");
    printf("%-8s | %8s | %8s | %6s\n","Rooms","FP16 us","BF16 us","Ratio");
    printf("---------|----------|----------|--------\n");
    int tr[]={64,256,1024,4096,8192};
    for(int t=0;t<5;t++){
        int rooms=tr[t]; dim3 g((rooms+7)/8);
        cudaEvent_t s1,e1,s2,e2;
        cudaEventCreate(&s1); cudaEventCreate(&e1);
        cudaEventCreate(&s2); cudaEventCreate(&e2);
        for(int i=0;i<100;i++) infer_fp16<<<g,256,0,s>>>(dw16,di16,do16,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s1,s);
        for(int i=0;i<2000;i++) infer_fp16<<<g,256,0,s>>>(dw16,di16,do16,rooms,DIM);
        cudaEventRecord(e1,s); cudaStreamSynchronize(s);
        float ms1; cudaEventElapsedTime(&ms1,s1,e1);
        for(int i=0;i<100;i++) infer_bf16<<<g,256,0,s>>>(dwbf,dibf,dobf,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s2,s);
        for(int i=0;i<2000;i++) infer_bf16<<<g,256,0,s>>>(dwbf,dibf,dobf,rooms,DIM);
        cudaEventRecord(e2,s); cudaStreamSynchronize(s);
        float ms2; cudaEventElapsedTime(&ms2,s2,e2);
        cudaEventDestroy(s1); cudaEventDestroy(e1);
        cudaEventDestroy(s2); cudaEventDestroy(e2);
        printf("%-8d | %6.2f   | %6.2f   | %.3f\n",rooms,ms1/2000*1000,ms2/2000*1000,ms2/ms1);
    }

    cudaStreamDestroy(s);
    cudaFree(dw16); cudaFree(di16); cudaFree(do16);
    cudaFree(dwbf); cudaFree(dibf); cudaFree(dobf);
    cudaFree(dw32); cudaFree(di32); cudaFree(do32);
    printf("\n=== Suite #58 Complete ===\n");
}
