#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#define DIM 256
#define WARMUP 500
#define ITERS 10000

__global__ void __launch_bounds__(256, 8)
infer_v7_lb(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    #pragma unroll 4
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void __launch_bounds__(256, 8)
infer_i8_lb(const signed char* w, const signed char* inp, float* out,
             const float* ws, float iscale, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    int room=rb+ri;
    int sum=0;
    #pragma unroll 4
    for(int i=lane;i<d;i+=32) sum+=(int)w[room*d+i]*(int)inp[i];
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[room]=(float)sum*ws[room]*iscale;
}

int main(){
    printf("=== Suite #69: Ultimate Combined Kernel ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);

    // Setup FP16
    half*dw16,*di16; float*do16;
    cudaMalloc(&dw16,(size_t)R*DIM*2); cudaMalloc(&di16,DIM*2); cudaMalloc(&do16,R*4);
    std::vector<half> hw(R*DIM), hi(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hi[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dw16,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(di16,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    // Setup INT8
    float imax=0;
    for(int i=0;i<DIM;i++){float v=fabsf(__half2float(hi[i]));if(v>imax)imax=v;}
    float iscale=imax/127.0f;
    std::vector<signed char> iq(DIM), wq(R*DIM);
    std::vector<float> ws(R);
    for(int i=0;i<DIM;i++){
        float v=__half2float(hi[i])/iscale;
        iq[i]=(signed char)(v>127?127:(v<-127?-127:(int)v));
    }
    for(int r=0;r<R;r++){
        float wmax=0;
        for(int d=0;d<DIM;d++){float v=fabsf(__half2float(hw[r*DIM+d]));if(v>wmax)wmax=v;}
        ws[r]=wmax/127.0f;
        float inv=ws[r]>0?127.0f/wmax:0;
        for(int d=0;d<DIM;d++){
            float v=__half2float(hw[r*DIM+d])*inv;
            wq[r*DIM+d]=(signed char)(v>127?127:(v<-127?-127:(int)v));
        }
    }
    signed char*dwq,*diq; float*dws,*doi8;
    cudaMalloc(&dwq,(size_t)R*DIM); cudaMalloc(&diq,DIM);
    cudaMalloc(&dws,R*4); cudaMalloc(&doi8,R*4);
    cudaMemcpy(dwq,wq.data(),R*DIM,cudaMemcpyHostToDevice);
    cudaMemcpy(diq,iq.data(),DIM,cudaMemcpyHostToDevice);
    cudaMemcpy(dws,ws.data(),R*4,cudaMemcpyHostToDevice);

    int l2_max=0;
    cudaDeviceGetAttribute(&l2_max, cudaDevAttrMaxPersistingL2CacheSize, 0);
    printf("  L2 persist max: %d KB\n", l2_max/1024);
    printf("  SMs: 8\n");
    printf("  FP16 weights: %d KB\n", R*DIM*2/1024);
    printf("  INT8 weights: %d KB\n", R*DIM/1024);

    // Test all combinations
    printf("\n  %-30s | %8s | %10s\n","Configuration","us","M qps");
    printf("  ------------------------------|----------|------------\n");

    // 1. FP16, no persist, no bounds (original V7)
    {cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
     cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7_lb<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7_lb<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     printf("  %-30s | %6.2f   | %8.1f\n","FP16 (lb, no persist)",ms/ITERS*1000,R/(ms/ITERS*1000));
    }

    // 2. FP16 + L2 persist (partial, 1K rooms)
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr=dw16;
    attr.accessPolicyWindow.num_bytes=(size_t)1024*DIM*2;
    attr.accessPolicyWindow.hitRatio=1.0f;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7_lb<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7_lb<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     printf("  %-30s | %6.2f   | %8.1f\n","FP16 (lb + L2 persist)",ms/ITERS*1000,R/(ms/ITERS*1000));
    }
    // Reset
    attr.accessPolicyWindow.base_ptr=0; attr.accessPolicyWindow.num_bytes=0;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);

    // 3. INT8, no persist
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_i8_lb<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_i8_lb<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     printf("  %-30s | %6.2f   | %8.1f\n","INT8 (lb, no persist)",ms/ITERS*1000,R/(ms/ITERS*1000));
    }

    // 4. INT8 + L2 persist (INT8 weights fit entirely!)
    attr.accessPolicyWindow.base_ptr=(void*)dwq;
    attr.accessPolicyWindow.num_bytes=(size_t)R*DIM;  // All INT8 weights
    attr.accessPolicyWindow.hitRatio=1.0f;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_i8_lb<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_i8_lb<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     printf("  %-30s | %6.2f   | %8.1f\n","INT8 (lb + L2 ALL persist)",ms/ITERS*1000,R/(ms/ITERS*1000));
    }

    // 5. INT8 + L2 + sustained 1M
    printf("\n  INT8 + L2 persist sustained (1M inferences):\n");
    for(int i=0;i<1000;i++) infer_i8_lb<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
    cudaStreamSynchronize(s);
    cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventRecord(s1,s);
    for(int i=0;i<1000000;i++) infer_i8_lb<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
    cudaEventRecord(e1,s); cudaStreamSynchronize(s);
    float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
    printf("    1M sustained: %.2f us avg, %.1f M qps\n", ms/1000000*1000, R/(ms/1000000*1000));
    printf("    Total time: %.2f seconds\n", ms/1000);

    // Reset
    attr.accessPolicyWindow.base_ptr=0; attr.accessPolicyWindow.num_bytes=0;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);

    cudaStreamDestroy(s);
    cudaFree(dw16); cudaFree(di16); cudaFree(do16);
    cudaFree(dwq); cudaFree(diq); cudaFree(dws); cudaFree(doi8);
    printf("\n=== Suite #69 Complete ===\n");
}
