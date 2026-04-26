/**
 * Suite #62: L2 Cache Persisting
 * CUDA 11+ has cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)
 * This pins accessed memory regions in L2 cache, preventing eviction.
 * For room inference where weights are reused across inferences,
 * this could eliminate cache misses entirely.
 * 
 * Also test: cudaStreamAttrValue with accessPolicyWindow
 * to explicitly control which memory ranges stay in L2.
 */
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

__global__ void infer_fp16_acc(const half* w, const half* inp, half* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    half sum=__float2half(0.0f);
    for(int i=lane;i<d;i+=32)
        sum=__hadd(sum,__hmul(w[(rb+ri)*d+i],inp[i]));
    // Warp shuffle in half
    half tmp=sum;
    for(int o=16;o>0;o>>=1) tmp=__hadd(tmp,__shfl_sync(0xffffffff, tmp, o));
    if(lane==0) out[rb+ri]=tmp;
}

int main(){
    printf("=== Suite #62: L2 Cache Persisting + FP16 Accum ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);

    half*dw,*di; float*do_; half*do16;
    cudaMalloc(&dw,(size_t)R*DIM*2); cudaMalloc(&di,DIM*2);
    cudaMalloc(&do_,R*4); cudaMalloc(&do16,R*2);
    std::vector<half> hw(R*DIM), hi(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hi[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dw,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(di,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    // Get current L2 cache size
    int l2_size=0;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrMaxPersistingL2CacheSize, 0);
    printf("  Max persisting L2: %d KB\n", l2_size/1024);
    size_t l2_actual=0;
    cudaDeviceGetLimit(&l2_actual, cudaLimitPersistingL2CacheSize);
    printf("  Current persisting L2: %d KB\n", (int)l2_actual/1024);

    // Test 1: Baseline (no persisting)
    printf("\n--- Test 1: Baseline (no L2 persist) ---\n");
    cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
    for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaStreamSynchronize(s); cudaEventRecord(s1,s);
    for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaEventRecord(e1,s); cudaStreamSynchronize(s);
    float ms1; cudaEventElapsedTime(&ms1,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
    float base_us=ms1/ITERS*1000;
    printf("  Baseline: %.2f us, %.1f M qps\n", base_us, R/base_us);

    // Test 2: Max L2 persisting
    printf("\n--- Test 2: Max L2 persist (%d KB) ---\n", l2_size/1024);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size);
    // Touch weights to load into persistent L2
    infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaStreamSynchronize(s);

    {cudaEvent_t s2,e2; cudaEventCreate(&s2); cudaEventCreate(&e2);
     for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s2,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e2,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s2,e2); cudaEventDestroy(s2); cudaEventDestroy(e2);
     printf("  Max persist: %.2f us, %.1f M qps (%.2fx baseline)\n",
         ms/ITERS*1000, R/(ms/ITERS*1000), base_us/(ms/ITERS*1000));
    }

    // Test 3: Stream access policy window (explicit persist)
    printf("\n--- Test 3: Stream Access Policy Window ---\n");
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0); // reset
    cudaStreamAttrValue attr;
    int num_rooms_persist=R; // persist all rooms
    size_t w_bytes=(size_t)R*DIM*sizeof(half);
    attr.accessPolicyWindow.base_ptr=dw;
    attr.accessPolicyWindow.num_bytes=w_bytes;
    attr.accessPolicyWindow.hitRatio=1.0f; // 100% hit ratio
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);

    {cudaEvent_t s3,e3; cudaEventCreate(&s3); cudaEventCreate(&e3);
     for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s3,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e3,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s3,e3); cudaEventDestroy(s3); cudaEventDestroy(e3);
     printf("  Policy window: %.2f us, %.1f M qps (%.2fx baseline)\n",
         ms/ITERS*1000, R/(ms/ITERS*1000), base_us/(ms/ITERS*1000));
    }

    // Reset policy
    attr.accessPolicyWindow.base_ptr=0;
    attr.accessPolicyWindow.num_bytes=0;
    attr.accessPolicyWindow.hitRatio=0.0f;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);

    // Test 4: Persist partial (only first 1024 rooms)
    printf("\n--- Test 4: Partial Persist (first 1024 rooms only) ---\n");
    attr.accessPolicyWindow.base_ptr=dw;
    attr.accessPolicyWindow.num_bytes=(size_t)1024*DIM*sizeof(half);
    attr.accessPolicyWindow.hitRatio=1.0f;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);

    {cudaEvent_t s4,e4; cudaEventCreate(&s4); cudaEventCreate(&e4);
     for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s4,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e4,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s4,e4); cudaEventDestroy(s4); cudaEventDestroy(e4);
     printf("  Partial persist: %.2f us, %.1f M qps (%.2fx baseline)\n",
         ms/ITERS*1000, R/(ms/ITERS*1000), base_us/(ms/ITERS*1000));
    }

    // Reset
    attr.accessPolicyWindow.base_ptr=0;
    attr.accessPolicyWindow.num_bytes=0;
    attr.accessPolicyWindow.hitRatio=0.0f;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);

    // Test 5: FP16 accumulation vs FP32
    printf("\n--- Test 5: FP16 Accumulation vs FP32 ---\n");
    printf("  Weight memory: %d KB\n", R*DIM*2/1024);
    printf("  L2 cache: %d KB\n", l2_size/1024);
    printf("  Weight %% of L2: %.1f%%\n", (float)(R*DIM*2)/l2_size*100);

    {cudaEvent_t s5,e5; cudaEventCreate(&s5); cudaEventCreate(&e5);
     for(int i=0;i<WARMUP;i++) infer_fp16_acc<<<grid,256,0,s>>>(dw,di,do16,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s5,s);
     for(int i=0;i<ITERS;i++) infer_fp16_acc<<<grid,256,0,s>>>(dw,di,do16,R,DIM);
     cudaEventRecord(e5,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s5,e5); cudaEventDestroy(s5); cudaEventDestroy(e5);
     float fp16_us=ms/ITERS*1000;
     printf("  FP16 accum: %.2f us, %.1f M qps\n", fp16_us, R/fp16_us);
    }

    // Test 6: Sustained inference (1M iterations) with max persist
    printf("\n--- Test 6: Sustained 1M inferences (max persist) ---\n");
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size);
    infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaStreamSynchronize(s);

    {cudaEvent_t s6,e6; cudaEventCreate(&s6); cudaEventCreate(&e6);
     cudaEventRecord(s6,s);
     for(int i=0;i<1000000;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e6,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s6,e6); cudaEventDestroy(s6); cudaEventDestroy(e6);
     float us=ms/1000000*1000;
     printf("  1M inferences: %.2f us avg, %.1f M qps\n", us, R/us);
     printf("  Total time: %.2f seconds\n", ms/1000);
    }

    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di); cudaFree(do_); cudaFree(do16);
    printf("\n=== Suite #62 Complete ===\n");
}
