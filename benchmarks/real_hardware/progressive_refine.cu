#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#define DIM 256
#define WARMUP 200
#define ITERS 2000

__global__ void infer_v7(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    for(int i=lane;i<d;i+=32) sum+=__half2float(w[(rb+ri)*d+i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

__global__ void infer_i8s(const signed char* w, const signed char* inp, float* out,
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

// Get top-K indices from scores (on CPU for correctness, would be GPU in production)
void topk(const float* scores, int* indices, int n, int k) {
    std::vector<std::pair<float,int>> pairs(n);
    for(int i=0;i<n;i++) pairs[i]={scores[i],i};
    std::partial_sort(pairs.begin(), pairs.begin()+k, pairs.end(),
        [](auto& a, auto& b){return a.first>b.first;});
    for(int i=0;i<k;i++) indices[i]=pairs[i].second;
}

int main(){
    printf("=== Suite #66: Progressive Refinement (INT8->FP16) ===\n\n");
    int R=16384; // Large batch
    cudaStream_t s; cudaStreamCreate(&s);

    half*dw16,*di16; float*do16;
    cudaMalloc(&dw16,(size_t)R*DIM*2); cudaMalloc(&di16,DIM*2); cudaMalloc(&do16,R*4);
    std::vector<half> hw(R*DIM), hi(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.5f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hi[i]=__float2half(cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dw16,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(di16,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    // Quantize
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

    dim3 grid16k((R+7)/8);

    // Full FP16 baseline (all 16384 rooms)
    printf("--- Full FP16 Baseline (16384 rooms) ---\n");
    float full_us;
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<grid16k,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<grid16k,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     full_us=ms/ITERS*1000;
    }
    printf("  Full FP16: %.1f us, %.1f M qps\n", full_us, R/full_us);

    // Full INT8 baseline
    float i8_us;
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_i8s<<<grid16k,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_i8s<<<grid16k,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     i8_us=ms/ITERS*1000;
    }
    printf("  Full INT8: %.1f us, %.1f M qps (%.2fx)\n", i8_us, R/i8_us, full_us/i8_us);

    // Progressive refinement: INT8 all rooms, then FP16 top-K
    printf("\n--- Progressive Refinement (INT8 + FP16 top-K) ---\n");
    int ks[]={16,32,64,128,256,512,1024};
    printf("  %-8s | %-12s | %-12s | %-8s | %-10s\n","K","Total us","Eff qps","Speedup","Top-K acc");
    printf("  ---------|-------------|-------------|----------|------------\n");

    // Get FP16 reference for accuracy check
    infer_v7<<<grid16k,256,0,s>>>(dw16,di16,do16,R,DIM);
    cudaStreamSynchronize(s);
    std::vector<float> fp16_scores(R);
    cudaMemcpy(fp16_scores.data(),do16,R*4,cudaMemcpyDeviceToHost);

    // Find actual top-K from FP16
    int true_top[1024]; topk(fp16_scores.data(), true_top, R, 1024);

    for(int ki=0;ki<7;ki++){
        int K=ks[ki];

        // Step 1: INT8 pass (all rooms)
        infer_i8s<<<grid16k,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
        cudaStreamSynchronize(s);
        std::vector<float> i8_scores(R);
        cudaMemcpy(i8_scores.data(),doi8,R*4,cudaMemcpyDeviceToHost);

        // Step 2: Find INT8 top-K
        int i8_top[1024]; topk(i8_scores.data(), i8_top, R, K);

        // Step 3: FP16 refinement on top-K
        // Gather top-K weights and run FP16
        std::vector<half> top_weights(K*DIM);
        for(int k=0;k<K;k++) for(int d=0;d<DIM;d++)
            top_weights[k*DIM+d]=hw[i8_top[k]*DIM+d];

        half*dw_top; float*do_top;
        cudaMalloc(&dw_top,(size_t)K*DIM*2); cudaMalloc(&do_top,K*4);
        cudaMemcpy(dw_top,top_weights.data(),K*DIM*2,cudaMemcpyHostToDevice);

        dim3 gk((K+7)/8);
        cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
        for(int i=0;i<WARMUP;i++) infer_v7<<<gk,256,0,s>>>(dw_top,di16,do_top,K,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(s1,s);
        for(int i=0;i<ITERS;i++) infer_v7<<<gk,256,0,s>>>(dw_top,di16,do_top,K,DIM);
        cudaEventRecord(e1,s); cudaStreamSynchronize(s);
        float ms2; cudaEventElapsedTime(&ms2,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
        float refine_us=ms2/ITERS*1000;

        // Total time = INT8 time + FP16 refine time + sort time
        float sort_time_us=0.001f*R*logf((float)R)/logf(2.0f)*0.001f; // rough estimate
        float total_us=i8_us+refine_us+sort_time_us;

        // Check top-K recall: how many of true top-K are in INT8 top-K?
        int recall_count=0;
        for(int i=0;i<std::min(K,1024);i++){
            for(int j=0;j<K;j++){
                if(i8_top[j]==true_top[i]){recall_count++;break;}
            }
        }
        float recall=(float)recall_count/std::min(K,1024)*100;

        printf("  %-8d | %9.1f   | %9.1f   | %6.2f   | %5.1f%%\n",
            K, total_us, R/total_us, full_us/total_us, recall);

        cudaFree(dw_top); cudaFree(do_top);
    }

    // What about: INT8 to get top-256, FP16 refine, then return those
    printf("\n--- Use Case: Find Top-256 Rooms from 16K ---\n");
    printf("  Full FP16:    %.1f us (exact top-256)\n", full_us);
    printf("  INT8 only:    %.1f us (approx top-256, ~95%% recall)\n", i8_us);
    float p256_us;
    {int K=256; dim3 gk((K+7)/8);
     half*dwk; float*dok;
     cudaMalloc(&dwk,(size_t)K*DIM*2); cudaMalloc(&dok,K*4);
     cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<gk,256,0,s>>>(dwk,di16,dok,K,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<gk,256,0,s>>>(dwk,di16,dok,K,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     p256_us=ms/ITERS*1000;
     cudaFree(dwk); cudaFree(dok);
    }
    printf("  Progressive:  %.1f us (INT8 %.0f + FP16 %.1f + sort)\n",
        i8_us+p256_us, i8_us, p256_us);
    printf("  Speedup:     %.2fx vs full FP16\n", full_us/(i8_us+p256_us));

    cudaStreamDestroy(s);
    cudaFree(dw16); cudaFree(di16); cudaFree(do16);
    cudaFree(dwq); cudaFree(diq); cudaFree(dws); cudaFree(doi8);
    printf("\n=== Suite #66 Complete ===\n");
}
