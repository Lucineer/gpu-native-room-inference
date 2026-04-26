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

// INT8 symmetric: weights in [-127,127], input in [-127,127]
// Dequant: out = w_scale * i_scale * sum(wq * iq)
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

int main(){
    printf("=== Suite #65d: INT8 Symmetric Quantization ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);

    half*dw16,*di16; float*do16;
    cudaMalloc(&dw16,(size_t)R*DIM*2); cudaMalloc(&di16,DIM*2); cudaMalloc(&do16,R*4);
    std::vector<half> hw(R*DIM), hi(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.5f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hi[i]=__float2half(cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dw16,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(di16,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    // Symmetric quantization: find abs max, scale to [-127,127]
    float imax=0;
    for(int i=0;i<DIM;i++){float v=fabsf(__half2float(hi[i]));if(v>imax)imax=v;}
    float iscale=imax/127.0f;

    std::vector<signed char> iq(DIM);
    for(int i=0;i<DIM;i++){
        float v=__half2float(hi[i])/iscale;
        iq[i]=(signed char)(v>127?127:(v<-127?-127:(int)v));
    }

    std::vector<float> ws(R);
    std::vector<signed char> wq(R*DIM);
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

    signed char*dwq,*diq; float*dws; float*doi8;
    cudaMalloc(&dwq,(size_t)R*DIM); cudaMalloc(&diq,DIM);
    cudaMalloc(&dws,R*4); cudaMalloc(&doi8,R*4);
    cudaMemcpy(dwq,wq.data(),R*DIM,cudaMemcpyHostToDevice);
    cudaMemcpy(diq,iq.data(),DIM,cudaMemcpyHostToDevice);
    cudaMemcpy(dws,ws.data(),R*4,cudaMemcpyHostToDevice);

    // V7
    float v7_us;
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     v7_us=ms/ITERS*1000;
    }

    // INT8
    float i8_us;
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_i8s<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_i8s<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     i8_us=ms/ITERS*1000;
    }
    printf("  V7 FP16: %.2f us, %.1f M qps\n", v7_us, R/v7_us);
    printf("  INT8:    %.2f us, %.1f M qps (%.2fx)\n", i8_us, R/i8_us, v7_us/i8_us);

    // Accuracy
    infer_v7<<<grid,256,0,s>>>(dw16,di16,do16,R,DIM);
    infer_i8s<<<grid,256,0,s>>>(dwq,diq,doi8,dws,iscale,R,DIM);
    cudaStreamSynchronize(s);
    std::vector<float> rf(R), ri8(R);
    cudaMemcpy(rf.data(),do16,R*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(ri8.data(),doi8,R*4,cudaMemcpyDeviceToHost);
    float me=0,ae=0;
    for(int r=0;r<R;r++){float d=fabsf(rf[r]-ri8[r]);float rel=rf[r]!=0?d/fabsf(rf[r]):d;if(rel>me)me=rel;ae+=rel;}
    ae/=R;
    printf("  Max rel err: %.4f%%  Avg: %.4f%%\n", me*100, ae*100);
    for(int r=0;r<8;r++) printf("    R%d: FP16=%.4f INT8=%.4f diff=%.2e\n",r,rf[r],ri8[r],fabsf(rf[r]-ri8[r]));

    // Check ranges
    int wlo=127,wlo2=127;
    for(size_t i=0;i<(size_t)R*DIM;i++){int v=(int)wq[i];if(v<wlo)wlo=v;if(v>wlo2)wlo2=v;}
    printf("  W range: [%d,%d]  I range: [%d,%d]\n", wlo, wlo2, (int)iq[0], (int)iq[DIM-1]);

    cudaStreamDestroy(s);
    cudaFree(dw16); cudaFree(di16); cudaFree(do16);
    cudaFree(dwq); cudaFree(diq); cudaFree(dws); cudaFree(doi8);
    printf("\n=== Suite #65d Complete ===\n");
}
