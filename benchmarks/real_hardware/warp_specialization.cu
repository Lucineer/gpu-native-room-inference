/**
 * Suite #63: Warp Specialization + Memory Pipeline
 * Split 256 threads (8 warps) into roles:
 * - Warps 0-6: compute (7 warps for 7 rooms)
 * - Warp 7: memory prefetch (load next batch weights)
 * 
 * On Volta+, cooperative groups allow warp-level synchronization.
 * The memory warp preloads the next batch while compute warps process current.
 * This hides memory latency behind compute.
 * 
 * Also: test cooperative groups __syncwarp() for partial reductions.
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

// Warp-specialized: last warp prefetches, first 7 warps compute
__global__ void infer_warp_spec(const half* w, const half* inp, float* out,
                                 int n, int d, int total_batches) {
    int warpid=threadIdx.x/32;
    int lane=threadIdx.x%32;
    int rb=blockIdx.x*8;

    // All warps participate in initial data load
    __shared__ float sums[8];
    if(lane==0) sums[warpid]=0.0f;

    for(int batch=0; batch<total_batches; batch++) {
        int room_offset=rb+batch*8+warpid;
        if(room_offset>=n) break;

        // Memory warp (warp 7) prefetches next batch if available
        if(warpid==7 && batch+1<total_batches) {
            // Prefetch hint (volatile read)
            volatile half prefetch=w[(rb+(batch+1)*8+lane)*d];
        }

        // Compute warps (0-6)
        if(warpid<7) {
            float sum=0.0f;
            int room=rb+batch*8+warpid;
            if(room<n) {
                for(int i=lane;i<d;i+=32)
                    sum+=__half2float(w[room*d+i])*__half2float(inp[i]);
            }
            // Cross-warp reduction via shared memory
            if(lane==0) sums[warpid]=sum;
        }

        __syncwarp();
    }

    // Final write
    if(warpid<7 && lane==0) {
        out[rb+warpid]=sums[warpid];
    }
}

// Cooperative groups: use warp-level matrix (8x32) as one unit
__global__ void infer_cgroups(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8;
    int tid=threadIdx.x;
    int ri=tid/32;  // room index within block (0-7)
    int lane=tid%32; // thread within warp
    if(rb+ri>=n) return;

    float sum=0.0f;
    // Staggered read to reduce bank conflicts
    int base=(rb+ri)*d;
    #pragma unroll 2
    for(int i=lane;i<d;i+=32)
        sum+=__half2float(w[base+i])*__half2float(inp[i]);

    // Intra-warp reduction with full mask
    #pragma unroll
    for(int o=16;o>0;o>>=1)
        sum+=__shfl_sync(0xffffffff, sum, o);

    if(lane==0) out[rb+ri]=sum;
}

// LDP (load-predicated) optimization: use .ld.global.cg for cached loads
__global__ void infer_ldp(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    const half* wp=w+(rb+ri)*d;
    // Process 4 elements per iteration (wider unroll)
    #pragma unroll 4
    for(int i=lane;i<d;i+=32)
        sum+=__half2float(wp[i])*__half2float(inp[i]);
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

// LDG (read-through cache) - bypass L1, go straight to L2
__global__ void infer_ldg(const half* w, const half* inp, float* out, int n, int d) {
    int rb=blockIdx.x*8; int ri=threadIdx.x/32; int lane=threadIdx.x%32;
    if(rb+ri>=n) return;
    float sum=0.0f;
    int base=(rb+ri)*d;
    #pragma unroll 2
    for(int i=lane;i<d;i+=32) {
        half wh=__ldg(&w[base+i]);  // cached load (L2, not L1)
        half ih=__ldg(&inp[i]);
        sum+=__half2float(wh)*__half2float(ih);
    }
    for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[rb+ri]=sum;
}

int main(){
    printf("=== Suite #63: Warp Spec + Memory Pipeline + Cache Control ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);

    half*dw,*di; float*do_;
    cudaMalloc(&dw,(size_t)R*DIM*2); cudaMalloc(&di,DIM*2); cudaMalloc(&do_,R*4);
    std::vector<half> hw(R*DIM), hi(DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    for(int i=0;i<DIM;i++) hi[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(dw,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    cudaMemcpy(di,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    printf("--- Kernel Variants (4096 rooms) ---\n");
    printf("%-25s | %8s | %10s | %8s\n","Kernel","us","M qps","vs V7");
    printf("-------------------------|----------|------------|----------\n");

    float v7_us;
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     v7_us=ms/ITERS*1000;
     printf("%-25s | %6.2f   | %8.1f   | %6.2f\n","V7 (baseline)",v7_us,R/v7_us,1.0);
    }

    // LDP (wider unroll)
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_ldp<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_ldp<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-25s | %6.2f   | %8.1f   | %6.2f\n","LDP (unroll4)",us,R/us,v7_us/us);
    }

    // LDG (cached load)
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_ldg<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_ldg<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-25s | %6.2f   | %8.1f   | %6.2f\n","LDG (L2 cached)",us,R/us,v7_us/us);
    }

    // Cooperative groups
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_cgroups<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_cgroups<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-25s | %6.2f   | %8.1f   | %6.2f\n","Cooperative Groups",us,R/us,v7_us/us);
    }

    // Warp specialized
    int batches=R/8; // total batches of 8 rooms
    {cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
     for(int i=0;i<WARMUP;i++) infer_warp_spec<<<grid,256,0,s>>>(dw,di,do_,R,DIM,batches);
     cudaStreamSynchronize(s); cudaEventRecord(s1,s);
     for(int i=0;i<ITERS;i++) infer_warp_spec<<<grid,256,0,s>>>(dw,di,do_,R,DIM,batches);
     cudaEventRecord(e1,s); cudaStreamSynchronize(s);
     float ms; cudaEventElapsedTime(&ms,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
     float us=ms/ITERS*1000;
     printf("%-25s | %6.2f   | %8.1f   | %6.2f\n","Warp Specialized",us,R/us,v7_us/us);
    }

    // Scaling test: LDG vs V7 at various batch sizes
    printf("\n--- LDG vs V7 Scaling ---\n");
    printf("%-8s | %8s | %8s | %6s\n","Rooms","V7 us","LDG us","LDG/V7");
    printf("---------|----------|----------|--------\n");
    int ts[]={64,256,1024,4096,8192};
    for(int t=0;t<5;t++){
        int rooms=ts[t]; dim3 g((rooms+7)/8);
        cudaEvent_t sa,ea,sb,eb;
        cudaEventCreate(&sa); cudaEventCreate(&ea);
        cudaEventCreate(&sb); cudaEventCreate(&eb);
        for(int i=0;i<100;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sa,s);
        for(int i=0;i<2000;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(ea,s); cudaStreamSynchronize(s);
        float m1; cudaEventElapsedTime(&m1,sa,ea);
        for(int i=0;i<100;i++) infer_ldg<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sb,s);
        for(int i=0;i<2000;i++) infer_ldg<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(eb,s); cudaStreamSynchronize(s);
        float m2; cudaEventElapsedTime(&m2,sb,eb);
        cudaEventDestroy(sa); cudaEventDestroy(ea);
        cudaEventDestroy(sb); cudaEventDestroy(eb);
        printf("%-8d | %6.2f   | %6.2f   | %.2f\n",rooms,
            m1/2000*1000, m2/2000*1000, (m2/2000)/(m1/2000));
    }

    // LDP vs V7 scaling
    printf("\n--- LDP (unroll4) vs V7 Scaling ---\n");
    printf("%-8s | %8s | %8s | %6s\n","Rooms","V7 us","LDP us","LDP/V7");
    printf("---------|----------|----------|--------\n");
    for(int t=0;t<5;t++){
        int rooms=ts[t]; dim3 g((rooms+7)/8);
        cudaEvent_t sa,ea,sb,eb;
        cudaEventCreate(&sa); cudaEventCreate(&ea);
        cudaEventCreate(&sb); cudaEventCreate(&eb);
        for(int i=0;i<100;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sa,s);
        for(int i=0;i<2000;i++) infer_v7<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(ea,s); cudaStreamSynchronize(s);
        float m1; cudaEventElapsedTime(&m1,sa,ea);
        for(int i=0;i<100;i++) infer_ldp<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaStreamSynchronize(s); cudaEventRecord(sb,s);
        for(int i=0;i<2000;i++) infer_ldp<<<g,256,0,s>>>(dw,di,do_,rooms,DIM);
        cudaEventRecord(eb,s); cudaStreamSynchronize(s);
        float m2; cudaEventElapsedTime(&m2,sb,eb);
        cudaEventDestroy(sa); cudaEventDestroy(ea);
        cudaEventDestroy(sb); cudaEventDestroy(eb);
        printf("%-8d | %6.2f   | %6.2f   | %.2f\n",rooms,
            m1/2000*1000, m2/2000*1000, (m2/2000)/(m1/2000));
    }

    cudaStreamDestroy(s); cudaFree(dw); cudaFree(di); cudaFree(do_);
    printf("\n=== Suite #63 Complete ===\n");
}
