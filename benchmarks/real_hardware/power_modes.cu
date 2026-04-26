#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_fp16.h>
#define DIM 256
__global__ void infer(const half* w, const half* inp, float* out, int n, int d) {
    int rb = blockIdx.x * 8; int ri = threadIdx.x / 32; int lane = threadIdx.x % 32;
    if (rb + ri >= n) return;
    float sum = 0.0f;
    for (int i = lane; i < d; i += 32)
        sum += __half2float(w[(rb + ri) * d + i]) * __half2float(inp[i]);
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) out[rb + ri] = sum;
}
int main() {
    printf("=== Suite #56: Power Modes & Clock Scaling ===\n\n");
    int ROOMS = 4096;
    half *d_w, *d_i; float *d_o;
    cudaMalloc(&d_w, (size_t)ROOMS * DIM * sizeof(half));
    cudaMalloc(&d_i, DIM * sizeof(half));
    cudaMalloc(&d_o, ROOMS * sizeof(float));
    std::vector<half> hw(ROOMS * DIM);
    for (size_t i = 0; i < (size_t)ROOMS * DIM; i++) hw[i] = __float2half(0.01f);
    cudaMemcpy(d_w, hw.data(), ROOMS * DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((ROOMS + 7) / 8);
    // Sustained
    printf("--- Sustained Load (10s, 4096 rooms, dim=256) ---\n");
    printf("Sec | us/iter | M qps\n");
    printf("----|---------|--------\n");
    auto t0 = std::chrono::steady_clock::now();
    int total = 0;
    for (int sec = 0; sec < 10; sec++) {
        auto ss = std::chrono::steady_clock::now(); int sc = 0;
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - ss).count() < 1000) {
            infer<<<grid, 256, 0, s>>>(d_w, d_i, d_o, ROOMS, DIM); sc++;
        }
        cudaStreamSynchronize(s);
        float sms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - ss).count();
        printf("%3d | %6.2f  | %6.1f\n", sec, sms / sc * 1000, ROOMS / (sms / sc * 1000));
        total += sc;
    }
    float ts = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
    printf("\nTotal: %.1fs, %d iters, %.1f M room-qps sustained\n", ts, total, ROOMS * total / ts / 1e6);
    // Latency dist
    printf("\n--- Latency Distribution (4096 rooms, 5000 samples) ---\n");
    std::vector<float> al(5000);
    cudaEvent_t se, ee; cudaEventCreate(&se); cudaEventCreate(&ee);
    for (int i = 0; i < 200; i++) infer<<<grid, 256, 0, s>>>(d_w, d_i, d_o, ROOMS, DIM);
    cudaStreamSynchronize(s);
    for (int i = 0; i < 5000; i++) {
        cudaEventRecord(se, s); infer<<<grid, 256, 0, s>>>(d_w, d_i, d_o, ROOMS, DIM);
        cudaEventRecord(ee, s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms, se, ee); al[i] = ms * 1000;
    }
    cudaEventDestroy(se); cudaEventDestroy(ee);
    std::sort(al.begin(), al.end());
    double dsum = 0; for (auto& v : al) dsum += v;
    printf("p50=%.2f p75=%.2f p90=%.2f p95=%.2f p99=%.2f p999=%.2f\n",
        al[2500], al[3750], al[4500], al[4750], al[4950], al[4995]);
    printf("min=%.2f max=%.2f avg=%.2f p99/p50=%.3f\n",
        al[0], al[4999], dsum / 5000, al[4950] / al[2500]);
    int outl = 0; for (auto& v : al) if (v > al[4950] * 2) outl++;
    printf("Outliers (>2x p99): %d (%.3f%%)\n", outl, 100.0 * outl / 5000);
    // Room scaling
    printf("\n--- Throughput by Room Count ---\n");
    printf("Rooms | us | M qps\n");
    printf("------|------|------\n");
    int tr[] = {1,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
    for (int t = 0; t < 15; t++) {
        dim3 g((tr[t]+7)/8);
        cudaEvent_t s3,e3; cudaEventCreate(&s3); cudaEventCreate(&e3);
        for (int i=0;i<200;i++) infer<<<g,256,0,s>>>(d_w,d_i,d_o,tr[t],DIM);
        cudaStreamSynchronize(s);
        cudaEventRecord(s3,s);
        for (int i=0;i<2000;i++) infer<<<g,256,0,s>>>(d_w,d_i,d_o,tr[t],DIM);
        cudaEventRecord(e3,s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms,s3,e3);
        cudaEventDestroy(s3); cudaEventDestroy(e3);
        printf("%6d | %6.2f | %6.1f\n", tr[t], ms/2000*1000, tr[t]/(ms/2000*1000));
    }
    cudaStreamDestroy(s); cudaFree(d_w); cudaFree(d_i); cudaFree(d_o);
    printf("\n=== Suite #56 Complete ===\n");
}
