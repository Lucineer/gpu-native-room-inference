#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda::wmma;

__global__ void tc_matvec(const half* w, const half* in, float* out, int nr) {
    __shared__ alignas(128) half sa[16], sb[16];
    fragment<matrix_a,16,16,16,half,row_major> af;
    fragment<matrix_b,16,16,16,half,col_major> bf;
    fragment<accumulator,16,16,16,float> cf;
    int rid = blockIdx.x, lane = threadIdx.x;
    if (rid >= nr) return;
    const half* ww = w + rid * 16 * 256;
    for (int n = 0; n < 16; n++) {
        float sum = 0.0f;
        for (int c = 0; c < 16; c++) {
            int koff = c * 16;
            if (lane < 16) {
                sa[lane] = in[koff + lane];
                sb[lane] = ww[n * 256 + koff + lane];
            }
            __syncwarp();
            load_matrix_sync(af, sa, 16);
            load_matrix_sync(bf, sb, 16);
            fill_fragment(cf, 0.0f);
            mma_sync(cf, af, bf, cf);
            if (lane == 0) sum += cf.x[0];
        }
        if (lane == 0) out[rid * 16 + n] = sum;
    }
}

__global__ void warp_matvec(const half* w, const half* in, float* out, int nr) {
    int rid = blockIdx.x, lane = threadIdx.x;
    if (rid >= nr) return;
    const half* ww = w + rid * 16 * 256;
    for (int n = 0; n < 16; n++) {
        float s = 0.0f;
        for (int k = lane; k < 256; k += 32)
            s += __half2float(__hmul(ww[n * 256 + k], in[k]));
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) out[rid * 16 + n] = s;
    }
}

int main() {
    int NR = 12;
    size_t wb = NR * 16 * 256 * sizeof(half);
    half *hw = (half*)malloc(wb);
    half *hi = (half*)malloc(256 * sizeof(half));
    float *h_tc = (float*)calloc(NR * 16, sizeof(float));
    float *h_w = (float*)calloc(NR * 16, sizeof(float));
    srand(42);
    for (size_t i = 0; i < NR * 16 * 256; i++)
        hw[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < 256; i++)
        hi[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    half *dw, *di; float *d1, *d2;
    cudaMalloc(&dw, wb); cudaMalloc(&di, 256 * sizeof(half));
    cudaMalloc(&d1, NR * 16 * sizeof(float));
    cudaMalloc(&d2, NR * 16 * sizeof(float));
    cudaMemcpy(dw, hw, wb, cudaMemcpyHostToDevice);
    cudaMemcpy(di, hi, 256 * sizeof(half), cudaMemcpyHostToDevice);

    tc_matvec<<<NR, 32>>>(dw, di, d1, NR);
    warp_matvec<<<NR, 32>>>(dw, di, d2, NR);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tc, d1, NR * 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w, d2, NR * 16 * sizeof(float), cudaMemcpyDeviceToHost);

    float maxd = 0;
    for (int i = 0; i < NR * 16; i++) {
        float d = fabsf(h_w[i] - h_tc[i]);
        if (d > maxd) maxd = d;
    }
    printf("12 rooms, 16 neurons, 256 weights: max_diff=%.6f %s\n", maxd, maxd < 0.001 ? "OK" : "BAD");
    if (maxd >= 0.001) {
        for (int r = 0; r < 3; r++)
            for (int n = 0; n < 4; n++) {
                int idx = r * 16 + n;
                printf("  [%d][%d] warp=%.6f tc=%.6f diff=%.4f\n", r, n, h_w[idx], h_tc[idx], fabsf(h_w[idx]-h_tc[idx]));
            }
    }
    cudaFree(dw); cudaFree(di); cudaFree(d1); cudaFree(d2);
    free(hw); free(hi); free(h_tc); free(h_w);
    return 0;
}
