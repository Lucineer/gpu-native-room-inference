#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cusparse.h>
#include <cublas_v2.h>
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

int main(){
    printf("=== Suite #60: cuSPARSE SpMV + cuBLAS GEMM vs Custom ===\n\n");
    int R=4096;
    cudaStream_t s; cudaStreamCreate(&s);
    dim3 grid((R+7)/8);

    half*dw,*di; float*do_;
    cudaMalloc(&dw,(size_t)R*DIM*2); cudaMalloc(&di,DIM*2); cudaMalloc(&do_,R*4);
    std::vector<half> hw(R*DIM);
    for(size_t i=0;i<(size_t)R*DIM;i++)
        hw[i]=__float2half(0.01f*(2.0f*(float)rand()/RAND_MAX-1.0f));
    cudaMemcpy(dw,hw.data(),R*DIM*2,cudaMemcpyHostToDevice);
    std::vector<half> hi(DIM);
    for(int i=0;i<DIM;i++) hi[i]=__float2half(0.5f*cosf((float)i/DIM*6.2832f));
    cudaMemcpy(di,hi.data(),DIM*2,cudaMemcpyHostToDevice);

    // V7 baseline
    printf("--- V7 Custom Kernel (baseline) ---\n");
    cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
    for(int i=0;i<WARMUP;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaStreamSynchronize(s); cudaEventRecord(s1,s);
    for(int i=0;i<ITERS;i++) infer_v7<<<grid,256,0,s>>>(dw,di,do_,R,DIM);
    cudaEventRecord(e1,s); cudaStreamSynchronize(s);
    float ms1; cudaEventElapsedTime(&ms1,s1,e1); cudaEventDestroy(s1); cudaEventDestroy(e1);
    printf("  V7: %.2f us, %.1f M qps\n", ms1/ITERS*1000, R/(ms1/ITERS*1000));

    // cuBLAS GEMM
    printf("\n--- cuBLAS GEMM ---\n");
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    float* d_out_blas;
    cudaMalloc(&d_out_blas, R*4);
    float alpha32=1.0f, beta32=0.0f;

    for(int i=0;i<50;i++){
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            R, 1, DIM, &alpha32,
            dw, CUDA_R_16F, R,
            di, CUDA_R_16F, DIM,
            &beta32, d_out_blas, CUDA_R_32F, R,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaStreamSynchronize(s);
    cudaEvent_t s2,e2; cudaEventCreate(&s2); cudaEventCreate(&e2);
    cudaEventRecord(s2,s);
    for(int i=0;i<ITERS;i++){
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            R, 1, DIM, &alpha32,
            dw, CUDA_R_16F, R,
            di, CUDA_R_16F, DIM,
            &beta32, d_out_blas, CUDA_R_32F, R,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(e2,s); cudaStreamSynchronize(s);
    float ms2; cudaEventElapsedTime(&ms2,s2,e2); cudaEventDestroy(s2); cudaEventDestroy(e2);
    printf("  cuBLAS: %.2f us, %.1f M qps\n", ms2/ITERS*1000, R/(ms2/ITERS*1000));
    printf("  vs V7: %.2fx\n", ms1/ms2);

    // cuSPARSE SpMV (dense matrix, not sparse)
    printf("\n--- cuSPARSE SpMV (dense CSR, no sparsity) ---\n");
    cusparseHandle_t cusparse;
    cusparseCreate(&cusparse);

    int nnz=R*DIM;
    int* d_row_ptr; int* d_col_ind; half* d_csr_vals;
    cudaMalloc(&d_row_ptr,(R+1)*4);
    cudaMalloc(&d_col_ind,nnz*4);
    cudaMalloc(&d_csr_vals,nnz*2);

    // Build CSR (dense, all elements)
    std::vector<int> row_ptr(R+1), col_ind(nnz);
    int idx=0;
    for(int r=0;r<R;r++){
        row_ptr[r]=idx;
        for(int d=0;d<DIM;d++){col_ind[idx]=d; idx++;}
    }
    row_ptr[R]=idx;
    cudaMemcpy(d_row_ptr,row_ptr.data(),(R+1)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind,col_ind.data(),nnz*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_vals,hw.data(),nnz*2,cudaMemcpyHostToDevice);

    // Create descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    float* d_out_sp;
    cudaMalloc(&d_out_sp, R*4);

    cusparseCreateCsr(&matA, R, DIM, nnz,
        d_row_ptr, d_col_ind, d_csr_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnVec(&vecX, DIM, di, CUDA_R_16F);
    cusparseCreateDnVec(&vecY, R, d_out_sp, CUDA_R_32F);

    // Buffer
    size_t bufSize=0;
    float alpha=1.0f, beta=0.0f;
    cusparseSpMV_bufferSize(cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
    void* dBuf=NULL;
    if(bufSize>0) cudaMalloc(&dBuf, bufSize);

    cusparseSpMV_preprocess(cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuf);

    for(int i=0;i<50;i++){
        cusparseSpMV(cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuf);
    }
    cudaStreamSynchronize(s);

    cudaEvent_t s3,e3; cudaEventCreate(&s3); cudaEventCreate(&e3);
    cudaEventRecord(s3,s);
    for(int i=0;i<ITERS;i++){
        cusparseSpMV(cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuf);
    }
    cudaEventRecord(e3,s); cudaStreamSynchronize(s);
    float ms3; cudaEventElapsedTime(&ms3,s3,e3); cudaEventDestroy(s3); cudaEventDestroy(e3);
    printf("  Dense CSR SpMV: %.2f us, %.1f M qps\n", ms3/ITERS*1000, R/(ms3/ITERS*1000));
    printf("  vs V7: %.2fx\n", ms1/ms3);

    // Now 50% sparse CSR (every other element zeroed)
    printf("\n--- cuSPARSE SpMV (50%% sparse, 2:4 pattern) ---\n");
    int nnz_half=R*DIM/2;
    std::vector<int> col_half(nnz_half);
    std::vector<half> vals_half(nnz_half);
    std::vector<int> row_half(R+1);
    idx=0;
    for(int r=0;r<R;r++){
        row_half[r]=idx;
        for(int d=0;d<DIM;d++){
            if((d%4)<2){ // keep 2 of every 4
                col_half[idx]=d;
                vals_half[idx]=hw[r*DIM+d];
                idx++;
            }
        }
    }
    row_half[R]=idx;

    int* d_row2,*d_col2; half* d_val2;
    cudaMalloc(&d_row2,(R+1)*4); cudaMalloc(&d_col2,nnz_half*4);
    cudaMalloc(&d_val2,nnz_half*2);
    cudaMemcpy(d_row2,row_half.data(),(R+1)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_col2,col_half.data(),nnz_half*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_val2,vals_half.data(),nnz_half*2,cudaMemcpyHostToDevice);

    cusparseSpMatDescr_t matA2;
    cusparseDnVecDescr_t vecY2;
    float* d_out_sp2;
    cudaMalloc(&d_out_sp2, R*4);
    cusparseCreateCsr(&matA2, R, DIM, nnz_half,
        d_row2, d_col2, d_val2,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnVec(&vecY2, R, d_out_sp2, CUDA_R_32F);

    size_t buf2=0;
    cusparseSpMV_bufferSize(cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA2, vecX, &beta, vecY2,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf2);
    void* dBuf2=NULL;
    if(buf2>0) cudaMalloc(&dBuf2, buf2);

    cusparseSpMV_preprocess(cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA2, vecX, &beta, vecY2,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuf2);

    for(int i=0;i<50;i++){
        cusparseSpMV(cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA2, vecX, &beta, vecY2,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuf2);
    }
    cudaStreamSynchronize(s);

    cudaEvent_t s4,e4; cudaEventCreate(&s4); cudaEventCreate(&e4);
    cudaEventRecord(s4,s);
    for(int i=0;i<ITERS;i++){
        cusparseSpMV(cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA2, vecX, &beta, vecY2,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuf2);
    }
    cudaEventRecord(e4,s); cudaStreamSynchronize(s);
    float ms4; cudaEventElapsedTime(&ms4,s4,e4); cudaEventDestroy(s4); cudaEventDestroy(e4);
    printf("  50%% sparse CSR: %.2f us, %.1f M qps\n", ms4/ITERS*1000, R/(ms4/ITERS*1000));
    printf("  vs V7: %.2fx\n", ms1/ms4);
    printf("  Memory: %d KB sparse vs %d KB dense (%.1fx)\n",
        (nnz_half*2+nnz_half*4+(R+1)*4)/1024, R*DIM*2/1024,
        (float)(R*DIM*2)/(nnz_half*2+nnz_half*4+(R+1)*4));

    // cuBLAS at various batch sizes
    printf("\n--- cuBLAS vs V7 at Various Batch Sizes ---\n");
    printf("%-8s | %8s | %8s | %6s\n","Rooms","V7 us","BLAS us","BLAS/V7");
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
        float msa; cudaEventElapsedTime(&msa,sa,ea);
        for(int i=0;i<100;i++){
            cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                rooms, 1, DIM, &alpha32,
                dw, CUDA_R_16F, rooms,
                di, CUDA_R_16F, DIM,
                &beta32, d_out_blas, CUDA_R_32F, rooms,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaStreamSynchronize(s); cudaEventRecord(sb,s);
        for(int i=0;i<2000;i++){
            cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                rooms, 1, DIM, &alpha32,
                dw, CUDA_R_16F, rooms,
                di, CUDA_R_16F, DIM,
                &beta32, d_out_blas, CUDA_R_32F, rooms,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(eb,s); cudaStreamSynchronize(s);
        float msb; cudaEventElapsedTime(&msb,sb,eb);
        cudaEventDestroy(sa); cudaEventDestroy(ea);
        cudaEventDestroy(sb); cudaEventDestroy(eb);
        printf("%-8d | %6.2f   | %6.2f   | %.2f\n",rooms,
            msa/2000*1000, msb/2000*1000, (msb/2000)/(msa/2000));
    }

    cublasDestroy(cublas);
    cusparseDestroySpMat(matA); cusparseDestroySpMat(matA2);
    cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecY2);
    cusparseDestroy(cusparse);
    cudaFree(dw); cudaFree(di); cudaFree(do_);
    cudaFree(d_out_blas); cudaFree(d_out_sp); cudaFree(d_out_sp2);
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_csr_vals);
    cudaFree(d_row2); cudaFree(d_col2); cudaFree(d_val2);
    if(dBuf) cudaFree(dBuf);
    if(dBuf2) cudaFree(dBuf2);
    cudaStreamDestroy(s);
    printf("\n=== Suite #60 Complete ===\n");
}
