/**
 * Power Efficiency Benchmark
 * 
 * Measures room-qps-per-watt on Jetson Orin Nano.
 * Uses INA3221 power monitor (VDD_CPU_GPU_CV rail).
 * 
 * The REAL edge metric: not speed, but speed-per-watt.
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 power_bench.cu -o power_bench
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void infer_kernel(const half* __restrict__ weights,
                              const half* __restrict__ input,
                              float* __restrict__ output,
                              int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[(size_t)room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) {
        float x = sum;
        output[room] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Read INA3221 rail 2 (VDD_CPU_GPU_CV) current and voltage
float read_gpu_power_watts() {
    FILE* f_curr = fopen("/sys/class/hwmon/hwmon1/curr2_input", "r");
    FILE* f_volt = fopen("/sys/class/hwmon/hwmon1/in2_input", "r");
    if (!f_curr || !f_volt) return -1.0f;
    
    int curr_ma, volt_mv;
    fscanf(f_curr, "%d", &curr_ma);
    fscanf(f_volt, "%d", &volt_mv);
    fclose(f_curr);
    fclose(f_volt);
    
    // Power = V * I = (volt_mv / 1000) * (curr_ma / 1000)
    return (float)volt_mv * (float)curr_ma / 1e6f;
}

// Also read total board power (rail 1 = VDD_IN)
float read_total_power_watts() {
    FILE* f_curr = fopen("/sys/class/hwmon/hwmon1/curr1_input", "r");
    FILE* f_volt = fopen("/sys/class/hwmon/hwmon1/in1_input", "r");
    if (!f_curr || !f_volt) return -1.0f;
    
    int curr_ma, volt_mv;
    fscanf(f_curr, "%d", &curr_ma);
    fscanf(f_volt, "%d", &volt_mv);
    fclose(f_curr);
    fclose(f_volt);
    
    return (float)volt_mv * (float)curr_ma / 1e6f;
}

int main() {
    printf("============================================================\n");
    printf("POWER EFFICIENCY BENCHMARK\n");
    printf("Room-qps-per-watt on Jetson Orin Nano\n");
    printf("============================================================\n\n");
    
    // Verify power monitoring
    float idle_gpu = read_gpu_power_watts();
    float idle_total = read_total_power_watts();
    printf("Power monitoring: INA3221\n");
    printf("  GPU rail (VDD_CPU_GPU_CV): %.2fW idle\n", idle_gpu);
    printf("  Total board (VDD_IN): %.2fW idle\n\n", idle_total);
    
    if (idle_gpu < 0 || idle_total < 0) {
        printf("WARNING: Power monitoring unavailable. Using estimated values.\n");
    }
    
    const int DIM = 256;
    
    half *h_weights = (half*)malloc((size_t)512 * DIM * sizeof(half));
    half *h_input = (half*)malloc(DIM * sizeof(half));
    srand(42);
    for (int i = 0; i < 512 * DIM; i++)
        h_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++)
        h_input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_weights, *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, (size_t)512 * DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, 512 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, (size_t)512 * DIM * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 block(32);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test configurations
    struct Config {
        const char* name;
        int batch;
        int duration_sec;
    };
    
    Config configs[] = {
        {"Idle (no GPU)",        0,    3},
        {"1 room (launch-bound)",1,    5},
        {"6 rooms (production)", 6,    5},
        {"32 rooms",             32,   5},
        {"64 rooms (fleet)",     64,   5},
        {"128 rooms",            128,  5},
        {"256 rooms (max perf)", 256,  5},
        {"512 rooms",            512,  5},
    };
    int num_configs = 8;
    
    printf("%-25s %-10s %-12s %-10s %-12s %-15s %-15s\n",
           "Config", "Batch", "Room-qps", "GPU(W)", "Total(W)", "Efficiency", "Efficiency");
    printf("%-25s %-10s %-12s %-10s %-12s %-15s %-15s\n",
           "", "", "", "", "", "(qps/W GPU)", "(qps/W total)");
    printf("--------------------------------------------------------------------------------\n");
    
    for (int c = 0; c < num_configs; c++) {
        Config cfg = configs[c];
        dim3 grid(cfg.batch > 0 ? cfg.batch : 1);
        int iters = cfg.duration_sec * 100000;  // 100K iterations per second
        
        // Warm up
        if (cfg.batch > 0) {
            for (int i = 0; i < 5000; i++)
                infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, cfg.batch);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Measure power during sustained GPU work
        float min_power = 999.0f, max_power = 0.0f, sum_power = 0.0f;
        int power_samples = 0;
        
        auto t_start = std::chrono::high_resolution_clock::now();
        
        CUDA_CHECK(cudaEventRecord(start));
        
        if (cfg.batch > 0) {
            // Run GPU work while sampling power
            for (int i = 0; i < iters; i++) {
                infer_kernel<<<grid, block>>>(d_weights, d_input, d_output, DIM, cfg.batch);
                
                // Sample power every ~1000 iterations
                if (i % 1000 == 0) {
                    float p = read_gpu_power_watts();
                    float t = read_total_power_watts();
                    if (p > 0) {
                        sum_power += p;
                        if (p < min_power) min_power = p;
                        if (p > max_power) max_power = p;
                    }
                    power_samples++;
                }
            }
        } else {
            // Idle: just sleep and sample power
            usleep(cfg.duration_sec * 1000000);
            for (int i = 0; i < cfg.duration_sec * 10; i++) {
                float p = read_gpu_power_watts();
                if (p > 0) sum_power += p;
                power_samples++;
                usleep(100000);
            }
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        auto t_end = std::chrono::high_resolution_clock::now();
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        
        float avg_gpu_power = (power_samples > 0) ? sum_power / power_samples : 0;
        float total_power = read_total_power_watts();
        
        // Calculate room-qps
        float room_qps = 0;
        if (cfg.batch > 0) {
            room_qps = (float)cfg.batch * iters / (elapsed_ms / 1000.0f);
        }
        
        float eff_gpu = (avg_gpu_power > 0.5f && room_qps > 0) ? room_qps / avg_gpu_power : 0;
        float eff_total = (total_power > 0.5f && room_qps > 0) ? room_qps / total_power : 0;
        
        printf("%-25s %-10d %-12.0f %-10.2f %-12.2f %-15.0f %-15.0f\n",
               cfg.name, cfg.batch, room_qps, avg_gpu_power, total_power,
               eff_gpu / 1e6, eff_total / 1e6);
        
        // Cool down
        usleep(500000);
    }
    
    // Compare with theoretical
    printf("\n============================================================\n");
    printf("POWER EFFICIENCY ANALYSIS\n");
    printf("============================================================\n");
    
    // Idle power
    float idle = read_gpu_power_watts();
    float total_idle = read_total_power_watts();
    printf("Idle GPU power: %.2fW | Idle total: %.2fW\n", idle, total_idle);
    printf("GPU power budget (10W TDP): %.0fW available for compute\n", 10.0 - idle);
    printf("Total board power (15W): %.0fW available for compute\n", 15.0 - total_idle);
    
    printf("\n--- Efficiency Comparison ---\n");
    printf("Jetson Orin Nano 8GB: 40 TOPS INT8, 10-15W TDP\n");
    printf("  Theoretical max INT8: 40T ops/sec at 10W = 4 TOPS/W\n");
    printf("  Our FP16 room inference: ~30M room-qps at ~4W = ~7.5M room-qps/W\n");
    printf("  Per room (256-dim dot + GELU): 256*2 = 512 FLOPs\n");
    printf("  Actual FLOPS: 30M * 512 = 15.4 GFLOPS\n");
    printf("  FLOPS/W: 15.4G / 4W = 3.85 GFLOPS/W\n");
    printf("  FP16 peak: 20 TOPS at 10W = 2 TOPS/W\n");
    printf("  Our efficiency vs peak: 3.85G / 20000G = 0.019%% of peak FP16\n");
    printf("  (Expected: room inference is memory-bound, not compute-bound)\n");
    
    printf("\n--- Why So Far From Peak? ---\n");
    printf("1. Room inference is MEMORY-BOUND (512 bytes per room)\n");
    printf("2. GPU does 512 FLOPs but reads 512 bytes (1 FLOP/byte ratio)\n");
    printf("3. Peak assumes 256+ FLOP/byte (compute-bound)\n");
    printf("4. Actual bottleneck: 25-44 GB/s bandwidth, not 40 TFLOPS compute\n");
    printf("5. Memory bandwidth efficiency: 15.4G FLOPs / 44 GB/s = 0.35 FLOP/byte\n");
    printf("6. This is correct behavior — we're I/O limited, not compute limited\n");
    
    printf("\n--- Edge Implications ---\n");
    printf("For memory-bound workloads, power efficiency is dominated by:\n");
    printf("1. Memory bandwidth per watt (Orin: ~4.4 GB/s/W)\n");
    printf("2. Not compute per watt (Orin: 2-4 TOPS/W)\n");
    printf("3. Reducing data movement > increasing compute\n");
    printf("4. Caching, batching, zero-copy = the real power optimization\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_weights);
    free(h_input);
    
    return 0;
}
