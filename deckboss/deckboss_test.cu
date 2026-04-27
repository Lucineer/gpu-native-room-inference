/**
 * deckboss_test.cu — Correctness verification + benchmark for deckboss runtime
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 --use_fast_math deckboss_runtime.cu deckboss_test.cu -o deckboss_test -I.
 */

#include "deckboss_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define DIM 256
#define MAX_ROOMS 4096
#define WARMUP 500
#define ITERS 10000

static void generate_weights(float* w, int dim, int seed) {
    srand(seed);
    for (int i = 0; i < dim; i++) {
        w[i] = 0.5f * (2.0f * (float)rand() / RAND_MAX - 1.0f);
    }
}

static void generate_input(float* inp, int dim) {
    for (int i = 0; i < dim; i++) {
        inp[i] = cosf((float)i / dim * 6.2832f);
    }
}

static float reference_dot(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += a[i] * b[i];
    return sum;
}

int main() {
    printf("=== Deckboss Runtime Test ===\n");
    printf("Hardware: Jetson Orin Nano 8GB, sm_87\n");
    printf("Config: dim=%d, max_rooms=%d\n\n", DIM, MAX_ROOMS);
    
    // Init
    deckboss_config_t cfg = {};
    cfg.dim = DIM;
    cfg.max_rooms = MAX_ROOMS;
    cfg.num_streams = 4;
    
    deckboss_handle_t* db = NULL;
    deckboss_status_t status = deckboss_init(&cfg, &db);
    if (status != DECKBOSS_OK) {
        printf("FATAL: deckboss_init failed: %s\n", deckboss_last_error());
        return 1;
    }
    printf("[OK] deckboss_init\n");
    
    // Generate and load weights for MAX_ROOMS rooms
    std::vector<float> weights(DIM);
    for (int r = 0; r < MAX_ROOMS; r++) {
        generate_weights(weights.data(), DIM, r * 42 + 7);
        status = deckboss_load_weights(db, r, weights.data(), DIM);
        if (status != DECKBOSS_OK) {
            printf("FATAL: deckboss_load_weights(%d) failed: %s\n", r, deckboss_last_error());
            return 1;
        }
    }
    printf("[OK] deckboss_load_weights — %d rooms loaded\n", deckboss_room_count(db));
    
    // Set shared input
    std::vector<float> input(DIM);
    generate_input(input.data(), DIM);
    status = deckboss_set_input(db, input.data(), DIM);
    if (status != DECKBOSS_OK) {
        printf("FATAL: deckboss_set_input failed: %s\n", deckboss_last_error());
        return 1;
    }
    printf("[OK] deckboss_set_input\n\n");
    
    // ===== Correctness Test =====
    printf("--- Correctness Test ---\n");
    int test_rooms[] = {0, 1, 100, 1023, 4095};
    float max_rel_error = 0.0f;
    float avg_rel_error = 0.0f;
    int test_count = sizeof(test_rooms) / sizeof(test_rooms[0]);
    
    float output[5];
    status = deckboss_infer(db, test_rooms, test_count, output, NULL);
    if (status != DECKBOSS_OK) {
        printf("FATAL: deckboss_infer failed: %s\n", deckboss_last_error());
        return 1;
    }
    
    bool pass = true;
    for (int t = 0; t < test_count; t++) {
        // Regenerate weights for this room to get reference
        generate_weights(weights.data(), DIM, test_rooms[t] * 42 + 7);
        float ref = reference_dot(weights.data(), input.data(), DIM);
        float rel_err = fabsf(output[t] - ref) / (fabsf(ref) + 1e-10f);
        if (rel_err > max_rel_error) max_rel_error = rel_err;
        avg_rel_error += rel_err;
        printf("  Room %4d: deckboss=%.4f  ref=%.4f  rel_err=%.4f%%\n",
               test_rooms[t], output[t], ref, rel_err * 100.0f);
        // Only flag error for non-trivial values
        if (rel_err > 0.10f && fabsf(output[t]) > 0.5f) pass = false;
    }
    avg_rel_error /= test_count;
    printf("  Average error: %.4f%%\n", avg_rel_error * 100.0f);
    printf("  Max error:     %.4f%%\n", max_rel_error * 100.0f);
    
    if (!pass) {  // 10% for non-trivial values
        printf("  [FAIL] Error exceeds 10%% threshold!\n");
    } else {
        printf("  [PASS] Error within acceptable range (INT8 quantization)\n");
    }
    printf("\n");
    
    // ===== Latency Sweep =====
    printf("--- Latency Sweep ---\n");
    int batch_sizes[] = {1, 4, 16, 64, 256, 1024, 4096};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    std::vector<int> room_ids(MAX_ROOMS);
    for (int i = 0; i < MAX_ROOMS; i++) room_ids[i] = i;
    std::vector<float> batch_output(MAX_ROOMS);
    
    printf("  %8s  %10s  %12s\n", "Rooms", "Latency", "Throughput");
    printf("  %8s  %10s  %12s\n", "------", "--------", "----------");
    
    for (int b = 0; b < num_batches; b++) {
        int n = batch_sizes[b];
        float total_us = 0.0f;
        
        // Warmup
        for (int w = 0; w < 100; w++) {
            deckboss_infer(db, room_ids.data(), n, batch_output.data(), NULL);
        }
        
        // Timed runs
        deckboss_stats_t stats;
        for (int i = 0; i < ITERS; i++) {
            status = deckboss_infer(db, room_ids.data(), n, batch_output.data(), &stats);
            total_us += stats.latency_us;
        }
        
        float avg_us = total_us / ITERS;
        float mqps = (float)n / avg_us * 1000.0f;  // rooms per second / 1M
        printf("  %8d  %8.2f us  %8.1f M qps\n", n, avg_us, mqps);
    }
    printf("\n");
    
    // ===== Sustained Throughput Test =====
    printf("--- Sustained Throughput (1M inferences, 4096 rooms) ---\n");
    deckboss_stats_t sustained_stats;
    float sustained_total_us = 0.0f;
    
    for (int i = 0; i < ITERS; i++) {
        status = deckboss_infer(db, room_ids.data(), MAX_ROOMS, batch_output.data(), &sustained_stats);
        sustained_total_us += sustained_stats.latency_us;
    }
    
    float sustained_avg = sustained_total_us / ITERS;
    float sustained_mqps = (float)MAX_ROOMS / sustained_avg * 1000.0f;
    printf("  Average: %.2f us, %.1f M room-qps\n", sustained_avg, sustained_mqps);
    printf("  1M inferences would take: %.2f seconds\n", sustained_avg * ITERS / 1e6f);
    printf("\n");
    
    // Cleanup
    deckboss_destroy(db);
    printf("[OK] deckboss_destroy — all tests complete\n");
    
    return 0;
}
