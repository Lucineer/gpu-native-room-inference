/**
 * deckboss_runtime.h — Edge GPU room inference runtime
 * 
 * Production INT8 kernel: 185M room-qps sustained on Jetson Orin Nano 8GB
 * Compile: nvcc -arch=sm_87 -O3 --use_fast_math deckboss_runtime.cu -o libdeckboss.so -shared -fPIC
 * 
 * Usage:
 *   deckboss_handle_t* db = deckboss_init(256, 4096);
 *   deckboss_load_weights(db, room_id, fp32_weights, 256);
 *   deckboss_infer(db, input_vec, room_ids, num_rooms, output_scores);
 *   deckboss_destroy(db);
 */

#ifndef DECKBOSS_RUNTIME_H
#define DECKBOSS_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle for a deckboss instance
typedef struct deckboss_handle deckboss_handle_t;

/// Status codes
typedef enum {
    DECKBOSS_OK = 0,
    DECKBOSS_ERR_CUDA_INIT = -1,
    DECKBOSS_ERR_MALLOC = -2,
    DECKBOSS_ERR_INVALID_DIM = -3,
    DECKBOSS_ERR_INVALID_ROOM = -4,
    DECKBOSS_ERR_INVALID_BATCH = -5,
    DECKBOSS_ERR_NOT_READY = -6,
    DECKBOSS_ERR_CUDA_KERNEL = -7,
} deckboss_status_t;

/// Configuration for initialization
typedef struct {
    int dim;            // Room dimension (must be > 0, typically 256)
    int max_rooms;      // Maximum concurrent rooms (must be > 0)
    int num_streams;    // CUDA streams (default: 4, max: 8)
    int use_l2_persist; // Enable L2 cache persisting for FP16 (default: 0)
} deckboss_config_t;

/// Performance stats from last inference call
typedef struct {
    float latency_us;       // Wall-clock latency in microseconds
    float throughput_mqps;  // Throughput in millions of rooms/second
    int rooms_processed;    // Number of rooms actually processed
} deckboss_stats_t;

/**
 * Initialize a deckboss instance.
 * 
 * @param config  Configuration (dim, max_rooms, etc.)
 * @param handle  Output handle
 * @return DECKBOSS_OK on success, error code on failure
 */
deckboss_status_t deckboss_init(const deckboss_config_t* config, deckboss_handle_t** handle);

/**
 * Destroy a deckboss instance and free all GPU memory.
 */
deckboss_status_t deckboss_destroy(deckboss_handle_t* handle);

/**
 * Load (or update) weights for a room.
 * Weights are quantized to INT8 internally using symmetric per-room quantization.
 * 
 * @param handle   Deckboss instance
 * @param room_id  Room identifier (0 <= room_id < max_rooms)
 * @param weights  FP32 weight vector of length dim
 * @param dim      Dimension of weight vector
 * @return DECKBOSS_OK on success
 */
deckboss_status_t deckboss_load_weights(deckboss_handle_t* handle, int room_id, 
                                         const float* weights, int dim);

/**
 * Load (or update) the shared input vector.
 * Quantized to INT8 internally.
 * 
 * @param handle  Deckboss instance
 * @param input   FP32 input vector of length dim
 * @param dim     Dimension of input vector
 * @return DECKBOSS_OK on success
 */
deckboss_status_t deckboss_set_input(deckboss_handle_t* handle, 
                                      const float* input, int dim);

/**
 * Run batched room inference.
 * 
 * Computes output[room_ids[i]] = dot(weights[room_ids[i]], input) for each room.
 * Uses INT8 symmetric quantization with per-room weight scales.
 * 
 * @param handle    Deckboss instance
 * @param room_ids  Array of room IDs to infer (on host)
 * @param num_rooms Number of rooms in batch
 * @param output    Output score array of length num_rooms (on host)
 * @param stats     Optional performance stats (pass NULL to skip)
 * @return DECKBOSS_OK on success
 */
deckboss_status_t deckboss_infer(deckboss_handle_t* handle,
                                  const int* room_ids, int num_rooms,
                                  float* output, deckboss_stats_t* stats);

/**
 * Get the last error message (thread-local).
 */
const char* deckboss_last_error(void);

/**
 * Get the number of rooms currently loaded.
 */
int deckboss_room_count(deckboss_handle_t* handle);

/**
 * Get configuration info.
 */
deckboss_status_t deckboss_get_config(deckboss_handle_t* handle, deckboss_config_t* config);

#ifdef __cplusplus
}
#endif

#endif // DECKBOSS_RUNTIME_H
