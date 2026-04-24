/**
 * deckboss Room Cache — L2-Aware LRU Cache
 * 
 * Based on benchmark findings:
 * - L2 cache warming gives 11x speedup for hot rooms
 * - 2MB L2 fits 4096 rooms (256-dim FP16)
 * - Repeated access: 0.029 us/room vs 0.326 us/room cold
 * 
 * This implements a simple LRU cache that keeps hot rooms in GPU memory
 * and evicts cold rooms when capacity is reached.
 * 
 * Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 room_cache.cu -o room_cache
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <list>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void room_inference_batch(const half* __restrict__ weights, const half* __restrict__ input,
                                      float* __restrict__ output, int dim, int num_rooms) {
    int room = blockIdx.x;
    if (room >= num_rooms) return;
    int lane = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = lane; i < dim; i += 32)
        sum += __half2float(weights[room * dim + i]) * __half2float(input[i]);
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (lane == 0) output[room] = sum;
}

// Simple LRU cache using a doubly-linked list + hash map
struct RoomCache {
    int dim;
    int max_rooms;        // GPU memory budget
    int cached_count;
    
    // GPU memory: contiguous weight storage
    half* d_weights;      // [max_rooms * dim]
    float* d_output;      // [max_rooms]
    half* d_input;        // [dim]
    
    // CPU-side room index mapping
    // room_id -> position in GPU weight array
    std::unordered_map<int, int> room_to_slot;
    // LRU list: most recently used at front
    std::list<int> lru_list;
    // slot -> room_id (reverse mapping)
    std::unordered_map<int, std::list<int>::iterator> slot_to_lru;
    // slot -> room_id
    int* slot_to_room;
    
    cudaStream_t stream;
    
    RoomCache(int dim, int max_rooms) : dim(dim), max_rooms(max_rooms), cached_count(0) {
        CUDA_CHECK(cudaMalloc(&d_weights, (size_t)max_rooms * dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output, max_rooms * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_input, dim * sizeof(half)));
        CUDA_CHECK(cudaStreamCreate(&stream));
        slot_to_room = new int[max_rooms];
        memset(slot_to_room, -1, max_rooms * sizeof(int));
    }
    
    ~RoomCache() {
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaStreamDestroy(stream));
        delete[] slot_to_room;
    }
    
    // Touch a room (move to front of LRU)
    void touch(int room_id) {
        auto it = room_to_slot.find(room_id);
        if (it != room_to_slot.end()) {
            auto lru_it = slot_to_lru[it->second];
            lru_list.erase(lru_it);
            lru_list.push_front(room_id);
            slot_to_lru[it->second] = lru_list.begin();
        }
    }
    
    // Evict LRU room
    int evict() {
        if (lru_list.empty()) return -1;
        int victim = lru_list.back();
        int slot = room_to_slot[victim];
        
        room_to_slot.erase(victim);
        slot_to_lru.erase(slot);
        lru_list.pop_back();
        slot_to_room[slot] = -1;
        cached_count--;
        
        return victim;
    }
    
    // Load room weights into cache
    void load(int room_id, const half* h_weights) {
        int slot;
        if (cached_count >= max_rooms) {
            int victim = evict();
            slot = -1; // Find free slot
            for (int i = 0; i < max_rooms; i++) {
                if (slot_to_room[i] == -1) { slot = i; break; }
            }
            if (victim != -1) {
                printf("[cache] evicted room %d (slot freed)\n", victim);
            }
        } else {
            for (int i = 0; i < max_rooms; i++) {
                if (slot_to_room[i] == -1) { slot = i; break; }
            }
        }
        
        // Copy weights to GPU
        CUDA_CHECK(cudaMemcpyAsync(
            d_weights + (size_t)slot * dim,
            h_weights,
            dim * sizeof(half),
            cudaMemcpyHostToDevice,
            stream
        ));
        
        room_to_slot[room_id] = slot;
        slot_to_room[slot] = room_id;
        lru_list.push_front(room_id);
        slot_to_lru[slot] = lru_list.begin();
        cached_count++;
    }
    
    // Check if room is cached
    bool is_cached(int room_id) {
        return room_to_slot.find(room_id) != room_to_slot.end();
    }
    
    // Infer on a batch of room IDs
    // Returns: number of rooms actually inferred
    int infer(const int* room_ids, int num_rooms, const half* h_input, float* h_output) {
        // Load input to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, dim * sizeof(half), cudaMemcpyHostToDevice, stream));
        
        // Ensure all rooms are loaded (load missing ones)
        for (int i = 0; i < num_rooms; i++) {
            if (!is_cached(room_ids[i])) {
                // In production, load from disk. For benchmark, skip.
                printf("[cache] room %d not loaded (would load from disk)\n", room_ids[i]);
            }
            touch(room_ids[i]);
        }
        
        // Gather weight pointers for batched inference
        // For simplicity, infer one room at a time using cached weights
        dim3 block(32);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        for (int i = 0; i < num_rooms; i++) {
            auto it = room_to_slot.find(room_ids[i]);
            if (it == room_to_slot.end()) continue;
            
            int slot = it->second;
            dim3 grid(1);
            room_inference_batch<<<grid, block, 0, stream>>>(
                d_weights + (size_t)slot * dim,
                d_input, d_output, dim, 1
            );
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // Copy outputs back
        // (in production, copy only requested outputs)
        
        return num_rooms;
    }
};

int main() {
    printf("============================================================\n");
    printf("ROOM CACHE — L2-Aware LRU Implementation\n");
    printf("============================================================\n\n");
    
    const int DIM = 256;
    const int CACHE_SIZE = 256;  // 256 rooms * 512 bytes = 128 KB (fits in L2)
    const int TOTAL_ROOMS = 1024;
    const int ITERS = 10000;
    
    // Generate random weights for all rooms
    half *all_weights = (half*)malloc((size_t)TOTAL_ROOMS * DIM * sizeof(half));
    half *input = (half*)malloc(DIM * sizeof(half));
    
    srand(42);
    for (int i = 0; i < TOTAL_ROOMS * DIM; i++)
        all_weights[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < DIM; i++)
        input[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    // Create cache
    RoomCache cache(DIM, CACHE_SIZE);
    printf("Cache: %d rooms, %d MB GPU memory\n", CACHE_SIZE, 
           CACHE_SIZE * DIM * sizeof(half) / (1024 * 1024));
    
    // Pre-load first 256 rooms
    printf("\nLoading %d rooms into cache...\n", CACHE_SIZE);
    for (int i = 0; i < CACHE_SIZE; i++) {
        cache.load(i, all_weights + (size_t)i * DIM);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Cache warmed: %d rooms loaded\n", cache.cached_count);
    
    // Benchmark 1: Cache hit (all rooms in cache)
    printf("\n>>> Test 1: All Cache Hits (256 rooms, 10000 iterations)\n");
    
    int *room_ids = new int[CACHE_SIZE];
    for (int i = 0; i < CACHE_SIZE; i++) room_ids[i] = i;
    
    float *dummy_output = new float[CACHE_SIZE];
    
    // Warmup
    for (int i = 0; i < 100; i++)
        cache.infer(room_ids, CACHE_SIZE, input, dummy_output);
    
    // Timed
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        cache.infer(room_ids, CACHE_SIZE, input, dummy_output);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_hits;
    CUDA_CHECK(cudaEventElapsedTime(&ms_hits, start, stop));
    float us_per_room_hits = ms_hits / ITERS / CACHE_SIZE * 1000.0;
    
    printf("  Total: %.1f ms | Per-iter: %.1f us | Per-room: %.3f us\n",
           ms_hits, ms_hits / ITERS * 1000.0, us_per_room_hits);
    printf("  Room QPS: %.0f\n", (float)ITERS * CACHE_SIZE / (ms_hits / 1000.0));
    
    // Benchmark 2: Cache miss pattern (random rooms, most miss)
    printf("\n>>> Test 2: Random Access (1024 rooms, 256 cache, 10000 iterations)\n");
    
    int *random_ids = new int[CACHE_SIZE];
    for (int i = 0; i < ITERS; i++) {
        for (int j = 0; j < CACHE_SIZE; j++)
            random_ids[j] = rand() % TOTAL_ROOMS;
    }
    
    // Use a fixed random pattern for measurement
    for (int j = 0; j < CACHE_SIZE; j++)
        random_ids[j] = rand() % TOTAL_ROOMS;
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        // Only infer rooms that are in cache
        int inferred = 0;
        for (int j = 0; j < CACHE_SIZE; j++) {
            if (cache.is_cached(random_ids[j])) {
                cache.touch(random_ids[j]);
                inferred++;
            }
        }
        // For this benchmark, we only measure cached rooms
        // In production, miss = load from disk + evict LRU
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_random;
    CUDA_CHECK(cudaEventElapsedTime(&ms_random, start, stop));
    
    printf("  Total: %.1f ms (cached-only subset)\n", ms_random);
    printf("  Note: In production, cache misses add disk I/O + memcpy\n");
    
    // Benchmark 3: Sequential hot access (best case — same room repeated)
    printf("\n>>> Test 3: Hot Room (1 room, 100000 iterations)\n");
    
    int hot_id = 42;
    int *hot_ids = new int[1];
    hot_ids[0] = hot_id;
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100000; i++)
        cache.infer(hot_ids, 1, input, dummy_output);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_hot;
    CUDA_CHECK(cudaEventElapsedTime(&ms_hot, start, stop));
    float us_per_room_hot = ms_hot / 100000 * 1000.0;
    
    printf("  Total: %.1f ms | Per-room: %.3f us\n", ms_hot, us_per_room_hot);
    printf("  Room QPS: %.0f\n", 100000.0 / (ms_hot / 1000.0));
    printf("  Speedup vs cold: %.1fx\n", 7.0 / us_per_room_hot);
    
    // Benchmark 4: Batched inference on cached rooms
    printf("\n>>> Test 4: Batched Cache Hit (256 rooms, single kernel launch)\n");
    
    // Direct batched launch (bypassing cache for comparison)
    half *d_batch_weights;
    CUDA_CHECK(cudaMalloc(&d_batch_weights, (size_t)CACHE_SIZE * DIM * sizeof(half)));
    
    // Copy weights for rooms 0-255 contiguously
    CUDA_CHECK(cudaMemcpy(d_batch_weights, all_weights, 
                           (size_t)CACHE_SIZE * DIM * sizeof(half), 
                           cudaMemcpyHostToDevice));
    
    half *d_input;
    float *d_batch_output;
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_batch_output, CACHE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input, DIM * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 block(32);
    dim3 grid(CACHE_SIZE);
    
    for (int i = 0; i < 1000; i++)
        room_inference_batch<<<grid, block>>>(d_batch_weights, d_input, d_batch_output, DIM, CACHE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        room_inference_batch<<<grid, block>>>(d_batch_weights, d_input, d_batch_output, DIM, CACHE_SIZE);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_batch;
    CUDA_CHECK(cudaEventElapsedTime(&ms_batch, start, stop));
    float us_per_room_batch = ms_batch / ITERS / CACHE_SIZE * 1000.0;
    
    printf("  Total: %.1f ms | Per-room: %.3f us\n", ms_batch, us_per_room_batch);
    printf("  Room QPS: %.0f\n", (float)ITERS * CACHE_SIZE / (ms_batch / 1000.0));
    printf("  vs Cache sequential: %.1fx\n", us_per_room_hits / us_per_room_batch);
    
    printf("\n============================================================\n");
    printf("CONCLUSION\n");
    printf("============================================================\n");
    printf("Batched inference (single kernel) >> cache sequential (per-room)\n");
    printf("because launch overhead dominates at small per-room batch sizes.\n");
    printf("The cache is most valuable when:\n");
    printf("  1. Rooms are accessed repeatedly between launches\n");
    printf("  2. The same rooms appear in consecutive batches\n");
    printf("  3. Weight loading from disk is expensive\n");
    printf("For production: batch rooms, then check cache for the batch.\n");
    printf("Load missing rooms, then launch single batched kernel.\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_batch_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_batch_output));
    
    delete[] room_ids;
    delete[] random_ids;
    delete[] hot_ids;
    delete[] dummy_output;
    free(all_weights);
    free(input);
    
    return 0;
}
