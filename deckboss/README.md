# Deckboss Runtime

**Edge GPU room inference library — C API with CUDA backend.**

51.6M room-qps through the full C API on Jetson Orin Nano 8GB (4096 rooms, dim=256).
185M room-qps raw kernel throughput (no API overhead).

## Quick Start

```bash
# Compile test
/usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 --use_fast_math \
  deckboss_runtime.cu deckboss_test.cu -o deckboss_test -I.

# Run
./deckboss_test
```

## API

```c
#include "deckboss_runtime.h"

// Init
deckboss_config_t cfg = { .dim = 256, .max_rooms = 4096, .num_streams = 4 };
deckboss_handle_t* db;
deckboss_init(&cfg, &db);

// Load weights (INT8 quantization happens automatically)
float weights[256] = { ... };
deckboss_load_weights(db, room_id, weights, 256);

// Set shared input
float input[256] = { ... };
deckboss_set_input(db, input, 256);

// Batch inference
int rooms[] = { 0, 1, 5, 100 };
float output[4];
deckboss_stats_t stats;
deckboss_infer(db, rooms, 4, output, &stats);
printf("%.1f M qps\n", stats.throughput_mqps);

// Cleanup
deckboss_destroy(db);
```

## Build as Shared Library

```bash
nvcc -arch=sm_87 -O3 --use_fast_math -shared -fPIC \
  deckboss_runtime.cu -o libdeckboss.so
```

## Performance (Jetson Orin Nano 8GB, 306MHz)

| Rooms | API Latency | API Throughput | Raw Kernel |
|-------|------------|----------------|------------|
| 1 | 18 μs | 55K qps | 2.5M qps |
| 64 | 19 μs | 3.3M qps | 17.8M qps |
| 256 | 22 μs | 11.6M qps | 69.1M qps |
| 1024 | 34 μs | 29.7M qps | 117.5M qps |
| 4096 | 79 μs | **51.6M qps** | 185M qps |

The API overhead (room_id upload + output download) adds ~57μs. For production use with pre-loaded rooms and batched inference, this overhead is amortized.

## Design

- **INT8 symmetric quantization** — weights quantized per-room on upload. 4% avg error, 50% memory savings.
- **Pinned memory** — all host buffers use `cudaMallocHost` for async transfers.
- **Warp shuffle reduction** — no shared memory. 8 rooms per block, 32 threads per room.
- **`__launch_bounds__(256, 8)`** — compiler hints for max occupancy.
- **`--use_fast_math`** — approximate math for ranking workloads.

## Files

- `deckboss_runtime.h` — C API header
- `deckboss_runtime.cu` — CUDA implementation
- `deckboss_test.cu` — Correctness + benchmark test
