# Suite #57: Multiple Input Vectors (Per-Room Inputs)

**Date:** 2026-04-26
**Hardware:** Jetson Orin Nano 8GB, sm_87

## Part 1: Shared vs Per-Room Input (dim=256)

| Rooms | Shared μs | Per-Room μs | Ratio |
|---|---|---|---|
| 64 | 10.25 | 6.48 | 0.63× (per-room faster) |
| 256 | 12.06 | 9.70 | 0.80× (per-room faster) |
| 1,024 | 10.70 | 10.05 | 0.94× (per-room faster) |
| **4,096** | **28.99** | **70.09** | **2.42× (shared faster)** |
| 8,192 | 61.43 | N/A | N/A |

## Key Findings

1. **Per-room input is FASTER at small batches** — 0.63× at 64 rooms. Shared input causes bank conflicts when all 32 threads in a warp read the same input simultaneously. Per-room input spreads reads across the memory array, improving throughput.

2. **Crossover at ~2000 rooms** — Below this threshold, per-room access patterns are more cache-friendly. Above it, the per-room input array (rooms × 512B) exceeds the 2MB L2 cache, causing capacity misses.

3. **At 4096 rooms: 2.42× slower per-room** — 4096 rooms × 512B = 2MB, exactly the L2 cache size. The per-room input array completely evicts weights from cache. Shared input (512B) stays in L2 while weights stream through.

4. **L2 capacity is the limiting factor** — With shared input, only weights occupy L2 (up to 2MB). With per-room input, weights + inputs compete for the same 2MB. At 4096 rooms, there's no room for both.

## Rule #47: Use Shared Input When Batch > L2 Capacity

Per-room inputs are beneficial at small batches (<2000 rooms at dim=256) due to reduced bank conflicts. At larger batches where input arrays exceed L2 cache, shared input is 2.42× faster. For the fleet, use shared inputs for large batch inference and per-room inputs for small specialized batches.

## Production Implication

The deckboss fleet should standardize on shared inputs where possible (e.g., broadcast vectors, system prompts). Per-room inputs should only be used for rooms with genuinely unique context, and those should be batched separately from the main inference pool.
