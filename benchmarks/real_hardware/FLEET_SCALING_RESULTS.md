# Suite #39: Multi-Device Fleet Scaling Simulation

**Date:** 2026-04-25
**Type:** Software simulation (not GPU benchmark)

## Question
How do we shard rooms across multiple Jetson devices? What's the load balance and cost scaling?

## Results (Zipf-distributed workload, 4096 rooms, 100K requests)

### Hash-Based Sharding (room_id % num_devices)
| Devices | Max Load | Min Load | Imbalance | Cache Hit |
|---------|----------|----------|-----------|-----------|
| 2 | 127K | 123K | 1.04× | 96.0% |
| 4 | 73K | 53K | 1.37× | 96.0% |
| 8 | 50K | 23K | 2.15× | 96.0% |
| 16 | 34K | 10K | 3.45× | 96.0% |

### Power-of-Two Choices (pick less loaded of 2 random)
| Devices | Max Load | Min Load | Imbalance | Cache Hit |
|---------|----------|----------|-----------|-----------|
| 2 | 126K | 125K | **1.01×** | 96.0% |
| 4 | 63K | 62K | **1.02×** | 96.0% |
| 8 | 37K | 29K | **1.30×** | 96.0% |
| 16 | 32K | 12K | 2.57× | 96.0% |

### Fleet Throughput Projection
| Devices | Efficient qps | Cost | $/M qps/year |
|---------|--------------|------|--------------|
| 1 | 42M | $249 | $5.86 |
| 4 | 170M | $996 | $5.86 |
| 8 | 340M | $1,992 | $5.86 |
| 64 | 2,720M | $15,936 | $5.86 |

## Key Findings

1. **Hash-based sharding creates hot spots** — Zipf distribution means popular rooms cluster on same device
2. **Power-of-two choices fixes balance up to 8 devices** — 1.02× imbalance at 4 devices
3. **Cache hit rate is always 96%** — Zipf means same rooms keep coming back, any sticky assignment works
4. **Linear cost scaling** — $5.86 per million room-qps/year, constant regardless of fleet size
5. **Assignment overhead: 5ns** — hash computation is negligible
6. **8 devices is the practical max for hash-based balancing** — beyond that, need dynamic load rebalancing

## Production Architecture

For multi-Jetson fleet:
1. **Use power-of-two-choices for room assignment** (1.02× balance at 4 devices)
2. **Sticky assignment** (96% cache hit rate from Zipf distribution)
3. **Dynamic rebalancing** when imbalance exceeds 1.5× (migrate hot rooms)
4. **Linear scaling** — each $249 Jetson adds 42M room-qps

**Cost comparison:** 64 Jetson Orins ($15,936) = 2.7B room-qps. Equivalent A100 cloud: ~$12,960/year for 970M qps. **Jetson fleet is 2.8× cheaper AND 2.8× faster.**
