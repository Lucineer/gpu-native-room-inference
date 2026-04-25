/**
 * Suite #39: Multi-Device Fleet Scaling Simulation
 * 
 * Suite #34 showed the Jetson caps at ~67M room-qps total.
 * Suite #38 estimated 10 Orins = 425M room-qps.
 * 
 * But how do we shard rooms across devices?
 * Options:
 * 1. Hash-based: room_id % num_devices → deterministic, simple
 * 2. Load-based: dynamic assignment based on device load
 * 3. Affinity-based: keep rooms on same device (cache locality)
 * 
 * This simulates the software architecture for multi-Jetson fleet inference.
 * Measures: assignment overhead, cache hit rate, load balance.
 * 
 * Hardware: Jetson Orin Nano 8GB (simulation)
 * Compile: g++ -O3 fleet_scaling.cpp -o fleet_scaling
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>

int main() {
    printf("=== Suite #39: Multi-Device Fleet Scaling Simulation ===\n\n");
    
    int num_devices[] = {1, 2, 4, 8, 16, 32, 64};
    int total_rooms = 4096;
    int iters = 100000;
    
    // Simulate request pattern: Zipf-like distribution
    // Some rooms are hot (frequent), most are cold (rare)
    std::mt19937 rng(42);
    
    // Generate Zipf-distributed room requests
    std::vector<double> weights(total_rooms);
    double sum = 0;
    for (int i = 0; i < total_rooms; i++) {
        weights[i] = 1.0 / (1.0 + i);  // Zipf with s=1
        sum += weights[i];
    }
    for (int i = 0; i < total_rooms; i++) weights[i] /= sum;
    
    std::discrete_distribution<int> zipf(weights.begin(), weights.end());
    
    // Generate request sequence
    std::vector<int> requests(iters);
    for (int i = 0; i < iters; i++)
        requests[i] = zipf(rng);
    
    // Room sizes (some rooms have more data)
    std::vector<int> room_weights(total_rooms);
    for (int i = 0; i < total_rooms; i++)
        room_weights[i] = 1 + (rng() % 4);  // 1-4 weight units
    
    // ========================================
    // Strategy 1: Hash-based (room_id % num_devices)
    // ========================================
    printf("--- Strategy 1: Hash-Based Sharding ---\n");
    printf("%-8s | %-10s | %-10s | %-10s | %-10s\n",
           "Devices", "Max Load", "Min Load", "Imbalance", "Cache Hit%%");
    printf("---------|------------|------------|------------|------------\n");
    
    for (int d = 0; d < 7; d++) {
        int nd = num_devices[d];
        std::vector<int> device_load(nd, 0);
        std::vector<int> cache_hits(nd, 0);
        std::vector<int> last_device(total_rooms, -1);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            int room = requests[i];
            int device = room % nd;
            device_load[device] += room_weights[room];
            
            if (last_device[room] == device) cache_hits[device]++;
            last_device[room] = device;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        int max_load = *std::max_element(device_load.begin(), device_load.end());
        int min_load = *std::min_element(device_load.begin(), device_load.end());
        double imbalance = (double)max_load / std::max(1, min_load);
        
        int total_hits = std::accumulate(cache_hits.begin(), cache_hits.end(), 0);
        double cache_hit_rate = 100.0 * total_hits / iters;
        
        double assign_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        
        printf("%-8d | %8d    | %8d    | %.3fx     | %.1f%% (%.1fns/req)\n",
               nd, max_load, min_load, imbalance, cache_hit_rate, assign_ns);
    }
    
    // ========================================
    // Strategy 2: Consistent Hashing (better balance)
    // ========================================
    printf("\n--- Strategy 2: Consistent Hashing ---\n");
    printf("%-8s | %-10s | %-10s | %-10s | %-10s\n",
           "Devices", "Max Load", "Min Load", "Imbalance", "Cache Hit%%");
    printf("---------|------------|------------|------------|------------\n");
    
    for (int d = 0; d < 7; d++) {
        int nd = num_devices[d];
        
        // Build hash ring with 100 virtual nodes per device
        struct VNode { unsigned hash; int device; };
        std::vector<VNode> ring;
        ring.reserve(nd * 100);
        for (int dev = 0; dev < nd; dev++) {
            for (int v = 0; v < 100; v++) {
                unsigned h = std::hash<std::string>{}(std::to_string(dev) + "_" + std::to_string(v));
                ring.push_back({h, dev});
            }
        }
        std::sort(ring.begin(), ring.end(), [](const VNode& a, const VNode& b) { return a.hash < b.hash; });
        
        std::vector<int> device_load(nd, 0);
        std::vector<int> cache_hits(nd, 0);
        std::vector<int> last_device(total_rooms, -1);
        
        for (int i = 0; i < iters; i++) {
            int room = requests[i];
            unsigned rh = std::hash<int>{}(room);
            
            // Binary search for device
            auto it = std::upper_bound(ring.begin(), ring.end(), rh,
                [](unsigned val, const VNode& vn) { return val < vn.hash; });
            if (it == ring.end()) it = ring.begin();
            int device = it->device;
            
            device_load[device] += room_weights[room];
            if (last_device[room] == device) cache_hits[device]++;
            last_device[room] = device;
        }
        
        int max_load = *std::max_element(device_load.begin(), device_load.end());
        int min_load = *std::min_element(device_load.begin(), device_load.end());
        double imbalance = (double)max_load / std::max(1, min_load);
        
        int total_hits = std::accumulate(cache_hits.begin(), cache_hits.end(), 0);
        double cache_hit_rate = 100.0 * total_hits / iters;
        
        printf("%-8d | %8d    | %8d    | %.3fx     | %.1f%%\n",
               nd, max_load, min_load, imbalance, cache_hit_rate);
    }
    
    // ========================================
    // Strategy 3: Power-of-Two Choices (best balance)
    // ========================================
    printf("\n--- Strategy 3: Power-of-Two Choices ---\n");
    printf("%-8s | %-10s | %-10s | %-10s | %-10s\n",
           "Devices", "Max Load", "Min Load", "Imbalance", "Cache Hit%%");
    printf("---------|------------|------------|------------|------------\n");
    
    for (int d = 0; d < 7; d++) {
        int nd = num_devices[d];
        std::vector<int> device_load(nd, 0);
        std::vector<int> cache_hits(nd, 0);
        std::vector<int> last_device(total_rooms, -1);
        
        for (int i = 0; i < iters; i++) {
            int room = requests[i];
            
            // Hash room to two candidate devices
            unsigned h1 = std::hash<int>{}(room + 0);
            unsigned h2 = std::hash<int>{}(room + 1);
            int dev1 = h1 % nd;
            int dev2 = h2 % nd;
            
            // Pick the less loaded one (or the cached one)
            int device;
            if (last_device[room] == dev1) device = dev1;
            else if (last_device[room] == dev2) device = dev2;
            else if (device_load[dev1] <= device_load[dev2]) device = dev1;
            else device = dev2;
            
            device_load[device] += room_weights[room];
            if (last_device[room] == device) cache_hits[device]++;
            last_device[room] = device;
        }
        
        int max_load = *std::max_element(device_load.begin(), device_load.end());
        int min_load = *std::min_element(device_load.begin(), device_load.end());
        double imbalance = (double)max_load / std::max(1, min_load);
        
        int total_hits = std::accumulate(cache_hits.begin(), cache_hits.end(), 0);
        double cache_hit_rate = 100.0 * total_hits / iters;
        
        printf("%-8d | %8d    | %8d    | %.3fx     | %.1f%%\n",
               nd, max_load, min_load, imbalance, cache_hit_rate);
    }
    
    // ========================================
    // Fleet Throughput Projection
    // ========================================
    printf("\n=== Fleet Throughput Projection ===\n");
    printf("Based on single-Jetson 67M room-qps hard cap (suite #34)\n\n");
    
    printf("%-8s | %-14s | %-14s | %-14s | %-14s\n",
           "Devices", "Theoretical", "Efficient", "Cost ($)", "Cost/qps (yr)");
    printf("---------|----------------|----------------|----------------|----------------\n");
    
    int device_costs[] = {249, 498, 996, 1992, 3984, 7968, 15936};
    for (int d = 0; d < 7; d++) {
        int nd = num_devices[d];
        float theoretical = nd * 67.0f;  // 67M per device
        float efficient = nd * 42.5f;    // 42.5M practical per device (single stream)
        float cost = device_costs[d];
        float cost_per_qps = cost / efficient;
        
        printf("%-8d | %10.0fM    | %10.0fM    | %12.0f   | $%.4f/M\n",
               nd, theoretical, efficient, cost, cost_per_qps);
    }
    
    printf("\n=== Suite #39 Complete ===\n");
    return 0;
}
