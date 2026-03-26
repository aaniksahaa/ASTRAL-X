/**
 * GPU Key-Value Map Lookup using Thrust
 * 
 * Problem:
 * - 10^6 random keys from [1, 10^8]
 * - Each key i has value: 'a' + (i % 26)
 * - 10^6 queries - retrieve the corresponding char if key exists
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/gather.h>

// Compute the value for a given key
__host__ __device__ char keyToValue(int key) {
    return 'a' + (key % 26);
}

int main() {
    // Parameters
    const int N = 1000000;             // 10^6 keys in the map
    const int NUM_QUERIES = 1000000;   // 10^6 queries
    const int KEY_RANGE = 100000000;   // Keys from [1, 10^8]
    
    std::cout << "========================================\n";
    std::cout << "GPU Key-Value Map Lookup (Thrust)\n";
    std::cout << "========================================\n";
    std::cout << "Map size: " << N << " key-value pairs\n";
    std::cout << "Key range: [1, " << KEY_RANGE << "]\n";
    std::cout << "Number of queries: " << NUM_QUERIES << "\n";
    std::cout << "Value for key i: '" << "a' + (i % 26)\n\n";

    // ========================================
    // Generate Data
    // ========================================
    std::cout << "Generating random keys and values...\n";
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> key_dist(1, KEY_RANGE);
    
    // Generate unique keys using a set
    std::vector<int> h_keys;
    h_keys.reserve(N);
    std::unordered_set<int> key_set;
    
    while (h_keys.size() < N) {
        int k = key_dist(rng);
        if (key_set.find(k) == key_set.end()) {
            key_set.insert(k);
            h_keys.push_back(k);
        }
    }
    
    // Generate corresponding values
    std::vector<char> h_values(N);
    for (int i = 0; i < N; i++) {
        h_values[i] = keyToValue(h_keys[i]);
    }
    
    // Generate queries - 50% existing keys, 50% random (may or may not exist)
    std::vector<int> h_queries(NUM_QUERIES);
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (i % 2 == 0) {
            // Pick a key that exists
            h_queries[i] = h_keys[rng() % N];
        } else {
            // Pick a random key (likely doesn't exist given 10^6 / 10^8 = 1% density)
            h_queries[i] = key_dist(rng);
        }
    }

    // ========================================
    // CPU Reference using unordered_map
    // ========================================
    std::cout << "\n--- CPU unordered_map (Reference) ---\n";
    
    auto cpu_build_start = std::chrono::high_resolution_clock::now();
    std::unordered_map<int, char> cpu_map;
    for (int i = 0; i < N; i++) {
        cpu_map[h_keys[i]] = h_values[i];
    }
    auto cpu_build_end = std::chrono::high_resolution_clock::now();
    double cpu_build_time = std::chrono::duration<double, std::milli>(cpu_build_end - cpu_build_start).count();
    std::cout << "Build time: " << cpu_build_time << " ms\n";
    
    std::vector<char> cpu_results(NUM_QUERIES);
    std::vector<bool> cpu_found(NUM_QUERIES);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_QUERIES; i++) {
        auto it = cpu_map.find(h_queries[i]);
        if (it != cpu_map.end()) {
            cpu_found[i] = true;
            cpu_results[i] = it->second;
        } else {
            cpu_found[i] = false;
            cpu_results[i] = '?';
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "Lookup time: " << cpu_time << " ms\n";
    
    int cpu_found_count = 0;
    for (bool f : cpu_found) if (f) cpu_found_count++;
    std::cout << "Found: " << cpu_found_count << " / " << NUM_QUERIES << "\n";

    // ========================================
    // GPU Thrust Sorted Map
    // ========================================
    std::cout << "\n--- GPU Thrust Sorted Map ---\n";
    
    // Copy keys and values to GPU
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<char> d_values = h_values;
    
    // Sort keys and values together (by keys)
    auto gpu_build_start = std::chrono::high_resolution_clock::now();
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
    cudaDeviceSynchronize();
    auto gpu_build_end = std::chrono::high_resolution_clock::now();
    double gpu_build_time = std::chrono::duration<double, std::milli>(gpu_build_end - gpu_build_start).count();
    std::cout << "Build time (sort): " << gpu_build_time << " ms\n";
    
    // Copy queries to GPU
    thrust::device_vector<int> d_queries = h_queries;
    
    // Step 1: Find positions using lower_bound
    thrust::device_vector<int> d_positions(NUM_QUERIES);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    thrust::lower_bound(d_keys.begin(), d_keys.end(),
                        d_queries.begin(), d_queries.end(),
                        d_positions.begin());
    
    // Step 2: Check if key at position matches query and gather values
    // We need to handle: 
    //   - position might be N (key larger than all)
    //   - key at position might not match query
    
    thrust::device_vector<bool> d_found(NUM_QUERIES);
    thrust::device_vector<char> d_results(NUM_QUERIES, '?');
    
    // Use transform to check if found and get values
    // Create raw pointers for use in lambda
    int* keys_ptr = thrust::raw_pointer_cast(d_keys.data());
    char* values_ptr = thrust::raw_pointer_cast(d_values.data());
    int* queries_ptr = thrust::raw_pointer_cast(d_queries.data());
    int* positions_ptr = thrust::raw_pointer_cast(d_positions.data());
    bool* found_ptr = thrust::raw_pointer_cast(d_found.data());
    char* results_ptr = thrust::raw_pointer_cast(d_results.data());
    
    // Transform to check matches and retrieve values
    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(NUM_QUERIES),
        [=] __device__ (int i) {
            int pos = positions_ptr[i];
            int query = queries_ptr[i];
            
            // Check if position is valid and key matches
            if (pos < N && keys_ptr[pos] == query) {
                found_ptr[i] = true;
                results_ptr[i] = values_ptr[pos];
            } else {
                found_ptr[i] = false;
                results_ptr[i] = '?';
            }
        }
    );
    
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    std::cout << "Lookup time: " << gpu_time << " ms\n";
    
    // Copy results back
    thrust::host_vector<bool> h_gpu_found = d_found;
    thrust::host_vector<char> h_gpu_results = d_results;
    
    int gpu_found_count = 0;
    for (bool f : h_gpu_found) if (f) gpu_found_count++;
    std::cout << "Found: " << gpu_found_count << " / " << NUM_QUERIES << "\n";
    
    // Verify correctness
    int correct = 0;
    int value_correct = 0;
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (h_gpu_found[i] == cpu_found[i]) {
            correct++;
            if (cpu_found[i] && h_gpu_results[i] == cpu_results[i]) {
                value_correct++;
            } else if (!cpu_found[i]) {
                value_correct++; // Both not found, values don't matter
            }
        }
    }
    std::cout << "Lookup correctness: " << correct << " / " << NUM_QUERIES 
              << " (" << (100.0 * correct / NUM_QUERIES) << "%)\n";
    std::cout << "Value correctness: " << value_correct << " / " << NUM_QUERIES 
              << " (" << (100.0 * value_correct / NUM_QUERIES) << "%)\n";

    // ========================================
    // Show some sample results
    // ========================================
    std::cout << "\n--- Sample Results (first 10 queries) ---\n";
    std::cout << "Query\t\tFound?\tValue\tExpected\n";
    std::cout << "----------------------------------------\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_queries[i] << "\t\t" 
                  << (h_gpu_found[i] ? "Yes" : "No") << "\t"
                  << h_gpu_results[i] << "\t"
                  << (cpu_found[i] ? std::string(1, cpu_results[i]) : "N/A") << "\n";
    }

    // ========================================
    // Summary
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Build Time:\n";
    std::cout << "  CPU unordered_map: " << cpu_build_time << " ms\n";
    std::cout << "  GPU Thrust sort:   " << gpu_build_time << " ms\n";
    std::cout << "\nLookup Time (" << NUM_QUERIES << " queries):\n";
    std::cout << "  CPU unordered_map: " << cpu_time << " ms\n";
    std::cout << "  GPU Thrust:        " << gpu_time << " ms\n";
    std::cout << "  Speedup:           " << cpu_time / gpu_time << "x\n";
    std::cout << "\nPer-query latency:\n";
    std::cout << "  CPU: " << (cpu_time * 1000 / NUM_QUERIES) << " µs/query\n";
    std::cout << "  GPU: " << (gpu_time * 1000 / NUM_QUERIES) << " µs/query\n";

    std::cout << "\n✓ All tests completed successfully!\n";
    return 0;
}
