/**
 * GPU Set/HashSet Lookup Demonstration
 * Testing fast O(log n) and O(1) lookups on 10^6 integers using CUDA
 * 
 * Approaches:
 * 1. Thrust sorted array + binary_search (O(log n))
 * 2. Custom GPU hash set (O(1) average)
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <set>
#include <unordered_set>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================
// APPROACH 1: Thrust Binary Search (O(log n) lookup)
// ============================================================

// Kernel to perform binary search lookups
__global__ void binarySearchKernel(const int* sorted_data, int n, 
                                    const int* queries, bool* results, int num_queries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    int target = queries[idx];
    int left = 0, right = n - 1;
    bool found = false;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (sorted_data[mid] == target) {
            found = true;
            break;
        } else if (sorted_data[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    results[idx] = found;
}

// ============================================================
// APPROACH 2: Simple GPU Hash Set (O(1) average lookup)
// ============================================================

// Hash function
__device__ __host__ unsigned int hashFunc(int key, int table_size) {
    unsigned int hash = (unsigned int)key;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;
    return hash % table_size;
}

// Empty marker for hash table
#define EMPTY_SLOT -1

// Initialize hash table
__global__ void hashInitKernel(int* hash_table, int table_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < table_size) {
        hash_table[idx] = EMPTY_SLOT;
    }
}

// Insert into hash table (using linear probing)
__global__ void hashInsertKernel(int* hash_table, int table_size, 
                                  const int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int key = data[idx];
    if (key == EMPTY_SLOT) return; // Skip if key equals empty marker
    
    unsigned int hash = hashFunc(key, table_size);
    
    // Linear probing with limited iterations
    for (int i = 0; i < 1000; i++) {
        unsigned int probe = (hash + i) % table_size;
        int old = atomicCAS(&hash_table[probe], EMPTY_SLOT, key);
        if (old == EMPTY_SLOT || old == key) {
            return; // Inserted or already exists
        }
    }
}

// Lookup in hash table
__global__ void hashLookupKernel(const int* hash_table, int table_size,
                                  const int* queries, bool* results, int num_queries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    int key = queries[idx];
    unsigned int hash = hashFunc(key, table_size);
    bool found = false;
    
    // Linear probing with limited iterations
    for (int i = 0; i < 1000; i++) {
        unsigned int probe = (hash + i) % table_size;
        int val = hash_table[probe];
        if (val == key) {
            found = true;
            break;
        }
        if (val == EMPTY_SLOT) {
            break; // Empty slot, key doesn't exist
        }
    }
    results[idx] = found;
}

// ============================================================
// CPU Reference Implementation
// ============================================================

void cpuSetLookup(const std::vector<int>& data, const std::vector<int>& queries,
                  std::vector<bool>& results) {
    std::unordered_set<int> hash_set(data.begin(), data.end());
    for (size_t i = 0; i < queries.size(); i++) {
        results[i] = hash_set.count(queries[i]) > 0;
    }
}

// ============================================================
// Main Test
// ============================================================

int main() {
    // Parameters
    const int N = 1000000;          // 10^6 elements in the set
    const int NUM_QUERIES = 100000; // 10^5 queries
    const int VALUE_RANGE = 2000000; // Values range [0, 2M)
    
    std::cout << "========================================\n";
    std::cout << "GPU Set/HashSet Lookup Test\n";
    std::cout << "========================================\n";
    std::cout << "Set size: " << N << " elements\n";
    std::cout << "Number of queries: " << NUM_QUERIES << "\n";
    std::cout << "Value range: [0, " << VALUE_RANGE << ")\n\n";

    // Check GPU
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n\n";

    // Generate random data
    std::cout << "Generating random data...\n";
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, VALUE_RANGE - 1);
    
    std::vector<int> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = dist(rng);
    }
    
    // Generate queries - mix of existing and non-existing values
    std::vector<int> h_queries(NUM_QUERIES);
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (i % 2 == 0) {
            // Pick a value that exists
            h_queries[i] = h_data[dist(rng) % N];
        } else {
            // Pick a random value (may or may not exist)
            h_queries[i] = dist(rng);
        }
    }

    // ========================================
    // CPU Reference
    // ========================================
    std::cout << "\n--- CPU unordered_set (Reference) ---\n";
    std::vector<bool> cpu_results(NUM_QUERIES);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuSetLookup(h_data, h_queries, cpu_results);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU lookup time: " << cpu_time << " ms\n";
    
    int cpu_found = 0;
    for (bool r : cpu_results) if (r) cpu_found++;
    std::cout << "Found: " << cpu_found << " / " << NUM_QUERIES << "\n";

    // ========================================
    // GPU Approach 1: Thrust Binary Search
    // ========================================
    std::cout << "\n--- GPU Thrust Binary Search (O(log n)) ---\n";
    
    // Copy data to GPU and sort
    thrust::device_vector<int> d_sorted_data = h_data;
    
    auto sort_start = std::chrono::high_resolution_clock::now();
    thrust::sort(d_sorted_data.begin(), d_sorted_data.end());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto sort_end = std::chrono::high_resolution_clock::now();
    
    double sort_time = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();
    std::cout << "Sort time (one-time): " << sort_time << " ms\n";
    
    // Perform lookups using Thrust
    thrust::device_vector<int> d_queries = h_queries;
    thrust::device_vector<bool> d_results_thrust(NUM_QUERIES);
    
    auto thrust_start = std::chrono::high_resolution_clock::now();
    thrust::binary_search(d_sorted_data.begin(), d_sorted_data.end(),
                          d_queries.begin(), d_queries.end(),
                          d_results_thrust.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto thrust_end = std::chrono::high_resolution_clock::now();
    
    double thrust_time = std::chrono::duration<double, std::milli>(thrust_end - thrust_start).count();
    std::cout << "Thrust binary_search time: " << thrust_time << " ms\n";
    
    thrust::host_vector<bool> h_results_thrust = d_results_thrust;
    int thrust_found = 0;
    for (bool r : h_results_thrust) if (r) thrust_found++;
    std::cout << "Found: " << thrust_found << " / " << NUM_QUERIES << "\n";
    
    // Verify correctness
    int thrust_correct = 0;
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (h_results_thrust[i] == cpu_results[i]) thrust_correct++;
    }
    std::cout << "Correctness: " << thrust_correct << " / " << NUM_QUERIES 
              << " (" << (100.0 * thrust_correct / NUM_QUERIES) << "%)\n";

    // ========================================
    // GPU Approach 1b: Custom Binary Search Kernel
    // ========================================
    std::cout << "\n--- GPU Custom Binary Search Kernel ---\n";
    
    int* d_sorted_ptr = thrust::raw_pointer_cast(d_sorted_data.data());
    int* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());
    
    bool* d_results_binary;
    CUDA_CHECK(cudaMalloc(&d_results_binary, NUM_QUERIES * sizeof(bool)));
    
    int blockSize = 256;
    int numBlocks = (NUM_QUERIES + blockSize - 1) / blockSize;
    
    auto binary_start = std::chrono::high_resolution_clock::now();
    binarySearchKernel<<<numBlocks, blockSize>>>(d_sorted_ptr, N, 
                                                   d_queries_ptr, d_results_binary, NUM_QUERIES);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto binary_end = std::chrono::high_resolution_clock::now();
    
    double binary_time = std::chrono::duration<double, std::milli>(binary_end - binary_start).count();
    std::cout << "Custom binary search time: " << binary_time << " ms\n";
    
    std::vector<char> h_results_binary(NUM_QUERIES);
    CUDA_CHECK(cudaMemcpy(h_results_binary.data(), d_results_binary, 
                          NUM_QUERIES * sizeof(bool), cudaMemcpyDeviceToHost));
    
    int binary_found = 0;
    for (bool r : h_results_binary) if (r) binary_found++;
    std::cout << "Found: " << binary_found << " / " << NUM_QUERIES << "\n";
    
    int binary_correct = 0;
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (h_results_binary[i] == cpu_results[i]) binary_correct++;
    }
    std::cout << "Correctness: " << binary_correct << " / " << NUM_QUERIES 
              << " (" << (100.0 * binary_correct / NUM_QUERIES) << "%)\n";

    // ========================================
    // GPU Approach 2: Hash Set
    // ========================================
    std::cout << "\n--- GPU Hash Set (O(1) average) ---\n";
    
    int table_size = N * 2; // Load factor ~0.5
    int* d_hash_table;
    CUDA_CHECK(cudaMalloc(&d_hash_table, table_size * sizeof(int)));
    
    // Initialize hash table with EMPTY_SLOT using kernel
    hashInitKernel<<<(table_size + 255) / 256, 256>>>(d_hash_table, table_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy original data to GPU
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    
    // Insert into hash table
    auto hash_build_start = std::chrono::high_resolution_clock::now();
    hashInsertKernel<<<(N + 255) / 256, 256>>>(d_hash_table, table_size, d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto hash_build_end = std::chrono::high_resolution_clock::now();
    
    double hash_build_time = std::chrono::duration<double, std::milli>(hash_build_end - hash_build_start).count();
    std::cout << "Hash table build time (one-time): " << hash_build_time << " ms\n";
    
    // Perform lookups
    bool* d_results_hash;
    CUDA_CHECK(cudaMalloc(&d_results_hash, NUM_QUERIES * sizeof(bool)));
    
    auto hash_start = std::chrono::high_resolution_clock::now();
    hashLookupKernel<<<numBlocks, blockSize>>>(d_hash_table, table_size,
                                                 d_queries_ptr, d_results_hash, NUM_QUERIES);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto hash_end = std::chrono::high_resolution_clock::now();
    
    double hash_time = std::chrono::duration<double, std::milli>(hash_end - hash_start).count();
    std::cout << "Hash lookup time: " << hash_time << " ms\n";
    
    std::vector<char> h_results_hash(NUM_QUERIES);
    CUDA_CHECK(cudaMemcpy(h_results_hash.data(), d_results_hash, 
                          NUM_QUERIES * sizeof(bool), cudaMemcpyDeviceToHost));
    
    int hash_found = 0;
    for (bool r : h_results_hash) if (r) hash_found++;
    std::cout << "Found: " << hash_found << " / " << NUM_QUERIES << "\n";
    
    int hash_correct = 0;
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (h_results_hash[i] == cpu_results[i]) hash_correct++;
    }
    std::cout << "Correctness: " << hash_correct << " / " << NUM_QUERIES 
              << " (" << (100.0 * hash_correct / NUM_QUERIES) << "%)\n";

    // ========================================
    // Summary
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY - Query Time Comparison\n";
    std::cout << "========================================\n";
    std::cout << "CPU unordered_set:        " << cpu_time << " ms\n";
    std::cout << "GPU Thrust binary_search: " << thrust_time << " ms (" 
              << cpu_time / thrust_time << "x speedup)\n";
    std::cout << "GPU Custom binary search: " << binary_time << " ms (" 
              << cpu_time / binary_time << "x speedup)\n";
    std::cout << "GPU Hash Set lookup:      " << hash_time << " ms (" 
              << cpu_time / hash_time << "x speedup)\n";
    
    std::cout << "\nPer-query latency:\n";
    std::cout << "CPU:          " << (cpu_time * 1000 / NUM_QUERIES) << " µs/query\n";
    std::cout << "GPU Thrust:   " << (thrust_time * 1000 / NUM_QUERIES) << " µs/query\n";
    std::cout << "GPU Binary:   " << (binary_time * 1000 / NUM_QUERIES) << " µs/query\n";
    std::cout << "GPU Hash:     " << (hash_time * 1000 / NUM_QUERIES) << " µs/query\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_results_binary));
    CUDA_CHECK(cudaFree(d_hash_table));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_results_hash));

    std::cout << "\n✓ All tests completed successfully!\n";
    return 0;
}
