/**
 * GPU Sorting Benchmark using CUDA Thrust
 * Compares GPU sorting (thrust::sort, thrust::stable_sort, radix sort)
 * against CPU sorting (std::sort, std::stable_sort) on various data sizes.
 *
 * Data types tested: int, float, (key, value) pairs
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

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
// Timing helpers
// ============================================================

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

// ============================================================
// Data generation helpers
// ============================================================

template<typename T>
std::vector<T> make_random_int_vec(int n, T lo, T hi, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(lo, hi);
    std::vector<T> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<float> make_random_float_vec(int n, float lo = 0.f, float hi = 1.f, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Nearly-sorted: sorted with k random swaps
std::vector<int> make_nearly_sorted(int n, int swaps = 1000, unsigned seed = 7) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < swaps; i++) std::swap(v[dist(rng)], v[dist(rng)]);
    return v;
}

// Reverse sorted
std::vector<int> make_reverse_sorted(int n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::reverse(v.begin(), v.end());
    return v;
}

// ============================================================
// Verification helpers
// ============================================================

template<typename T>
bool verify_sorted(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

// ============================================================
// Benchmark: thrust::sort  (int)
// ============================================================

struct BenchResult {
    double cpu_introsort_ms;  // std::sort (introsort, O(n log n))
    double cpu_mergesort_ms;  // std::stable_sort (merge sort, O(n log n))
    double gpu_sort_ms;       // thrust::sort
    double gpu_stable_ms;     // thrust::stable_sort
    double gpu_h2d_ms;        // host-to-device transfer
    double gpu_d2h_ms;        // device-to-host transfer
    bool   correct_sort;
    bool   correct_stable;
};

BenchResult bench_int_sort(const std::vector<int>& h_input, const std::string& label) {
    const int N = (int)h_input.size();
    BenchResult res{};

    // --- CPU std::sort (introsort) ---
    std::vector<int> cpu_sorted = h_input;
    {
        double t0 = now_ms();
        std::sort(cpu_sorted.begin(), cpu_sorted.end());
        res.cpu_introsort_ms = now_ms() - t0;
    }

    // --- CPU std::stable_sort (merge sort) ---
    {
        std::vector<int> tmp = h_input;
        double t0 = now_ms();
        std::stable_sort(tmp.begin(), tmp.end());
        res.cpu_mergesort_ms = now_ms() - t0;
    }

    // --- H2D transfer ---
    double t0 = now_ms();
    thrust::device_vector<int> d_data = h_input;
    CUDA_CHECK(cudaDeviceSynchronize());
    res.gpu_h2d_ms = now_ms() - t0;

    // --- GPU thrust::sort ---
    {
        thrust::device_vector<int> d_tmp = d_data;
        CUDA_CHECK(cudaDeviceSynchronize());
        t0 = now_ms();
        thrust::sort(d_tmp.begin(), d_tmp.end());
        CUDA_CHECK(cudaDeviceSynchronize());
        res.gpu_sort_ms = now_ms() - t0;

        double t1 = now_ms();
        thrust::host_vector<int> h_out = d_tmp;
        CUDA_CHECK(cudaDeviceSynchronize());
        res.gpu_d2h_ms = now_ms() - t1;

        res.correct_sort = verify_sorted(cpu_sorted, std::vector<int>(h_out.begin(), h_out.end()));
    }

    // --- GPU thrust::stable_sort ---
    {
        thrust::device_vector<int> d_tmp = d_data;
        CUDA_CHECK(cudaDeviceSynchronize());
        t0 = now_ms();
        thrust::stable_sort(d_tmp.begin(), d_tmp.end());
        CUDA_CHECK(cudaDeviceSynchronize());
        res.gpu_stable_ms = now_ms() - t0;

        thrust::host_vector<int> h_out = d_tmp;
        res.correct_stable = verify_sorted(cpu_sorted, std::vector<int>(h_out.begin(), h_out.end()));
    }

    // Print row: label | CPU introsort | CPU mergesort | GPU sort (speedup vs introsort) | GPU stable (speedup vs mergesort)
    double speedup_sort   = res.cpu_introsort_ms / res.gpu_sort_ms;
    double speedup_stable = res.cpu_mergesort_ms / res.gpu_stable_ms;

    std::cout << std::left  << std::setw(18) << label
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(9)  << res.cpu_introsort_ms << " ms"
              << std::setw(9)  << res.cpu_mergesort_ms << " ms"
              << std::setw(9)  << res.gpu_sort_ms      << " ms"
              << std::setw(7)  << std::setprecision(1) << speedup_sort   << "x"
              << std::setw(9)  << std::setprecision(2) << res.gpu_stable_ms << " ms"
              << std::setw(7)  << std::setprecision(1) << speedup_stable << "x"
              << "  " << (res.correct_sort ? "OK" : "FAIL")
              << "/" << (res.correct_stable ? "OK" : "FAIL")
              << "\n";

    return res;
}

// ============================================================
// Benchmark: float sort
// ============================================================

void bench_float_sort(const std::vector<float>& h_input, const std::string& label) {
    const int N = (int)h_input.size();

    std::vector<float> cpu_sorted = h_input;
    double t0 = now_ms();
    std::sort(cpu_sorted.begin(), cpu_sorted.end());
    double cpu_ms = now_ms() - t0;

    thrust::device_vector<float> d_tmp = h_input;
    CUDA_CHECK(cudaDeviceSynchronize());

    t0 = now_ms();
    thrust::sort(d_tmp.begin(), d_tmp.end());
    CUDA_CHECK(cudaDeviceSynchronize());
    double gpu_ms = now_ms() - t0;

    thrust::host_vector<float> h_out = d_tmp;
    bool ok = verify_sorted(cpu_sorted, std::vector<float>(h_out.begin(), h_out.end()));

    std::cout << std::left  << std::setw(20) << label
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << cpu_ms << " ms"
              << std::setw(10) << gpu_ms << " ms"
              << std::setw(8)  << std::setprecision(1) << (cpu_ms / gpu_ms) << "x"
              << "   " << (ok ? "OK" : "FAIL") << "\n";
}

// ============================================================
// Benchmark: key-value sort (argsort / sort by key)
// ============================================================

void bench_key_value_sort(int N, const std::string& label) {
    // Generate random keys, values = original indices (= argsort)
    std::vector<int> h_keys = make_random_int_vec<int>(N, 0, N * 10);
    std::vector<int> h_vals(N);
    std::iota(h_vals.begin(), h_vals.end(), 0);

    // CPU reference: stable argsort by key
    std::vector<int> cpu_vals = h_vals;
    std::vector<int> cpu_keys = h_keys;
    double t0 = now_ms();
    std::stable_sort(cpu_vals.begin(), cpu_vals.end(),
                     [&](int a, int b){ return h_keys[a] < h_keys[b]; });
    double cpu_ms = now_ms() - t0;

    // GPU: thrust::sort_by_key
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_vals = h_vals;
    CUDA_CHECK(cudaDeviceSynchronize());

    t0 = now_ms();
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    double gpu_ms = now_ms() - t0;

    // Verify: keys should be non-decreasing after sort
    thrust::host_vector<int> h_out_keys = d_keys;
    bool ok = true;
    for (int i = 1; i < N; i++)
        if (h_out_keys[i] < h_out_keys[i-1]) { ok = false; break; }

    std::cout << std::left  << std::setw(20) << label
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << cpu_ms << " ms"
              << std::setw(10) << gpu_ms << " ms"
              << std::setw(8)  << std::setprecision(1) << (cpu_ms / gpu_ms) << "x"
              << "   " << (ok ? "OK" : "FAIL") << "\n";
}

// ============================================================
// Main
// ============================================================

int main() {
    // Device info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "========================================\n";
    std::cout << "GPU Sorting Benchmark (Thrust vs CPU)\n";
    std::cout << "========================================\n";
    std::cout << "GPU: " << prop.name
              << "  |  SM count: " << prop.multiProcessorCount
              << "  |  Mem: " << (prop.totalGlobalMem >> 20) << " MB\n\n";

    // ----------------------------------------
    // Section 1: Integer sort across sizes
    // ----------------------------------------
    std::cout << "--- Integer Sort (random uniform) ---\n";
    std::cout << std::left  << std::setw(18) << "Dataset"
              << std::right << std::setw(11) << "CPU introsort"
              << std::setw(11) << "CPU mergesort"
              << std::setw(11) << "GPU sort"
              << std::setw(7)  << "Spdup"
              << std::setw(11) << "GPU stable"
              << std::setw(7)  << "Spdup"
              << "  Correct\n";
    std::cout << std::string(90, '-') << "\n";

    for (int N : {100000, 1000000, 5000000, 10000000, 50000000}) {
        std::string label = std::to_string(N / 1000) + "K ints";
        auto data = make_random_int_vec<int>(N, 0, N * 10);
        bench_int_sort(data, label);
    }

    // ----------------------------------------
    // Section 2: Different data patterns (N=10M)
    // ----------------------------------------
    const int N_PAT = 10000000;
    std::cout << "\n--- Integer Sort Patterns (N = 10M) ---\n";
    std::cout << std::left  << std::setw(18) << "Pattern"
              << std::right << std::setw(11) << "CPU introsort"
              << std::setw(11) << "CPU mergesort"
              << std::setw(11) << "GPU sort"
              << std::setw(7)  << "Spdup"
              << std::setw(11) << "GPU stable"
              << std::setw(7)  << "Spdup"
              << "  Correct\n";
    std::cout << std::string(90, '-') << "\n";

    bench_int_sort(make_random_int_vec<int>(N_PAT, 0, N_PAT * 10), "random");
    bench_int_sort(make_nearly_sorted(N_PAT, 1000),                 "nearly sorted");
    bench_int_sort(make_reverse_sorted(N_PAT),                      "reverse sorted");
    {
        // All same value → worst case for many algorithms
        std::vector<int> all_same(N_PAT, 42);
        bench_int_sort(all_same, "all same");
    }
    {
        // Low cardinality: only 256 distinct values
        auto data = make_random_int_vec<int>(N_PAT, 0, 255);
        bench_int_sort(data, "low cardinality");
    }

    // ----------------------------------------
    // Section 3: Float sort
    // ----------------------------------------
    std::cout << "\n--- Float Sort (random uniform, various sizes) ---\n";
    std::cout << std::left  << std::setw(20) << "Dataset"
              << std::right << std::setw(12) << "CPU sort"
              << std::setw(12) << "GPU sort"
              << std::setw(8)  << "Speedup"
              << "   Correct\n";
    std::cout << std::string(60, '-') << "\n";

    for (int N : {1000000, 10000000, 50000000}) {
        std::string label = std::to_string(N / 1000) + "K floats";
        bench_float_sort(make_random_float_vec(N), label);
    }

    // ----------------------------------------
    // Section 4: Key-value (sort_by_key)
    // ----------------------------------------
    std::cout << "\n--- Key-Value Sort (thrust::sort_by_key) ---\n";
    std::cout << std::left  << std::setw(20) << "Dataset"
              << std::right << std::setw(12) << "CPU stable"
              << std::setw(12) << "GPU sort_by_key"
              << std::setw(8)  << "Speedup"
              << "   Correct\n";
    std::cout << std::string(60, '-') << "\n";

    for (int N : {1000000, 10000000, 50000000}) {
        std::string label = std::to_string(N / 1000) + "K pairs";
        bench_key_value_sort(N, label);
    }

    // ----------------------------------------
    // Section 5: Total-time breakdown (H2D + sort + D2H) for 10M ints
    // ----------------------------------------
    std::cout << "\n--- Full Pipeline Breakdown (10M ints, random) ---\n";
    {
        const int N = 10000000;
        auto h_input = make_random_int_vec<int>(N, 0, N * 10);

        // CPU introsort
        std::vector<int> cpu_sorted = h_input;
        double t0 = now_ms();
        std::sort(cpu_sorted.begin(), cpu_sorted.end());
        double cpu_ms = now_ms() - t0;

        // CPU mergesort
        std::vector<int> cpu_ms_tmp = h_input;
        double t0b = now_ms();
        std::stable_sort(cpu_ms_tmp.begin(), cpu_ms_tmp.end());
        double cpu_merge_ms = now_ms() - t0b;

        // GPU: measure each phase separately
        t0 = now_ms();
        thrust::device_vector<int> d_data = h_input;
        CUDA_CHECK(cudaDeviceSynchronize());
        double h2d_ms = now_ms() - t0;

        t0 = now_ms();
        thrust::sort(d_data.begin(), d_data.end());
        CUDA_CHECK(cudaDeviceSynchronize());
        double sort_ms = now_ms() - t0;

        t0 = now_ms();
        thrust::host_vector<int> h_out = d_data;
        CUDA_CHECK(cudaDeviceSynchronize());
        double d2h_ms = now_ms() - t0;

        double total_gpu = h2d_ms + sort_ms + d2h_ms;
        bool ok = verify_sorted(cpu_sorted, std::vector<int>(h_out.begin(), h_out.end()));

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CPU std::sort (introsort):        " << cpu_ms       << " ms\n";
        std::cout << "CPU std::stable_sort (mergesort): " << cpu_merge_ms << " ms\n";
        std::cout << "GPU H2D transfer:                 " << h2d_ms       << " ms\n";
        std::cout << "GPU thrust::sort:                 " << sort_ms      << " ms\n";
        std::cout << "GPU D2H transfer:                 " << d2h_ms       << " ms\n";
        std::cout << "GPU total pipeline:               " << total_gpu    << " ms  ("
                  << std::setprecision(1) << (cpu_ms / total_gpu) << "x vs introsort, "
                  << (cpu_merge_ms / total_gpu) << "x vs mergesort)\n";
        std::cout << "GPU sort only:                    " << sort_ms      << " ms  ("
                  << (cpu_ms / sort_ms) << "x vs introsort, "
                  << (cpu_merge_ms / sort_ms) << "x vs mergesort)\n";
        std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "All benchmarks completed.\n";
    std::cout << "========================================\n";
    return 0;
}
