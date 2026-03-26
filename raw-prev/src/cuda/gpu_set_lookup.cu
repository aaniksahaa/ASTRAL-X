#include <stdint.h>

extern "C" __global__ void gpu_set_lookup(
    const uint64_t* query_keys,
    const uint64_t* table_keys,
    const int table_size,
    int* found_flags,
    int query_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= query_count) {
        return;
    }

    uint64_t q = query_keys[idx];
    int found = 0;
    for (int i = 0; i < table_size; i++) {
        if (table_keys[i] == q) {
            found = 1;
            break;
        }
    }
    found_flags[idx] = found;
}
