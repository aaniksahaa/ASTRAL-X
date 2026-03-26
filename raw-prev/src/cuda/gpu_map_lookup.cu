#include <stdint.h>

extern "C" __global__ void gpu_map_lookup(
    const uint64_t* query_keys,
    const uint64_t* table_keys,
    const int* table_values,
    const int table_size,
    int* out_values,
    int query_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= query_count) {
        return;
    }

    uint64_t q = query_keys[idx];
    int value = -1;
    for (int i = 0; i < table_size; i++) {
        if (table_keys[i] == q) {
            value = table_values[i];
            break;
        }
    }
    out_values[idx] = value;
}
