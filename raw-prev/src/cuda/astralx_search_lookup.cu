#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static const int INPUT_MAGIC = 0x41534C31; // ASL1
static const int OUTPUT_MAGIC = 0x41534C32; // ASL2

static inline int32_t readIntBE(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) return -1;
    return (int32_t(b[0]) << 24) | (int32_t(b[1]) << 16) | (int32_t(b[2]) << 8) | int32_t(b[3]);
}

static inline uint64_t readLongBE(std::ifstream& in) {
    unsigned char b[8];
    in.read(reinterpret_cast<char*>(b), 8);
    if (!in) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | uint64_t(b[i]);
    return v;
}

static inline void writeIntBE(std::ofstream& out, int32_t v) {
    out.put((char)((v >> 24) & 0xFF));
    out.put((char)((v >> 16) & 0xFF));
    out.put((char)((v >> 8) & 0xFF));
    out.put((char)(v & 0xFF));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <output.bin>\n";
        return 2;
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open input\n";
        return 3;
    }

    int magic = readIntBE(in);
    int N = readIntBE(in);
    int Q = readIntBE(in);
    if (magic != INPUT_MAGIC || N < 0 || Q < 0) {
        std::cerr << "Invalid input header\n";
        return 4;
    }

    thrust::host_vector<uint64_t> hKeys(N);
    thrust::host_vector<int> hIds(N);
    for (int i = 0; i < N; ++i) {
        hKeys[i] = readLongBE(in);
        hIds[i] = readIntBE(in);
    }

    thrust::host_vector<uint64_t> hQueries(Q);
    for (int i = 0; i < Q; ++i) {
        hQueries[i] = readLongBE(in);
    }

    if (!in) {
        std::cerr << "Input payload truncated\n";
        return 5;
    }

    thrust::device_vector<uint64_t> dKeys = hKeys;
    thrust::device_vector<int> dIds = hIds;
    thrust::sort_by_key(dKeys.begin(), dKeys.end(), dIds.begin());

    thrust::device_vector<uint64_t> dQueries = hQueries;
    thrust::device_vector<int> dLo(Q), dHi(Q);

    thrust::lower_bound(dKeys.begin(), dKeys.end(), dQueries.begin(), dQueries.end(), dLo.begin());
    thrust::upper_bound(dKeys.begin(), dKeys.end(), dQueries.begin(), dQueries.end(), dHi.begin());

    thrust::host_vector<int> hSortedIds = dIds;
    thrust::host_vector<int> hLo = dLo;
    thrust::host_vector<int> hHi = dHi;

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        std::cerr << "Cannot open output\n";
        return 6;
    }

    writeIntBE(out, OUTPUT_MAGIC);
    writeIntBE(out, N);
    writeIntBE(out, Q);
    for (int i = 0; i < N; ++i) writeIntBE(out, hSortedIds[i]);
    for (int i = 0; i < Q; ++i) {
        writeIntBE(out, hLo[i]);
        writeIntBE(out, hHi[i]);
    }

    return 0;
}
