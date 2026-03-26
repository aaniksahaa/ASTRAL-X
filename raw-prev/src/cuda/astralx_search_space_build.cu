#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static const int IN_MAGIC = 0x41534231;   // ASB1
static const int OUT_MAGIC = 0x41534232;  // ASB2

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

__global__ void emitSplitsKernel(
    const int* qA,
    const int* qB,
    const int* qLo,
    const int* qHi,
    int Q,
    const int* sortedIdx,
    const int* clusterIds,
    const uint64_t* bitsets,
    int words,
    int* outTriples,
    int outCap,
    int* outCount,
    int* overflow
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q) return;

    int a = qA[q];
    int b = qB[q];
    int lo = qLo[q];
    int hi = qHi[q];

    const uint64_t* bitsA = bitsets + (size_t)a * words;
    const uint64_t* bitsB = bitsets + (size_t)b * words;

    int idB = clusterIds[b];

    for (int p = lo; p < hi; ++p) {
        int c = sortedIdx[p];
        if (b == c) continue;

        int idC = clusterIds[c];
        if (idB > idC) continue; // deterministic dedup

        const uint64_t* bitsC = bitsets + (size_t)c * words;

        bool ok = true;
        for (int w = 0; w < words; ++w) {
            uint64_t inter = bitsB[w] & bitsC[w];
            if (inter != 0ULL) {
                ok = false;
                break;
            }
            uint64_t uni = bitsB[w] | bitsC[w];
            if (uni != bitsA[w]) {
                ok = false;
                break;
            }
        }

        if (!ok) continue;

        int pos = atomicAdd(outCount, 1);
        if (pos < outCap) {
            int base = pos * 3;
            outTriples[base + 0] = a;
            outTriples[base + 1] = b;
            outTriples[base + 2] = c;
        } else {
            atomicExch(overflow, 1);
        }
    }
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
    int C = readIntBE(in);
    int T = readIntBE(in);
    int words = readIntBE(in);
    int Q = readIntBE(in);
    int outCap = readIntBE(in);

    if (magic != IN_MAGIC || C <= 0 || words <= 0 || Q < 0 || outCap <= 0) {
        std::cerr << "Invalid input header\n";
        return 4;
    }

    thrust::host_vector<uint64_t> hSig(C);
    thrust::host_vector<int> hClusterIds(C);
    for (int i = 0; i < C; ++i) {
        hSig[i] = readLongBE(in);
        hClusterIds[i] = readIntBE(in);
    }

    std::vector<uint64_t> hBits((size_t)C * words);
    for (size_t i = 0; i < hBits.size(); ++i) {
        hBits[i] = readLongBE(in);
    }

    thrust::host_vector<int> hQA(Q), hQB(Q);
    thrust::host_vector<uint64_t> hQKey(Q);
    for (int i = 0; i < Q; ++i) {
        hQA[i] = readIntBE(in);
        hQB[i] = readIntBE(in);
        hQKey[i] = readLongBE(in);
    }

    if (!in) {
        std::cerr << "Input truncated\n";
        return 5;
    }

    thrust::device_vector<uint64_t> dSig = hSig;
    thrust::device_vector<int> dSortedIdx(C);
    thrust::sequence(dSortedIdx.begin(), dSortedIdx.end());
    thrust::sort_by_key(dSig.begin(), dSig.end(), dSortedIdx.begin());

    thrust::device_vector<uint64_t> dQKey = hQKey;
    thrust::device_vector<int> dLo(Q), dHi(Q);
    thrust::lower_bound(dSig.begin(), dSig.end(), dQKey.begin(), dQKey.end(), dLo.begin());
    thrust::upper_bound(dSig.begin(), dSig.end(), dQKey.begin(), dQKey.end(), dHi.begin());

    thrust::device_vector<int> dQA = hQA;
    thrust::device_vector<int> dQB = hQB;
    thrust::device_vector<int> dClusterIds = hClusterIds;

    uint64_t* dBits = nullptr;
    cudaMalloc(&dBits, hBits.size() * sizeof(uint64_t));
    cudaMemcpy(dBits, hBits.data(), hBits.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int* dOutTriples = nullptr;
    int* dOutCount = nullptr;
    int* dOverflow = nullptr;
    cudaMalloc(&dOutTriples, outCap * 3 * sizeof(int));
    cudaMalloc(&dOutCount, sizeof(int));
    cudaMalloc(&dOverflow, sizeof(int));
    cudaMemset(dOutCount, 0, sizeof(int));
    cudaMemset(dOverflow, 0, sizeof(int));

    int block = 256;
    int grid = (Q + block - 1) / block;
    emitSplitsKernel<<<grid, block>>>(
        thrust::raw_pointer_cast(dQA.data()),
        thrust::raw_pointer_cast(dQB.data()),
        thrust::raw_pointer_cast(dLo.data()),
        thrust::raw_pointer_cast(dHi.data()),
        Q,
        thrust::raw_pointer_cast(dSortedIdx.data()),
        thrust::raw_pointer_cast(dClusterIds.data()),
        dBits,
        words,
        dOutTriples,
        outCap,
        dOutCount,
        dOverflow
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err) << "\n";
        return 6;
    }

    int hCount = 0;
    int hOverflow = 0;
    cudaMemcpy(&hCount, dOutCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hOverflow, dOverflow, sizeof(int), cudaMemcpyDeviceToHost);

    int used = hCount;
    if (used > outCap) used = outCap;
    std::vector<int> hTriples((size_t)used * 3);
    if (used > 0) {
        cudaMemcpy(hTriples.data(), dOutTriples, hTriples.size() * sizeof(int), cudaMemcpyDeviceToHost);
    }

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        std::cerr << "Cannot open output\n";
        return 7;
    }
    writeIntBE(out, OUT_MAGIC);
    writeIntBE(out, used);
    writeIntBE(out, hOverflow);
    for (size_t i = 0; i < hTriples.size(); ++i) {
        writeIntBE(out, hTriples[i]);
    }

    cudaFree(dBits);
    cudaFree(dOutTriples);
    cudaFree(dOutCount);
    cudaFree(dOverflow);

    return 0;
}
