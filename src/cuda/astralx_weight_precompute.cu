#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static const int INPUT_MAGIC = 0x41585731; // AWX1

struct ClusterDesc {
    int treeIndex;
    int left;
    int right;
    int flags; // bit0 localComp, bit1 globalComp, bit2 allTaxa
    int size;
};

static inline int32_t readIntBE(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) return -1;
    return (int32_t(b[0]) << 24) | (int32_t(b[1]) << 16) | (int32_t(b[2]) << 8) | int32_t(b[3]);
}

static inline bool checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

static inline int levelsForSigma(int sigmaExclusive) {
    int v = std::max(1, sigmaExclusive - 1);
    int levels = 0;
    while (v > 0) {
        levels++;
        v >>= 1;
    }
    return std::max(1, levels);
}

static void buildWaveletForPair(
    int i,
    int j,
    int K,
    int N,
    int levels,
    const std::vector<int>& present,
    const std::vector<int>& ordering,
    const std::vector<int>& position,
    std::vector<int>& pairPrefixOffset,
    std::vector<int>& pairZeroOffset,
    std::vector<int>& prefixPool,
    std::vector<int>& zeroPool
) {
    int len = present[i];
    int pairId = i * K + j;
    if (i == j || len <= 0) {
        pairPrefixOffset[pairId] = -1;
        pairZeroOffset[pairId] = -1;
        return;
    }

    std::vector<int> Y(len);
    for (int p = 0; p < len; ++p) {
        int taxon = ordering[i * N + p];
        int posj = (taxon < 0) ? -1 : position[j * N + taxon];
        Y[p] = posj + 1; // missing => 0, present => 1..present[j]
    }

    int pOff = (int)prefixPool.size();
    int zOff = (int)zeroPool.size();
    pairPrefixOffset[pairId] = pOff;
    pairZeroOffset[pairId] = zOff;

    prefixPool.resize(prefixPool.size() + levels * (len + 1), 0);
    zeroPool.resize(zeroPool.size() + levels, 0);

    std::vector<int> curr = Y;
    std::vector<int> next(len);

    for (int lvl = 0; lvl < levels; ++lvl) {
        int bit = levels - 1 - lvl;
        int base = pOff + lvl * (len + 1);

        int zeros = 0;
        prefixPool[base] = 0;
        for (int t = 0; t < len; ++t) {
            int b = (curr[t] >> bit) & 1;
            prefixPool[base + t + 1] = prefixPool[base + t] + b;
            if (b == 0) zeros++;
        }
        zeroPool[zOff + lvl] = zeros;

        int z = 0, o = zeros;
        for (int t = 0; t < len; ++t) {
            int v = curr[t];
            int b = (v >> bit) & 1;
            if (b == 0) next[z++] = v;
            else next[o++] = v;
        }
        curr.swap(next);
    }
}

__device__ __forceinline__ int wmLessThan(
    int l,
    int r,
    int x,
    int len,
    int levels,
    const int* prefix,
    const int* zeros
) {
    int cnt = 0;
    for (int lvl = 0; lvl < levels; ++lvl) {
        int bit = levels - 1 - lvl;
        int xb = (x >> bit) & 1;

        int base = lvl * (len + 1);
        int onesL = prefix[base + l];
        int onesR = prefix[base + r];
        int zerosL = l - onesL;
        int zerosR = r - onesR;

        if (xb) {
            cnt += (zerosR - zerosL);
            l = zeros[lvl] + onesL;
            r = zeros[lvl] + onesR;
        } else {
            l = zerosL;
            r = zerosR;
        }
    }
    return cnt;
}

__device__ __forceinline__ int wmRangeFreq(
    int l,
    int r,
    int lower,
    int upper,
    int len,
    int levels,
    const int* prefix,
    const int* zeros
) {
    if (l < 0) l = 0;
    if (r > len) r = len;
    if (l >= r || lower >= upper) return 0;
    int a = wmLessThan(l, r, upper, len, levels, prefix, zeros);
    int b = wmLessThan(l, r, lower, len, levels, prefix, zeros);
    return a - b;
}

__device__ __forceinline__ int rangeRange(
    int ti,
    int l1,
    int r1,
    int tj,
    int l2,
    int r2,
    int K,
    int levels,
    const int* present,
    const int* pairPrefixOffset,
    const int* pairZeroOffset,
    const int* prefixPool,
    const int* zeroPool
) {
    if (l1 > r1 || l2 > r2) return 0;
    if (ti == tj) {
        int lo = max(l1, l2);
        int hi = min(r1, r2);
        return max(0, hi - lo + 1);
    }

    int pairId = ti * K + tj;
    int pOff = pairPrefixOffset[pairId];
    int zOff = pairZeroOffset[pairId];
    if (pOff < 0 || zOff < 0) return 0;

    int len = present[ti];
    const int* pref = prefixPool + pOff;
    const int* zeros = zeroPool + zOff;

    return wmRangeFreq(l1, r1 + 1, l2 + 1, r2 + 2, len, levels, pref, zeros);
}

__device__ __forceinline__ int localSizeNoGlobal(
    const ClusterDesc& c,
    int N,
    const int* present
) {
    (void)N;
    int range = c.right - c.left + 1;
    if ((c.flags & 1) == 0) return range;
    return present[c.treeIndex] - range;
}

__device__ __forceinline__ int localIntersection(
    const ClusterDesc& a,
    const ClusterDesc& b,
    int N,
    int K,
    int levels,
    const int* present,
    const int* pairPrefixOffset,
    const int* pairZeroOffset,
    const int* prefixPool,
    const int* zeroPool,
    const int* sharedUniverse
) {
    (void)N;
    bool al = (a.flags & 1) != 0;
    bool bl = (b.flags & 1) != 0;

    auto rr = [&](int i, int l1, int r1, int j, int l2, int r2) {
        return rangeRange(i, l1, r1, j, l2, r2, K, levels, present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool);
    };

    if (!al && !bl) {
        return rr(a.treeIndex, a.left, a.right, b.treeIndex, b.left, b.right);
    }

    if (al && !bl) {
        int uiRj = rr(a.treeIndex, 0, present[a.treeIndex] - 1, b.treeIndex, b.left, b.right);
        int rra = rr(a.treeIndex, a.left, a.right, b.treeIndex, b.left, b.right);
        return uiRj - rra;
    }

    if (!al && bl) {
        int riUj = rr(a.treeIndex, a.left, a.right, b.treeIndex, 0, present[b.treeIndex] - 1);
        int rra = rr(a.treeIndex, a.left, a.right, b.treeIndex, b.left, b.right);
        return riUj - rra;
    }

    int uiUj = sharedUniverse[a.treeIndex * K + b.treeIndex];
    int riUj = rr(a.treeIndex, a.left, a.right, b.treeIndex, 0, present[b.treeIndex] - 1);
    int uiRj = rr(a.treeIndex, 0, present[a.treeIndex] - 1, b.treeIndex, b.left, b.right);
    int rra = rr(a.treeIndex, a.left, a.right, b.treeIndex, b.left, b.right);
    return uiUj - riUj - uiRj + rra;
}

__device__ __forceinline__ int clusterIntersection(
    const ClusterDesc& a,
    const ClusterDesc& b,
    int N,
    int K,
    int levels,
    const int* present,
    const int* pairPrefixOffset,
    const int* pairZeroOffset,
    const int* prefixPool,
    const int* zeroPool,
    const int* sharedUniverse
) {
    bool aAll = (a.flags & 4) != 0;
    bool bAll = (b.flags & 4) != 0;

    if (aAll && bAll) return N;
    if (aAll) return b.size;
    if (bAll) return a.size;

    int localA = localSizeNoGlobal(a, N, present);
    int localB = localSizeNoGlobal(b, N, present);

    int localAB = localIntersection(
        a, b, N, K, levels,
        present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool, sharedUniverse);

    bool ag = (a.flags & 2) != 0;
    bool bg = (b.flags & 2) != 0;

    if (!ag && !bg) return localAB;
    if (ag && !bg) return localB - localAB;
    if (!ag && bg) return localA - localAB;
    return N - localA - localB + localAB;
}

__device__ __forceinline__ double term(int a, int b, int c) {
    return ((double)(a + b + c - 3) / 2.0) * double(a) * double(b) * double(c);
}

__global__ void weightKernel(
    const ClusterDesc* candL,
    const ClusterDesc* candR,
    const ClusterDesc* partA,
    const ClusterDesc* partB,
    const ClusterDesc* partL,
    const int* freq,
    int C,
    int P,
    int N,
    int K,
    int levels,
    const int* present,
    const int* pairPrefixOffset,
    const int* pairZeroOffset,
    const int* prefixPool,
    const int* zeroPool,
    const int* sharedUniverse,
    double* outW
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= C) return;

    ClusterDesc x = candL[cid];
    ClusterDesc y = candR[cid];

    double total = 0.0;
    for (int pid = 0; pid < P; ++pid) {
        ClusterDesc a = partA[pid];
        ClusterDesc b = partB[pid];
        ClusterDesc lgt = partL[pid];

        int xA = clusterIntersection(x, a, N, K, levels, present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool, sharedUniverse);
        int xB = clusterIntersection(x, b, N, K, levels, present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool, sharedUniverse);
        int xL = clusterIntersection(x, lgt, N, K, levels, present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool, sharedUniverse);

        int yA = clusterIntersection(y, a, N, K, levels, present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool, sharedUniverse);
        int yB = clusterIntersection(y, b, N, K, levels, present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool, sharedUniverse);
        int yL = clusterIntersection(y, lgt, N, K, levels, present, pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool, sharedUniverse);

        int sa = a.size;
        int sb = b.size;
        int sl = lgt.size;
        int sc = sl - sa - sb;

        int xC = xL - xA - xB;
        int yC = yL - yA - yB;

        int zA = sa - xA - yA;
        int zB = sb - xB - yB;
        int zC = sc - xC - yC;

        double qi =
            term(xA, yB, zC) +
            term(xA, yC, zB) +
            term(xB, yA, zC) +
            term(xB, yC, zA) +
            term(xC, yA, zB) +
            term(xC, yB, zA);

        total += 0.5 * qi * double(freq[pid]);
    }

    outW[cid] = total;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <output.txt>\n";
        return 2;
    }

    std::string inPath = argv[1];
    std::string outPath = argv[2];

    std::ifstream in(inPath, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open input file: " << inPath << "\n";
        return 3;
    }

    int magic = readIntBE(in);
    int C = readIntBE(in);
    int P = readIntBE(in);
    int N = readIntBE(in);
    int K = readIntBE(in);

    if (magic != INPUT_MAGIC) {
        std::cerr << "Invalid input magic\n";
        return 4;
    }
    if (C <= 0 || P <= 0 || N <= 0 || K <= 0) {
        std::cerr << "Invalid dimensions C=" << C << " P=" << P << " N=" << N << " K=" << K << "\n";
        return 5;
    }

    int levels = levelsForSigma(N + 2);

    std::vector<int> present(K);
    for (int i = 0; i < K; ++i) present[i] = readIntBE(in);

    std::vector<int> ordering(K * N);
    for (int i = 0; i < K * N; ++i) ordering[i] = readIntBE(in);

    std::vector<int> position(K * N);
    for (int i = 0; i < K * N; ++i) position[i] = readIntBE(in);

    std::vector<ClusterDesc> hCandL(C), hCandR(C), hPartA(P), hPartB(P), hPartL(P);
    auto readClusters = [&](std::vector<ClusterDesc>& arr) {
        for (auto& c : arr) {
            c.treeIndex = readIntBE(in);
            c.left = readIntBE(in);
            c.right = readIntBE(in);
            c.flags = readIntBE(in);
            c.size = readIntBE(in);
        }
    };
    readClusters(hCandL);
    readClusters(hCandR);
    readClusters(hPartA);
    readClusters(hPartB);
    readClusters(hPartL);

    std::vector<int> hFreq(P);
    for (int i = 0; i < P; ++i) hFreq[i] = readIntBE(in);

    if (!in) {
        std::cerr << "Failed reading input payload\n";
        return 6;
    }

    // Build wavelet matrices for all ordered tree pairs (i,j)
    std::vector<int> pairPrefixOffset(K * K, -1);
    std::vector<int> pairZeroOffset(K * K, -1);
    std::vector<int> prefixPool;
    std::vector<int> zeroPool;
    prefixPool.reserve((size_t)K * K * levels * (N + 1));
    zeroPool.reserve((size_t)K * K * levels);

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            buildWaveletForPair(i, j, K, N, levels, present, ordering, position,
                                pairPrefixOffset, pairZeroOffset, prefixPool, zeroPool);
        }
    }

    // shared universe intersections |Ui ∩ Uj|
    std::vector<int> sharedUniverse(K * K, 0);
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            if (i == j) {
                sharedUniverse[i * K + j] = present[i];
                continue;
            }
            int cnt = 0;
            for (int t = 0; t < N; ++t) {
                int pi = position[i * N + t];
                int pj = position[j * N + t];
                if (pi >= 0 && pj >= 0) cnt++;
            }
            sharedUniverse[i * K + j] = cnt;
        }
    }

    ClusterDesc *dCandL = nullptr, *dCandR = nullptr, *dPartA = nullptr, *dPartB = nullptr, *dPartL = nullptr;
    int *dFreq = nullptr, *dPresent = nullptr, *dPairPrefixOffset = nullptr, *dPairZeroOffset = nullptr;
    int *dPrefixPool = nullptr, *dZeroPool = nullptr, *dSharedUniverse = nullptr;
    double* dOut = nullptr;

    if (!checkCuda(cudaMalloc(&dCandL, C * sizeof(ClusterDesc)), "cudaMalloc dCandL")) return 10;
    if (!checkCuda(cudaMalloc(&dCandR, C * sizeof(ClusterDesc)), "cudaMalloc dCandR")) return 10;
    if (!checkCuda(cudaMalloc(&dPartA, P * sizeof(ClusterDesc)), "cudaMalloc dPartA")) return 10;
    if (!checkCuda(cudaMalloc(&dPartB, P * sizeof(ClusterDesc)), "cudaMalloc dPartB")) return 10;
    if (!checkCuda(cudaMalloc(&dPartL, P * sizeof(ClusterDesc)), "cudaMalloc dPartL")) return 10;
    if (!checkCuda(cudaMalloc(&dFreq, P * sizeof(int)), "cudaMalloc dFreq")) return 10;
    if (!checkCuda(cudaMalloc(&dPresent, K * sizeof(int)), "cudaMalloc dPresent")) return 10;
    if (!checkCuda(cudaMalloc(&dPairPrefixOffset, K * K * sizeof(int)), "cudaMalloc dPairPrefixOffset")) return 10;
    if (!checkCuda(cudaMalloc(&dPairZeroOffset, K * K * sizeof(int)), "cudaMalloc dPairZeroOffset")) return 10;
    if (!checkCuda(cudaMalloc(&dPrefixPool, prefixPool.size() * sizeof(int)), "cudaMalloc dPrefixPool")) return 10;
    if (!checkCuda(cudaMalloc(&dZeroPool, zeroPool.size() * sizeof(int)), "cudaMalloc dZeroPool")) return 10;
    if (!checkCuda(cudaMalloc(&dSharedUniverse, K * K * sizeof(int)), "cudaMalloc dSharedUniverse")) return 10;
    if (!checkCuda(cudaMalloc(&dOut, C * sizeof(double)), "cudaMalloc dOut")) return 10;

    if (!checkCuda(cudaMemcpy(dCandL, hCandL.data(), C * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy candL")) return 11;
    if (!checkCuda(cudaMemcpy(dCandR, hCandR.data(), C * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy candR")) return 11;
    if (!checkCuda(cudaMemcpy(dPartA, hPartA.data(), P * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy partA")) return 11;
    if (!checkCuda(cudaMemcpy(dPartB, hPartB.data(), P * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy partB")) return 11;
    if (!checkCuda(cudaMemcpy(dPartL, hPartL.data(), P * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy partL")) return 11;
    if (!checkCuda(cudaMemcpy(dFreq, hFreq.data(), P * sizeof(int), cudaMemcpyHostToDevice), "copy freq")) return 11;
    if (!checkCuda(cudaMemcpy(dPresent, present.data(), K * sizeof(int), cudaMemcpyHostToDevice), "copy present")) return 11;
    if (!checkCuda(cudaMemcpy(dPairPrefixOffset, pairPrefixOffset.data(), K * K * sizeof(int), cudaMemcpyHostToDevice), "copy pairPrefixOffset")) return 11;
    if (!checkCuda(cudaMemcpy(dPairZeroOffset, pairZeroOffset.data(), K * K * sizeof(int), cudaMemcpyHostToDevice), "copy pairZeroOffset")) return 11;
    if (!checkCuda(cudaMemcpy(dPrefixPool, prefixPool.data(), prefixPool.size() * sizeof(int), cudaMemcpyHostToDevice), "copy prefixPool")) return 11;
    if (!checkCuda(cudaMemcpy(dZeroPool, zeroPool.data(), zeroPool.size() * sizeof(int), cudaMemcpyHostToDevice), "copy zeroPool")) return 11;
    if (!checkCuda(cudaMemcpy(dSharedUniverse, sharedUniverse.data(), K * K * sizeof(int), cudaMemcpyHostToDevice), "copy sharedUniverse")) return 11;

    int blockSize = 128;
    int gridSize = (C + blockSize - 1) / blockSize;
    weightKernel<<<gridSize, blockSize>>>(
        dCandL,
        dCandR,
        dPartA,
        dPartB,
        dPartL,
        dFreq,
        C,
        P,
        N,
        K,
        levels,
        dPresent,
        dPairPrefixOffset,
        dPairZeroOffset,
        dPrefixPool,
        dZeroPool,
        dSharedUniverse,
        dOut
    );
    if (!checkCuda(cudaGetLastError(), "kernel launch")) return 12;
    if (!checkCuda(cudaDeviceSynchronize(), "kernel sync")) return 12;

    std::vector<double> hOut(C);
    if (!checkCuda(cudaMemcpy(hOut.data(), dOut, C * sizeof(double), cudaMemcpyDeviceToHost), "copy out")) return 13;

    std::ofstream out(outPath);
    if (!out) {
        std::cerr << "Cannot open output file: " << outPath << "\n";
        return 14;
    }
    out.setf(std::ios::fixed);
    out.precision(17);
    for (int i = 0; i < C; ++i) {
        out << hOut[i] << '\n';
    }

    cudaFree(dCandL);
    cudaFree(dCandR);
    cudaFree(dPartA);
    cudaFree(dPartB);
    cudaFree(dPartL);
    cudaFree(dFreq);
    cudaFree(dPresent);
    cudaFree(dPairPrefixOffset);
    cudaFree(dPairZeroOffset);
    cudaFree(dPrefixPool);
    cudaFree(dZeroPool);
    cudaFree(dSharedUniverse);
    cudaFree(dOut);

    return 0;
}
