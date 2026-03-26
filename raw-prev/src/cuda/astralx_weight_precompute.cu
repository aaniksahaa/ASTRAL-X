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

static void buildWaveletToAnchor(
    int tree,
    int anchor,
    int K,
    int N,
    int levels,
    const std::vector<int>& present,
    const std::vector<int>& ordering,
    const std::vector<int>& position,
    std::vector<int>& prefixOffsetByTree,
    std::vector<int>& zeroOffsetByTree,
    std::vector<int>& prefixPool,
    std::vector<int>& zeroPool
) {
    if (tree == anchor || present[tree] <= 0) {
        prefixOffsetByTree[tree] = -1;
        zeroOffsetByTree[tree] = -1;
        return;
    }

    int len = present[tree];
    std::vector<int> Y(len);
    for (int p = 0; p < len; ++p) {
        int taxon = ordering[tree * N + p];
        int posAnchor = (taxon < 0) ? -1 : position[anchor * N + taxon];
        Y[p] = posAnchor + 1; // missing => 0, present => 1..present[anchor]
    }

    int pOff = (int)prefixPool.size();
    int zOff = (int)zeroPool.size();
    prefixOffsetByTree[tree] = pOff;
    zeroOffsetByTree[tree] = zOff;

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

__device__ __forceinline__ int rangeRangeToAnchor(
    int ti,
    int l1,
    int r1,
    int anchor,
    int l2,
    int r2,
    int levels,
    const int* present,
    const int* prefixOffsetByTree,
    const int* zeroOffsetByTree,
    const int* prefixPool,
    const int* zeroPool
) {
    if (l1 > r1 || l2 > r2) return 0;

    if (ti == anchor) {
        int lo = max(l1, l2);
        int hi = min(r1, r2);
        return max(0, hi - lo + 1);
    }

    int pOff = prefixOffsetByTree[ti];
    int zOff = zeroOffsetByTree[ti];
    if (pOff < 0 || zOff < 0) return 0;

    int len = present[ti];
    const int* pref = prefixPool + pOff;
    const int* zeros = zeroPool + zOff;
    return wmRangeFreq(l1, r1 + 1, l2 + 1, r2 + 2, len, levels, pref, zeros);
}

__device__ __forceinline__ int localSizeNoGlobal(
    const ClusterDesc& c,
    const int* present
) {
    int range = c.right - c.left + 1;
    if ((c.flags & 1) == 0) return range;
    return present[c.treeIndex] - range;
}

__device__ __forceinline__ int localIntersectionToAnchor(
    const ClusterDesc& cand,
    const ClusterDesc& part,
    int anchor,
    int levels,
    const int* present,
    const int* prefixOffsetByTree,
    const int* zeroOffsetByTree,
    const int* prefixPool,
    const int* zeroPool,
    const int* sharedUniverseToAnchor
) {
    bool candLocalComp = (cand.flags & 1) != 0;
    bool partLocalComp = (part.flags & 1) != 0;

    auto rr = [&](int l1, int r1, int l2, int r2) {
        return rangeRangeToAnchor(cand.treeIndex, l1, r1, anchor, l2, r2,
                                  levels, present, prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool);
    };

    if (!candLocalComp && !partLocalComp) {
        return rr(cand.left, cand.right, part.left, part.right);
    }

    if (candLocalComp && !partLocalComp) {
        int uiRj = rr(0, present[cand.treeIndex] - 1, part.left, part.right);
        int rrBoth = rr(cand.left, cand.right, part.left, part.right);
        return uiRj - rrBoth;
    }

    if (!candLocalComp && partLocalComp) {
        int riUj = rr(cand.left, cand.right, 0, present[anchor] - 1);
        int rrBoth = rr(cand.left, cand.right, part.left, part.right);
        return riUj - rrBoth;
    }

    int uiUj = sharedUniverseToAnchor[cand.treeIndex];
    int riUj = rr(cand.left, cand.right, 0, present[anchor] - 1);
    int uiRj = rr(0, present[cand.treeIndex] - 1, part.left, part.right);
    int rrBoth = rr(cand.left, cand.right, part.left, part.right);
    return uiUj - riUj - uiRj + rrBoth;
}

__device__ __forceinline__ int clusterIntersectionToAnchor(
    const ClusterDesc& cand,
    const ClusterDesc& part,
    int N,
    int anchor,
    int levels,
    const int* present,
    const int* prefixOffsetByTree,
    const int* zeroOffsetByTree,
    const int* prefixPool,
    const int* zeroPool,
    const int* sharedUniverseToAnchor
) {
    bool candAll = (cand.flags & 4) != 0;
    bool partAll = (part.flags & 4) != 0;
    if (candAll && partAll) return N;
    if (candAll) return part.size;
    if (partAll) return cand.size;

    int localCand = localSizeNoGlobal(cand, present);
    int localPart = localSizeNoGlobal(part, present);

    int localBoth = localIntersectionToAnchor(
        cand, part, anchor, levels,
        present, prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool, sharedUniverseToAnchor);

    bool candGlobalComp = (cand.flags & 2) != 0;
    bool partGlobalComp = (part.flags & 2) != 0;

    if (!candGlobalComp && !partGlobalComp) return localBoth;
    if (candGlobalComp && !partGlobalComp) return localPart - localBoth;
    if (!candGlobalComp && partGlobalComp) return localCand - localBoth;
    return N - localCand - localPart + localBoth;
}

__device__ __forceinline__ double term(int a, int b, int c) {
    return ((double)(a + b + c - 3) / 2.0) * double(a) * double(b) * double(c);
}

__global__ void weightKernelAnchor(
    const ClusterDesc* candL,
    const ClusterDesc* candR,
    int C,
    const ClusterDesc* partA,
    const ClusterDesc* partB,
    const ClusterDesc* partL,
    const int* freq,
    int Panchor,
    int N,
    int anchor,
    int levels,
    const int* present,
    const int* prefixOffsetByTree,
    const int* zeroOffsetByTree,
    const int* prefixPool,
    const int* zeroPool,
    const int* sharedUniverseToAnchor,
    double* outW
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= C) return;

    ClusterDesc x = candL[cid];
    ClusterDesc y = candR[cid];

    double partial = 0.0;
    for (int pid = 0; pid < Panchor; ++pid) {
        ClusterDesc a = partA[pid];
        ClusterDesc b = partB[pid];
        ClusterDesc lgt = partL[pid];

        int xA = clusterIntersectionToAnchor(x, a, N, anchor, levels, present,
                                             prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool, sharedUniverseToAnchor);
        int xB = clusterIntersectionToAnchor(x, b, N, anchor, levels, present,
                                             prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool, sharedUniverseToAnchor);
        int xL = clusterIntersectionToAnchor(x, lgt, N, anchor, levels, present,
                                             prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool, sharedUniverseToAnchor);

        int yA = clusterIntersectionToAnchor(y, a, N, anchor, levels, present,
                                             prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool, sharedUniverseToAnchor);
        int yB = clusterIntersectionToAnchor(y, b, N, anchor, levels, present,
                                             prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool, sharedUniverseToAnchor);
        int yL = clusterIntersectionToAnchor(y, lgt, N, anchor, levels, present,
                                             prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool, sharedUniverseToAnchor);

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

        partial += 0.5 * qi * double(freq[pid]);
    }

    outW[cid] += partial;
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

    // Group partition indices by anchor tree index
    std::vector<std::vector<int>> partIdsByAnchor(K);
    for (int pid = 0; pid < P; ++pid) {
        int anchor = hPartA[pid].treeIndex;
        if (anchor < 0 || anchor >= K) {
            std::cerr << "Invalid partition anchor index: " << anchor << "\n";
            return 7;
        }
        partIdsByAnchor[anchor].push_back(pid);
    }

    // Device-global candidate and output buffers
    ClusterDesc *dCandL = nullptr, *dCandR = nullptr;
    double* dOut = nullptr;
    if (!checkCuda(cudaMalloc(&dCandL, C * sizeof(ClusterDesc)), "cudaMalloc dCandL")) return 10;
    if (!checkCuda(cudaMalloc(&dCandR, C * sizeof(ClusterDesc)), "cudaMalloc dCandR")) return 10;
    if (!checkCuda(cudaMalloc(&dOut, C * sizeof(double)), "cudaMalloc dOut")) return 10;
    if (!checkCuda(cudaMemcpy(dCandL, hCandL.data(), C * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy candL")) return 11;
    if (!checkCuda(cudaMemcpy(dCandR, hCandR.data(), C * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy candR")) return 11;
    if (!checkCuda(cudaMemset(dOut, 0, C * sizeof(double)), "memset out")) return 11;

    int blockSize = 128;
    int gridSize = (C + blockSize - 1) / blockSize;

    // Reused device pointer for present counts
    int* dPresent = nullptr;
    if (!checkCuda(cudaMalloc(&dPresent, K * sizeof(int)), "cudaMalloc dPresent")) return 10;
    if (!checkCuda(cudaMemcpy(dPresent, present.data(), K * sizeof(int), cudaMemcpyHostToDevice), "copy present")) return 11;

    // Iterate anchor-by-anchor to keep memory O(k * n * log n)
    for (int anchor = 0; anchor < K; ++anchor) {
        const std::vector<int>& ids = partIdsByAnchor[anchor];
        if (ids.empty()) continue;

        // Build wavelets for (tree -> anchor)
        std::vector<int> prefixOffsetByTree(K, -1);
        std::vector<int> zeroOffsetByTree(K, -1);
        std::vector<int> prefixPool;
        std::vector<int> zeroPool;
        prefixPool.reserve((size_t)K * levels * (N + 1));
        zeroPool.reserve((size_t)K * levels);

        for (int tree = 0; tree < K; ++tree) {
            buildWaveletToAnchor(tree, anchor, K, N, levels, present, ordering, position,
                                 prefixOffsetByTree, zeroOffsetByTree, prefixPool, zeroPool);
        }

        // shared universe |U_tree ∩ U_anchor|
        std::vector<int> sharedUniverseToAnchor(K, 0);
        for (int tree = 0; tree < K; ++tree) {
            if (tree == anchor) {
                sharedUniverseToAnchor[tree] = present[anchor];
                continue;
            }
            int cnt = 0;
            for (int t = 0; t < N; ++t) {
                int pi = position[tree * N + t];
                int pa = position[anchor * N + t];
                if (pi >= 0 && pa >= 0) cnt++;
            }
            sharedUniverseToAnchor[tree] = cnt;
        }

        // Compact partition block for this anchor
        int Panchor = (int)ids.size();
        std::vector<ClusterDesc> pA(Panchor), pB(Panchor), pL(Panchor);
        std::vector<int> pF(Panchor);
        for (int u = 0; u < Panchor; ++u) {
            int pid = ids[u];
            pA[u] = hPartA[pid];
            pB[u] = hPartB[pid];
            pL[u] = hPartL[pid];
            pF[u] = hFreq[pid];
        }

        // Device allocations for anchor block
        ClusterDesc *dPartA = nullptr, *dPartB = nullptr, *dPartL = nullptr;
        int *dFreq = nullptr, *dPrefixOffsetByTree = nullptr, *dZeroOffsetByTree = nullptr;
        int *dPrefixPool = nullptr, *dZeroPool = nullptr, *dSharedUniverseToAnchor = nullptr;

        if (!checkCuda(cudaMalloc(&dPartA, Panchor * sizeof(ClusterDesc)), "cudaMalloc dPartA")) return 10;
        if (!checkCuda(cudaMalloc(&dPartB, Panchor * sizeof(ClusterDesc)), "cudaMalloc dPartB")) return 10;
        if (!checkCuda(cudaMalloc(&dPartL, Panchor * sizeof(ClusterDesc)), "cudaMalloc dPartL")) return 10;
        if (!checkCuda(cudaMalloc(&dFreq, Panchor * sizeof(int)), "cudaMalloc dFreq")) return 10;
        if (!checkCuda(cudaMalloc(&dPrefixOffsetByTree, K * sizeof(int)), "cudaMalloc dPrefixOffsetByTree")) return 10;
        if (!checkCuda(cudaMalloc(&dZeroOffsetByTree, K * sizeof(int)), "cudaMalloc dZeroOffsetByTree")) return 10;
        if (!checkCuda(cudaMalloc(&dPrefixPool, std::max<size_t>(1, prefixPool.size()) * sizeof(int)), "cudaMalloc dPrefixPool")) return 10;
        if (!checkCuda(cudaMalloc(&dZeroPool, std::max<size_t>(1, zeroPool.size()) * sizeof(int)), "cudaMalloc dZeroPool")) return 10;
        if (!checkCuda(cudaMalloc(&dSharedUniverseToAnchor, K * sizeof(int)), "cudaMalloc dSharedUniverseToAnchor")) return 10;

        if (!checkCuda(cudaMemcpy(dPartA, pA.data(), Panchor * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy partA")) return 11;
        if (!checkCuda(cudaMemcpy(dPartB, pB.data(), Panchor * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy partB")) return 11;
        if (!checkCuda(cudaMemcpy(dPartL, pL.data(), Panchor * sizeof(ClusterDesc), cudaMemcpyHostToDevice), "copy partL")) return 11;
        if (!checkCuda(cudaMemcpy(dFreq, pF.data(), Panchor * sizeof(int), cudaMemcpyHostToDevice), "copy freq")) return 11;
        if (!checkCuda(cudaMemcpy(dPrefixOffsetByTree, prefixOffsetByTree.data(), K * sizeof(int), cudaMemcpyHostToDevice), "copy prefixOffsetByTree")) return 11;
        if (!checkCuda(cudaMemcpy(dZeroOffsetByTree, zeroOffsetByTree.data(), K * sizeof(int), cudaMemcpyHostToDevice), "copy zeroOffsetByTree")) return 11;
        if (!prefixPool.empty() && !checkCuda(cudaMemcpy(dPrefixPool, prefixPool.data(), prefixPool.size() * sizeof(int), cudaMemcpyHostToDevice), "copy prefixPool")) return 11;
        if (!zeroPool.empty() && !checkCuda(cudaMemcpy(dZeroPool, zeroPool.data(), zeroPool.size() * sizeof(int), cudaMemcpyHostToDevice), "copy zeroPool")) return 11;
        if (!checkCuda(cudaMemcpy(dSharedUniverseToAnchor, sharedUniverseToAnchor.data(), K * sizeof(int), cudaMemcpyHostToDevice), "copy sharedUniverseToAnchor")) return 11;

        weightKernelAnchor<<<gridSize, blockSize>>>(
            dCandL,
            dCandR,
            C,
            dPartA,
            dPartB,
            dPartL,
            dFreq,
            Panchor,
            N,
            anchor,
            levels,
            dPresent,
            dPrefixOffsetByTree,
            dZeroOffsetByTree,
            dPrefixPool,
            dZeroPool,
            dSharedUniverseToAnchor,
            dOut
        );

        if (!checkCuda(cudaGetLastError(), "kernel launch")) return 12;
        if (!checkCuda(cudaDeviceSynchronize(), "kernel sync")) return 12;

        cudaFree(dPartA);
        cudaFree(dPartB);
        cudaFree(dPartL);
        cudaFree(dFreq);
        cudaFree(dPrefixOffsetByTree);
        cudaFree(dZeroOffsetByTree);
        cudaFree(dPrefixPool);
        cudaFree(dZeroPool);
        cudaFree(dSharedUniverseToAnchor);
    }

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
    cudaFree(dPresent);
    cudaFree(dOut);

    return 0;
}
