// ASTRAL-X branch-support (quartet-frequency local PP) CUDA kernel.
//
// Mirrors astralx.support.BranchSupport.scoreBranch bit for bit:
//   kernel 1 (one thread per (branch, gene chunk)): for each gene tree, count
//     the quartets around the branch's quadripartition supporting each of the
//     three resolutions with one post-order pass, and store the per-gene
//     fractions f_i / (2 * m_g) as doubles;
//   kernel 2 (one thread per branch): add the per-gene fractions in gene order,
//     exactly as the CPU loop does, so the sums are IEEE-identical.
// Integer counts are exact (64-bit); only the final fraction is a double.
#include <cuda_runtime.h>
#include <jni.h>
#include <cstdio>
#include <cstdint>

#ifndef ASTRALX_MIN_CUDA_CC
#define ASTRALX_MIN_CUDA_CC 0
#endif

typedef unsigned long long ull;

#define SUPPORT_MAX_STACK 1024

__device__ __forceinline__ int popAnd(const ull* __restrict__ a, const ull* __restrict__ b, int W) {
    int c = 0;
    for (int w = 0; w < W; w++) c += __popcll(a[w] & b[w]);
    return c;
}

__device__ __forceinline__ long long pairingTerm(long long x, long long y, long long z, long long w,
                                                 long long sx, long long sy, long long sz, long long sw,
                                                 long long pxy, long long pzw) {
    return x * y * ((sz - z) * (sw - w) - pzw + z * w)
         + z * w * ((sx - x) * (sy - y) - pxy + x * y);
}

template<int CAP>
__global__ void supportFractionKernel(
    const ull* __restrict__ clusterBits,   // [branch][4][W]
    const ull* __restrict__ geneBits,      // [gene][W]
    const int* __restrict__ tokens,
    const int* __restrict__ offsets,       // numTrees + 1
    int numBranches, int numTrees, int W, int chunks,
    double* __restrict__ frac,             // [branch][gene][3]
    unsigned char* __restrict__ eff)       // [branch][gene]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int branch = idx / chunks, chunk = idx - branch * chunks;
    if (branch >= numBranches) return;
    int gBeg = (int)(((long long)numTrees * chunk) / chunks);
    int gEnd = (int)(((long long)numTrees * (chunk + 1)) / chunks);

    const ull* c0 = clusterBits + (size_t)branch * 4 * W;
    const ull* c1 = c0 + W;
    const ull* c2 = c1 + W;
    const ull* c3 = c2 + W;
    int stack[CAP * 4];

    for (int g = gBeg; g < gEnd; g++) {
        const ull* gb = geneBits + (size_t)g * W;
        int t0 = popAnd(c0, gb, W), t1 = popAnd(c1, gb, W);
        int t2 = popAnd(c2, gb, W), t3 = popAnd(c3, gb, W);
        long long mi = (long long)t0 * t1 * t2 * t3;
        size_t fo = ((size_t)branch * numTrees + g) * 3;
        if (mi == 0) {
            eff[(size_t)branch * numTrees + g] = 0;
            frac[fo] = 0.0; frac[fo + 1] = 0.0; frac[fo + 2] = 0.0;
            continue;
        }
        eff[(size_t)branch * numTrees + g] = 1;
        long long f1 = 0, f2 = 0, f3 = 0;
        int top = 0;
        for (int i = offsets[g], end = offsets[g + 1]; i < end; i++) {
            int tok = tokens[i];
            if (tok >= 0) {
                int e = top << 2;
                int wi = tok >> 6, bi = tok & 63;
                stack[e]     = (int)((c0[wi] >> bi) & 1ULL);
                stack[e + 1] = (int)((c1[wi] >> bi) & 1ULL);
                stack[e + 2] = (int)((c2[wi] >> bi) & 1ULL);
                stack[e + 3] = (int)((c3[wi] >> bi) & 1ULL);
                top++;
                continue;
            }
            int k = -tok;
            int base = top - k;
            int bs = base << 2;
            int u0 = 0, u1 = 0, u2 = 0, u3 = 0;
            for (int j = 0; j < k; j++) {
                int e = bs + (j << 2);
                u0 += stack[e]; u1 += stack[e + 1]; u2 += stack[e + 2]; u3 += stack[e + 3];
            }
            long long r0 = t0 - u0, r1 = t1 - u1, r2 = t2 - u2, r3 = t3 - u3;
            long long p01 = r0 * r1, p23 = r2 * r3;
            long long p02 = r0 * r2, p13 = r1 * r3;
            long long p03 = r0 * r3, p12 = r1 * r2;
            for (int j = 0; j < k; j++) {
                int e = bs + (j << 2);
                long long a = stack[e], bb = stack[e + 1], c = stack[e + 2], dd = stack[e + 3];
                p01 += a * bb; p23 += c * dd;
                p02 += a * c;  p13 += bb * dd;
                p03 += a * dd; p12 += bb * c;
            }
            f1 += pairingTerm(r0, r1, r2, r3, t0, t1, t2, t3, p01, p23);
            f2 += pairingTerm(r0, r2, r1, r3, t0, t2, t1, t3, p02, p13);
            f3 += pairingTerm(r0, r3, r1, r2, t0, t3, t1, t2, p03, p12);
            for (int j = 0; j < k; j++) {
                int e = bs + (j << 2);
                long long a = stack[e], bb = stack[e + 1], c = stack[e + 2], dd = stack[e + 3];
                f1 += pairingTerm(a, bb, c, dd, t0, t1, t2, t3, p01, p23);
                f2 += pairingTerm(a, c, bb, dd, t0, t2, t1, t3, p02, p13);
                f3 += pairingTerm(a, dd, bb, c, t0, t3, t1, t2, p03, p12);
            }
            stack[bs] = u0; stack[bs + 1] = u1; stack[bs + 2] = u2; stack[bs + 3] = u3;
            top = base + 1;
        }
        double denom = 2.0 * (double)mi;
        frac[fo]     = (double)f1 / denom;
        frac[fo + 1] = (double)f2 / denom;
        frac[fo + 2] = (double)f3 / denom;
    }
}

// Sequential per-branch accumulation in gene order (bit-identical to the CPU loop).
__global__ void supportSumKernel(const double* __restrict__ frac, const unsigned char* __restrict__ eff,
                                 int numBranches, int numTrees, double* __restrict__ out)
{
    int branch = blockIdx.x * blockDim.x + threadIdx.x;
    if (branch >= numBranches) return;
    double w1 = 0.0, w2 = 0.0, w3 = 0.0;
    int effn = 0;
    const double* f = frac + (size_t)branch * numTrees * 3;
    const unsigned char* e = eff + (size_t)branch * numTrees;
    for (int g = 0; g < numTrees; g++) {
        if (!e[g]) continue;
        effn++;
        w1 += f[(size_t)g * 3];
        w2 += f[(size_t)g * 3 + 1];
        w3 += f[(size_t)g * 3 + 2];
    }
    out[(size_t)branch * 4]     = w1;
    out[(size_t)branch * 4 + 1] = w2;
    out[(size_t)branch * 4 + 2] = w3;
    out[(size_t)branch * 4 + 3] = (double)effn;
}

extern "C" {

JNIEXPORT jstring JNICALL
Java_astralx_gpu_GPUBranchSupport_queryStatus(JNIEnv* env, jclass)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return env->NewStringUTF("no CUDA device");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cc = prop.major * 10 + prop.minor;
    if (cc < ASTRALX_MIN_CUDA_CC) return env->NewStringUTF("GPU compute capability too old for this build");
    return env->NewStringUTF("ok");
}

/**
 * Returns double[numBranches * 4] = (f1, f2, f3, effn) per branch, or NULL when
 * the GPU path is not usable (the caller then uses the CPU path).
 */
JNIEXPORT jdoubleArray JNICALL
Java_astralx_gpu_GPUBranchSupport_computeSupportGPU(
    JNIEnv* env, jclass,
    jlongArray jClusterBits,    // numBranches * 4 * W
    jlongArray jGeneBits,       // numTrees * W
    jintArray  jTokens,
    jintArray  jOffsets,        // numTrees + 1
    jint numBranches, jint numTrees, jint W, jint maxFrontier, jint vramCapMiB)
{
    if (numBranches <= 0 || numTrees <= 0) return NULL;
    if (maxFrontier < 1 || maxFrontier > SUPPORT_MAX_STACK) {
        fprintf(stderr, "[ASTRAL-X GPU] branch support: stack frontier %d exceeds %d, using CPU\n",
                maxFrontier, SUPPORT_MAX_STACK);
        return NULL;
    }
    int cap = maxFrontier <= 32 ? 32 : maxFrontier <= 64 ? 64 : maxFrontier <= 128 ? 128
            : maxFrontier <= 256 ? 256 : SUPPORT_MAX_STACK;

    jlong* hCluster = env->GetLongArrayElements(jClusterBits, NULL);
    jlong* hGene    = env->GetLongArrayElements(jGeneBits, NULL);
    jint*  hTokens  = env->GetIntArrayElements(jTokens, NULL);
    jint*  hOffsets = env->GetIntArrayElements(jOffsets, NULL);
    size_t tokLen = (size_t) env->GetArrayLength(jTokens);

    // Resident data.
    ull *dGene = NULL; int *dTokens = NULL, *dOffsets = NULL;
    bool ok = cudaMalloc(&dGene, (size_t)numTrees * W * sizeof(ull)) == cudaSuccess
           && cudaMalloc(&dTokens, (tokLen ? tokLen : 1) * sizeof(int)) == cudaSuccess
           && cudaMalloc(&dOffsets, (size_t)(numTrees + 1) * sizeof(int)) == cudaSuccess;
    if (ok) {
        cudaMemcpy(dGene, hGene, (size_t)numTrees * W * sizeof(ull), cudaMemcpyHostToDevice);
        if (tokLen) cudaMemcpy(dTokens, hTokens, tokLen * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dOffsets, hOffsets, (size_t)(numTrees + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Batch over branches so that cluster bits + per-gene fractions fit the cap.
    size_t perBranch = (size_t)4 * W * sizeof(ull) + (size_t)numTrees * (3 * sizeof(double) + 1) + 4 * sizeof(double);
    size_t cap_bytes = (size_t)(vramCapMiB > 0 ? vramCapMiB : 512) * 1024ULL * 1024ULL;
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    if (freeB > 64ULL * 1024 * 1024 && cap_bytes > freeB - 64ULL * 1024 * 1024) cap_bytes = freeB - 64ULL * 1024 * 1024;
    int batch = (int)(cap_bytes / (perBranch ? perBranch : 1));
    if (batch < 1) batch = 1;
    if (batch > numBranches) batch = numBranches;

    // Thread count target: enough (branch, chunk) pairs to fill the device.
    int chunks = 1;
    {
        int target = 16384;
        if ((long long)batch < target) chunks = (int)((target + batch - 1) / batch);
        if (chunks > numTrees) chunks = numTrees;
        if (chunks < 1) chunks = 1;
    }

    ull* dCluster = NULL; double* dFrac = NULL; unsigned char* dEff = NULL; double* dOut = NULL;
    ok = ok && cudaMalloc(&dCluster, (size_t)batch * 4 * W * sizeof(ull)) == cudaSuccess
            && cudaMalloc(&dFrac, (size_t)batch * numTrees * 3 * sizeof(double)) == cudaSuccess
            && cudaMalloc(&dEff, (size_t)batch * numTrees) == cudaSuccess
            && cudaMalloc(&dOut, (size_t)batch * 4 * sizeof(double)) == cudaSuccess;

    jdoubleArray result = NULL;
    double* hOut = ok ? new double[(size_t)numBranches * 4] : NULL;
    fprintf(stderr, "[ASTRAL-X GPU] branch support: %d branch(es), %d gene trees, W=%d, "
            "stack cap %d, batch %d, %d gene chunk(s) per branch\n",
            numBranches, numTrees, W, cap, batch, chunks);

    for (int b0 = 0; ok && b0 < numBranches; b0 += batch) {
        int nb = (numBranches - b0 < batch) ? numBranches - b0 : batch;
        cudaMemcpy(dCluster, hCluster + (size_t)b0 * 4 * W, (size_t)nb * 4 * W * sizeof(ull), cudaMemcpyHostToDevice);
        long long threads = (long long)nb * chunks;
        int block = 128;
        int grid = (int)((threads + block - 1) / block);
        switch (cap) {
            case 32:  supportFractionKernel<32><<<grid, block>>>(dCluster, dGene, dTokens, dOffsets, nb, numTrees, W, chunks, dFrac, dEff); break;
            case 64:  supportFractionKernel<64><<<grid, block>>>(dCluster, dGene, dTokens, dOffsets, nb, numTrees, W, chunks, dFrac, dEff); break;
            case 128: supportFractionKernel<128><<<grid, block>>>(dCluster, dGene, dTokens, dOffsets, nb, numTrees, W, chunks, dFrac, dEff); break;
            case 256: supportFractionKernel<256><<<grid, block>>>(dCluster, dGene, dTokens, dOffsets, nb, numTrees, W, chunks, dFrac, dEff); break;
            default:  supportFractionKernel<SUPPORT_MAX_STACK><<<grid, block>>>(dCluster, dGene, dTokens, dOffsets, nb, numTrees, W, chunks, dFrac, dEff); break;
        }
        int grid2 = (nb + 127) / 128;
        supportSumKernel<<<grid2, 128>>>(dFrac, dEff, nb, numTrees, dOut);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ASTRAL-X GPU] branch support kernel failed: %s; using CPU\n", cudaGetErrorString(err));
            ok = false;
            break;
        }
        cudaMemcpy(hOut + (size_t)b0 * 4, dOut, (size_t)nb * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    }

    if (ok) {
        result = env->NewDoubleArray((jsize)((size_t)numBranches * 4));
        env->SetDoubleArrayRegion(result, 0, (jsize)((size_t)numBranches * 4), hOut);
    }
    delete[] hOut;
    cudaFree(dCluster); cudaFree(dFrac); cudaFree(dEff); cudaFree(dOut);
    cudaFree(dGene); cudaFree(dTokens); cudaFree(dOffsets);
    env->ReleaseLongArrayElements(jClusterBits, hCluster, JNI_ABORT);
    env->ReleaseLongArrayElements(jGeneBits, hGene, JNI_ABORT);
    env->ReleaseIntArrayElements(jTokens, hTokens, JNI_ABORT);
    env->ReleaseIntArrayElements(jOffsets, hOffsets, JNI_ABORT);
    return result;
}

} // extern "C"
