/**
 * ASTRAL-X GPU weight calculation kernel (CUDA + JNI).
 *
 * One CUDA thread per candidate split.  Each thread iterates over every
 * gene-tree tripartition and accumulates 2*QI weighted by tripartition
 * frequency.  The result array (twoScores) is divided by 2 on the Java side.
 *
 * Data layout (mirrors STELAR-X compact pattern):
 *   orderings[t * numTaxa + pos]   = taxon id at postorder position pos in tree t
 *   invIndex [t * numTaxa + taxon] = postorder position of taxon in tree t (-1 if absent)
 *
 * Split layout (10 ints per split):
 *   [0] loTreeIdx  [1] loLeft  [2] loRight  [3] loComplement  [4] loSize
 *   [5] hiTreeIdx  [6] hiLeft  [7] hiRight  [8] hiComplement  [9] hiSize
 *
 * Partition layout (9 ints per partition):
 *   [0] treeIdx  [1] lo1  [2] hi1  [3] lo2  [4] hi2
 *   [5] sz1  [6] sz2  [7] sz3  [8] frequency
 *
 * Batching:
 *   Static data (orderings, invIndex, parts) is uploaded to the device ONCE.
 *   Splits are processed in adaptive batches whose size is derived from free
 *   VRAM after the static upload.  This bounds peak VRAM at:
 *
 *     static:  numTrees×numTaxa×8B + numParts×36B
 *     dynamic: batchSize × 48B  (10 ints split-in + 1 long long score-out)
 *
 *   batchSizeHint semantics (passed from Java):
 *      0  — auto: query cudaMemGetInfo, use 75% of remaining free VRAM
 *     -1  — no batching: single launch with all splits (original behaviour)
 *     >0  — manual override: use exactly this value as batchSize
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <jni.h>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/**
 * Range intersection: count taxa in [loA,hiA) of tree tA that also appear
 * in [loB,hiB) of tree tB.  Iterates the smaller range for efficiency.
 */
__device__ int coreIntersect(
    int tA, int loA, int hiA,
    int tB, int loB, int hiB,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int numTaxa)
{
    int szA = hiA - loA, szB = hiB - loB;
    int count = 0;
    if (szA <= szB) {
        for (int pos = loA; pos < hiA; pos++) {
            int taxon = orderings[tA * numTaxa + pos];
            int posB  = invIndex [tB * numTaxa + taxon];
            if (posB >= loB && posB < hiB) count++;
        }
    } else {
        for (int pos = loB; pos < hiB; pos++) {
            int taxon = orderings[tB * numTaxa + pos];
            int posA  = invIndex [tA * numTaxa + taxon];
            if (posA >= loA && posA < hiA) count++;
        }
    }
    return count;
}

/**
 * Intersection with optional complement.  If cComp==1, actual set is the
 * complement of [loC, hiC) within tree tC, so
 *   |comp(C) ∩ M| = |M| - |C ∩ M|
 */
__device__ int intersect(
    int tGT, int loGT, int hiGT,
    int tC,  int loC,  int hiC, int cComp, int szGTRange,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int numTaxa)
{
    int raw = coreIntersect(tGT, loGT, hiGT, tC, loC, hiC, orderings, invIndex, numTaxa);
    return cComp ? (szGTRange - raw) : raw;
}

// ---------------------------------------------------------------------------
// Main kernel: 1 thread per split in current batch
// ---------------------------------------------------------------------------

__global__ void computeWeightsKernel(
    const int* __restrict__ splits,    // curBatch * 10
    const int* __restrict__ parts,     // numParts  * 9
    const int* __restrict__ orderings, // numTrees  * numTaxa
    const int* __restrict__ invIndex,  // numTrees  * numTaxa
    int curBatch,
    int numParts,
    int numTaxa,
    int totalN,
    long long* __restrict__ twoScores  // output: curBatch entries
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curBatch) return;

    // Load split
    const int* sp = splits + idx * 10;
    int loTree = sp[0], loLeft = sp[1], loRight = sp[2], loComp = sp[3], sizeA = sp[4];
    int hiTree = sp[5], hiLeft = sp[6], hiRight = sp[7], hiComp = sp[8], sizeB = sp[9];

    int sizeC = totalN - sizeA - sizeB;
    if (sizeC < 0) { twoScores[idx] = 0LL; return; }

    // 6 permutations: (pi,pj,pk) index into arrays a[], b[], c[]
    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    long long twoScore = 0LL;

    for (int j = 0; j < numParts; j++) {
        const int* pt = parts + j * 9;
        int tGT = pt[0];
        int lo1 = pt[1], hi1 = pt[2];
        int lo2 = pt[3], hi2 = pt[4];
        int sz1 = pt[5], sz2 = pt[6], sz3 = pt[7];
        int freq = pt[8];

        // 4 core intersections
        int a0 = intersect(tGT, lo1, hi1, loTree, loLeft, loRight, loComp, sz1, orderings, invIndex, numTaxa);
        int a1 = intersect(tGT, lo2, hi2, loTree, loLeft, loRight, loComp, sz2, orderings, invIndex, numTaxa);
        int b0 = intersect(tGT, lo1, hi1, hiTree, hiLeft, hiRight, hiComp, sz1, orderings, invIndex, numTaxa);
        int b1 = intersect(tGT, lo2, hi2, hiTree, hiLeft, hiRight, hiComp, sz2, orderings, invIndex, numTaxa);

        // Row sums: for incomplete gene trees L_GT < totalN, so |A∩Lg_GT| != sizeA
        int L_GT = sz1 + sz2 + sz3;
        int lgA, lgB;
        if (L_GT == totalN) {
            lgA = sizeA;
            lgB = sizeB;
        } else {
            int coreA = coreIntersect(tGT, 0, L_GT, loTree, loLeft, loRight, orderings, invIndex, numTaxa);
            lgA = loComp ? (L_GT - coreA) : coreA;
            int coreB = coreIntersect(tGT, 0, L_GT, hiTree, hiLeft, hiRight, orderings, invIndex, numTaxa);
            lgB = hiComp ? (L_GT - coreB) : coreB;
        }

        // Derive remaining 5 (c2 uses column constraint on M3, not row C)
        int a2 = lgA  - a0 - a1;
        int b2 = lgB  - b0 - b1;
        int c0 = sz1  - a0 - b0;
        int c1 = sz2  - a1 - b1;
        int c2 = sz3  - a2 - b2;

        if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

        // 2*QI = sum over 6 perms (i,j,k): a[i]*b[j]*c[k]*(a[i]+b[j]+c[k]-3)
        long long a[3] = {a0, a1, a2};
        long long b[3] = {b0, b1, b2};
        long long c[3] = {c0, c1, c2};

        long long twoQI = 0LL;
        for (int p = 0; p < 6; p++) {
            long long ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
            long long s  = ai + bj + ck - 3;
            if (s > 0) twoQI += ai * bj * ck * s;
        }
        twoScore += (long long)freq * twoQI;
    }

    twoScores[idx] = twoScore;
}

// ---------------------------------------------------------------------------
// JNI entry point
// ---------------------------------------------------------------------------

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_astralx_gpu_GPUWeightCalculator_computeWeightsGPU(
    JNIEnv* env, jclass cls,
    jintArray jSplits, jintArray jParts,
    jintArray jOrderings, jintArray jInvIndex,
    jint numSplits, jint numParts,
    jint numTrees, jint numTaxa, jint totalN,
    jint batchSizeHint)
{
    // -------------------------------------------------------------------------
    // Pin host arrays
    // -------------------------------------------------------------------------
    jint* hSplits    = env->GetIntArrayElements(jSplits,    NULL);
    jint* hParts     = env->GetIntArrayElements(jParts,     NULL);
    jint* hOrderings = env->GetIntArrayElements(jOrderings, NULL);
    jint* hInvIndex  = env->GetIntArrayElements(jInvIndex,  NULL);

    // -------------------------------------------------------------------------
    // Upload static data ONCE (orderings, invIndex, parts stay resident)
    // -------------------------------------------------------------------------
    int      *dParts, *dOrderings, *dInvIndex;

    size_t partsSz    = (size_t)numParts  *  9 * sizeof(int);
    size_t orderingSz = (size_t)numTrees  * numTaxa * sizeof(int);

    cudaMalloc(&dParts,     partsSz);
    cudaMalloc(&dOrderings, orderingSz);
    cudaMalloc(&dInvIndex,  orderingSz);

    cudaMemcpy(dParts,     hParts,     partsSz,    cudaMemcpyHostToDevice);
    cudaMemcpy(dOrderings, hOrderings, orderingSz, cudaMemcpyHostToDevice);
    cudaMemcpy(dInvIndex,  hInvIndex,  orderingSz, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // Determine batch size
    //   batchSizeHint == -1  → no batching (single launch, original behaviour)
    //   batchSizeHint ==  0  → auto: fill 75% of remaining free VRAM
    //   batchSizeHint >   0  → manual override
    // -------------------------------------------------------------------------
    int batchSize;

    if (batchSizeHint == -1) {
        // No batching: process all splits in one launch
        batchSize = numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] batching disabled — single launch, %d splits\n",
                numSplits);
    } else if (batchSizeHint > 0) {
        // Manual override
        batchSize = (batchSizeHint < numSplits) ? batchSizeHint : numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] manual batch size: %d  (numSplits=%d)\n",
                batchSize, numSplits);
    } else {
        // Auto: query free VRAM after static upload
        size_t freeVRAM = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeVRAM, &totalVRAM);
        // Reserve 25% headroom for driver, kernel stack, page tables
        size_t usable = (size_t)((double)freeVRAM * 0.75);
        // 48 bytes per split: 10 ints (40 B split data) + 8 B score
        size_t perSplitBytes = 10 * sizeof(int) + sizeof(long long);
        long long autoSize = (long long)(usable / perSplitBytes);
        if (autoSize < 1) autoSize = 1;
        if (autoSize > (long long)numSplits) autoSize = (long long)numSplits;
        batchSize = (int)autoSize;
        fprintf(stderr,
            "[ASTRAL-X GPU] adaptive batch: freeVRAM=%.2f GB, usable=%.2f GB, "
            "perSplit=%zu B → batchSize=%d  (numSplits=%d, numBatches=%d)\n",
            freeVRAM / 1e9, usable / 1e9, perSplitBytes,
            batchSize, numSplits,
            (numSplits + batchSize - 1) / batchSize);
    }

    // -------------------------------------------------------------------------
    // Allocate batch-local device buffers (with halving fallback on OOM)
    // -------------------------------------------------------------------------
    int*       dSplits    = NULL;
    long long* dTwoScores = NULL;

    while (batchSize > 0) {
        size_t splitBufSz = (size_t)batchSize * 10 * sizeof(int);
        size_t scoreBufSz = (size_t)batchSize * sizeof(long long);
        cudaError_t e1 = cudaMalloc(&dSplits,    splitBufSz);
        cudaError_t e2 = cudaMalloc(&dTwoScores, scoreBufSz);
        if (e1 == cudaSuccess && e2 == cudaSuccess) break;
        // Partial allocation: free any that succeeded before retrying
        if (dSplits)    { cudaFree(dSplits);    dSplits    = NULL; }
        if (dTwoScores) { cudaFree(dTwoScores); dTwoScores = NULL; }
        batchSize /= 2;
        fprintf(stderr, "[ASTRAL-X GPU] cudaMalloc failed, retrying with batchSize=%d\n",
                batchSize);
    }
    if (batchSize <= 0 || dSplits == NULL || dTwoScores == NULL) {
        fprintf(stderr, "[ASTRAL-X GPU] FATAL: cannot allocate GPU batch buffers\n");
        cudaFree(dParts); cudaFree(dOrderings); cudaFree(dInvIndex);
        env->ReleaseIntArrayElements(jSplits,    hSplits,    JNI_ABORT);
        env->ReleaseIntArrayElements(jParts,     hParts,     JNI_ABORT);
        env->ReleaseIntArrayElements(jOrderings, hOrderings, JNI_ABORT);
        env->ReleaseIntArrayElements(jInvIndex,  hInvIndex,  JNI_ABORT);
        return NULL;
    }

    // -------------------------------------------------------------------------
    // Host result buffer — accumulates scores across all batches
    // -------------------------------------------------------------------------
    long long* hTwoScores = new long long[numSplits]();   // zero-initialised

    // -------------------------------------------------------------------------
    // Batch loop: stream splits in, stream scores out
    // -------------------------------------------------------------------------
    int blockSize  = 256;
    int numBatches = (numSplits + batchSize - 1) / batchSize;

    for (int b = 0; b < numBatches; b++) {
        int offset   = b * batchSize;
        int curBatch = (offset + batchSize <= numSplits) ? batchSize : (numSplits - offset);

        // Upload this batch's splits
        cudaMemcpy(dSplits,
                   hSplits + (size_t)offset * 10,
                   (size_t)curBatch * 10 * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Launch kernel for this batch
        int gridSize = (curBatch + blockSize - 1) / blockSize;
        computeWeightsKernel<<<gridSize, blockSize>>>(
            dSplits, dParts, dOrderings, dInvIndex,
            curBatch, numParts, numTaxa, totalN,
            dTwoScores);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ASTRAL-X GPU] kernel error (batch %d/%d): %s\n",
                    b + 1, numBatches, cudaGetErrorString(err));
        }

        // Copy this batch's scores back to host
        cudaMemcpy(hTwoScores + offset,
                   dTwoScores,
                   (size_t)curBatch * sizeof(long long),
                   cudaMemcpyDeviceToHost);

        // Progress (only shown when there are multiple batches)
        if (numBatches > 1) {
            fprintf(stderr,
                "\r[ASTRAL-X GPU] weight batch %d/%d  (splits %d–%d, %.1f%%)",
                b + 1, numBatches,
                offset, offset + curBatch - 1,
                100.0 * (b + 1) / numBatches);
            fflush(stderr);
        }
    }
    if (numBatches > 1) {
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    // -------------------------------------------------------------------------
    // Build Java long[] result
    // -------------------------------------------------------------------------
    jlongArray result = env->NewLongArray(numSplits);
    env->SetLongArrayRegion(result, 0, numSplits, (jlong*)hTwoScores);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    delete[] hTwoScores;
    cudaFree(dSplits);
    cudaFree(dTwoScores);
    cudaFree(dParts);
    cudaFree(dOrderings);
    cudaFree(dInvIndex);

    env->ReleaseIntArrayElements(jSplits,    hSplits,    JNI_ABORT);
    env->ReleaseIntArrayElements(jParts,     hParts,     JNI_ABORT);
    env->ReleaseIntArrayElements(jOrderings, hOrderings, JNI_ABORT);
    env->ReleaseIntArrayElements(jInvIndex,  hInvIndex,  JNI_ABORT);

    return result;
}

} // extern "C"
