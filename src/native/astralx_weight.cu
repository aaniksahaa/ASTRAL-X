/**
 * ASTRAL-X GPU weight calculation kernel (CUDA + JNI).
 *
 * PREFIX-SUM TREE-DP FORMULATION
 * ------------------------------
 * One CUDA *thread block* per candidate split.  The block loops over every gene
 * tree on-device.  For each tree it builds, in shared memory, two prefix-sum
 * arrays over the tree's leaf postorder array — one for each side (A, B) of the
 * candidate split:
 *
 *     prefixA[p] = number of the first p leaves (in this tree's postorder) that
 *                  belong to cluster A
 *
 * A gene-tree tripartition is a contiguous postorder leaf interval [lo,hi) split
 * at mid, so every core intersection becomes an O(1) prefix difference:
 *
 *     |M1 ∩ A| = prefixA[mid] - prefixA[lo]
 *     |M2 ∩ A| = prefixA[hi]  - prefixA[mid]
 *     |Lg ∩ A| = prefixA[L]                      (row sum; free for incomplete trees)
 *
 * The remaining 5 entries of the 3×3 matrix are derived by the row/column
 * constraints (see DESIGN/intersection-optimization.md), then 2*QI is summed
 * over all internal nodes weighted by 1 (every node contributes once — no
 * tripartition dedup; identical trees may carry a multiplicity, see host code).
 *
 * This replaces the old element-by-element coreIntersect() walk: per (split,
 * tree) cost is now exactly O(L) regardless of tree balance, with no scattered
 * membership probes inside the hot loop.
 *
 * Membership test (leaf taxon t ∈ cluster A):
 *     posA = invIndex[aTree*numTaxa + t]
 *     inA  = (posA in [aLo, aHi)) XOR aComp
 * Cluster exemplar trees are *completed* (full taxon set), so invIndex is always
 * valid — no missing-taxon special case in the membership test.
 *
 * Data layout:
 *   orderings[t*numTaxa + pos]   = taxon id at postorder leaf position pos in tree t
 *   invIndex [t*numTaxa + taxon] = postorder position of taxon in tree t (-1 if absent)
 *
 * Split layout (10 ints per split):
 *   [0] aTree  [1] aLo  [2] aHi  [3] aComp  [4] aSize
 *   [5] bTree  [6] bLo  [7] bHi  [8] bComp  [9] bSize
 *
 * Per-tree node CSR (static):
 *   nodeOffset[g] .. nodeOffset[g+1]  index into nodeData for tree g's internal nodes
 *   nodeData[3*ni + {0,1,2}] = (lo, mid, hi)  leaf-interval of internal node ni
 *   partLeafCount[g] = L (leaf count of gene tree g)
 *
 * Batching:
 *   Static data (orderings, invIndex, nodeData, nodeOffset, partLeafCount) is
 *   uploaded ONCE.  Splits are processed in adaptive batches; per-split device
 *   memory is 40 B in + 8 B out (unchanged from the old kernel), so the existing
 *   VRAM-budget logic carries over verbatim.
 *
 *   batchSizeHint semantics (passed from Java):
 *      0  — auto: query cudaMemGetInfo, use vramFraction of remaining free VRAM
 *     -1  — no batching: single launch with all splits
 *     >0  — manual override: use exactly this value as batchSize
 */

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <jni.h>

// Fixed block size.  Must match the static reduction buffer below and the
// dynamic shared-memory scan area sized on the host.
#define WB_BLOCK 256

// ---------------------------------------------------------------------------
// Device helper: cooperative prefix-sum of cluster membership over a tree's
// leaves, written into pX[0..L].  Uses scan[] (WB_BLOCK ints) as scratch.
//
//   pX[p] = number of leaves among the first p (postorder) that are in the
//           cluster (clLo,clHi,clComp) of tree clBase.
//
// All threads of the block must call this uniformly (it issues __syncthreads).
// ---------------------------------------------------------------------------
__device__ void buildPrefix(
    int* __restrict__ pX, int* __restrict__ scan, int L,
    size_t gBase, size_t clBase, int clLo, int clHi, int clComp,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int tid, int nthreads)
{
    if (L <= 0) {
        if (tid == 0) pX[0] = 0;
        __syncthreads();
        return;
    }

    int chunk = (L + nthreads - 1) / nthreads;   // ceil
    int start = tid * chunk;
    int end   = start + chunk;
    if (start > L) start = L;
    if (end   > L) end   = L;

    // Pass A: write indicator into pX[start..end), accumulate this chunk's sum.
    int sum = 0;
    for (int p = start; p < end; p++) {
        int t   = orderings[gBase + (size_t)p];
        int pos = invIndex[clBase + (size_t)t];
        int in  = (pos >= clLo && pos < clHi) ? 1 : 0;
        in     ^= clComp;          // clComp is 0/1
        pX[p]   = in;
        sum    += in;
    }
    scan[tid] = sum;
    __syncthreads();

    // Inclusive scan of chunk sums (Hillis-Steele over WB_BLOCK elements).
    for (int off = 1; off < nthreads; off <<= 1) {
        int v = (tid >= off) ? scan[tid - off] : 0;
        __syncthreads();
        if (tid >= off) scan[tid] += v;
        __syncthreads();
    }
    int excl = scan[tid] - sum;   // exclusive offset = sum of all previous chunks

    // Pass B: convert per-chunk indicators to global prefix (value BEFORE p).
    int acc = excl;
    for (int p = start; p < end; p++) {
        int v = pX[p];
        pX[p] = acc;
        acc  += v;
    }
    if (start < L && end == L) pX[L] = acc;   // total row sum at the very end

    __syncthreads();
}

// ---------------------------------------------------------------------------
// Main kernel: one block per split in the current batch.
// ---------------------------------------------------------------------------
__global__ void computeWeightsKernel(
    const int* __restrict__ splits,        // curBatch * 10
    const int* __restrict__ nodeData,      // totalNodes * 3
    const int* __restrict__ nodeOffset,    // numPartTrees + 1
    const int* __restrict__ partLeafCount, // numPartTrees
    const int* __restrict__ orderings,     // numGpuTrees * numTaxa
    const int* __restrict__ invIndex,      // numGpuTrees * numTaxa
    int curBatch,
    int numPartTrees,
    int partTreeOffset,
    int maxLeafCount,
    int numTaxa,
    int totalN,
    long long* __restrict__ twoScores      // output: curBatch entries
)
{
    int s = blockIdx.x;
    if (s >= curBatch) return;

    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    // Shared layout: pA[maxLeafCount+1], pB[maxLeafCount+1], scan[WB_BLOCK]
    extern __shared__ int smem[];
    int stride = maxLeafCount + 1;
    int* pA   = smem;
    int* pB   = smem + stride;
    int* scan = smem + 2 * stride;

    __shared__ long long red[WB_BLOCK];

    // Load split.
    const int* sp = splits + (size_t)s * 10;
    int aTree = sp[0], aLo = sp[1], aHi = sp[2], aComp = sp[3], aSize = sp[4];
    int bTree = sp[5], bLo = sp[6], bHi = sp[7], bComp = sp[8], bSize = sp[9];

    // Invalid / overlapping split → zero (defensive; real DP splits are disjoint).
    if (aSize + bSize > totalN) {
        if (tid == 0) twoScores[s] = 0LL;
        return;
    }

    size_t aBase = (size_t)aTree * numTaxa;
    size_t bBase = (size_t)bTree * numTaxa;

    // 6 permutations for 2*QI.
    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    long long threadAccum = 0LL;

    for (int g = 0; g < numPartTrees; g++) {
        int    L     = partLeafCount[g];
        size_t gBase = (size_t)(partTreeOffset + g) * numTaxa;

        buildPrefix(pA, scan, L, gBase, aBase, aLo, aHi, aComp, orderings, invIndex, tid, nthreads);
        buildPrefix(pB, scan, L, gBase, bBase, bLo, bHi, bComp, orderings, invIndex, tid, nthreads);

        int lgA = pA[L];
        int lgB = pB[L];

        int nbeg = nodeOffset[g];
        int nend = nodeOffset[g + 1];
        for (int ni = nbeg + tid; ni < nend; ni += nthreads) {
            size_t nb = (size_t)ni * 3;
            int lo  = nodeData[nb];
            int mid = nodeData[nb + 1];
            int hi  = nodeData[nb + 2];

            int a0 = pA[mid] - pA[lo];
            int a1 = pA[hi]  - pA[mid];
            int b0 = pB[mid] - pB[lo];
            int b1 = pB[hi]  - pB[mid];

            int sz1 = mid - lo;
            int sz2 = hi  - mid;
            int sz3 = L   - (hi - lo);

            int a2 = lgA - a0 - a1;
            int b2 = lgB - b0 - b1;
            int c0 = sz1 - a0 - b0;
            int c1 = sz2 - a1 - b1;
            int c2 = sz3 - a2 - b2;

            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

            long long a[3] = {a0, a1, a2};
            long long b[3] = {b0, b1, b2};
            long long c[3] = {c0, c1, c2};

            long long twoQI = 0LL;
            #pragma unroll
            for (int p = 0; p < 6; p++) {
                long long ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
                long long su = ai + bj + ck - 3;
                if (su > 0) twoQI += ai * bj * ck * su;
            }
            threadAccum += twoQI;
        }

        __syncthreads();   // pA/pB reused next iteration; ensure node loop done
    }

    // Block reduction of threadAccum → twoScores[s].
    red[tid] = threadAccum;
    __syncthreads();
    for (int off = nthreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) twoScores[s] = red[0];
}

// ---------------------------------------------------------------------------
// Progress-bar helpers (host-side, used in the batch loop)
// ---------------------------------------------------------------------------

static double wb_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Format a duration in seconds as "4s", "1m23s", "2h05m"
static void wb_fmt_duration(double secs, char* buf, int buflen) {
    int s = (int)secs;
    if (s < 60)
        snprintf(buf, buflen, "%ds", s);
    else if (s < 3600)
        snprintf(buf, buflen, "%dm%02ds", s / 60, s % 60);
    else
        snprintf(buf, buflen, "%dh%02dm", s / 3600, (s % 3600) / 60);
}

static int wb_use_color(void) {
    if (getenv("NO_COLOR"))    return 0;
    if (getenv("FORCE_COLOR")) return 1;
    return 0;
}

#define WB_BAR_W 28
static void wb_build_bar(char* buf, int done, int total) {
    int filled = (total > 0) ? (int)((double)done / total * WB_BAR_W + 0.5) : 0;
    if (filled > WB_BAR_W) filled = WB_BAR_W;
    int pos = 0;
    for (int i = 0; i < WB_BAR_W; i++) {
        if (i < filled) {
            buf[pos++] = '\xe2'; buf[pos++] = '\x96'; buf[pos++] = '\x88'; // █
        } else {
            buf[pos++] = '\xe2'; buf[pos++] = '\x96'; buf[pos++] = '\x91'; // ░
        }
    }
    buf[pos] = '\0';
}

// ---------------------------------------------------------------------------

extern "C" {

// ---------------------------------------------------------------------------
// queryVRAMMiB: lightweight VRAM probe for Java-side phase logging
// ---------------------------------------------------------------------------
JNIEXPORT jlongArray JNICALL
Java_astralx_gpu_GPUWeightCalculator_queryVRAMMiB(JNIEnv* env, jclass cls)
{
    size_t freeBytes = 0, totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) return NULL;
    jlong data[2] = {
        (jlong)(freeBytes  / (1024ULL * 1024ULL)),
        (jlong)(totalBytes / (1024ULL * 1024ULL))
    };
    jlongArray result = env->NewLongArray(2);
    if (!result) return NULL;
    env->SetLongArrayRegion(result, 0, 2, data);
    return result;
}

JNIEXPORT jlongArray JNICALL
Java_astralx_gpu_GPUWeightCalculator_computeWeightsGPU(
    JNIEnv* env, jclass cls,
    jintArray jSplits, jintArray jNodeData, jintArray jNodeOffset,
    jintArray jPartLeafCount, jintArray jOrderings, jintArray jInvIndex,
    jint numSplits, jint numPartTrees, jint partTreeOffset, jint maxLeafCount,
    jint numGpuTrees, jint numTaxa,
    jint batchSizeHint, jdouble vramFraction)
{
    // -------------------------------------------------------------------------
    // Pin host arrays
    // -------------------------------------------------------------------------
    jint* hSplits        = env->GetIntArrayElements(jSplits,        NULL);
    jint* hNodeData      = env->GetIntArrayElements(jNodeData,      NULL);
    jint* hNodeOffset    = env->GetIntArrayElements(jNodeOffset,    NULL);
    jint* hPartLeafCount = env->GetIntArrayElements(jPartLeafCount, NULL);
    jint* hOrderings     = env->GetIntArrayElements(jOrderings,     NULL);
    jint* hInvIndex      = env->GetIntArrayElements(jInvIndex,      NULL);

    jsize nodeDataLen = env->GetArrayLength(jNodeData);   // = totalNodes * 3

    // -------------------------------------------------------------------------
    // Shared-memory feasibility check (opt-in beyond 48 KB if the device allows)
    // -------------------------------------------------------------------------
    size_t sharedBytes = ((size_t)2 * (maxLeafCount + 1) + WB_BLOCK) * sizeof(int);
    {
        int maxOptin = 0;
        cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if ((int)sharedBytes > maxOptin) {
            fprintf(stderr,
                "[ASTRAL-X GPU] weight: required shared memory %.1f KB exceeds device "
                "opt-in limit %.1f KB (maxLeafCount=%d) — falling back to CPU.\n",
                sharedBytes / 1024.0, maxOptin / 1024.0, (int)maxLeafCount);
            env->ReleaseIntArrayElements(jSplits,        hSplits,        JNI_ABORT);
            env->ReleaseIntArrayElements(jNodeData,      hNodeData,      JNI_ABORT);
            env->ReleaseIntArrayElements(jNodeOffset,    hNodeOffset,    JNI_ABORT);
            env->ReleaseIntArrayElements(jPartLeafCount, hPartLeafCount, JNI_ABORT);
            env->ReleaseIntArrayElements(jOrderings,     hOrderings,     JNI_ABORT);
            env->ReleaseIntArrayElements(jInvIndex,      hInvIndex,      JNI_ABORT);
            return NULL;   // Java falls back to the CPU path
        }
        // Opt in to larger dynamic shared memory if needed (default cap is 48 KB).
        if (sharedBytes > 49152) {
            cudaFuncSetAttribute(computeWeightsKernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sharedBytes);
        }
    }

    // -------------------------------------------------------------------------
    // Upload static data ONCE (orderings, invIndex, node CSR stay resident)
    // -------------------------------------------------------------------------
    int *dNodeData, *dNodeOffset, *dPartLeafCount, *dOrderings, *dInvIndex;

    size_t nodeDataSz   = (size_t)nodeDataLen          * sizeof(int);
    size_t nodeOffsetSz = (size_t)(numPartTrees + 1)   * sizeof(int);
    size_t partLeafSz   = (size_t)numPartTrees         * sizeof(int);
    size_t orderingSz   = (size_t)numGpuTrees * numTaxa * sizeof(int);

    cudaMalloc(&dNodeData,      nodeDataSz);
    cudaMalloc(&dNodeOffset,    nodeOffsetSz);
    cudaMalloc(&dPartLeafCount, partLeafSz);
    cudaMalloc(&dOrderings,     orderingSz);
    cudaMalloc(&dInvIndex,      orderingSz);

    cudaMemcpy(dNodeData,      hNodeData,      nodeDataSz,   cudaMemcpyHostToDevice);
    cudaMemcpy(dNodeOffset,    hNodeOffset,    nodeOffsetSz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPartLeafCount, hPartLeafCount, partLeafSz,   cudaMemcpyHostToDevice);
    cudaMemcpy(dOrderings,     hOrderings,     orderingSz,   cudaMemcpyHostToDevice);
    cudaMemcpy(dInvIndex,      hInvIndex,      orderingSz,   cudaMemcpyHostToDevice);

    // Analytical VRAM budget: show exactly what is resident on-device
    {
        size_t staticTotal = nodeDataSz + nodeOffsetSz + partLeafSz + 2 * orderingSz;
        size_t freeAfterStatic = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeAfterStatic, &totalVRAM);
        fprintf(stderr,
            "[ASTRAL-X GPU] weight static data uploaded (prefix-sum tree-DP):\n"
            "  orderings   : %6.1f MB\n"
            "  invIndex    : %6.1f MB\n"
            "  nodeData    : %6.1f MB  (%d internal nodes × 3 ints)\n"
            "  nodeOffset  : %6.1f MB\n"
            "  shared/block: %6.1f KB  (maxLeafCount=%d)\n"
            "  ─────────────────────\n"
            "  static total : %6.1f MB   (VRAM free after: %.1f MB / %.1f MB)\n",
            orderingSz / 1e6, orderingSz / 1e6,
            nodeDataSz / 1e6, nodeDataLen / 3,
            nodeOffsetSz / 1e6,
            sharedBytes / 1024.0, (int)maxLeafCount,
            staticTotal / 1e6, freeAfterStatic / 1e6, totalVRAM / 1e6);
        fflush(stderr);
    }

    // -------------------------------------------------------------------------
    // Determine batch size (per-split footprint unchanged: 40 B in + 8 B out)
    // -------------------------------------------------------------------------
    int batchSize;

    if (batchSizeHint == -1) {
        batchSize = numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] batching disabled — single launch, %d splits\n",
                numSplits);
    } else if (batchSizeHint > 0) {
        batchSize = (batchSizeHint < numSplits) ? batchSizeHint : numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] manual batch size: %d  (numSplits=%d)\n",
                batchSize, numSplits);
    } else {
        size_t freeVRAM = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeVRAM, &totalVRAM);
        size_t usable = (size_t)((double)freeVRAM * (double)vramFraction);
        size_t perSplitBytes = 10 * sizeof(int) + sizeof(long long);
        long long autoSize = (long long)(usable / perSplitBytes);
        if (autoSize < 1) autoSize = 1;
        if (autoSize > (long long)numSplits) autoSize = (long long)numSplits;
        batchSize = (int)autoSize;
        fprintf(stderr,
            "[ASTRAL-X GPU] adaptive batch: freeVRAM=%.2f GB, occupancy=%.0f%%, "
            "usable=%.2f GB, perSplit=%zu B → batchSize=%d  (numSplits=%d, numBatches=%d)\n",
            freeVRAM / 1e9, (double)vramFraction * 100.0, usable / 1e9,
            perSplitBytes, batchSize, numSplits,
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
        if (dSplits)    { cudaFree(dSplits);    dSplits    = NULL; }
        if (dTwoScores) { cudaFree(dTwoScores); dTwoScores = NULL; }
        batchSize /= 2;
        fprintf(stderr, "[ASTRAL-X GPU] cudaMalloc failed, retrying with batchSize=%d\n",
                batchSize);
    }
    if (batchSize <= 0 || dSplits == NULL || dTwoScores == NULL) {
        fprintf(stderr, "[ASTRAL-X GPU] FATAL: cannot allocate GPU batch buffers\n");
        cudaFree(dNodeData); cudaFree(dNodeOffset); cudaFree(dPartLeafCount);
        cudaFree(dOrderings); cudaFree(dInvIndex);
        env->ReleaseIntArrayElements(jSplits,        hSplits,        JNI_ABORT);
        env->ReleaseIntArrayElements(jNodeData,      hNodeData,      JNI_ABORT);
        env->ReleaseIntArrayElements(jNodeOffset,    hNodeOffset,    JNI_ABORT);
        env->ReleaseIntArrayElements(jPartLeafCount, hPartLeafCount, JNI_ABORT);
        env->ReleaseIntArrayElements(jOrderings,     hOrderings,     JNI_ABORT);
        env->ReleaseIntArrayElements(jInvIndex,      hInvIndex,      JNI_ABORT);
        return NULL;
    }

    {
        int numBatchesPlan = (numSplits + batchSize - 1) / batchSize;
        size_t splitBufMB = (size_t)batchSize * 10 * sizeof(int);
        size_t scoreBufMB = (size_t)batchSize * sizeof(long long);
        fprintf(stderr,
            "[ASTRAL-X GPU] weight batch buffers:\n"
            "  splits buf  : %6.1f MB  (%d splits × 40 B)\n"
            "  scores buf  : %6.1f MB  (%d splits × 8 B)\n"
            "  batches     : %d  (batchSize=%d, numSplits=%d)\n",
            splitBufMB / 1e6, batchSize,
            scoreBufMB / 1e6, batchSize,
            numBatchesPlan, batchSize, numSplits);
        fflush(stderr);
    }

    // -------------------------------------------------------------------------
    // Host result buffer — accumulates scores across all batches
    // -------------------------------------------------------------------------
    long long* hTwoScores = new long long[numSplits]();   // zero-initialised

    // -------------------------------------------------------------------------
    // Batch loop: stream splits in, stream scores out
    // -------------------------------------------------------------------------
    int    numBatches = (numSplits + batchSize - 1) / batchSize;
    double t_loop_start = wb_now_sec();
    const char* GRN = wb_use_color() ? "\033[32m" : "";
    const char* RST = wb_use_color() ? "\033[0m"  : "";
    char   bar_buf[WB_BAR_W * 3 + 1];

    for (int b = 0; b < numBatches; b++) {
        int offset   = b * batchSize;
        int curBatch = (offset + batchSize <= numSplits) ? batchSize : (numSplits - offset);

        cudaMemcpy(dSplits,
                   hSplits + (size_t)offset * 10,
                   (size_t)curBatch * 10 * sizeof(int),
                   cudaMemcpyHostToDevice);

        // One block per split; WB_BLOCK threads cooperate per split.
        computeWeightsKernel<<<curBatch, WB_BLOCK, sharedBytes>>>(
            dSplits, dNodeData, dNodeOffset, dPartLeafCount, dOrderings, dInvIndex,
            curBatch, numPartTrees, partTreeOffset, maxLeafCount, numTaxa, numTaxa,
            dTwoScores);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ASTRAL-X GPU] kernel error (batch %d/%d): %s\n",
                    b + 1, numBatches, cudaGetErrorString(err));
        }

        cudaMemcpy(hTwoScores + offset,
                   dTwoScores,
                   (size_t)curBatch * sizeof(long long),
                   cudaMemcpyDeviceToHost);

        // ── tqdm-style progress bar (multi-batch only) ────────────────────
        if (numBatches > 1) {
            double elapsed  = wb_now_sec() - t_loop_start;
            double avg_sec  = elapsed / (b + 1);
            int    rem      = numBatches - (b + 1);
            double pct      = 100.0 * (b + 1) / numBatches;
            wb_build_bar(bar_buf, b + 1, numBatches);

            if (rem == 0) {
                char dur_buf[32];
                wb_fmt_duration(elapsed, dur_buf, sizeof(dur_buf));
                fprintf(stderr,
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  100%%  done in %s"
                    "                    \n",
                    GRN, RST, GRN, bar_buf, RST, numBatches, numBatches, dur_buf);
            } else {
                char eta_buf[32];
                wb_fmt_duration(avg_sec * rem, eta_buf, sizeof(eta_buf));
                fprintf(stderr,
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  %5.1f%%  "
                    "%.2fs/batch  ETA: %-8s",
                    GRN, RST, GRN, bar_buf, RST, b + 1, numBatches, pct, avg_sec, eta_buf);
            }
            fflush(stderr);
        }
    }   // end batch loop

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
    cudaFree(dNodeData);
    cudaFree(dNodeOffset);
    cudaFree(dPartLeafCount);
    cudaFree(dOrderings);
    cudaFree(dInvIndex);

    env->ReleaseIntArrayElements(jSplits,        hSplits,        JNI_ABORT);
    env->ReleaseIntArrayElements(jNodeData,      hNodeData,      JNI_ABORT);
    env->ReleaseIntArrayElements(jNodeOffset,    hNodeOffset,    JNI_ABORT);
    env->ReleaseIntArrayElements(jPartLeafCount, hPartLeafCount, JNI_ABORT);
    env->ReleaseIntArrayElements(jOrderings,     hOrderings,     JNI_ABORT);
    env->ReleaseIntArrayElements(jInvIndex,      hInvIndex,      JNI_ABORT);

    return result;
}

} // extern "C"
