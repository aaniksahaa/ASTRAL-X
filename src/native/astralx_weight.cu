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
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <jni.h>

// Fixed block size.  Must match the static reduction buffer below and the
// dynamic shared-memory scan area sized on the host.
#define WB_BLOCK 256
#define WB_SMALL_QI_MAX_N 491

// Simple-tree-walk per-thread postorder stack cap (in triples). The Java side
// measures the exact maximum postorder evaluation frontier of the scoring trees;
// the GPU path is used whenever that measured frontier fits the compiled maximum,
// regardless of the total taxon count.  Launch dispatch selects the smallest of
// 32/64/128/256/512 entries that fits, so the usual private stack is much smaller
// than the worst-case maximum.
#define WB_TW_STACK_CAP 512

// ---------------------------------------------------------------------------
// Adaptive accumulator transport.
//
// Scores are accumulated either as exact 64-bit integers (long long) or, for
// very large taxon sets where the exact value overflows, as 64-bit floating
// point (double).  Both are returned through the same long long[] transport:
//   - long long: stored verbatim (exact 2·score).
//   - double:    stored as its IEEE-754 bit pattern via __double_as_longlong;
//                the Java side recovers it with Double.longBitsToDouble.
// The template accumulator type (ACC) selects the path at compile time; the
// host launches the matching instantiation based on the useDouble flag.
// ---------------------------------------------------------------------------
__device__ inline void storeTwoScore(long long* out, int idx, long long v) { out[idx] = v; }
__device__ inline void storeTwoScore(long long* out, int idx, double    v) { out[idx] = __double_as_longlong(v); }

// Exact binary-node 2*QI.  For totalN <= 491, each individual permutation
// term is bounded by floor-balanced x*y*z*(totalN-3) <= INT_MAX.  The guarded
// specialization therefore uses full-rate 32-bit products for each term and
// widens only the six-term sum.  Larger inputs retain the original ACC-width
// arithmetic.  SMALL_QI is a compile-time dispatch flag, so the generic kernel
// pays no runtime branch or additional live state.
template<typename ACC, bool SMALL_QI>
__device__ __forceinline__ ACC binaryTwoQI(
    int a0, int a1, int a2, int b0, int b1, int b2, int c0, int c1, int c2)
{
    if (SMALL_QI) {
        long long q = 0;
        int t;
        t = a0 * b1; t *= c2; t *= a0 + b1 + c2 - 3; q += t;
        t = a0 * b2; t *= c1; t *= a0 + b2 + c1 - 3; q += t;
        t = a1 * b0; t *= c2; t *= a1 + b0 + c2 - 3; q += t;
        t = a1 * b2; t *= c0; t *= a1 + b2 + c0 - 3; q += t;
        t = a2 * b0; t *= c1; t *= a2 + b0 + c1 - 3; q += t;
        t = a2 * b1; t *= c0; t *= a2 + b1 + c0 - 3; q += t;
        return (ACC)q;
    }

    ACC q = (ACC)0;
    ACC A0 = (ACC)a0, A1 = (ACC)a1, A2 = (ACC)a2;
    ACC B0 = (ACC)b0, B1 = (ACC)b1, B2 = (ACC)b2;
    ACC C0 = (ACC)c0, C1 = (ACC)c1, C2 = (ACC)c2;
    q += A0 * B1 * C2 * (A0 + B1 + C2 - 3);
    q += A0 * B2 * C1 * (A0 + B2 + C1 - 3);
    q += A1 * B0 * C2 * (A1 + B0 + C2 - 3);
    q += A1 * B2 * C0 * (A1 + B2 + C0 - 3);
    q += A2 * B0 * C1 * (A2 + B0 + C1 - 3);
    q += A2 * B1 * C0 * (A2 + B1 + C0 - 3);
    return q;
}

// ---------------------------------------------------------------------------
// Emulated 128-bit signed integer for exact, overflow-free accumulation at very
// large taxon counts.  CUDA device code has no native __int128, so we carry a
// {low (unsigned), high (signed)} pair and implement only the few operations the
// score loop needs — all from full-rate integer instructions (no throttled FP64).
//
// All score operands are non-negative (intersection counts, frequencies), so the
// multiplies use unsigned 64×64→128 (__umul64hi); the signed high word only
// matters for the DP's sentinel on the Java side.
//
// Magnitude budget (n ≤ ~1e5, genes ≤ ~1e4):
//   ai·bj·ck      ≤ ~2^51   (fits signed 64-bit)
//   (ai·bj·ck)·su  → up to ~2^70   (needs 128-bit; one __umul64hi)
//   2·QI = Σ6 terms, freq·2·QI, and the per-split block sum all fit in 128 bits.
// ---------------------------------------------------------------------------
struct I128 { unsigned long long lo; long long hi; };

__device__ inline I128 i128_zero() { I128 r; r.lo = 0ULL; r.hi = 0LL; return r; }

__device__ inline I128 i128_add(I128 a, I128 b) {
    I128 r;
    r.lo = a.lo + b.lo;
    long long carry = (r.lo < a.lo) ? 1LL : 0LL;   // unsigned wrap ⇒ carry
    r.hi = a.hi + b.hi + carry;
    return r;
}

// Exact 64×64→128 product of two non-negative values.
__device__ inline I128 i128_mul_u64(unsigned long long a, unsigned long long b) {
    I128 r;
    r.lo = a * b;
    r.hi = (long long) __umul64hi(a, b);
    return r;
}

// this · small non-negative scalar (true product fits in 128 bits for our budget).
__device__ inline I128 i128_mul_scalar(I128 a, unsigned long long f) {
    I128 lop = i128_mul_u64(a.lo, f);          // 128-bit product of the low word
    long long hiAdd = a.hi * (long long) f;    // high-word contribution (fits 64-bit here)
    I128 r; r.lo = lop.lo; r.hi = lop.hi + hiAdd; return r;
}

// INT128 transport: two longs per split — [2*idx] = low (unsigned bits), [2*idx+1] = high.
__device__ inline void storeTwoScoreI128(long long* out, int idx, I128 v) {
    out[(size_t)idx * 2]     = (long long) v.lo;
    out[(size_t)idx * 2 + 1] = v.hi;
}

// ---------------------------------------------------------------------------
// Device helper: cooperatively build BOTH cluster-membership prefix rows over a
// tree's leaves.  Fusing A/B means the gene-tree ordering is loaded only once.
// Chunk totals are scanned with warp shuffles; scanA/scanB hold only the eight
// warp totals (WB_BLOCK slots are reserved to keep the dynamic layout simple).
//
//   pX[p] = number of leaves among the first p (postorder) that are in the
//           cluster (clLo,clHi,clComp) of tree clBase.
//
// All threads of the block must call this uniformly (it issues __syncthreads).
// ---------------------------------------------------------------------------
__device__ void buildPrefixPair(
    int* __restrict__ pA, int* __restrict__ pB,
    int* __restrict__ scanA, int* __restrict__ scanB, int L,
    size_t gBase,
    size_t aBase, int aLo, int aHi, int aComp, int aRngOff, int aRngCnt,
    size_t bBase, int bLo, int bHi, int bComp, int bRngOff, int bRngCnt,
    const int* __restrict__ rangeData,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int tid, int nthreads)
{
    if (L <= 0) {
        if (tid == 0) { pA[0] = 0; pB[0] = 0; }
        __syncthreads();
        return;
    }

    int chunk = (L + nthreads - 1) / nthreads;   // ceil
    int start = tid * chunk;
    int end   = start + chunk;
    if (start > L) start = L;
    if (end   > L) end   = L;

    // Pass A: write both indicator rows and accumulate both chunk totals.
    int sumA = 0, sumB = 0;
    for (int p = start; p < end; p++) {
        int t = orderings[gBase + (size_t)p];

        int posA = invIndex[aBase + (size_t)t];
        int inA;
        if (aRngCnt == 0) {
            inA = (posA >= aLo && posA < aHi) ? 1 : 0;
        } else {
            inA = 0;
            for (int r = 0; r < aRngCnt; r++) {
                int rlo = rangeData[2 * (aRngOff + r)];
                int rhi = rangeData[2 * (aRngOff + r) + 1];
                if (posA >= rlo && posA < rhi) { inA = 1; break; }
            }
        }
        inA ^= aComp;

        int posB = invIndex[bBase + (size_t)t];
        int inB;
        if (bRngCnt == 0) {
            inB = (posB >= bLo && posB < bHi) ? 1 : 0;
        } else {
            inB = 0;
            for (int r = 0; r < bRngCnt; r++) {
                int rlo = rangeData[2 * (bRngOff + r)];
                int rhi = rangeData[2 * (bRngOff + r) + 1];
                if (posB >= rlo && posB < rhi) { inB = 1; break; }
            }
        }
        inB ^= bComp;

        pA[p] = inA; pB[p] = inB;
        sumA += inA; sumB += inB;
    }

    // Inclusive scan within each warp (all WB_BLOCK threads are active here).
    int warpA = sumA, warpB = sumB;
    int lane = tid & 31;
    int warp = tid >> 5;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int va = __shfl_up_sync(0xffffffffu, warpA, off);
        int vb = __shfl_up_sync(0xffffffffu, warpB, off);
        if (lane >= off) { warpA += va; warpB += vb; }
    }
    if (lane == 31) { scanA[warp] = warpA; scanB[warp] = warpB; }
    __syncthreads();

    // First warp scans the (at most eight) warp totals.
    int numWarps = (nthreads + 31) >> 5;
    if (warp == 0) {
        int blockA = (lane < numWarps) ? scanA[lane] : 0;
        int blockB = (lane < numWarps) ? scanB[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int va = __shfl_up_sync(0xffffffffu, blockA, off);
            int vb = __shfl_up_sync(0xffffffffu, blockB, off);
            if (lane >= off) { blockA += va; blockB += vb; }
        }
        if (lane < numWarps) { scanA[lane] = blockA; scanB[lane] = blockB; }
    }
    __syncthreads();

    int priorWarpA = (warp == 0) ? 0 : scanA[warp - 1];
    int priorWarpB = (warp == 0) ? 0 : scanB[warp - 1];
    int exclA = priorWarpA + warpA - sumA;
    int exclB = priorWarpB + warpB - sumB;

    // Pass B: convert both per-chunk indicator rows to global exclusive prefixes.
    int accA = exclA, accB = exclB;
    for (int p = start; p < end; p++) {
        int va = pA[p], vb = pB[p];
        pA[p] = accA; pB[p] = accB;
        accA += va; accB += vb;
    }
    if (start < L && end == L) { pA[L] = accA; pB[L] = accB; }

    __syncthreads();
}

// ---------------------------------------------------------------------------
// Polytomy (d>3) per-node QI on the prefix-sum path (polytomy-design.md §3.8.4b).
// Reuses the SAME pA/pB prefix arrays already built for tree g (and its lgA/lgB).
// Each thread grid-strides over tree g's poly nodes; each computes the full O(d)
// QI with O(1) working memory (child parts via O(1) prefix differences).  Returns
// this thread's partial accumulation, to be added into threadAccum.
//
// Poly CSR (bucketed by exemplar tree):
//   polyTreeOffset[g]..[g+1]      poly nodes of tree g
//   polyBoundOffset[pn]..[pn+1]   range into polyBounds; length d (the degree)
//   polyBounds[base + 0..d-1]     boundary list; child i = [b[i],b[i+1]) (i=0..d-2),
//                                  part d-1 = complement Lg \ [b[0],b[d-1])
//   polyFreq[pn]                  occurrence count
// ---------------------------------------------------------------------------
template<typename ACC>
__device__ ACC scorePolyNodes(
    int g, int L, int lgA, int lgB,
    const int* __restrict__ pA, const int* __restrict__ pB,
    const int* __restrict__ polyTreeOffset,
    const int* __restrict__ polyBoundOffset,
    const int* __restrict__ polyBounds,
    const int* __restrict__ polyFreq,
    int tid, int nthreads)
{
    ACC acc = (ACC) 0;
    int pbeg = polyTreeOffset[g], pend = polyTreeOffset[g + 1];
    for (int pn = pbeg + tid; pn < pend; pn += nthreads) {
        int base = polyBoundOffset[pn];
        int d    = polyBoundOffset[pn + 1] - base;
        int b0   = polyBounds[base];
        int bD   = polyBounds[base + d - 1];

        // Pass 1: global marginals over all d parts (child parts via prefix diffs).
        ACC Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        int sumA = 0, sumB = 0;
        for (int i = 0; i < d - 1; i++) {
            int lo = polyBounds[base + i], hi = polyBounds[base + i + 1];
            int ai = pA[hi] - pA[lo];
            int bi = pB[hi] - pB[lo];
            int ci = (hi - lo) - ai - bi;            // ≥ 0 by construction
            Sa += ai; Sb += bi; Sc += ci;
            Sab += (ACC) ai * bi; Sac += (ACC) ai * ci; Sbc += (ACC) bi * ci;
            sumA += ai; sumB += bi;
        }
        int aC = lgA - sumA;
        int bC = lgB - sumB;
        int szC = L - (bD - b0);
        int cC = szC - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) continue;    // incomplete-tree row mismatch → skip node
        Sa += aC; Sb += bC; Sc += cC;
        Sab += (ACC) aC * bC; Sac += (ACC) aC * cC; Sbc += (ACC) bC * cC;

        // Pass 2: O(d) QI (recompute child parts; complement reused).
        ACC twoQI = (ACC) 0;
        for (int i = 0; i < d; i++) {
            int ai, bi, ci;
            if (i < d - 1) {
                int lo = polyBounds[base + i], hi = polyBounds[base + i + 1];
                ai = pA[hi] - pA[lo]; bi = pB[hi] - pB[lo]; ci = (hi - lo) - ai - bi;
            } else { ai = aC; bi = bC; ci = cC; }
            ACC A = ai, B = bi, C = ci;
            twoQI += A * (A - 1) * ((Sb - B) * (Sc - C) - Sbc + B * C);
            twoQI += B * (B - 1) * ((Sa - A) * (Sc - C) - Sac + A * C);
            twoQI += C * (C - 1) * ((Sa - A) * (Sb - B) - Sab + A * B);
        }
        acc += (ACC) polyFreq[pn] * twoQI;
    }
    return acc;
}

// INT128 twin of scorePolyNodes.  Marginals (Sa..Sbc) and each bracket fit in 64-bit
// (≤ n²); the weight·bracket product (≤ n⁴) and the accumulation are 128-bit.
__device__ I128 scorePolyNodesI128(
    int g, int L, int lgA, int lgB,
    const int* __restrict__ pA, const int* __restrict__ pB,
    const int* __restrict__ polyTreeOffset,
    const int* __restrict__ polyBoundOffset,
    const int* __restrict__ polyBounds,
    const int* __restrict__ polyFreq,
    int tid, int nthreads)
{
    I128 acc = i128_zero();
    int pbeg = polyTreeOffset[g], pend = polyTreeOffset[g + 1];
    for (int pn = pbeg + tid; pn < pend; pn += nthreads) {
        int base = polyBoundOffset[pn];
        int d    = polyBoundOffset[pn + 1] - base;
        int b0   = polyBounds[base];
        int bD   = polyBounds[base + d - 1];

        long long Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        int sumA = 0, sumB = 0;
        for (int i = 0; i < d - 1; i++) {
            int lo = polyBounds[base + i], hi = polyBounds[base + i + 1];
            int ai = pA[hi] - pA[lo];
            int bi = pB[hi] - pB[lo];
            int ci = (hi - lo) - ai - bi;
            Sa += ai; Sb += bi; Sc += ci;
            Sab += (long long) ai * bi; Sac += (long long) ai * ci; Sbc += (long long) bi * ci;
            sumA += ai; sumB += bi;
        }
        int aC = lgA - sumA;
        int bC = lgB - sumB;
        int szC = L - (bD - b0);
        int cC = szC - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) continue;
        Sa += aC; Sb += bC; Sc += cC;
        Sab += (long long) aC * bC; Sac += (long long) aC * cC; Sbc += (long long) bC * cC;

        I128 twoQI = i128_zero();
        for (int i = 0; i < d; i++) {
            long long ai, bi, ci;
            if (i < d - 1) {
                int lo = polyBounds[base + i], hi = polyBounds[base + i + 1];
                ai = pA[hi] - pA[lo]; bi = pB[hi] - pB[lo]; ci = (hi - lo) - ai - bi;
            } else { ai = aC; bi = bC; ci = cC; }
            long long brA = (Sb - bi) * (Sc - ci) - Sbc + bi * ci;   // each ≥ 0, ≤ n²
            long long brB = (Sa - ai) * (Sc - ci) - Sac + ai * ci;
            long long brC = (Sa - ai) * (Sb - bi) - Sab + ai * bi;
            long long wA = ai * (ai - 1), wB = bi * (bi - 1), wC = ci * (ci - 1);
            if (wA > 0 && brA > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wA, (unsigned long long) brA));
            if (wB > 0 && brB > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wB, (unsigned long long) brB));
            if (wC > 0 && brC > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wC, (unsigned long long) brC));
        }
        acc = i128_add(acc, i128_mul_scalar(twoQI, (unsigned long long) polyFreq[pn]));
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Score one split.  pA/pB are the two prefix buffers (in shared memory for the
// fast path, or in a per-block global slot for the large-L path); scanA/scanB are
// the shared-memory warp-total scratch used by buildPrefixPair.
//
// Called once per block (shared mode) or repeatedly via a grid-stride loop
// (global mode).  Issues __syncthreads, so all threads must call it uniformly.
// ---------------------------------------------------------------------------
template<typename ACC, bool SMALL_QI>
__device__ void scoreSplit(
    int s,
    const int* __restrict__ splits,
    const int* __restrict__ splitRangeMeta,
    const int* __restrict__ rangeData,
    const int* __restrict__ nodeData,
    const int* __restrict__ nodeFreq,
    const int* __restrict__ nodeOffset,
    const int* __restrict__ partLeafCount,
    const int* __restrict__ polyTreeOffset,
    const int* __restrict__ polyBoundOffset,
    const int* __restrict__ polyBounds,
    const int* __restrict__ polyFreq,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int numPartTrees, int partTreeOffset, int numTaxa, int totalN,
    int* __restrict__ pA, int* __restrict__ pB,
    int* __restrict__ scanA, int* __restrict__ scanB,
    int tid, int nthreads,
    long long* __restrict__ twoScores)
{
    __shared__ ACC red[WB_BLOCK];

    const int* sp = splits + (size_t)s * 10;
    int aTree = sp[0], aLo = sp[1], aHi = sp[2], aComp = sp[3], aSize = sp[4];
    int bTree = sp[5], bLo = sp[6], bHi = sp[7], bComp = sp[8], bSize = sp[9];
    // Multi-range descriptor: [aRngOff, aRngCnt, bRngOff, bRngCnt]; cnt==0 ⇒ single-range.
    const int* rm = splitRangeMeta + (size_t)s * 4;
    int aRngOff = rm[0], aRngCnt = rm[1], bRngOff = rm[2], bRngCnt = rm[3];

    // Invalid / overlapping split → zero (defensive; real DP splits are disjoint).
    // (0LL is also the bit pattern of +0.0, so it decodes correctly in both modes.)
    if (aSize + bSize > totalN) {
        if (tid == 0) twoScores[s] = 0LL;
        return;   // uniform across the block (same aSize/bSize for all threads)
    }

    size_t aBase = (size_t)aTree * numTaxa;
    size_t bBase = (size_t)bTree * numTaxa;

    ACC threadAccum = (ACC) 0;

    for (int g = 0; g < numPartTrees; g++) {
        int nbeg = nodeOffset[g];
        int nend = nodeOffset[g + 1];
        int pbeg = polyTreeOffset[g];
        int pend = polyTreeOffset[g + 1];
        if (nbeg == nend && pbeg == pend) continue;   // no binary AND no poly nodes (uniform skip)

        int    L     = partLeafCount[g];
        size_t gBase = (size_t)(partTreeOffset + g) * numTaxa;

        buildPrefixPair(pA, pB, scanA, scanB, L, gBase,
            aBase, aLo, aHi, aComp, aRngOff, aRngCnt,
            bBase, bLo, bHi, bComp, bRngOff, bRngCnt,
            rangeData, orderings, invIndex, tid, nthreads);

        int lgA = pA[L];
        int lgB = pB[L];

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

            ACC twoQI = binaryTwoQI<ACC, SMALL_QI>(
                a0, a1, a2, b0, b1, b2, c0, c1, c2);
            threadAccum += (ACC) nodeFreq[ni] * twoQI;   // weight by occurrence count
        }

        // Polytomy (d>3) nodes of this tree — reuse the SAME pA/pB/lgA/lgB.
        if (pbeg != pend)
            threadAccum += scorePolyNodes<ACC>(g, L, lgA, lgB, pA, pB,
                polyTreeOffset, polyBoundOffset, polyBounds, polyFreq, tid, nthreads);

        __syncthreads();   // pA/pB reused next iteration; ensure both loops done
    }

    // Block reduction of threadAccum → twoScores[s].
    red[tid] = threadAccum;
    __syncthreads();
    for (int off = nthreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) storeTwoScore(twoScores, s, red[0]);
    __syncthreads();   // red fully consumed before a global-mode reuse
}

// ---------------------------------------------------------------------------
// Main kernel.
//   GLOBAL=false : prefix buffers live in dynamic shared memory; one block per
//                  split (grid = curBatch).  Fast path, capped at L that fits.
//   GLOBAL=true  : prefix buffers live in a per-block slot of gPrefix (global
//                  memory); grid is capped to the resident-block count and each
//                  block grid-strides over splits.  Large-L path, bounded VRAM.
//
// Dynamic shared layout:
//   GLOBAL=false : pA[stride], pB[stride], scanA[WB_BLOCK], scanB[WB_BLOCK]
//   GLOBAL=true  : scanA[WB_BLOCK], scanB[WB_BLOCK]     (pA/pB in gPrefix)
// ---------------------------------------------------------------------------
template<bool GLOBAL, typename ACC, bool SMALL_QI>
__global__ void computeWeightsKernel(
    const int* __restrict__ splits,
    const int* __restrict__ splitRangeMeta,
    const int* __restrict__ rangeData,
    const int* __restrict__ nodeData,
    const int* __restrict__ nodeFreq,
    const int* __restrict__ nodeOffset,
    const int* __restrict__ partLeafCount,
    const int* __restrict__ polyTreeOffset,
    const int* __restrict__ polyBoundOffset,
    const int* __restrict__ polyBounds,
    const int* __restrict__ polyFreq,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int curBatch,
    int numPartTrees,
    int partTreeOffset,
    int prefixStride,                      // = maxLeafCount + 1
    int numTaxa,
    int totalN,
    int* __restrict__ gPrefix,             // global prefix pool (GLOBAL only)
    long long* __restrict__ twoScores,
    int* __restrict__ dProgress)           // splits-completed counter (host-polled)
{
    extern __shared__ int smem[];
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    if (GLOBAL) {
        int* scanA = smem;
        int* scanB = smem + WB_BLOCK;
        int* pA   = gPrefix + (size_t)blockIdx.x * 2 * prefixStride;
        int* pB   = pA + prefixStride;
        for (int s = blockIdx.x; s < curBatch; s += gridDim.x) {
            scoreSplit<ACC, SMALL_QI>(s, splits, splitRangeMeta, rangeData, nodeData, nodeFreq, nodeOffset, partLeafCount,
                       polyTreeOffset, polyBoundOffset, polyBounds, polyFreq,
                       orderings, invIndex, numPartTrees, partTreeOffset,
                       numTaxa, totalN, pA, pB, scanA, scanB, tid, nthreads, twoScores);
            if (tid == 0 && dProgress) atomicAdd(dProgress, 1);   // one per finished split
        }
    } else {
        int* pA   = smem;
        int* pB   = smem + prefixStride;
        int* scanA = smem + 2 * prefixStride;
        int* scanB = scanA + WB_BLOCK;
        int s = blockIdx.x;
        if (s < curBatch) {
            scoreSplit<ACC, SMALL_QI>(s, splits, splitRangeMeta, rangeData, nodeData, nodeFreq, nodeOffset, partLeafCount,
                       polyTreeOffset, polyBoundOffset, polyBounds, polyFreq,
                       orderings, invIndex, numPartTrees, partTreeOffset,
                       numTaxa, totalN, pA, pB, scanA, scanB, tid, nthreads, twoScores);
            if (tid == 0 && dProgress) atomicAdd(dProgress, 1);   // one per finished split
        }
    }
}

// ---------------------------------------------------------------------------
// INT128 variant of scoreSplit — exact 128-bit accumulation (overflow-free at
// very large n) using only full-rate integer instructions.  Structurally
// identical to scoreSplit<ACC>; only the QI products and accumulators are 128-bit.
// twoScores is the 2-wide INT128 transport (two longs per split).
// ---------------------------------------------------------------------------
__device__ void scoreSplitI128(
    int s,
    const int* __restrict__ splits,
    const int* __restrict__ splitRangeMeta,
    const int* __restrict__ rangeData,
    const int* __restrict__ nodeData,
    const int* __restrict__ nodeFreq,
    const int* __restrict__ nodeOffset,
    const int* __restrict__ partLeafCount,
    const int* __restrict__ polyTreeOffset,
    const int* __restrict__ polyBoundOffset,
    const int* __restrict__ polyBounds,
    const int* __restrict__ polyFreq,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int numPartTrees, int partTreeOffset, int numTaxa, int totalN,
    int* __restrict__ pA, int* __restrict__ pB,
    int* __restrict__ scanA, int* __restrict__ scanB,
    int tid, int nthreads,
    long long* __restrict__ twoScores)
{
    __shared__ I128 red[WB_BLOCK];

    const int* sp = splits + (size_t)s * 10;
    int aTree = sp[0], aLo = sp[1], aHi = sp[2], aComp = sp[3], aSize = sp[4];
    int bTree = sp[5], bLo = sp[6], bHi = sp[7], bComp = sp[8], bSize = sp[9];
    const int* rm = splitRangeMeta + (size_t)s * 4;
    int aRngOff = rm[0], aRngCnt = rm[1], bRngOff = rm[2], bRngCnt = rm[3];

    if (aSize + bSize > totalN) {
        if (tid == 0) storeTwoScoreI128(twoScores, s, i128_zero());
        return;
    }

    size_t aBase = (size_t)aTree * numTaxa;
    size_t bBase = (size_t)bTree * numTaxa;

    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    I128 threadAccum = i128_zero();

    for (int g = 0; g < numPartTrees; g++) {
        int nbeg = nodeOffset[g];
        int nend = nodeOffset[g + 1];
        int pbeg = polyTreeOffset[g];
        int pend = polyTreeOffset[g + 1];
        if (nbeg == nend && pbeg == pend) continue;

        int    L     = partLeafCount[g];
        size_t gBase = (size_t)(partTreeOffset + g) * numTaxa;

        buildPrefixPair(pA, pB, scanA, scanB, L, gBase,
            aBase, aLo, aHi, aComp, aRngOff, aRngCnt,
            bBase, bLo, bHi, bComp, bRngOff, bRngCnt,
            rangeData, orderings, invIndex, tid, nthreads);

        int lgA = pA[L];
        int lgB = pB[L];

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

            I128 twoQI = i128_zero();
            #pragma unroll
            for (int p = 0; p < 6; p++) {
                long long ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
                long long su = ai + bj + ck - 3;
                if (su > 0) {
                    long long abc = ai * bj * ck;   // ≤ ~2^51, fits 64-bit
                    twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long)abc,
                                                         (unsigned long long)su));
                }
            }
            threadAccum = i128_add(threadAccum,
                                   i128_mul_scalar(twoQI, (unsigned long long) nodeFreq[ni]));
        }

        // Polytomy (d>3) nodes of this tree — reuse the SAME pA/pB/lgA/lgB.
        if (pbeg != pend)
            threadAccum = i128_add(threadAccum,
                scorePolyNodesI128(g, L, lgA, lgB, pA, pB,
                    polyTreeOffset, polyBoundOffset, polyBounds, polyFreq, tid, nthreads));

        __syncthreads();
    }

    red[tid] = threadAccum;
    __syncthreads();
    for (int off = nthreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = i128_add(red[tid], red[tid + off]);
        __syncthreads();
    }
    if (tid == 0) storeTwoScoreI128(twoScores, s, red[0]);
    __syncthreads();
}

// INT128 kernel wrapper (mirrors computeWeightsKernel<GLOBAL, ACC>).
template<bool GLOBAL>
__global__ void computeWeightsKernelI128(
    const int* __restrict__ splits,
    const int* __restrict__ splitRangeMeta,
    const int* __restrict__ rangeData,
    const int* __restrict__ nodeData,
    const int* __restrict__ nodeFreq,
    const int* __restrict__ nodeOffset,
    const int* __restrict__ partLeafCount,
    const int* __restrict__ polyTreeOffset,
    const int* __restrict__ polyBoundOffset,
    const int* __restrict__ polyBounds,
    const int* __restrict__ polyFreq,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int curBatch,
    int numPartTrees,
    int partTreeOffset,
    int prefixStride,
    int numTaxa,
    int totalN,
    int* __restrict__ gPrefix,
    long long* __restrict__ twoScores,
    int* __restrict__ dProgress)
{
    extern __shared__ int smem[];
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    if (GLOBAL) {
        int* scanA = smem;
        int* scanB = smem + WB_BLOCK;
        int* pA   = gPrefix + (size_t)blockIdx.x * 2 * prefixStride;
        int* pB   = pA + prefixStride;
        for (int s = blockIdx.x; s < curBatch; s += gridDim.x) {
            scoreSplitI128(s, splits, splitRangeMeta, rangeData, nodeData, nodeFreq, nodeOffset, partLeafCount,
                           polyTreeOffset, polyBoundOffset, polyBounds, polyFreq,
                           orderings, invIndex, numPartTrees, partTreeOffset,
                           numTaxa, totalN, pA, pB, scanA, scanB, tid, nthreads, twoScores);
            if (tid == 0 && dProgress) atomicAdd(dProgress, 1);
        }
    } else {
        int* pA   = smem;
        int* pB   = smem + prefixStride;
        int* scanA = smem + 2 * prefixStride;
        int* scanB = scanA + WB_BLOCK;
        int s = blockIdx.x;
        if (s < curBatch) {
            scoreSplitI128(s, splits, splitRangeMeta, rangeData, nodeData, nodeFreq, nodeOffset, partLeafCount,
                           polyTreeOffset, polyBoundOffset, polyBounds, polyFreq,
                           orderings, invIndex, numPartTrees, partTreeOffset,
                           numTaxa, totalN, pA, pB, scanA, scanB, tid, nthreads, twoScores);
            if (tid == 0 && dProgress) atomicAdd(dProgress, 1);
        }
    }
}

// ===========================================================================
// LEGACY "smaller-side traversal" path (activated by --weight-intersection-method
// smaller-side-traversal).  One thread per split, ZERO per-thread state, NO prefix sums:
// each of the 4 core intersections is counted by walking the smaller of the two
// ranges element-by-element.  Completely independent of the prefix-sum path above
// — no shared/global prefix memory is touched here.
// ===========================================================================

// Range intersection: count taxa in [loA,hiA) of tree tA that also appear in
// [loB,hiB) of tree tB.  Iterates the SMALLER range for efficiency.
__device__ int ssCoreIntersect(
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
            int taxon = orderings[(size_t)tA * numTaxa + pos];
            int posB  = invIndex [(size_t)tB * numTaxa + taxon];
            if (posB >= loB && posB < hiB) count++;
        }
    } else {
        for (int pos = loB; pos < hiB; pos++) {
            int taxon = orderings[(size_t)tB * numTaxa + pos];
            int posA  = invIndex [(size_t)tA * numTaxa + taxon];
            if (posA >= loA && posA < hiA) count++;
        }
    }
    return count;
}

// Intersection with optional complement of the cluster side.
__device__ int ssIntersect(
    int tGT, int loGT, int hiGT,
    int tC,  int loC,  int hiC, int cComp, int szGTRange,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int numTaxa)
{
    int raw = ssCoreIntersect(tGT, loGT, hiGT, tC, loC, hiC, orderings, invIndex, numTaxa);
    return cComp ? (szGTRange - raw) : raw;
}

// Multi-range-aware |M_range ∩ cluster|: rCnt==0 ⇒ single-range fast path; else sum
// ssCoreIntersect over the cluster's disjoint ranges (multi-range-cluster-design §5.3).
__device__ int ssIntersectSide(
    int tGT, int loGT, int hiGT,
    int cTree, int cLo, int cHi, int cComp, int szGTRange,
    int rOff, int rCnt, const int* __restrict__ rangeData,
    const int* __restrict__ orderings, const int* __restrict__ invIndex, int numTaxa)
{
    if (rCnt == 0)
        return ssIntersect(tGT, loGT, hiGT, cTree, cLo, cHi, cComp, szGTRange, orderings, invIndex, numTaxa);
    int core = 0;
    for (int r = 0; r < rCnt; r++) {
        int rlo = rangeData[2 * (rOff + r)];
        int rhi = rangeData[2 * (rOff + r) + 1];
        core += ssCoreIntersect(tGT, loGT, hiGT, cTree, rlo, rhi, orderings, invIndex, numTaxa);
    }
    return cComp ? (szGTRange - core) : core;
}

// Multi-range-aware row sum |cluster ∩ Lg_GT| for incomplete gene trees.
__device__ int ssRowSum(
    int tGT, int L_GT, int cTree, int cLo, int cHi, int cComp,
    int rOff, int rCnt, const int* __restrict__ rangeData,
    const int* __restrict__ orderings, const int* __restrict__ invIndex, int numTaxa)
{
    int core = 0;
    if (rCnt == 0) {
        core = ssCoreIntersect(tGT, 0, L_GT, cTree, cLo, cHi, orderings, invIndex, numTaxa);
    } else {
        for (int r = 0; r < rCnt; r++) {
            int rlo = rangeData[2 * (rOff + r)];
            int rhi = rangeData[2 * (rOff + r) + 1];
            core += ssCoreIntersect(tGT, 0, L_GT, cTree, rlo, rhi, orderings, invIndex, numTaxa);
        }
    }
    return cComp ? (L_GT - core) : core;
}

// ---------------------------------------------------------------------------
// Smaller-side polytomy (d>3) scoring — two-pass-with-rewalk (polytomy-design.md
// §3.8.4d).  Pass 1 walks the d-1 child ranges (×2 for A,B) to accumulate the
// global marginals; pass 2 re-walks them and applies the per-part O(d) formula —
// reusing the IDENTICAL arithmetic as the prefix-sum/CPU paths (trivially correct
// in LONG/DOUBLE/INT128).  Poly nodes are rare, so the extra walk is negligible.
//
// Smaller-side poly CSR:
//   ssPolyMeta[3*pn] = {treeIdx(+partTreeOffset), L_GT, freq}
//   ssPolyBoundOffset[pn]..[pn+1]   range into ssPolyBounds, length d
//   ssPolyBounds[base + 0..d-1]     child i = [b[i],b[i+1]); part d-1 = complement
// ---------------------------------------------------------------------------
template<typename ACC, bool CACHE_ROWS>
__device__ ACC ssScorePoly(
    int loTree, int loLeft, int loRight, int loComp, int sizeA, int aRngOff, int aRngCnt,
    int hiTree, int hiLeft, int hiRight, int hiComp, int sizeB, int bRngOff, int bRngCnt,
    const int* __restrict__ rangeData,
    const int* __restrict__ ssPolyMeta,
    const int* __restrict__ ssPolyBoundOffset,
    const int* __restrict__ ssPolyBounds,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int numPolyParts, int numTaxa, int totalN)
{
    ACC twoScore = (ACC) 0;
    int cachedTree = -1, cachedLgA = 0, cachedLgB = 0;
    for (int pn = 0; pn < numPolyParts; pn++) {
        int tGT  = ssPolyMeta[3 * pn];
        int L_GT = ssPolyMeta[3 * pn + 1];
        int freq = ssPolyMeta[3 * pn + 2];
        int base = ssPolyBoundOffset[pn];
        int d    = ssPolyBoundOffset[pn + 1] - base;
        int b0   = ssPolyBounds[base];
        int bD   = ssPolyBounds[base + d - 1];

        int lgA, lgB;
        if (L_GT == totalN) { lgA = sizeA; lgB = sizeB; }
        else {
            if (CACHE_ROWS && tGT != cachedTree) {
                cachedLgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedLgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedTree = tGT;
            }
            if (CACHE_ROWS) { lgA = cachedLgA; lgB = cachedLgB; }
            else {
                lgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                lgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
            }
        }

        // Pass 1: marginals over all d parts (child parts walked once).
        ACC Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        int sumA = 0, sumB = 0;
        for (int i = 0; i < d - 1; i++) {
            int lo = ssPolyBounds[base + i], hi = ssPolyBounds[base + i + 1], sz = hi - lo;
            int ai = ssIntersectSide(tGT, lo, hi, loTree, loLeft, loRight, loComp, sz, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
            int bi = ssIntersectSide(tGT, lo, hi, hiTree, hiLeft, hiRight, hiComp, sz, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
            int ci = sz - ai - bi;
            Sa += ai; Sb += bi; Sc += ci;
            Sab += (ACC) ai * bi; Sac += (ACC) ai * ci; Sbc += (ACC) bi * ci;
            sumA += ai; sumB += bi;
        }
        int aC = lgA - sumA, bC = lgB - sumB, szC = L_GT - (bD - b0), cC = szC - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) continue;
        Sa += aC; Sb += bC; Sc += cC;
        Sab += (ACC) aC * bC; Sac += (ACC) aC * cC; Sbc += (ACC) bC * cC;

        // Pass 2: re-walk child ranges + O(d) formula (complement reused).
        ACC twoQI = (ACC) 0;
        for (int i = 0; i < d; i++) {
            int ai, bi, ci;
            if (i < d - 1) {
                int lo = ssPolyBounds[base + i], hi = ssPolyBounds[base + i + 1], sz = hi - lo;
                ai = ssIntersectSide(tGT, lo, hi, loTree, loLeft, loRight, loComp, sz, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                bi = ssIntersectSide(tGT, lo, hi, hiTree, hiLeft, hiRight, hiComp, sz, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
                ci = sz - ai - bi;
            } else { ai = aC; bi = bC; ci = cC; }
            ACC A = ai, B = bi, C = ci;
            twoQI += A * (A - 1) * ((Sb - B) * (Sc - C) - Sbc + B * C);
            twoQI += B * (B - 1) * ((Sa - A) * (Sc - C) - Sac + A * C);
            twoQI += C * (C - 1) * ((Sa - A) * (Sb - B) - Sab + A * B);
        }
        twoScore += (ACC) freq * twoQI;
    }
    return twoScore;
}

// INT128 twin of ssScorePoly.
template<bool CACHE_ROWS>
__device__ I128 ssScorePolyI128(
    int loTree, int loLeft, int loRight, int loComp, int sizeA, int aRngOff, int aRngCnt,
    int hiTree, int hiLeft, int hiRight, int hiComp, int sizeB, int bRngOff, int bRngCnt,
    const int* __restrict__ rangeData,
    const int* __restrict__ ssPolyMeta,
    const int* __restrict__ ssPolyBoundOffset,
    const int* __restrict__ ssPolyBounds,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int numPolyParts, int numTaxa, int totalN)
{
    I128 twoScore = i128_zero();
    int cachedTree = -1, cachedLgA = 0, cachedLgB = 0;
    for (int pn = 0; pn < numPolyParts; pn++) {
        int tGT  = ssPolyMeta[3 * pn];
        int L_GT = ssPolyMeta[3 * pn + 1];
        int freq = ssPolyMeta[3 * pn + 2];
        int base = ssPolyBoundOffset[pn];
        int d    = ssPolyBoundOffset[pn + 1] - base;
        int b0   = ssPolyBounds[base];
        int bD   = ssPolyBounds[base + d - 1];

        int lgA, lgB;
        if (L_GT == totalN) { lgA = sizeA; lgB = sizeB; }
        else {
            if (CACHE_ROWS && tGT != cachedTree) {
                cachedLgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedLgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedTree = tGT;
            }
            if (CACHE_ROWS) { lgA = cachedLgA; lgB = cachedLgB; }
            else {
                lgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                lgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
            }
        }

        long long Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        int sumA = 0, sumB = 0;
        for (int i = 0; i < d - 1; i++) {
            int lo = ssPolyBounds[base + i], hi = ssPolyBounds[base + i + 1], sz = hi - lo;
            int ai = ssIntersectSide(tGT, lo, hi, loTree, loLeft, loRight, loComp, sz, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
            int bi = ssIntersectSide(tGT, lo, hi, hiTree, hiLeft, hiRight, hiComp, sz, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
            int ci = sz - ai - bi;
            Sa += ai; Sb += bi; Sc += ci;
            Sab += (long long) ai * bi; Sac += (long long) ai * ci; Sbc += (long long) bi * ci;
            sumA += ai; sumB += bi;
        }
        int aC = lgA - sumA, bC = lgB - sumB, szC = L_GT - (bD - b0), cC = szC - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) continue;
        Sa += aC; Sb += bC; Sc += cC;
        Sab += (long long) aC * bC; Sac += (long long) aC * cC; Sbc += (long long) bC * cC;

        I128 twoQI = i128_zero();
        for (int i = 0; i < d; i++) {
            long long ai, bi, ci;
            if (i < d - 1) {
                int lo = ssPolyBounds[base + i], hi = ssPolyBounds[base + i + 1], sz = hi - lo;
                ai = ssIntersectSide(tGT, lo, hi, loTree, loLeft, loRight, loComp, sz, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                bi = ssIntersectSide(tGT, lo, hi, hiTree, hiLeft, hiRight, hiComp, sz, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
                ci = sz - ai - bi;
            } else { ai = aC; bi = bC; ci = cC; }
            long long brA = (Sb - bi) * (Sc - ci) - Sbc + bi * ci;
            long long brB = (Sa - ai) * (Sc - ci) - Sac + ai * ci;
            long long brC = (Sa - ai) * (Sb - bi) - Sab + ai * bi;
            long long wA = ai * (ai - 1), wB = bi * (bi - 1), wC = ci * (ci - 1);
            if (wA > 0 && brA > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wA, (unsigned long long) brA));
            if (wB > 0 && brB > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wB, (unsigned long long) brB));
            if (wC > 0 && brC > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wC, (unsigned long long) brC));
        }
        twoScore = i128_add(twoScore, i128_mul_scalar(twoQI, (unsigned long long) freq));
    }
    return twoScore;
}

// One thread per split; loop all deduplicated tripartitions (parts, 9 ints each).
template<typename ACC, bool CACHE_ROWS>
__global__ void computeWeightsSmallerSideKernel(
    const int* __restrict__ splits,    // curBatch * 10
    const int* __restrict__ splitRangeMeta, // curBatch * 4  [aOff,aCnt,bOff,bCnt]
    const int* __restrict__ rangeData,      // resident flat [lo,hi] pairs
    const int* __restrict__ parts,     // numParts  * 9
    const int* __restrict__ ssPolyMeta,        // numPolyParts * 3 {treeIdx,L_GT,freq}
    const int* __restrict__ ssPolyBoundOffset, // numPolyParts + 1
    const int* __restrict__ ssPolyBounds,      // Σ d boundary positions
    const int* __restrict__ orderings, // numGpuTrees * numTaxa
    const int* __restrict__ invIndex,  // numGpuTrees * numTaxa
    int curBatch,
    int numParts,
    int numPolyParts,
    int numTaxa,
    int totalN,
    long long* __restrict__ twoScores,
    int* __restrict__ dProgress)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curBatch) return;

    const int* sp = splits + (size_t)idx * 10;
    int loTree = sp[0], loLeft = sp[1], loRight = sp[2], loComp = sp[3], sizeA = sp[4];
    int hiTree = sp[5], hiLeft = sp[6], hiRight = sp[7], hiComp = sp[8], sizeB = sp[9];
    const int* rm = splitRangeMeta + (size_t)idx * 4;
    int aRngOff = rm[0], aRngCnt = rm[1], bRngOff = rm[2], bRngCnt = rm[3];

    int sizeC = totalN - sizeA - sizeB;
    if (sizeC < 0) { twoScores[idx] = 0LL; return; }   // 0LL == bits of +0.0 in both modes

    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    ACC twoScore = (ACC) 0;
    int cachedTree = -1, cachedLgA = 0, cachedLgB = 0;

    for (int j = 0; j < numParts; j++) {
        const int* pt = parts + (size_t)j * 9;
        int tGT = pt[0];
        int lo1 = pt[1], hi1 = pt[2];
        int lo2 = pt[3], hi2 = pt[4];
        int sz1 = pt[5], sz2 = pt[6], sz3 = pt[7];
        int freq = pt[8];

        int a0 = ssIntersectSide(tGT, lo1, hi1, loTree, loLeft, loRight, loComp, sz1, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
        int a1 = ssIntersectSide(tGT, lo2, hi2, loTree, loLeft, loRight, loComp, sz2, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
        int b0 = ssIntersectSide(tGT, lo1, hi1, hiTree, hiLeft, hiRight, hiComp, sz1, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
        int b1 = ssIntersectSide(tGT, lo2, hi2, hiTree, hiLeft, hiRight, hiComp, sz2, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);

        int L_GT = sz1 + sz2 + sz3;
        int lgA, lgB;
        if (L_GT == totalN) {
            lgA = sizeA;
            lgB = sizeB;
        } else {
            if (CACHE_ROWS && tGT != cachedTree) {
                cachedLgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedLgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedTree = tGT;
            }
            if (CACHE_ROWS) { lgA = cachedLgA; lgB = cachedLgB; }
            else {
                lgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                lgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
            }
        }

        int a2 = lgA - a0 - a1;
        int b2 = lgB - b0 - b1;
        int c0 = sz1 - a0 - b0;
        int c1 = sz2 - a1 - b1;
        int c2 = sz3 - a2 - b2;

        if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

        ACC a[3] = {(ACC)a0, (ACC)a1, (ACC)a2};
        ACC b[3] = {(ACC)b0, (ACC)b1, (ACC)b2};
        ACC c[3] = {(ACC)c0, (ACC)c1, (ACC)c2};

        ACC twoQI = (ACC) 0;
        #pragma unroll
        for (int p = 0; p < 6; p++) {
            ACC ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
            ACC su = ai + bj + ck - 3;
            if (su > 0) twoQI += ai * bj * ck * su;
        }
        twoScore += (ACC) freq * twoQI;
    }

    // Polytomy (d>3) parts — two-pass-rewalk O(d) QI.
    if (numPolyParts > 0)
        twoScore += ssScorePoly<ACC, CACHE_ROWS>(
            loTree, loLeft, loRight, loComp, sizeA, aRngOff, aRngCnt,
            hiTree, hiLeft, hiRight, hiComp, sizeB, bRngOff, bRngCnt,
            rangeData, ssPolyMeta, ssPolyBoundOffset, ssPolyBounds,
            orderings, invIndex, numPolyParts, numTaxa, totalN);

    storeTwoScore(twoScores, idx, twoScore);

    // Warp-aggregated progress bump: one atomic per warp (counts its active lanes).
    if (dProgress) {
        unsigned act = __activemask();
        if ((threadIdx.x & 31) == (__ffs(act) - 1)) atomicAdd(dProgress, __popc(act));
    }
}

// INT128 variant of the smaller-side kernel (one thread per split, exact 128-bit).
template<bool CACHE_ROWS>
__global__ void computeWeightsSmallerSideKernelI128(
    const int* __restrict__ splits,
    const int* __restrict__ splitRangeMeta,
    const int* __restrict__ rangeData,
    const int* __restrict__ parts,
    const int* __restrict__ ssPolyMeta,
    const int* __restrict__ ssPolyBoundOffset,
    const int* __restrict__ ssPolyBounds,
    const int* __restrict__ orderings,
    const int* __restrict__ invIndex,
    int curBatch,
    int numParts,
    int numPolyParts,
    int numTaxa,
    int totalN,
    long long* __restrict__ twoScores,
    int* __restrict__ dProgress)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curBatch) return;

    const int* sp = splits + (size_t)idx * 10;
    int loTree = sp[0], loLeft = sp[1], loRight = sp[2], loComp = sp[3], sizeA = sp[4];
    int hiTree = sp[5], hiLeft = sp[6], hiRight = sp[7], hiComp = sp[8], sizeB = sp[9];
    const int* rm = splitRangeMeta + (size_t)idx * 4;
    int aRngOff = rm[0], aRngCnt = rm[1], bRngOff = rm[2], bRngCnt = rm[3];

    int sizeC = totalN - sizeA - sizeB;
    if (sizeC < 0) { storeTwoScoreI128(twoScores, idx, i128_zero()); return; }

    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    I128 twoScore = i128_zero();
    int cachedTree = -1, cachedLgA = 0, cachedLgB = 0;

    for (int j = 0; j < numParts; j++) {
        const int* pt = parts + (size_t)j * 9;
        int tGT = pt[0];
        int lo1 = pt[1], hi1 = pt[2];
        int lo2 = pt[3], hi2 = pt[4];
        int sz1 = pt[5], sz2 = pt[6], sz3 = pt[7];
        int freq = pt[8];

        int a0 = ssIntersectSide(tGT, lo1, hi1, loTree, loLeft, loRight, loComp, sz1, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
        int a1 = ssIntersectSide(tGT, lo2, hi2, loTree, loLeft, loRight, loComp, sz2, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
        int b0 = ssIntersectSide(tGT, lo1, hi1, hiTree, hiLeft, hiRight, hiComp, sz1, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
        int b1 = ssIntersectSide(tGT, lo2, hi2, hiTree, hiLeft, hiRight, hiComp, sz2, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);

        int L_GT = sz1 + sz2 + sz3;
        int lgA, lgB;
        if (L_GT == totalN) {
            lgA = sizeA;
            lgB = sizeB;
        } else {
            if (CACHE_ROWS && tGT != cachedTree) {
                cachedLgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedLgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
                cachedTree = tGT;
            }
            if (CACHE_ROWS) { lgA = cachedLgA; lgB = cachedLgB; }
            else {
                lgA = ssRowSum(tGT, L_GT, loTree, loLeft, loRight, loComp, aRngOff, aRngCnt, rangeData, orderings, invIndex, numTaxa);
                lgB = ssRowSum(tGT, L_GT, hiTree, hiLeft, hiRight, hiComp, bRngOff, bRngCnt, rangeData, orderings, invIndex, numTaxa);
            }
        }

        int a2 = lgA - a0 - a1;
        int b2 = lgB - b0 - b1;
        int c0 = sz1 - a0 - b0;
        int c1 = sz2 - a1 - b1;
        int c2 = sz3 - a2 - b2;

        if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

        long long a[3] = {a0, a1, a2};
        long long b[3] = {b0, b1, b2};
        long long c[3] = {c0, c1, c2};

        I128 twoQI = i128_zero();
        #pragma unroll
        for (int p = 0; p < 6; p++) {
            long long ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
            long long su = ai + bj + ck - 3;
            if (su > 0) {
                long long abc = ai * bj * ck;
                twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long)abc,
                                                     (unsigned long long)su));
            }
        }
        twoScore = i128_add(twoScore, i128_mul_scalar(twoQI, (unsigned long long) freq));
    }

    // Polytomy (d>3) parts — two-pass-rewalk O(d) QI (exact 128-bit).
    if (numPolyParts > 0)
        twoScore = i128_add(twoScore, ssScorePolyI128<CACHE_ROWS>(
            loTree, loLeft, loRight, loComp, sizeA, aRngOff, aRngCnt,
            hiTree, hiLeft, hiRight, hiComp, sizeB, bRngOff, bRngCnt,
            rangeData, ssPolyMeta, ssPolyBoundOffset, ssPolyBounds,
            orderings, invIndex, numPolyParts, numTaxa, totalN));

    storeTwoScoreI128(twoScores, idx, twoScore);

    // Warp-aggregated progress bump: one atomic per warp (counts its active lanes).
    if (dProgress) {
        unsigned act = __activemask();
        if ((threadIdx.x & 31) == (__ffs(act) - 1)) atomicAdd(dProgress, __popc(act));
    }
}

// ===========================================================================
// BITSET path (low-taxa fast option).
//
// Every cluster and every gene-tree part is a global-taxon bitset of W = ceil(n/64)
// 64-bit words.  One thread per split loads its A/B cluster bitsets from the resident
// pool (indexed by a per-split cluster id) and, for each part, computes each core
// intersection as popcount(A & M) over W words.  No orderings/invIndex/prefix arrays
// are used in the score loop.  The 3×3 derivation, poly two-pass formula, and QI math
// are identical to the other kernels, so scores are bit-identical.
//
// Layouts:
//   splits[idx*4]        = {aCid, bCid, aSize, bSize}   (per-batch, streamed)
//   clusterBits[cid*W]   = W-word global-taxon set       (resident; cid 0 = empty)
//   partM1/partM2[j*W]   = binary part child bitsets      (resident)
//   partMeta[j*5]        = {lgTree, sz1, sz2, sz3, freq}
//   geneLgBits[g*W]      = gene-tree present-taxa mask     (resident)
//   polyMeta[pn*5]       = {lgTree, d, lastSize, freq, L_GT}
//   polyChildOffset[pn]  = CSR row pointer into polyChildBits/polyChildSize
//   polyChildBits[c*W]   = poly child bitset;  polyChildSize[c] = |child|
// ===========================================================================

__device__ inline int bs_popAnd(const unsigned long long* __restrict__ x,
                                 const unsigned long long* __restrict__ y, int W) {
    int c = 0;
    for (int k = 0; k < W; k++) c += __popcll(x[k] & y[k]);
    return c;
}

// Gather candidate A/B bitsets into a warp-coalesced per-batch layout:
//   out[(side*W + word)*batch + split].
// All one-thread-per-split kernels then read adjacent addresses across a warp.
__global__ void gatherCandidateBits(
    const int* __restrict__ splits,
    const unsigned long long* __restrict__ clusterBits,
    unsigned long long* __restrict__ out,
    int batch, int W)
{
    size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * W;
    if (q >= total) return;
    size_t word = q / batch;
    int split = (int)(q - word * batch);
    const int* sp = splits + (size_t)split * 4;
    out[(size_t)word * batch + split] =
        clusterBits[(size_t)sp[0] * W + word];
    out[((size_t)W + word) * batch + split] =
        clusterBits[(size_t)sp[1] * W + word];
}

__device__ inline int bs_popAndCandidate(
    const unsigned long long* __restrict__ candidate,
    const unsigned long long* __restrict__ mask,
    int W, int candidateStride)
{
    int c = 0;
    for (int k = 0; k < W; k++)
        c += __popcll(candidate[(size_t)k * candidateStride] & mask[k]);
    return c;
}

template<typename ACC>
__global__ void computeWeightsBitsetKernel(
    const int* __restrict__ splits,                       // curBatch * 4
    const unsigned long long* __restrict__ clusterBits,
    const unsigned long long* __restrict__ partM1,
    const unsigned long long* __restrict__ partM2,
    const int* __restrict__ partMeta,                     // numParts * 5
    const unsigned long long* __restrict__ geneLgBits,
    const int* __restrict__ polyMeta,                     // numPoly * 5
    const int* __restrict__ polyChildOffset,              // numPoly + 1
    const unsigned long long* __restrict__ polyChildBits,
    const int* __restrict__ polyChildSize,
    int curBatch, int numParts, int numPoly, int W, int totalN,
    long long* __restrict__ twoScores, int* __restrict__ dProgress)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curBatch) return;

    const int* sp = splits + (size_t)idx * 4;
    int aCid = sp[0], bCid = sp[1], aSize = sp[2], bSize = sp[3];
    if (totalN - aSize - bSize < 0) { twoScores[idx] = 0LL; return; }

    const unsigned long long* A = clusterBits + (size_t)aCid * W;
    const unsigned long long* B = clusterBits + (size_t)bCid * W;

    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    ACC twoScore = (ACC) 0;

    for (int j = 0; j < numParts; j++) {
        const unsigned long long* M1 = partM1 + (size_t)j * W;
        const unsigned long long* M2 = partM2 + (size_t)j * W;
        const int* pm = partMeta + (size_t)j * 5;
        int lgTree = pm[0], sz1 = pm[1], sz2 = pm[2], sz3 = pm[3], freq = pm[4];

        int a0 = bs_popAnd(A, M1, W);
        int a1 = bs_popAnd(A, M2, W);
        int b0 = bs_popAnd(B, M1, W);
        int b1 = bs_popAnd(B, M2, W);

        int L_GT = sz1 + sz2 + sz3, lgA, lgB;
        if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
        else { const unsigned long long* Lg = geneLgBits + (size_t)lgTree * W;
               lgA = bs_popAnd(A, Lg, W); lgB = bs_popAnd(B, Lg, W); }

        int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1;
        int c0 = sz1 - a0 - b0, c1 = sz2 - a1 - b1, c2 = sz3 - a2 - b2;
        if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

        ACC a[3] = {(ACC)a0, (ACC)a1, (ACC)a2};
        ACC b[3] = {(ACC)b0, (ACC)b1, (ACC)b2};
        ACC c[3] = {(ACC)c0, (ACC)c1, (ACC)c2};
        ACC twoQI = (ACC) 0;
        #pragma unroll
        for (int p = 0; p < 6; p++) {
            ACC ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
            ACC su = ai + bj + ck - 3;
            if (su > 0) twoQI += ai * bj * ck * su;
        }
        twoScore += (ACC) freq * twoQI;
    }

    // Polytomy (d>3) parts — two-pass-rewalk O(d) QI (reuses A/B bitsets).
    for (int pn = 0; pn < numPoly; pn++) {
        const int* pm = polyMeta + (size_t)pn * 5;
        int lgTree = pm[0], d = pm[1], lastSize = pm[2], freq = pm[3], L_GT = pm[4];
        int cbeg = polyChildOffset[pn];

        ACC Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        int sumA = 0, sumB = 0;
        for (int i = 0; i < d - 1; i++) {
            const unsigned long long* Mi = polyChildBits + (size_t)(cbeg + i) * W;
            int szi = polyChildSize[cbeg + i];
            int ai = bs_popAnd(A, Mi, W), bi = bs_popAnd(B, Mi, W), ci = szi - ai - bi;
            Sa += ai; Sb += bi; Sc += ci;
            Sab += (ACC) ai * bi; Sac += (ACC) ai * ci; Sbc += (ACC) bi * ci;
            sumA += ai; sumB += bi;
        }
        int lgA, lgB;
        if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
        else { const unsigned long long* Lg = geneLgBits + (size_t)lgTree * W;
               lgA = bs_popAnd(A, Lg, W); lgB = bs_popAnd(B, Lg, W); }
        int aC = lgA - sumA, bC = lgB - sumB, cC = lastSize - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) continue;
        Sa += aC; Sb += bC; Sc += cC;
        Sab += (ACC) aC * bC; Sac += (ACC) aC * cC; Sbc += (ACC) bC * cC;

        ACC twoQI = (ACC) 0;
        for (int i = 0; i < d; i++) {
            ACC ai, bi, ci;
            if (i < d - 1) {
                const unsigned long long* Mi = polyChildBits + (size_t)(cbeg + i) * W;
                int szi = polyChildSize[cbeg + i];
                int aii = bs_popAnd(A, Mi, W), bii = bs_popAnd(B, Mi, W);
                ai = aii; bi = bii; ci = szi - aii - bii;
            } else { ai = aC; bi = bC; ci = cC; }
            twoQI += ai * (ai - 1) * ((Sb - bi) * (Sc - ci) - Sbc + bi * ci);
            twoQI += bi * (bi - 1) * ((Sa - ai) * (Sc - ci) - Sac + ai * ci);
            twoQI += ci * (ci - 1) * ((Sa - ai) * (Sb - bi) - Sab + ai * bi);
        }
        twoScore += (ACC) freq * twoQI;
    }

    storeTwoScore(twoScores, idx, twoScore);

    if (dProgress) {
        unsigned act = __activemask();
        if ((threadIdx.x & 31) == (__ffs(act) - 1)) atomicAdd(dProgress, __popc(act));
    }
}

// INT128 twin of the bitset kernel (one thread per split, exact 128-bit).
__global__ void computeWeightsBitsetKernelI128(
    const int* __restrict__ splits,
    const unsigned long long* __restrict__ clusterBits,
    const unsigned long long* __restrict__ partM1,
    const unsigned long long* __restrict__ partM2,
    const int* __restrict__ partMeta,
    const unsigned long long* __restrict__ geneLgBits,
    const int* __restrict__ polyMeta,
    const int* __restrict__ polyChildOffset,
    const unsigned long long* __restrict__ polyChildBits,
    const int* __restrict__ polyChildSize,
    int curBatch, int numParts, int numPoly, int W, int totalN,
    long long* __restrict__ twoScores, int* __restrict__ dProgress)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curBatch) return;

    const int* sp = splits + (size_t)idx * 4;
    int aCid = sp[0], bCid = sp[1], aSize = sp[2], bSize = sp[3];
    if (totalN - aSize - bSize < 0) { storeTwoScoreI128(twoScores, idx, i128_zero()); return; }

    const unsigned long long* A = clusterBits + (size_t)aCid * W;
    const unsigned long long* B = clusterBits + (size_t)bCid * W;

    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    I128 twoScore = i128_zero();

    for (int j = 0; j < numParts; j++) {
        const unsigned long long* M1 = partM1 + (size_t)j * W;
        const unsigned long long* M2 = partM2 + (size_t)j * W;
        const int* pm = partMeta + (size_t)j * 5;
        int lgTree = pm[0], sz1 = pm[1], sz2 = pm[2], sz3 = pm[3], freq = pm[4];

        int a0 = bs_popAnd(A, M1, W);
        int a1 = bs_popAnd(A, M2, W);
        int b0 = bs_popAnd(B, M1, W);
        int b1 = bs_popAnd(B, M2, W);

        int L_GT = sz1 + sz2 + sz3, lgA, lgB;
        if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
        else { const unsigned long long* Lg = geneLgBits + (size_t)lgTree * W;
               lgA = bs_popAnd(A, Lg, W); lgB = bs_popAnd(B, Lg, W); }

        int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1;
        int c0 = sz1 - a0 - b0, c1 = sz2 - a1 - b1, c2 = sz3 - a2 - b2;
        if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

        long long a[3] = {a0, a1, a2};
        long long b[3] = {b0, b1, b2};
        long long c[3] = {c0, c1, c2};
        I128 twoQI = i128_zero();
        #pragma unroll
        for (int p = 0; p < 6; p++) {
            long long ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
            long long su = ai + bj + ck - 3;
            if (su > 0) {
                long long abc = ai * bj * ck;
                twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) abc, (unsigned long long) su));
            }
        }
        twoScore = i128_add(twoScore, i128_mul_scalar(twoQI, (unsigned long long) freq));
    }

    for (int pn = 0; pn < numPoly; pn++) {
        const int* pm = polyMeta + (size_t)pn * 5;
        int lgTree = pm[0], d = pm[1], lastSize = pm[2], freq = pm[3], L_GT = pm[4];
        int cbeg = polyChildOffset[pn];

        long long Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        int sumA = 0, sumB = 0;
        for (int i = 0; i < d - 1; i++) {
            const unsigned long long* Mi = polyChildBits + (size_t)(cbeg + i) * W;
            int szi = polyChildSize[cbeg + i];
            int ai = bs_popAnd(A, Mi, W), bi = bs_popAnd(B, Mi, W), ci = szi - ai - bi;
            Sa += ai; Sb += bi; Sc += ci;
            Sab += (long long) ai * bi; Sac += (long long) ai * ci; Sbc += (long long) bi * ci;
            sumA += ai; sumB += bi;
        }
        int lgA, lgB;
        if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
        else { const unsigned long long* Lg = geneLgBits + (size_t)lgTree * W;
               lgA = bs_popAnd(A, Lg, W); lgB = bs_popAnd(B, Lg, W); }
        int aC = lgA - sumA, bC = lgB - sumB, cC = lastSize - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) continue;
        Sa += aC; Sb += bC; Sc += cC;
        Sab += (long long) aC * bC; Sac += (long long) aC * cC; Sbc += (long long) bC * cC;

        I128 twoQI = i128_zero();
        for (int i = 0; i < d; i++) {
            long long ai, bi, ci;
            if (i < d - 1) {
                const unsigned long long* Mi = polyChildBits + (size_t)(cbeg + i) * W;
                int szi = polyChildSize[cbeg + i];
                int aii = bs_popAnd(A, Mi, W), bii = bs_popAnd(B, Mi, W);
                ai = aii; bi = bii; ci = szi - aii - bii;
            } else { ai = aC; bi = bC; ci = cC; }
            long long brA = (Sb - bi) * (Sc - ci) - Sbc + bi * ci;
            long long brB = (Sa - ai) * (Sc - ci) - Sac + ai * ci;
            long long brC = (Sa - ai) * (Sb - bi) - Sab + ai * bi;
            long long wA = ai * (ai - 1), wB = bi * (bi - 1), wC = ci * (ci - 1);
            if (wA > 0 && brA > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wA, (unsigned long long) brA));
            if (wB > 0 && brB > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wB, (unsigned long long) brB));
            if (wC > 0 && brC > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wC, (unsigned long long) brC));
        }
        twoScore = i128_add(twoScore, i128_mul_scalar(twoQI, (unsigned long long) freq));
    }

    storeTwoScoreI128(twoScores, idx, twoScore);

    if (dProgress) {
        unsigned act = __activemask();
        if ((threadIdx.x & 31) == (__ffs(act) - 1)) atomicAdd(dProgress, __popc(act));
    }
}

// ===========================================================================
// SIMPLE-TREE-WALK path (many-candidate fast option).
//
// One thread per split walks the resident flat postorder token stream of all gene
// trees sequentially, maintaining a small private stack of (nA,nB,nS) triples
// = (|node∩A|, |node∩B|, |node|).  Tokens: leaf = taxon id (>=0), internal =
// -childCount; the root is not emitted.  Every non-root internal node's
// tripartition is scored in O(1) from its children — same QI arithmetic as the
// other kernels, so bit-identical.  No prefix arrays, no dedup.  Per-tree lgA/lgB
// come from popcount(A/B & geneLgBits[g]); complete trees short-circuit to aSize/bSize.
// ===========================================================================

template<typename ACC, bool SMALL_QI, int STACK_CAP>
__global__ void computeWeightsTreeWalkKernel(
    const int* __restrict__ splits,                     // curBatch * 4
    const unsigned long long* __restrict__ candidateBits,
    const unsigned long long* __restrict__ geneLgBits,
    const int* __restrict__ nodeStream,
    const int* __restrict__ treeNodeOffset,
    const int* __restrict__ leafCount,
    int curBatch, int numTrees, int W, int totalN,
    long long* __restrict__ twoScores, int* __restrict__ dProgress)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curBatch) return;

    const int* sp = splits + (size_t)idx * 4;
    int aSize = sp[2], bSize = sp[3];
    if (totalN - aSize - bSize < 0) { twoScores[idx] = 0LL; return; }

    const unsigned long long* A = candidateBits + idx;
    const unsigned long long* B = candidateBits + (size_t)W * curBatch + idx;

    int stack[STACK_CAP * 3];
    ACC twoScore = (ACC) 0;

    for (int g = 0; g < numTrees; g++) {
        int segBeg = treeNodeOffset[g], segEnd = treeNodeOffset[g + 1];
        if (segBeg == segEnd) continue;
        int LgSize = leafCount[g];
        int lgA, lgB;
        if (LgSize == totalN) { lgA = aSize; lgB = bSize; }
        else { const unsigned long long* Lg = geneLgBits + (size_t)g * W;
               lgA = bs_popAndCandidate(A, Lg, W, curBatch);
               lgB = bs_popAndCandidate(B, Lg, W, curBatch); }

        int top = 0;
        for (int i = segBeg; i < segEnd; i++) {
            int tok = nodeStream[i];
            if (tok >= 0) {                                   // leaf
                int inA = (int)((A[(size_t)(tok >> 6) * curBatch] >> (tok & 63)) & 1ULL);
                int inB = (int)((B[(size_t)(tok >> 6) * curBatch] >> (tok & 63)) & 1ULL);
                int e = top * 3;
                stack[e] = inA; stack[e + 1] = inB; stack[e + 2] = 1;
                top++;
            } else {                                          // internal (k children)
                int k = -tok;
                int cbase = top - k;
                if (k == 2) {
                    int e0 = cbase * 3;
                    int a0 = stack[e0],     b0 = stack[e0 + 1], s0 = stack[e0 + 2];
                    int a1 = stack[e0 + 3], b1 = stack[e0 + 4], s1 = stack[e0 + 5];
                    int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1;
                    int sz3 = LgSize - s0 - s1;
                    int c0 = s0 - a0 - b0, c1 = s1 - a1 - b1, c2 = sz3 - a2 - b2;
                    if (!(a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0)) {
                        ACC twoQI = binaryTwoQI<ACC, SMALL_QI>(
                            a0, a1, a2, b0, b1, b2, c0, c1, c2);
                        twoScore += twoQI;
                    }
                    stack[e0] = a0 + a1; stack[e0 + 1] = b0 + b1; stack[e0 + 2] = s0 + s1;
                    top = cbase + 1;
                } else {                                       // polytomy k>=3
                    ACC Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
                    int sumA = 0, sumB = 0, sumS = 0;
                    for (int j = 0; j < k; j++) {
                        int e = (cbase + j) * 3;
                        int aj = stack[e], bj = stack[e + 1], sj = stack[e + 2];
                        int cj = sj - aj - bj;
                        Sa += aj; Sb += bj; Sc += cj;
                        Sab += (ACC) aj * bj; Sac += (ACC) aj * cj; Sbc += (ACC) bj * cj;
                        sumA += aj; sumB += bj; sumS += sj;
                    }
                    int aC = lgA - sumA, bC = lgB - sumB, szC = LgSize - sumS, cC = szC - aC - bC;
                    if (!(aC < 0 || bC < 0 || cC < 0)) {
                        Sa += aC; Sb += bC; Sc += cC;
                        Sab += (ACC) aC * bC; Sac += (ACC) aC * cC; Sbc += (ACC) bC * cC;
                        ACC twoQI = (ACC) 0;
                        for (int j = 0; j < k; j++) {
                            int e = (cbase + j) * 3;
                            ACC aj = stack[e], bj = stack[e + 1], cj = (ACC)(stack[e + 2] - stack[e] - stack[e + 1]);
                            twoQI += aj * (aj - 1) * ((Sb - bj) * (Sc - cj) - Sbc + bj * cj);
                            twoQI += bj * (bj - 1) * ((Sa - aj) * (Sc - cj) - Sac + aj * cj);
                            twoQI += cj * (cj - 1) * ((Sa - aj) * (Sb - bj) - Sab + aj * bj);
                        }
                        { ACC aj = aC, bj = bC, cj = cC;
                          twoQI += aj * (aj - 1) * ((Sb - bj) * (Sc - cj) - Sbc + bj * cj);
                          twoQI += bj * (bj - 1) * ((Sa - aj) * (Sc - cj) - Sac + aj * cj);
                          twoQI += cj * (cj - 1) * ((Sa - aj) * (Sb - bj) - Sab + aj * bj); }
                        twoScore += twoQI;
                    }
                    stack[cbase * 3] = sumA; stack[cbase * 3 + 1] = sumB; stack[cbase * 3 + 2] = sumS;
                    top = cbase + 1;
                }
            }
        }
    }

    storeTwoScore(twoScores, idx, twoScore);
    if (dProgress) {
        unsigned act = __activemask();
        if ((threadIdx.x & 31) == (__ffs(act) - 1)) atomicAdd(dProgress, __popc(act));
    }
}

// INT128 twin of the tree-walk kernel (one thread per split, exact 128-bit).
template<int STACK_CAP>
__global__ void computeWeightsTreeWalkKernelI128(
    const int* __restrict__ splits,
    const unsigned long long* __restrict__ candidateBits,
    const unsigned long long* __restrict__ geneLgBits,
    const int* __restrict__ nodeStream,
    const int* __restrict__ treeNodeOffset,
    const int* __restrict__ leafCount,
    int curBatch, int numTrees, int W, int totalN,
    long long* __restrict__ twoScores, int* __restrict__ dProgress)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curBatch) return;

    const int* sp = splits + (size_t)idx * 4;
    int aSize = sp[2], bSize = sp[3];
    if (totalN - aSize - bSize < 0) { storeTwoScoreI128(twoScores, idx, i128_zero()); return; }

    const unsigned long long* A = candidateBits + idx;
    const unsigned long long* B = candidateBits + (size_t)W * curBatch + idx;

    const int PI[6] = {0, 0, 1, 1, 2, 2};
    const int PJ[6] = {1, 2, 0, 2, 0, 1};
    const int PK[6] = {2, 1, 2, 0, 1, 0};

    int stack[STACK_CAP * 3];
    I128 twoScore = i128_zero();

    for (int g = 0; g < numTrees; g++) {
        int segBeg = treeNodeOffset[g], segEnd = treeNodeOffset[g + 1];
        if (segBeg == segEnd) continue;
        int LgSize = leafCount[g];
        int lgA, lgB;
        if (LgSize == totalN) { lgA = aSize; lgB = bSize; }
        else { const unsigned long long* Lg = geneLgBits + (size_t)g * W;
               lgA = bs_popAndCandidate(A, Lg, W, curBatch);
               lgB = bs_popAndCandidate(B, Lg, W, curBatch); }

        int top = 0;
        for (int i = segBeg; i < segEnd; i++) {
            int tok = nodeStream[i];
            if (tok >= 0) {                                   // leaf
                int inA = (int)((A[(size_t)(tok >> 6) * curBatch] >> (tok & 63)) & 1ULL);
                int inB = (int)((B[(size_t)(tok >> 6) * curBatch] >> (tok & 63)) & 1ULL);
                int e = top * 3;
                stack[e] = inA; stack[e + 1] = inB; stack[e + 2] = 1;
                top++;
            } else {
                int k = -tok;
                int cbase = top - k;
                if (k == 2) {
                    int e0 = cbase * 3;
                    int a0 = stack[e0],     b0 = stack[e0 + 1], s0 = stack[e0 + 2];
                    int a1 = stack[e0 + 3], b1 = stack[e0 + 4], s1 = stack[e0 + 5];
                    int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1;
                    int sz3 = LgSize - s0 - s1;
                    int c0 = s0 - a0 - b0, c1 = s1 - a1 - b1, c2 = sz3 - a2 - b2;
                    if (!(a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0)) {
                        long long a[3] = {a0, a1, a2};
                        long long b[3] = {b0, b1, b2};
                        long long c[3] = {c0, c1, c2};
                        I128 twoQI = i128_zero();
                        #pragma unroll
                        for (int p = 0; p < 6; p++) {
                            long long ai = a[PI[p]], bj = b[PJ[p]], ck = c[PK[p]];
                            long long su = ai + bj + ck - 3;
                            if (su > 0) {
                                long long abc = ai * bj * ck;
                                twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long)abc, (unsigned long long)su));
                            }
                        }
                        twoScore = i128_add(twoScore, twoQI);
                    }
                    stack[e0] = a0 + a1; stack[e0 + 1] = b0 + b1; stack[e0 + 2] = s0 + s1;
                    top = cbase + 1;
                } else {
                    long long Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
                    int sumA = 0, sumB = 0, sumS = 0;
                    for (int j = 0; j < k; j++) {
                        int e = (cbase + j) * 3;
                        int aj = stack[e], bj = stack[e + 1], sj = stack[e + 2];
                        int cj = sj - aj - bj;
                        Sa += aj; Sb += bj; Sc += cj;
                        Sab += (long long) aj * bj; Sac += (long long) aj * cj; Sbc += (long long) bj * cj;
                        sumA += aj; sumB += bj; sumS += sj;
                    }
                    int aC = lgA - sumA, bC = lgB - sumB, szC = LgSize - sumS, cC = szC - aC - bC;
                    if (!(aC < 0 || bC < 0 || cC < 0)) {
                        Sa += aC; Sb += bC; Sc += cC;
                        Sab += (long long) aC * bC; Sac += (long long) aC * cC; Sbc += (long long) bC * cC;
                        I128 twoQI = i128_zero();
                        for (int j = 0; j <= k; j++) {
                            long long aj, bj, cj;
                            if (j < k) { int e = (cbase + j) * 3;
                                         aj = stack[e]; bj = stack[e + 1]; cj = (long long)stack[e + 2] - aj - bj; }
                            else       { aj = aC; bj = bC; cj = cC; }
                            long long brA = (Sb - bj) * (Sc - cj) - Sbc + bj * cj;
                            long long brB = (Sa - aj) * (Sc - cj) - Sac + aj * cj;
                            long long brC = (Sa - aj) * (Sb - bj) - Sab + aj * bj;
                            long long wA = aj * (aj - 1), wB = bj * (bj - 1), wC = cj * (cj - 1);
                            if (wA > 0 && brA > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wA, (unsigned long long) brA));
                            if (wB > 0 && brB > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wB, (unsigned long long) brB));
                            if (wC > 0 && brC > 0) twoQI = i128_add(twoQI, i128_mul_u64((unsigned long long) wC, (unsigned long long) brC));
                        }
                        twoScore = i128_add(twoScore, twoQI);
                    }
                    stack[cbase * 3] = sumA; stack[cbase * 3 + 1] = sumB; stack[cbase * 3 + 2] = sumS;
                    top = cbase + 1;
                }
            }
        }
    }

    storeTwoScoreI128(twoScores, idx, twoScore);
    if (dProgress) {
        unsigned act = __activemask();
        if ((threadIdx.x & 31) == (__ffs(act) - 1)) atomicAdd(dProgress, __popc(act));
    }
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
// Intra-kernel progress: poll a device counter while a kernel runs on kStream,
// printing a time-paced single line.  The kernel bumps *dProgress once per split
// it finishes (negligible cost, no change to grid/occupancy); we read it on a
// SEPARATE stream so the poll never stalls the kernel (both must be non-default,
// since the legacy default stream implicitly synchronizes with all streams).
//
// Output: a single carriage-return-overwritten, colorized line (only the latest
// update stays on screen; collapses on a terminal even through `tee`).  Coloured by
// default ([GPU] green, count cyan, percent yellow); set NO_COLOR to disable.
// Cadence default: 1 s; override precedence is
// --gpu-progress-interval > ASTRALX_GPU_PROGRESS_SEC > default.  Returns the kernel's
// terminal cudaStreamQuery status (cudaSuccess once finished).
// ---------------------------------------------------------------------------
static cudaError_t wb_poll_progress(cudaStream_t kStream, cudaStream_t pollStream,
                                    const int* dProgress, int* hPinned, int total,
                                    const char* label, double flagSec) {
    double interval = 1.0;                                     // default report cadence (s)
    const char* ev = getenv("ASTRALX_GPU_PROGRESS_SEC");       // env override
    if (ev) { double v = atof(ev); if (v > 0.0) interval = v; }
    if (flagSec > 0.0) interval = flagSec;                     // --gpu-progress-interval wins

    bool col = (getenv("NO_COLOR") == NULL);     // progress is colorized by default
    const char* GRN = col ? "\033[32m" : "";     // [GPU] + bar
    const char* CYN = col ? "\033[36m" : "";     // done/total
    const char* YEL = col ? "\033[33m" : "";     // percent
    const char* DIM = col ? "\033[2m"  : "";     // elapsed / ETA
    const char* RST = col ? "\033[0m"  : "";
    char bar[WB_BAR_W * 3 + 1];
    double t0 = wb_now_sec();
    double lastPrint = t0;
    bool   printed = false;

    while (true) {
        cudaError_t q = cudaStreamQuery(kStream);
        if (q != cudaErrorNotReady) {                 // finished (or error)
            if (printed) { fprintf(stderr, "\n"); fflush(stderr); }   // finalize the line
            return q;
        }
        struct timespec ts = { 0, 100L * 1000L * 1000L };  // 100 ms slice (responsive)
        nanosleep(&ts, NULL);

        double now = wb_now_sec();
        if (now - t0 < interval || now - lastPrint < interval) continue;
        lastPrint = now;

        *hPinned = 0;
        cudaMemcpyAsync(hPinned, dProgress, sizeof(int), cudaMemcpyDeviceToHost, pollStream);
        cudaStreamSynchronize(pollStream);
        int done = *hPinned;
        if (done < 0) done = 0;
        if (done > total) done = total;
        double frac    = (total > 0) ? (double) done / total : 0.0;
        double elapsed = now - t0;
        double eta     = (frac > 1e-6) ? elapsed * (1.0 - frac) / frac : 0.0;
        char eb[32], etb[32];
        wb_fmt_duration(elapsed, eb, sizeof eb);
        wb_fmt_duration(eta,     etb, sizeof etb);
        wb_build_bar(bar, done, total);
        // Single carriage-return-overwritten line (keeps only the latest update on
        // screen — collapses cleanly on a terminal, even through `tee`).
        fprintf(stderr,
                "\r  %s[GPU]%s %s  %s[%s]%s  %s%d/%d%s (%s%.1f%%%s)  %s%s elapsed · ETA %s%s    ",
                GRN, RST, label, GRN, bar, RST,
                CYN, done, total, RST, YEL, frac * 100.0, RST,
                DIM, eb, etb, RST);
        fflush(stderr);
        printed = true;
    }
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
    jintArray jSplits, jintArray jSplitRangeMeta, jintArray jRangeData,
    jintArray jNodeData, jintArray jNodeFreq, jintArray jNodeOffset,
    jintArray jPartLeafCount,
    jintArray jPolyTreeOffset, jintArray jPolyBoundOffset, jintArray jPolyBounds, jintArray jPolyFreq,
    jintArray jOrderings, jintArray jInvIndex,
    jint numSplits, jint numPartTrees, jint partTreeOffset, jint maxLeafCount,
    jint numGpuTrees, jint numTaxa,
    jint batchSizeHint, jdouble vramFraction, jint scoreMode, jdouble progressIntervalSec)
{
    // scoreMode: 0 = LONG (exact int64), 1 = DOUBLE (bit-packed), 2 = INT128 (2 longs/split)
    bool useDouble = (scoreMode == 1);
    bool useI128   = (scoreMode == 2);
    int  scoresPerSplit = useI128 ? 2 : 1;   // INT128 transports two longs per split
    fprintf(stderr, "[ASTRAL-X GPU] weight accumulator: %s\n",
            useI128   ? "INT128 (exact 128-bit integer)"
          : useDouble ? "DOUBLE (64-bit float, overflow-safe)"
                      : "LONG (exact 64-bit integer)");
    // -------------------------------------------------------------------------
    // Pin host arrays
    // -------------------------------------------------------------------------
    jint* hSplits        = env->GetIntArrayElements(jSplits,        NULL);
    jint* hSplitRangeMeta= env->GetIntArrayElements(jSplitRangeMeta,NULL);
    jint* hRangeData     = env->GetIntArrayElements(jRangeData,     NULL);
    jint* hNodeData      = env->GetIntArrayElements(jNodeData,      NULL);
    jint* hNodeFreq      = env->GetIntArrayElements(jNodeFreq,      NULL);
    jint* hNodeOffset    = env->GetIntArrayElements(jNodeOffset,    NULL);
    jint* hPartLeafCount = env->GetIntArrayElements(jPartLeafCount, NULL);
    jint* hPolyTreeOffset= env->GetIntArrayElements(jPolyTreeOffset,NULL);
    jint* hPolyBoundOffset=env->GetIntArrayElements(jPolyBoundOffset,NULL);
    jint* hPolyBounds    = env->GetIntArrayElements(jPolyBounds,    NULL);
    jint* hPolyFreq      = env->GetIntArrayElements(jPolyFreq,      NULL);
    jint* hOrderings     = env->GetIntArrayElements(jOrderings,     NULL);
    jint* hInvIndex      = env->GetIntArrayElements(jInvIndex,      NULL);

    jsize rangeDataLen = env->GetArrayLength(jRangeData);  // = 2 * (#multi-range ranges)
    jsize nodeDataLen = env->GetArrayLength(jNodeData);   // = totalNodes * 3
    jsize polyBoundsLen   = env->GetArrayLength(jPolyBounds);   // = Σ degree over poly nodes
    jsize numPolyNodes    = env->GetArrayLength(jPolyFreq);     // = #unique poly partitions

    // -------------------------------------------------------------------------
    // Adaptive mode selection: keep the fast shared-memory path whenever the two
    // prefix arrays fit a block's shared memory; otherwise spill them to a
    // bounded global-memory pool (large-L path).
    // -------------------------------------------------------------------------
    int    stride            = maxLeafCount + 1;
    size_t sharedBytesShared = ((size_t)2 * stride + 2 * WB_BLOCK) * sizeof(int); // pA+pB+warp scans
    size_t sharedBytesGlobal = (size_t)2 * WB_BLOCK * sizeof(int);                // warp scans only
    size_t redBytes          = (size_t)WB_BLOCK * sizeof(long long);          // static red[]

    int maxOptin = 0;
    cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    // Shared path must fit both the dynamic (pA,pB,scan) and the static red[].
    bool useShared = (sharedBytesShared + redBytes) <= (size_t)maxOptin;
    // Debug override: force the large-L global path even when shared would fit,
    // so the global path can be validated on small inputs.
    if (getenv("ASTRALX_WEIGHT_FORCE_GLOBAL")) useShared = false;
    size_t sharedBytes = useShared ? sharedBytesShared : sharedBytesGlobal;

    if (useShared && sharedBytesShared > 49152) {
        // Opt in to larger dynamic shared memory (default cap is 48 KB).
        // Set on whichever accumulator instantiation will actually launch.
        if (useI128)
            cudaFuncSetAttribute(computeWeightsKernelI128<false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sharedBytesShared);
        else if (useDouble)
            cudaFuncSetAttribute(computeWeightsKernel<false, double, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sharedBytesShared);
        else if (numTaxa <= WB_SMALL_QI_MAX_N)
            cudaFuncSetAttribute(computeWeightsKernel<false, long long, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sharedBytesShared);
        else
            cudaFuncSetAttribute(computeWeightsKernel<false, long long, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sharedBytesShared);
    }

    // -------------------------------------------------------------------------
    // Upload static data ONCE (orderings, invIndex, node CSR stay resident)
    // -------------------------------------------------------------------------
    int *dNodeData, *dNodeFreq, *dNodeOffset, *dPartLeafCount, *dOrderings, *dInvIndex;
    int *dRangeData;   // resident flat [lo,hi] pairs for multi-range split sides
    int *dPolyTreeOffset, *dPolyBoundOffset, *dPolyBounds, *dPolyFreq;  // polytomy CSR

    size_t nodeDataSz   = (size_t)nodeDataLen          * sizeof(int);
    size_t nodeFreqSz   = (size_t)(nodeDataLen / 3)    * sizeof(int);   // numUnique entries
    size_t nodeOffsetSz = (size_t)(numPartTrees + 1)   * sizeof(int);
    size_t partLeafSz   = (size_t)numPartTrees         * sizeof(int);
    size_t orderingSz   = (size_t)numGpuTrees * numTaxa * sizeof(int);
    // Guard empty (no multi-range clusters): allocate ≥1 int so cudaMalloc/pointer is valid.
    size_t rangeDataSz  = (size_t)(rangeDataLen > 0 ? rangeDataLen : 1) * sizeof(int);
    // Polytomy CSR sizes (all ≥1 for valid pointers; empty ⇒ kernel poly loop is a no-op).
    size_t polyTreeOffSz   = (size_t)(numPartTrees + 1)                         * sizeof(int);
    size_t polyBoundOffSz  = (size_t)(numPolyNodes + 1)                         * sizeof(int);
    size_t polyBoundsSz    = (size_t)(polyBoundsLen > 0 ? polyBoundsLen : 1)    * sizeof(int);
    size_t polyFreqSz      = (size_t)(numPolyNodes  > 0 ? numPolyNodes  : 1)    * sizeof(int);

    cudaMalloc(&dNodeData,      nodeDataSz);
    cudaMalloc(&dNodeFreq,      nodeFreqSz);
    cudaMalloc(&dNodeOffset,    nodeOffsetSz);
    cudaMalloc(&dPartLeafCount, partLeafSz);
    cudaMalloc(&dOrderings,     orderingSz);
    cudaMalloc(&dInvIndex,      orderingSz);
    cudaMalloc(&dRangeData,     rangeDataSz);
    cudaMalloc(&dPolyTreeOffset,  polyTreeOffSz);
    cudaMalloc(&dPolyBoundOffset, polyBoundOffSz);
    cudaMalloc(&dPolyBounds,      polyBoundsSz);
    cudaMalloc(&dPolyFreq,        polyFreqSz);

    cudaMemcpy(dNodeData,      hNodeData,      nodeDataSz,   cudaMemcpyHostToDevice);
    cudaMemcpy(dNodeFreq,      hNodeFreq,      nodeFreqSz,   cudaMemcpyHostToDevice);
    cudaMemcpy(dNodeOffset,    hNodeOffset,    nodeOffsetSz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPartLeafCount, hPartLeafCount, partLeafSz,   cudaMemcpyHostToDevice);
    cudaMemcpy(dOrderings,     hOrderings,     orderingSz,   cudaMemcpyHostToDevice);
    cudaMemcpy(dInvIndex,      hInvIndex,      orderingSz,   cudaMemcpyHostToDevice);
    if (rangeDataLen > 0)
        cudaMemcpy(dRangeData, hRangeData, (size_t)rangeDataLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyTreeOffset,  hPolyTreeOffset,  polyTreeOffSz,  cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyBoundOffset, hPolyBoundOffset, polyBoundOffSz, cudaMemcpyHostToDevice);
    if (polyBoundsLen > 0)
        cudaMemcpy(dPolyBounds, hPolyBounds, (size_t)polyBoundsLen * sizeof(int), cudaMemcpyHostToDevice);
    if (numPolyNodes > 0)
        cudaMemcpy(dPolyFreq,   hPolyFreq,   (size_t)numPolyNodes  * sizeof(int), cudaMemcpyHostToDevice);

    // Analytical VRAM budget: show exactly what is resident on-device
    {
        size_t staticTotal = nodeDataSz + nodeOffsetSz + partLeafSz + 2 * orderingSz;
        size_t freeAfterStatic = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeAfterStatic, &totalVRAM);
        fprintf(stderr,
            "[ASTRAL-X GPU] weight static data uploaded (prefix-sum tree-DP):\n"
            "  orderings   : %6.1f MB\n"
            "  invIndex    : %6.1f MB\n"
            "  nodeData    : %6.1f MB  (%d unique tripartitions × 3 ints + freq)\n"
            "  nodeOffset  : %6.1f MB\n"
            "  prefix mode : %s  (maxLeafCount=%d, shared/block=%.1f KB)\n"
            "  ─────────────────────\n"
            "  static total : %6.1f MB   (VRAM free after: %.1f MB / %.1f MB)\n",
            orderingSz / 1e6, orderingSz / 1e6,
            nodeDataSz / 1e6, nodeDataLen / 3,
            nodeOffsetSz / 1e6,
            useShared ? "SHARED" : "GLOBAL (large-L)", (int)maxLeafCount,
            sharedBytes / 1024.0,
            staticTotal / 1e6, freeAfterStatic / 1e6, totalVRAM / 1e6);
        fflush(stderr);
    }

    // -------------------------------------------------------------------------
    // Large-L path: allocate a bounded global prefix pool — one (2·stride) slot
    // per *resident* block (NOT per split), so memory stays O(residentBlocks·L).
    // Each resident block grid-strides over the splits, reusing its own slot.
    // -------------------------------------------------------------------------
    int*  dPrefix     = NULL;
    int   maxResident = 0;
    if (!useShared) {
        int numSM = 0, blocksPerSM = 0;
        cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
        if (useI128)
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocksPerSM, computeWeightsKernelI128<true>, WB_BLOCK, sharedBytesGlobal);
        else if (useDouble)
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocksPerSM, computeWeightsKernel<true, double, false>, WB_BLOCK, sharedBytesGlobal);
        else if (numTaxa <= WB_SMALL_QI_MAX_N)
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocksPerSM, computeWeightsKernel<true, long long, true>, WB_BLOCK, sharedBytesGlobal);
        else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocksPerSM, computeWeightsKernel<true, long long, false>, WB_BLOCK, sharedBytesGlobal);
        if (blocksPerSM < 1) blocksPerSM = 1;
        maxResident = numSM * blocksPerSM;
        if (maxResident < 1)         maxResident = 1;
        if (maxResident > numSplits) maxResident = numSplits;

        size_t slotInts = (size_t)2 * stride;
        while (maxResident > 0) {
            size_t poolSz = (size_t)maxResident * slotInts * sizeof(int);
            if (cudaMalloc(&dPrefix, poolSz) == cudaSuccess) break;
            dPrefix = NULL;
            maxResident /= 2;
        }
        if (dPrefix == NULL) {
            fprintf(stderr, "[ASTRAL-X GPU] weight: FATAL — cannot allocate global prefix pool\n");
            cudaFree(dNodeData); cudaFree(dNodeFreq); cudaFree(dNodeOffset);
            cudaFree(dPartLeafCount); cudaFree(dOrderings); cudaFree(dInvIndex);
            cudaFree(dRangeData);
            cudaFree(dPolyTreeOffset); cudaFree(dPolyBoundOffset); cudaFree(dPolyBounds); cudaFree(dPolyFreq);
            env->ReleaseIntArrayElements(jSplits,        hSplits,        JNI_ABORT);
            env->ReleaseIntArrayElements(jSplitRangeMeta,hSplitRangeMeta,JNI_ABORT);
            env->ReleaseIntArrayElements(jRangeData,     hRangeData,     JNI_ABORT);
            env->ReleaseIntArrayElements(jNodeData,      hNodeData,      JNI_ABORT);
            env->ReleaseIntArrayElements(jNodeFreq,      hNodeFreq,      JNI_ABORT);
            env->ReleaseIntArrayElements(jNodeOffset,    hNodeOffset,    JNI_ABORT);
            env->ReleaseIntArrayElements(jPartLeafCount, hPartLeafCount, JNI_ABORT);
            env->ReleaseIntArrayElements(jPolyTreeOffset, hPolyTreeOffset, JNI_ABORT);
            env->ReleaseIntArrayElements(jPolyBoundOffset,hPolyBoundOffset,JNI_ABORT);
            env->ReleaseIntArrayElements(jPolyBounds,    hPolyBounds,    JNI_ABORT);
            env->ReleaseIntArrayElements(jPolyFreq,      hPolyFreq,      JNI_ABORT);
            env->ReleaseIntArrayElements(jOrderings,     hOrderings,     JNI_ABORT);
            env->ReleaseIntArrayElements(jInvIndex,      hInvIndex,      JNI_ABORT);
            return NULL;   // truly infeasible → Java CPU fallback
        }
        fprintf(stderr,
            "[ASTRAL-X GPU] weight: GLOBAL prefix path — shared needed %.1f KB > %.1f KB cap; "
            "resident blocks=%d, global pool=%.1f MB\n",
            sharedBytesShared / 1024.0, maxOptin / 1024.0, maxResident,
            (double)maxResident * slotInts * sizeof(int) / 1e6);
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
        size_t perSplitBytes = 10 * sizeof(int) + scoresPerSplit * sizeof(long long);
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
    int*       dSplits        = NULL;
    int*       dSplitRangeMeta = NULL;   // batched: curBatch * 4
    long long* dTwoScores     = NULL;

    while (batchSize > 0) {
        size_t splitBufSz = (size_t)batchSize * 10 * sizeof(int);
        size_t metaBufSz  = (size_t)batchSize * 4  * sizeof(int);
        size_t scoreBufSz = (size_t)batchSize * scoresPerSplit * sizeof(long long);
        cudaError_t e1 = cudaMalloc(&dSplits,         splitBufSz);
        cudaError_t e2 = cudaMalloc(&dTwoScores,      scoreBufSz);
        cudaError_t e3 = cudaMalloc(&dSplitRangeMeta, metaBufSz);
        if (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess) break;
        if (dSplits)         { cudaFree(dSplits);         dSplits         = NULL; }
        if (dTwoScores)      { cudaFree(dTwoScores);      dTwoScores      = NULL; }
        if (dSplitRangeMeta) { cudaFree(dSplitRangeMeta); dSplitRangeMeta = NULL; }
        batchSize /= 2;
        fprintf(stderr, "[ASTRAL-X GPU] cudaMalloc failed, retrying with batchSize=%d\n",
                batchSize);
    }
    if (batchSize <= 0 || dSplits == NULL || dTwoScores == NULL) {
        fprintf(stderr, "[ASTRAL-X GPU] FATAL: cannot allocate GPU batch buffers\n");
        cudaFree(dNodeData); cudaFree(dNodeFreq); cudaFree(dNodeOffset);
        cudaFree(dPartLeafCount); cudaFree(dOrderings); cudaFree(dInvIndex);
        cudaFree(dRangeData); if (dSplitRangeMeta) cudaFree(dSplitRangeMeta);
        cudaFree(dPolyTreeOffset); cudaFree(dPolyBoundOffset); cudaFree(dPolyBounds); cudaFree(dPolyFreq);
        if (dPrefix) cudaFree(dPrefix);
        env->ReleaseIntArrayElements(jSplits,        hSplits,        JNI_ABORT);
        env->ReleaseIntArrayElements(jSplitRangeMeta,hSplitRangeMeta,JNI_ABORT);
        env->ReleaseIntArrayElements(jRangeData,     hRangeData,     JNI_ABORT);
        env->ReleaseIntArrayElements(jNodeData,      hNodeData,      JNI_ABORT);
        env->ReleaseIntArrayElements(jNodeFreq,      hNodeFreq,      JNI_ABORT);
        env->ReleaseIntArrayElements(jNodeOffset,    hNodeOffset,    JNI_ABORT);
        env->ReleaseIntArrayElements(jPartLeafCount, hPartLeafCount, JNI_ABORT);
        env->ReleaseIntArrayElements(jPolyTreeOffset, hPolyTreeOffset, JNI_ABORT);
        env->ReleaseIntArrayElements(jPolyBoundOffset,hPolyBoundOffset,JNI_ABORT);
        env->ReleaseIntArrayElements(jPolyBounds,    hPolyBounds,    JNI_ABORT);
        env->ReleaseIntArrayElements(jPolyFreq,      hPolyFreq,      JNI_ABORT);
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
    long long* hTwoScores = new long long[(size_t)numSplits * scoresPerSplit]();   // zero-initialised

    // Intra-kernel progress: a device splits-completed counter polled from the host.
    // Kernel runs on wbStream; the counter is read on a SEPARATE pollStream (both
    // non-default, so the poll never serializes with the kernel).
    cudaStream_t wbStream = 0, pollStream = 0;
    cudaStreamCreate(&wbStream);
    cudaStreamCreate(&pollStream);
    int* dProgress = NULL; int* hProgress = NULL;
    cudaMalloc(&dProgress, sizeof(int));
    cudaHostAlloc((void**)&hProgress, sizeof(int), cudaHostAllocDefault);

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
        cudaMemcpy(dSplitRangeMeta,
                   hSplitRangeMeta + (size_t)offset * 4,
                   (size_t)curBatch * 4 * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Reset the progress counter on wbStream (ordered before the kernel below).
        cudaMemsetAsync(dProgress, 0, sizeof(int), wbStream);

        if (useShared) {
            // Fast path: one block per split; prefix arrays in shared memory.
            if (useI128)
                computeWeightsKernelI128<false><<<curBatch, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    NULL, dTwoScores, dProgress);
            else if (useDouble)
                computeWeightsKernel<false, double, false><<<curBatch, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    NULL, dTwoScores, dProgress);
            else if (numTaxa <= WB_SMALL_QI_MAX_N)
                computeWeightsKernel<false, long long, true><<<curBatch, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    NULL, dTwoScores, dProgress);
            else
                computeWeightsKernel<false, long long, false><<<curBatch, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    NULL, dTwoScores, dProgress);
        } else {
            // Large-L path: resident-capped grid grid-strides over splits;
            // prefix arrays in the bounded global pool (slot = blockIdx.x).
            int gridDim = (curBatch < maxResident) ? curBatch : maxResident;
            if (useI128)
                computeWeightsKernelI128<true><<<gridDim, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    dPrefix, dTwoScores, dProgress);
            else if (useDouble)
                computeWeightsKernel<true, double, false><<<gridDim, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    dPrefix, dTwoScores, dProgress);
            else if (numTaxa <= WB_SMALL_QI_MAX_N)
                computeWeightsKernel<true, long long, true><<<gridDim, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    dPrefix, dTwoScores, dProgress);
            else
                computeWeightsKernel<true, long long, false><<<gridDim, WB_BLOCK, sharedBytes, wbStream>>>(
                    dSplits, dSplitRangeMeta, dRangeData, dNodeData, dNodeFreq, dNodeOffset, dPartLeafCount,
                    dPolyTreeOffset, dPolyBoundOffset, dPolyBounds, dPolyFreq, dOrderings, dInvIndex,
                    curBatch, numPartTrees, partTreeOffset, stride, numTaxa, numTaxa,
                    dPrefix, dTwoScores, dProgress);
        }

        // Poll the splits-completed counter while the kernel runs (time-paced,
        // single-line), then make sure it has fully finished.
        char wbLabel[64];
        snprintf(wbLabel, sizeof wbLabel,
                 (numBatches > 1) ? "weight batch %d/%d" : "weight", b + 1, numBatches);
        cudaError_t err = wb_poll_progress(wbStream, pollStream, dProgress, hProgress, curBatch, wbLabel, progressIntervalSec);
        cudaError_t serr = cudaStreamSynchronize(wbStream);
        if (err == cudaErrorNotReady || err == cudaSuccess) err = serr;
        if (err != cudaSuccess) {
            fprintf(stderr, "[ASTRAL-X GPU] kernel error (batch %d/%d): %s\n",
                    b + 1, numBatches, cudaGetErrorString(err));
        }

        cudaMemcpy(hTwoScores + (size_t)offset * scoresPerSplit,
                   dTwoScores,
                   (size_t)curBatch * scoresPerSplit * sizeof(long long),
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
    jsize outLen = (jsize)((size_t)numSplits * scoresPerSplit);
    jlongArray result = env->NewLongArray(outLen);
    env->SetLongArrayRegion(result, 0, outLen, (jlong*)hTwoScores);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    delete[] hTwoScores;
    cudaFree(dSplits);
    cudaFree(dSplitRangeMeta);
    cudaFree(dRangeData);
    cudaFree(dTwoScores);
    cudaFree(dNodeData);
    cudaFree(dNodeFreq);
    cudaFree(dNodeOffset);
    cudaFree(dPartLeafCount);
    cudaFree(dOrderings);
    cudaFree(dInvIndex);
    cudaFree(dPolyTreeOffset);
    cudaFree(dPolyBoundOffset);
    cudaFree(dPolyBounds);
    cudaFree(dPolyFreq);
    if (dPrefix) cudaFree(dPrefix);
    cudaFree(dProgress);
    cudaFreeHost(hProgress);
    cudaStreamDestroy(wbStream);
    cudaStreamDestroy(pollStream);

    env->ReleaseIntArrayElements(jSplits,        hSplits,        JNI_ABORT);
    env->ReleaseIntArrayElements(jSplitRangeMeta,hSplitRangeMeta,JNI_ABORT);
    env->ReleaseIntArrayElements(jRangeData,     hRangeData,     JNI_ABORT);
    env->ReleaseIntArrayElements(jNodeData,      hNodeData,      JNI_ABORT);
    env->ReleaseIntArrayElements(jNodeFreq,      hNodeFreq,      JNI_ABORT);
    env->ReleaseIntArrayElements(jNodeOffset,    hNodeOffset,    JNI_ABORT);
    env->ReleaseIntArrayElements(jPartLeafCount, hPartLeafCount, JNI_ABORT);
    env->ReleaseIntArrayElements(jPolyTreeOffset, hPolyTreeOffset, JNI_ABORT);
    env->ReleaseIntArrayElements(jPolyBoundOffset,hPolyBoundOffset,JNI_ABORT);
    env->ReleaseIntArrayElements(jPolyBounds,    hPolyBounds,    JNI_ABORT);
    env->ReleaseIntArrayElements(jPolyFreq,      hPolyFreq,      JNI_ABORT);
    env->ReleaseIntArrayElements(jOrderings,     hOrderings,     JNI_ABORT);
    env->ReleaseIntArrayElements(jInvIndex,      hInvIndex,      JNI_ABORT);

    return result;
}

// ---------------------------------------------------------------------------
// LEGACY JNI entry point: smaller-side traversal (no prefix sums).
// ---------------------------------------------------------------------------
JNIEXPORT jlongArray JNICALL
Java_astralx_gpu_GPUWeightCalculator_computeWeightsSmallerSideGPU(
    JNIEnv* env, jclass cls,
    jintArray jSplits, jintArray jSplitRangeMeta, jintArray jRangeData,
    jintArray jParts,
    jintArray jSsPolyMeta, jintArray jSsPolyBoundOffset, jintArray jSsPolyBounds,
    jintArray jOrderings, jintArray jInvIndex,
    jint numSplits, jint numParts, jint numPolyParts, jint numGpuTrees, jint numTaxa, jint totalN,
    jint batchSizeHint, jdouble vramFraction, jint scoreMode, jdouble progressIntervalSec)
{
    bool useDouble = (scoreMode == 1);
    bool useI128   = (scoreMode == 2);
    int  scoresPerSplit = useI128 ? 2 : 1;
    fprintf(stderr, "[ASTRAL-X GPU] weight accumulator: %s\n",
            useI128   ? "INT128 (exact 128-bit integer)"
          : useDouble ? "DOUBLE (64-bit float, overflow-safe)"
                      : "LONG (exact 64-bit integer)");
    jint* hSplits    = env->GetIntArrayElements(jSplits,    NULL);
    jint* hSplitRangeMeta = env->GetIntArrayElements(jSplitRangeMeta, NULL);
    jint* hRangeData = env->GetIntArrayElements(jRangeData, NULL);
    jint* hParts     = env->GetIntArrayElements(jParts,     NULL);
    jint* hSsPolyMeta       = env->GetIntArrayElements(jSsPolyMeta,       NULL);
    jint* hSsPolyBoundOffset= env->GetIntArrayElements(jSsPolyBoundOffset,NULL);
    jint* hSsPolyBounds     = env->GetIntArrayElements(jSsPolyBounds,     NULL);
    jint* hOrderings = env->GetIntArrayElements(jOrderings, NULL);
    jint* hInvIndex  = env->GetIntArrayElements(jInvIndex,  NULL);
    jsize rangeDataLen   = env->GetArrayLength(jRangeData);
    jsize ssPolyBoundsLen= env->GetArrayLength(jSsPolyBounds);

    bool cacheRows = false;
    for (int j = 0; j < numParts && !cacheRows; j++) {
        const jint* pt = hParts + (size_t)j * 9;
        cacheRows = (pt[5] + pt[6] + pt[7] != totalN);
    }
    for (int p = 0; p < numPolyParts && !cacheRows; p++)
        cacheRows = (hSsPolyMeta[(size_t)p * 3 + 1] != totalN);
    if (getenv("ASTRALX_WEIGHT_DISABLE_ROW_CACHE")) cacheRows = false;

    // --- Upload static data once (parts, poly CSR, orderings, invIndex, rangeData) ---
    int *dParts, *dOrderings, *dInvIndex, *dRangeData;
    int *dSsPolyMeta, *dSsPolyBoundOffset, *dSsPolyBounds;
    size_t partsSz    = (size_t)numParts  * 9 * sizeof(int);
    size_t orderingSz = (size_t)numGpuTrees * numTaxa * sizeof(int);
    size_t rangeDataSz = (size_t)(rangeDataLen > 0 ? rangeDataLen : 1) * sizeof(int);
    size_t ssPolyMetaSz   = (size_t)(numPolyParts > 0 ? numPolyParts * 3 : 1) * sizeof(int);
    size_t ssPolyBoundOffSz = (size_t)(numPolyParts + 1) * sizeof(int);
    size_t ssPolyBoundsSz   = (size_t)(ssPolyBoundsLen > 0 ? ssPolyBoundsLen : 1) * sizeof(int);

    cudaMalloc(&dParts,     partsSz);
    cudaMalloc(&dOrderings, orderingSz);
    cudaMalloc(&dInvIndex,  orderingSz);
    cudaMalloc(&dRangeData, rangeDataSz);
    cudaMalloc(&dSsPolyMeta,       ssPolyMetaSz);
    cudaMalloc(&dSsPolyBoundOffset,ssPolyBoundOffSz);
    cudaMalloc(&dSsPolyBounds,     ssPolyBoundsSz);
    cudaMemcpy(dParts,     hParts,     partsSz,    cudaMemcpyHostToDevice);
    cudaMemcpy(dOrderings, hOrderings, orderingSz, cudaMemcpyHostToDevice);
    cudaMemcpy(dInvIndex,  hInvIndex,  orderingSz, cudaMemcpyHostToDevice);
    if (rangeDataLen > 0)
        cudaMemcpy(dRangeData, hRangeData, (size_t)rangeDataLen * sizeof(int), cudaMemcpyHostToDevice);
    if (numPolyParts > 0) {
        cudaMemcpy(dSsPolyMeta,   hSsPolyMeta,   (size_t)numPolyParts * 3 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dSsPolyBounds, hSsPolyBounds, (size_t)ssPolyBoundsLen  * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dSsPolyBoundOffset, hSsPolyBoundOffset, ssPolyBoundOffSz, cudaMemcpyHostToDevice);

    {
        size_t staticTotal = partsSz + 2 * orderingSz;
        size_t freeAfterStatic = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeAfterStatic, &totalVRAM);
        fprintf(stderr,
            "[ASTRAL-X GPU] weight static data uploaded (smaller-side traversal, no prefix sums):\n"
            "  orderings : %6.1f MB\n"
            "  invIndex  : %6.1f MB\n"
            "  parts     : %6.1f MB  (%d unique tripartitions × 9 ints)\n"
            "  ─────────────────────\n"
            "  static total : %6.1f MB   (VRAM free after: %.1f MB / %.1f MB)\n",
            orderingSz / 1e6, orderingSz / 1e6, partsSz / 1e6, numParts,
            staticTotal / 1e6, freeAfterStatic / 1e6, totalVRAM / 1e6);
        fflush(stderr);
    }

    // --- Determine batch size (per-split: 40 B in + 8 B out) ---
    int batchSize;
    if (batchSizeHint == -1) {
        batchSize = numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] batching disabled — single launch, %d splits\n", numSplits);
    } else if (batchSizeHint > 0) {
        batchSize = (batchSizeHint < numSplits) ? batchSizeHint : numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] manual batch size: %d  (numSplits=%d)\n", batchSize, numSplits);
    } else {
        size_t freeVRAM = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeVRAM, &totalVRAM);
        size_t usable = (size_t)((double)freeVRAM * (double)vramFraction);
        size_t perSplitBytes = 10 * sizeof(int) + scoresPerSplit * sizeof(long long);
        long long autoSize = (long long)(usable / perSplitBytes);
        if (autoSize < 1) autoSize = 1;
        if (autoSize > (long long)numSplits) autoSize = (long long)numSplits;
        batchSize = (int)autoSize;
        fprintf(stderr,
            "[ASTRAL-X GPU] adaptive batch: freeVRAM=%.2f GB, occupancy=%.0f%%, usable=%.2f GB, "
            "perSplit=%zu B → batchSize=%d  (numSplits=%d, numBatches=%d)\n",
            freeVRAM / 1e9, (double)vramFraction * 100.0, usable / 1e9,
            perSplitBytes, batchSize, numSplits, (numSplits + batchSize - 1) / batchSize);
    }

    int*       dSplits        = NULL;
    int*       dSplitRangeMeta = NULL;
    long long* dTwoScores     = NULL;
    while (batchSize > 0) {
        size_t splitBufSz = (size_t)batchSize * 10 * sizeof(int);
        size_t metaBufSz  = (size_t)batchSize * 4  * sizeof(int);
        size_t scoreBufSz = (size_t)batchSize * scoresPerSplit * sizeof(long long);
        cudaError_t e1 = cudaMalloc(&dSplits,         splitBufSz);
        cudaError_t e2 = cudaMalloc(&dTwoScores,      scoreBufSz);
        cudaError_t e3 = cudaMalloc(&dSplitRangeMeta, metaBufSz);
        if (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess) break;
        if (dSplits)         { cudaFree(dSplits);         dSplits         = NULL; }
        if (dTwoScores)      { cudaFree(dTwoScores);      dTwoScores      = NULL; }
        if (dSplitRangeMeta) { cudaFree(dSplitRangeMeta); dSplitRangeMeta = NULL; }
        batchSize /= 2;
        fprintf(stderr, "[ASTRAL-X GPU] cudaMalloc failed, retrying with batchSize=%d\n", batchSize);
    }
    if (batchSize <= 0 || dSplits == NULL || dTwoScores == NULL) {
        fprintf(stderr, "[ASTRAL-X GPU] FATAL: cannot allocate GPU batch buffers\n");
        cudaFree(dParts); cudaFree(dOrderings); cudaFree(dInvIndex); cudaFree(dRangeData);
        cudaFree(dSsPolyMeta); cudaFree(dSsPolyBoundOffset); cudaFree(dSsPolyBounds);
        if (dSplitRangeMeta) cudaFree(dSplitRangeMeta);
        env->ReleaseIntArrayElements(jSplits,    hSplits,    JNI_ABORT);
        env->ReleaseIntArrayElements(jSplitRangeMeta, hSplitRangeMeta, JNI_ABORT);
        env->ReleaseIntArrayElements(jRangeData, hRangeData, JNI_ABORT);
        env->ReleaseIntArrayElements(jParts,     hParts,     JNI_ABORT);
        env->ReleaseIntArrayElements(jSsPolyMeta,       hSsPolyMeta,       JNI_ABORT);
        env->ReleaseIntArrayElements(jSsPolyBoundOffset,hSsPolyBoundOffset,JNI_ABORT);
        env->ReleaseIntArrayElements(jSsPolyBounds,     hSsPolyBounds,     JNI_ABORT);
        env->ReleaseIntArrayElements(jOrderings, hOrderings, JNI_ABORT);
        env->ReleaseIntArrayElements(jInvIndex,  hInvIndex,  JNI_ABORT);
        return NULL;
    }

    long long* hTwoScores = new long long[(size_t)numSplits * scoresPerSplit]();

    // Intra-kernel progress counter + dedicated streams (see prefix-sum path).
    cudaStream_t wbStream = 0, pollStream = 0;
    cudaStreamCreate(&wbStream);
    cudaStreamCreate(&pollStream);
    int* dProgress = NULL; int* hProgress = NULL;
    cudaMalloc(&dProgress, sizeof(int));
    cudaHostAlloc((void**)&hProgress, sizeof(int), cudaHostAllocDefault);

    int    blockSize  = WB_BLOCK;
    int    numBatches = (numSplits + batchSize - 1) / batchSize;
    double t_loop_start = wb_now_sec();
    const char* GRN = wb_use_color() ? "\033[32m" : "";
    const char* RST = wb_use_color() ? "\033[0m"  : "";
    char   bar_buf[WB_BAR_W * 3 + 1];

    for (int b = 0; b < numBatches; b++) {
        int offset   = b * batchSize;
        int curBatch = (offset + batchSize <= numSplits) ? batchSize : (numSplits - offset);

        cudaMemcpy(dSplits, hSplits + (size_t)offset * 10,
                   (size_t)curBatch * 10 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dSplitRangeMeta, hSplitRangeMeta + (size_t)offset * 4,
                   (size_t)curBatch * 4 * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemsetAsync(dProgress, 0, sizeof(int), wbStream);

        int gridSize = (curBatch + blockSize - 1) / blockSize;
        if (useI128 && cacheRows)
            computeWeightsSmallerSideKernelI128<true><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dSplitRangeMeta, dRangeData, dParts,
                dSsPolyMeta, dSsPolyBoundOffset, dSsPolyBounds, dOrderings, dInvIndex,
                curBatch, numParts, numPolyParts, numTaxa, totalN, dTwoScores, dProgress);
        else if (useI128)
            computeWeightsSmallerSideKernelI128<false><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dSplitRangeMeta, dRangeData, dParts,
                dSsPolyMeta, dSsPolyBoundOffset, dSsPolyBounds, dOrderings, dInvIndex,
                curBatch, numParts, numPolyParts, numTaxa, totalN, dTwoScores, dProgress);
        else if (useDouble && cacheRows)
            computeWeightsSmallerSideKernel<double, true><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dSplitRangeMeta, dRangeData, dParts,
                dSsPolyMeta, dSsPolyBoundOffset, dSsPolyBounds, dOrderings, dInvIndex,
                curBatch, numParts, numPolyParts, numTaxa, totalN, dTwoScores, dProgress);
        else if (useDouble)
            computeWeightsSmallerSideKernel<double, false><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dSplitRangeMeta, dRangeData, dParts,
                dSsPolyMeta, dSsPolyBoundOffset, dSsPolyBounds, dOrderings, dInvIndex,
                curBatch, numParts, numPolyParts, numTaxa, totalN, dTwoScores, dProgress);
        else if (cacheRows)
            computeWeightsSmallerSideKernel<long long, true><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dSplitRangeMeta, dRangeData, dParts,
                dSsPolyMeta, dSsPolyBoundOffset, dSsPolyBounds, dOrderings, dInvIndex,
                curBatch, numParts, numPolyParts, numTaxa, totalN, dTwoScores, dProgress);
        else
            computeWeightsSmallerSideKernel<long long, false><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dSplitRangeMeta, dRangeData, dParts,
                dSsPolyMeta, dSsPolyBoundOffset, dSsPolyBounds, dOrderings, dInvIndex,
                curBatch, numParts, numPolyParts, numTaxa, totalN, dTwoScores, dProgress);

        char wbLabel[64];
        snprintf(wbLabel, sizeof wbLabel,
                 (numBatches > 1) ? "weight batch %d/%d" : "weight", b + 1, numBatches);
        cudaError_t err = wb_poll_progress(wbStream, pollStream, dProgress, hProgress, curBatch, wbLabel, progressIntervalSec);
        cudaError_t serr = cudaStreamSynchronize(wbStream);
        if (err == cudaErrorNotReady || err == cudaSuccess) err = serr;
        if (err != cudaSuccess) {
            fprintf(stderr, "[ASTRAL-X GPU] kernel error (batch %d/%d): %s\n",
                    b + 1, numBatches, cudaGetErrorString(err));
        }

        cudaMemcpy(hTwoScores + (size_t)offset * scoresPerSplit, dTwoScores,
                   (size_t)curBatch * scoresPerSplit * sizeof(long long), cudaMemcpyDeviceToHost);

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
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  100%%  done in %s                    \n",
                    GRN, RST, GRN, bar_buf, RST, numBatches, numBatches, dur_buf);
            } else {
                char eta_buf[32];
                wb_fmt_duration(avg_sec * rem, eta_buf, sizeof(eta_buf));
                fprintf(stderr,
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  %5.1f%%  %.2fs/batch  ETA: %-8s",
                    GRN, RST, GRN, bar_buf, RST, b + 1, numBatches, pct, avg_sec, eta_buf);
            }
            fflush(stderr);
        }
    }

    jsize outLen = (jsize)((size_t)numSplits * scoresPerSplit);
    jlongArray result = env->NewLongArray(outLen);
    env->SetLongArrayRegion(result, 0, outLen, (jlong*)hTwoScores);

    delete[] hTwoScores;
    cudaFree(dSplits);
    cudaFree(dSplitRangeMeta);
    cudaFree(dTwoScores);
    cudaFree(dParts);
    cudaFree(dOrderings);
    cudaFree(dInvIndex);
    cudaFree(dRangeData);
    cudaFree(dSsPolyMeta);
    cudaFree(dSsPolyBoundOffset);
    cudaFree(dSsPolyBounds);
    cudaFree(dProgress);
    cudaFreeHost(hProgress);
    cudaStreamDestroy(wbStream);
    cudaStreamDestroy(pollStream);

    env->ReleaseIntArrayElements(jSplits,    hSplits,    JNI_ABORT);
    env->ReleaseIntArrayElements(jSplitRangeMeta, hSplitRangeMeta, JNI_ABORT);
    env->ReleaseIntArrayElements(jRangeData, hRangeData, JNI_ABORT);
    env->ReleaseIntArrayElements(jParts,     hParts,     JNI_ABORT);
    env->ReleaseIntArrayElements(jSsPolyMeta,       hSsPolyMeta,       JNI_ABORT);
    env->ReleaseIntArrayElements(jSsPolyBoundOffset,hSsPolyBoundOffset,JNI_ABORT);
    env->ReleaseIntArrayElements(jSsPolyBounds,     hSsPolyBounds,     JNI_ABORT);
    env->ReleaseIntArrayElements(jOrderings, hOrderings, JNI_ABORT);
    env->ReleaseIntArrayElements(jInvIndex,  hInvIndex,  JNI_ABORT);

    return result;
}

JNIEXPORT jlongArray JNICALL
Java_astralx_gpu_GPUWeightCalculator_computeWeightsBitsetGPU(
    JNIEnv* env, jclass cls,
    jintArray jSplits,
    jlongArray jClusterBits,
    jlongArray jPartM1, jlongArray jPartM2,
    jintArray jPartMeta,
    jlongArray jGeneLgBits,
    jintArray jPolyMeta, jintArray jPolyChildOffset,
    jlongArray jPolyChildBits, jintArray jPolyChildSize,
    jint numSplits, jint numClusters, jint numParts, jint numPoly, jint numPartTrees,
    jint wordsPerSet, jint numTaxa,
    jint batchSizeHint, jdouble vramFraction, jint scoreMode, jdouble progressIntervalSec)
{
    bool useDouble = (scoreMode == 1);
    bool useI128   = (scoreMode == 2);
    int  scoresPerSplit = useI128 ? 2 : 1;
    int  W = wordsPerSet;
    int  totalN = numTaxa;
    fprintf(stderr, "[ASTRAL-X GPU] weight accumulator: %s  (bitset, W=%d words)\n",
            useI128   ? "INT128 (exact 128-bit integer)"
          : useDouble ? "DOUBLE (64-bit float, overflow-safe)"
                      : "LONG (exact 64-bit integer)", W);

    jint*  hSplits         = env->GetIntArrayElements(jSplits, NULL);
    jlong* hClusterBits    = env->GetLongArrayElements(jClusterBits, NULL);
    jlong* hPartM1         = env->GetLongArrayElements(jPartM1, NULL);
    jlong* hPartM2         = env->GetLongArrayElements(jPartM2, NULL);
    jint*  hPartMeta       = env->GetIntArrayElements(jPartMeta, NULL);
    jlong* hGeneLgBits     = env->GetLongArrayElements(jGeneLgBits, NULL);
    jint*  hPolyMeta       = env->GetIntArrayElements(jPolyMeta, NULL);
    jint*  hPolyChildOffset= env->GetIntArrayElements(jPolyChildOffset, NULL);
    jlong* hPolyChildBits  = env->GetLongArrayElements(jPolyChildBits, NULL);
    jint*  hPolyChildSize  = env->GetIntArrayElements(jPolyChildSize, NULL);

    size_t clusterLen   = (size_t) env->GetArrayLength(jClusterBits);
    size_t partM1Len    = (size_t) env->GetArrayLength(jPartM1);
    size_t partM2Len    = (size_t) env->GetArrayLength(jPartM2);
    size_t partMetaLen  = (size_t) env->GetArrayLength(jPartMeta);
    size_t geneLgLen    = (size_t) env->GetArrayLength(jGeneLgBits);
    size_t polyMetaLen  = (size_t) env->GetArrayLength(jPolyMeta);
    size_t polyOffLen   = (size_t) env->GetArrayLength(jPolyChildOffset);
    size_t polyCBitsLen = (size_t) env->GetArrayLength(jPolyChildBits);
    size_t polyCSizeLen = (size_t) env->GetArrayLength(jPolyChildSize);

    // --- Upload resident data once ---
    unsigned long long *dClusterBits, *dPartM1, *dPartM2, *dGeneLgBits, *dPolyChildBits;
    int *dPartMeta, *dPolyMeta, *dPolyChildOffset, *dPolyChildSize;
    #define MB_(n) ((size_t)((n) > 0 ? (n) : 1))
    size_t clusterSz = MB_(clusterLen)   * sizeof(unsigned long long);
    size_t pm1Sz     = MB_(partM1Len)    * sizeof(unsigned long long);
    size_t pm2Sz     = MB_(partM2Len)    * sizeof(unsigned long long);
    size_t geneSz    = MB_(geneLgLen)    * sizeof(unsigned long long);
    size_t pcbSz     = MB_(polyCBitsLen) * sizeof(unsigned long long);
    size_t pMetaSz   = MB_(partMetaLen)  * sizeof(int);
    size_t polyMetaSz= MB_(polyMetaLen)  * sizeof(int);
    size_t polyOffSz = MB_(polyOffLen)   * sizeof(int);
    size_t polyCSzSz = MB_(polyCSizeLen) * sizeof(int);

    cudaMalloc(&dClusterBits,     clusterSz);
    cudaMalloc(&dPartM1,          pm1Sz);
    cudaMalloc(&dPartM2,          pm2Sz);
    cudaMalloc(&dGeneLgBits,      geneSz);
    cudaMalloc(&dPolyChildBits,   pcbSz);
    cudaMalloc(&dPartMeta,        pMetaSz);
    cudaMalloc(&dPolyMeta,        polyMetaSz);
    cudaMalloc(&dPolyChildOffset, polyOffSz);
    cudaMalloc(&dPolyChildSize,   polyCSzSz);

    if (clusterLen)   cudaMemcpy(dClusterBits,   hClusterBits,   clusterLen  * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (partM1Len)    cudaMemcpy(dPartM1,        hPartM1,        partM1Len   * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (partM2Len)    cudaMemcpy(dPartM2,        hPartM2,        partM2Len   * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (geneLgLen)    cudaMemcpy(dGeneLgBits,    hGeneLgBits,    geneLgLen   * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (polyCBitsLen) cudaMemcpy(dPolyChildBits, hPolyChildBits, polyCBitsLen* sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (partMetaLen)  cudaMemcpy(dPartMeta,      hPartMeta,      partMetaLen * sizeof(int), cudaMemcpyHostToDevice);
    if (polyMetaLen)  cudaMemcpy(dPolyMeta,      hPolyMeta,      polyMetaLen * sizeof(int), cudaMemcpyHostToDevice);
    if (polyOffLen)   cudaMemcpy(dPolyChildOffset,hPolyChildOffset,polyOffLen* sizeof(int), cudaMemcpyHostToDevice);
    if (polyCSizeLen) cudaMemcpy(dPolyChildSize, hPolyChildSize, polyCSizeLen* sizeof(int), cudaMemcpyHostToDevice);
    #undef MB_

    {
        size_t residentTotal = clusterSz + pm1Sz + pm2Sz + geneSz + pcbSz
                             + pMetaSz + polyMetaSz + polyOffSz + polyCSzSz;
        size_t freeAfterStatic = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeAfterStatic, &totalVRAM);
        fprintf(stderr,
            "[ASTRAL-X GPU] weight resident data uploaded (bitset, W=%d):\n"
            "  clusterBits : %6.1f MB  (%d clusters × %d words)\n"
            "  partM1+M2   : %6.1f MB  (%d binary parts)\n"
            "  geneLgBits  : %6.1f MB  (%d gene trees)\n"
            "  polyChild   : %6.1f MB  (%d poly parts)\n"
            "  ─────────────────────\n"
            "  resident total : %6.1f MB   (VRAM free after: %.1f MB / %.1f MB)\n",
            W, clusterSz / 1e6, numClusters, W, (pm1Sz + pm2Sz) / 1e6, numParts,
            geneSz / 1e6, numPartTrees, pcbSz / 1e6, numPoly,
            residentTotal / 1e6, freeAfterStatic / 1e6, totalVRAM / 1e6);
        fflush(stderr);
    }

    // --- Determine batch size (per-split: 16 B in + 8/16 B out) ---
    int batchSize;
    if (batchSizeHint == -1) {
        batchSize = numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] batching disabled — single launch, %d splits\n", numSplits);
    } else if (batchSizeHint > 0) {
        batchSize = (batchSizeHint < numSplits) ? batchSizeHint : numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] manual batch size: %d  (numSplits=%d)\n", batchSize, numSplits);
    } else {
        size_t freeVRAM = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeVRAM, &totalVRAM);
        size_t usable = (size_t)((double)freeVRAM * (double)vramFraction);
        size_t perSplitBytes = 4 * sizeof(int) + scoresPerSplit * sizeof(long long);
        long long autoSize = (long long)(usable / perSplitBytes);
        if (autoSize < 1) autoSize = 1;
        if (autoSize > (long long)numSplits) autoSize = (long long)numSplits;
        batchSize = (int)autoSize;
        fprintf(stderr,
            "[ASTRAL-X GPU] adaptive batch: freeVRAM=%.2f GB, occupancy=%.0f%%, usable=%.2f GB, "
            "perSplit=%zu B → batchSize=%d  (numSplits=%d, numBatches=%d)\n",
            freeVRAM / 1e9, (double)vramFraction * 100.0, usable / 1e9,
            perSplitBytes, batchSize, numSplits, (numSplits + batchSize - 1) / batchSize);
    }

    int*       dSplits    = NULL;
    long long* dTwoScores = NULL;
    while (batchSize > 0) {
        size_t splitBufSz = (size_t)batchSize * 4 * sizeof(int);
        size_t scoreBufSz = (size_t)batchSize * scoresPerSplit * sizeof(long long);
        cudaError_t e1 = cudaMalloc(&dSplits,    splitBufSz);
        cudaError_t e2 = cudaMalloc(&dTwoScores, scoreBufSz);
        if (e1 == cudaSuccess && e2 == cudaSuccess) break;
        if (dSplits)    { cudaFree(dSplits);    dSplits    = NULL; }
        if (dTwoScores) { cudaFree(dTwoScores); dTwoScores = NULL; }
        batchSize /= 2;
        fprintf(stderr, "[ASTRAL-X GPU] cudaMalloc failed, retrying with batchSize=%d\n", batchSize);
    }
    if (batchSize <= 0 || dSplits == NULL || dTwoScores == NULL) {
        fprintf(stderr, "[ASTRAL-X GPU] FATAL: cannot allocate GPU batch buffers (bitset)\n");
        cudaFree(dClusterBits); cudaFree(dPartM1); cudaFree(dPartM2); cudaFree(dGeneLgBits);
        cudaFree(dPolyChildBits); cudaFree(dPartMeta); cudaFree(dPolyMeta);
        cudaFree(dPolyChildOffset); cudaFree(dPolyChildSize);
        env->ReleaseIntArrayElements(jSplits, hSplits, JNI_ABORT);
        env->ReleaseLongArrayElements(jClusterBits, hClusterBits, JNI_ABORT);
        env->ReleaseLongArrayElements(jPartM1, hPartM1, JNI_ABORT);
        env->ReleaseLongArrayElements(jPartM2, hPartM2, JNI_ABORT);
        env->ReleaseIntArrayElements(jPartMeta, hPartMeta, JNI_ABORT);
        env->ReleaseLongArrayElements(jGeneLgBits, hGeneLgBits, JNI_ABORT);
        env->ReleaseIntArrayElements(jPolyMeta, hPolyMeta, JNI_ABORT);
        env->ReleaseIntArrayElements(jPolyChildOffset, hPolyChildOffset, JNI_ABORT);
        env->ReleaseLongArrayElements(jPolyChildBits, hPolyChildBits, JNI_ABORT);
        env->ReleaseIntArrayElements(jPolyChildSize, hPolyChildSize, JNI_ABORT);
        return NULL;
    }

    long long* hTwoScores = new long long[(size_t)numSplits * scoresPerSplit]();

    cudaStream_t wbStream = 0, pollStream = 0;
    cudaStreamCreate(&wbStream);
    cudaStreamCreate(&pollStream);
    int* dProgress = NULL; int* hProgress = NULL;
    cudaMalloc(&dProgress, sizeof(int));
    cudaHostAlloc((void**)&hProgress, sizeof(int), cudaHostAllocDefault);

    int    blockSize  = WB_BLOCK;
    int    numBatches = (numSplits + batchSize - 1) / batchSize;
    double t_loop_start = wb_now_sec();
    const char* GRN = wb_use_color() ? "\033[32m" : "";
    const char* RST = wb_use_color() ? "\033[0m"  : "";
    char   bar_buf[WB_BAR_W * 3 + 1];

    for (int b = 0; b < numBatches; b++) {
        int offset   = b * batchSize;
        int curBatch = (offset + batchSize <= numSplits) ? batchSize : (numSplits - offset);

        cudaMemcpy(dSplits, hSplits + (size_t)offset * 4,
                   (size_t)curBatch * 4 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemsetAsync(dProgress, 0, sizeof(int), wbStream);

        int gridSize = (curBatch + blockSize - 1) / blockSize;
        if (useI128)
            computeWeightsBitsetKernelI128<<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dClusterBits, dPartM1, dPartM2, dPartMeta, dGeneLgBits,
                dPolyMeta, dPolyChildOffset, dPolyChildBits, dPolyChildSize,
                curBatch, numParts, numPoly, W, totalN, dTwoScores, dProgress);
        else if (useDouble)
            computeWeightsBitsetKernel<double><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dClusterBits, dPartM1, dPartM2, dPartMeta, dGeneLgBits,
                dPolyMeta, dPolyChildOffset, dPolyChildBits, dPolyChildSize,
                curBatch, numParts, numPoly, W, totalN, dTwoScores, dProgress);
        else
            computeWeightsBitsetKernel<long long><<<gridSize, blockSize, 0, wbStream>>>(
                dSplits, dClusterBits, dPartM1, dPartM2, dPartMeta, dGeneLgBits,
                dPolyMeta, dPolyChildOffset, dPolyChildBits, dPolyChildSize,
                curBatch, numParts, numPoly, W, totalN, dTwoScores, dProgress);

        char wbLabel[64];
        snprintf(wbLabel, sizeof wbLabel,
                 (numBatches > 1) ? "weight batch %d/%d" : "weight", b + 1, numBatches);
        cudaError_t err = wb_poll_progress(wbStream, pollStream, dProgress, hProgress, curBatch, wbLabel, progressIntervalSec);
        cudaError_t serr = cudaStreamSynchronize(wbStream);
        if (err == cudaErrorNotReady || err == cudaSuccess) err = serr;
        if (err != cudaSuccess) {
            fprintf(stderr, "[ASTRAL-X GPU] kernel error (bitset batch %d/%d): %s\n",
                    b + 1, numBatches, cudaGetErrorString(err));
        }

        cudaMemcpy(hTwoScores + (size_t)offset * scoresPerSplit, dTwoScores,
                   (size_t)curBatch * scoresPerSplit * sizeof(long long), cudaMemcpyDeviceToHost);

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
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  100%%  done in %s                    \n",
                    GRN, RST, GRN, bar_buf, RST, numBatches, numBatches, dur_buf);
            } else {
                char eta_buf[32];
                wb_fmt_duration(avg_sec * rem, eta_buf, sizeof(eta_buf));
                fprintf(stderr,
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  %5.1f%%  %.2fs/batch  ETA: %-8s",
                    GRN, RST, GRN, bar_buf, RST, b + 1, numBatches, pct, avg_sec, eta_buf);
            }
            fflush(stderr);
        }
    }

    jsize outLen = (jsize)((size_t)numSplits * scoresPerSplit);
    jlongArray result = env->NewLongArray(outLen);
    env->SetLongArrayRegion(result, 0, outLen, (jlong*)hTwoScores);

    delete[] hTwoScores;
    cudaFree(dSplits); cudaFree(dTwoScores);
    cudaFree(dClusterBits); cudaFree(dPartM1); cudaFree(dPartM2); cudaFree(dGeneLgBits);
    cudaFree(dPolyChildBits); cudaFree(dPartMeta); cudaFree(dPolyMeta);
    cudaFree(dPolyChildOffset); cudaFree(dPolyChildSize);
    cudaFree(dProgress); cudaFreeHost(hProgress);
    cudaStreamDestroy(wbStream); cudaStreamDestroy(pollStream);

    env->ReleaseIntArrayElements(jSplits, hSplits, JNI_ABORT);
    env->ReleaseLongArrayElements(jClusterBits, hClusterBits, JNI_ABORT);
    env->ReleaseLongArrayElements(jPartM1, hPartM1, JNI_ABORT);
    env->ReleaseLongArrayElements(jPartM2, hPartM2, JNI_ABORT);
    env->ReleaseIntArrayElements(jPartMeta, hPartMeta, JNI_ABORT);
    env->ReleaseLongArrayElements(jGeneLgBits, hGeneLgBits, JNI_ABORT);
    env->ReleaseIntArrayElements(jPolyMeta, hPolyMeta, JNI_ABORT);
    env->ReleaseIntArrayElements(jPolyChildOffset, hPolyChildOffset, JNI_ABORT);
    env->ReleaseLongArrayElements(jPolyChildBits, hPolyChildBits, JNI_ABORT);
    env->ReleaseIntArrayElements(jPolyChildSize, hPolyChildSize, JNI_ABORT);

    return result;
}

JNIEXPORT jlongArray JNICALL
Java_astralx_gpu_GPUWeightCalculator_computeWeightsTreeWalkGPU(
    JNIEnv* env, jclass cls,
    jintArray jSplits,
    jlongArray jClusterBits,
    jlongArray jGeneLgBits,
    jintArray jNodeStream,
    jintArray jTreeNodeOffset,
    jintArray jLeafCount,
    jint numSplits, jint numClusters, jint numTrees,
    jint wordsPerSet, jint numTaxa, jint maxFrontier,
    jint batchSizeHint, jdouble vramFraction, jint scoreMode, jdouble progressIntervalSec)
{
    // Defense in depth: Java measures the exact token-stream frontier before
    // building resident data, but native code independently enforces the compiled
    // maximum before dispatching the smallest fitting private-array variant.
    if (maxFrontier < 1 || maxFrontier > WB_TW_STACK_CAP) {
        fprintf(stderr,
                "[ASTRAL-X GPU] tree-walk: measured frontier=%d outside stack cap %d → CPU fallback\n",
                maxFrontier, WB_TW_STACK_CAP);
        return NULL;
    }

    bool useDouble = (scoreMode == 1);
    bool useI128   = (scoreMode == 2);
    int  scoresPerSplit = useI128 ? 2 : 1;
    int  W = wordsPerSet;
    int  totalN = numTaxa;
    fprintf(stderr, "[ASTRAL-X GPU] weight accumulator: %s  (simple-tree-walk, W=%d words)\n",
            useI128   ? "INT128 (exact 128-bit integer)"
          : useDouble ? "DOUBLE (64-bit float, overflow-safe)"
                      : "LONG (exact 64-bit integer)", W);

    jint*  hSplits         = env->GetIntArrayElements(jSplits, NULL);
    jlong* hClusterBits    = env->GetLongArrayElements(jClusterBits, NULL);
    jlong* hGeneLgBits     = env->GetLongArrayElements(jGeneLgBits, NULL);
    jint*  hNodeStream     = env->GetIntArrayElements(jNodeStream, NULL);
    jint*  hTreeNodeOffset = env->GetIntArrayElements(jTreeNodeOffset, NULL);
    jint*  hLeafCount      = env->GetIntArrayElements(jLeafCount, NULL);

    size_t clusterLen  = (size_t) env->GetArrayLength(jClusterBits);
    size_t geneLgLen   = (size_t) env->GetArrayLength(jGeneLgBits);
    size_t nodeStreamLen = (size_t) env->GetArrayLength(jNodeStream);

    // --- Upload resident data once ---
    unsigned long long *dClusterBits, *dGeneLgBits;
    int *dNodeStream, *dTreeNodeOffset, *dLeafCount;
    #define MB_(n) ((size_t)((n) > 0 ? (n) : 1))
    size_t clusterSz    = MB_(clusterLen)    * sizeof(unsigned long long);
    size_t geneSz       = MB_(geneLgLen)     * sizeof(unsigned long long);
    size_t nodeStreamSz = MB_(nodeStreamLen) * sizeof(int);
    size_t treeOffSz    = (size_t)(numTrees + 1) * sizeof(int);
    size_t leafCntSz    = MB_(numTrees)      * sizeof(int);

    cudaMalloc(&dClusterBits,    clusterSz);
    cudaMalloc(&dGeneLgBits,     geneSz);
    cudaMalloc(&dNodeStream,     nodeStreamSz);
    cudaMalloc(&dTreeNodeOffset, treeOffSz);
    cudaMalloc(&dLeafCount,      leafCntSz);
    if (clusterLen)    cudaMemcpy(dClusterBits, hClusterBits, clusterLen    * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (geneLgLen)     cudaMemcpy(dGeneLgBits,  hGeneLgBits,  geneLgLen     * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (nodeStreamLen) cudaMemcpy(dNodeStream,  hNodeStream,  nodeStreamLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dTreeNodeOffset, hTreeNodeOffset, treeOffSz, cudaMemcpyHostToDevice);
    if (numTrees > 0)  cudaMemcpy(dLeafCount, hLeafCount, (size_t)numTrees * sizeof(int), cudaMemcpyHostToDevice);
    #undef MB_

    {
        size_t residentTotal = clusterSz + geneSz + nodeStreamSz + treeOffSz + leafCntSz;
        size_t freeAfterStatic = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeAfterStatic, &totalVRAM);
        fprintf(stderr,
            "[ASTRAL-X GPU] weight resident data uploaded (simple-tree-walk, W=%d):\n"
            "  clusterBits : %6.1f MB  (%d clusters × %d words)\n"
            "  geneLgBits  : %6.1f MB  (%d gene trees)\n"
            "  nodeStream  : %6.1f MB  (%zu tokens)\n"
            "  ─────────────────────\n"
            "  resident total : %6.1f MB   (VRAM free after: %.1f MB / %.1f MB)\n",
            W, clusterSz / 1e6, numClusters, W, geneSz / 1e6, numTrees,
            nodeStreamSz / 1e6, nodeStreamLen, residentTotal / 1e6,
            freeAfterStatic / 1e6, totalVRAM / 1e6);
        fflush(stderr);
    }

    // --- Determine batch size (metadata + transposed A/B candidate bits + score) ---
    int batchSize;
    if (batchSizeHint == -1) {
        batchSize = numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] batching disabled — single launch, %d splits\n", numSplits);
    } else if (batchSizeHint > 0) {
        batchSize = (batchSizeHint < numSplits) ? batchSizeHint : numSplits;
        fprintf(stderr, "[ASTRAL-X GPU] manual batch size: %d  (numSplits=%d)\n", batchSize, numSplits);
    } else {
        size_t freeVRAM = 0, totalVRAM = 0;
        cudaMemGetInfo(&freeVRAM, &totalVRAM);
        size_t usable = (size_t)((double)freeVRAM * (double)vramFraction);
        size_t perSplitBytes = 4 * sizeof(int)
                             + (size_t)2 * W * sizeof(unsigned long long)
                             + scoresPerSplit * sizeof(long long);
        long long autoSize = (long long)(usable / perSplitBytes);
        if (autoSize < 1) autoSize = 1;
        if (autoSize > (long long)numSplits) autoSize = (long long)numSplits;
        batchSize = (int)autoSize;
        fprintf(stderr,
            "[ASTRAL-X GPU] adaptive batch: freeVRAM=%.2f GB, occupancy=%.0f%%, usable=%.2f GB, "
            "perSplit=%zu B → batchSize=%d  (numSplits=%d, numBatches=%d)\n",
            freeVRAM / 1e9, (double)vramFraction * 100.0, usable / 1e9,
            perSplitBytes, batchSize, numSplits, (numSplits + batchSize - 1) / batchSize);
    }

    int*       dSplits        = NULL;
    long long* dTwoScores     = NULL;
    unsigned long long* dCandidateBits = NULL;
    while (batchSize > 0) {
        size_t splitBufSz = (size_t)batchSize * 4 * sizeof(int);
        size_t scoreBufSz = (size_t)batchSize * scoresPerSplit * sizeof(long long);
        size_t candidateBufSz = (size_t)batchSize * 2 * W * sizeof(unsigned long long);
        cudaError_t e1 = cudaMalloc(&dSplits,        splitBufSz);
        cudaError_t e2 = cudaMalloc(&dTwoScores,     scoreBufSz);
        cudaError_t e3 = cudaMalloc(&dCandidateBits, candidateBufSz);
        if (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess) break;
        if (dSplits)    { cudaFree(dSplits);    dSplits    = NULL; }
        if (dTwoScores) { cudaFree(dTwoScores); dTwoScores = NULL; }
        if (dCandidateBits) { cudaFree(dCandidateBits); dCandidateBits = NULL; }
        batchSize /= 2;
        fprintf(stderr, "[ASTRAL-X GPU] cudaMalloc failed, retrying with batchSize=%d\n", batchSize);
    }
    if (batchSize <= 0 || dSplits == NULL || dTwoScores == NULL || dCandidateBits == NULL) {
        fprintf(stderr, "[ASTRAL-X GPU] FATAL: cannot allocate GPU batch buffers (tree-walk)\n");
        cudaFree(dClusterBits); cudaFree(dGeneLgBits); cudaFree(dNodeStream);
        cudaFree(dTreeNodeOffset); cudaFree(dLeafCount);
        env->ReleaseIntArrayElements(jSplits, hSplits, JNI_ABORT);
        env->ReleaseLongArrayElements(jClusterBits, hClusterBits, JNI_ABORT);
        env->ReleaseLongArrayElements(jGeneLgBits, hGeneLgBits, JNI_ABORT);
        env->ReleaseIntArrayElements(jNodeStream, hNodeStream, JNI_ABORT);
        env->ReleaseIntArrayElements(jTreeNodeOffset, hTreeNodeOffset, JNI_ABORT);
        env->ReleaseIntArrayElements(jLeafCount, hLeafCount, JNI_ABORT);
        return NULL;
    }

    long long* hTwoScores = new long long[(size_t)numSplits * scoresPerSplit]();

    cudaStream_t wbStream = 0, pollStream = 0;
    cudaStreamCreate(&wbStream);
    cudaStreamCreate(&pollStream);
    int* dProgress = NULL; int* hProgress = NULL;
    cudaMalloc(&dProgress, sizeof(int));
    cudaHostAlloc((void**)&hProgress, sizeof(int), cudaHostAllocDefault);

    int    blockSize  = WB_BLOCK;
    int    numBatches = (numSplits + batchSize - 1) / batchSize;
    double t_loop_start = wb_now_sec();
    const char* GRN = wb_use_color() ? "\033[32m" : "";
    const char* RST = wb_use_color() ? "\033[0m"  : "";
    char   bar_buf[WB_BAR_W * 3 + 1];

    for (int b = 0; b < numBatches; b++) {
        int offset   = b * batchSize;
        int curBatch = (offset + batchSize <= numSplits) ? batchSize : (numSplits - offset);

        cudaMemcpy(dSplits, hSplits + (size_t)offset * 4,
                   (size_t)curBatch * 4 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemsetAsync(dProgress, 0, sizeof(int), wbStream);

        size_t gatherTotal = (size_t)curBatch * W;
        unsigned int gatherGrid = (unsigned int)((gatherTotal + WB_BLOCK - 1) / WB_BLOCK);
        gatherCandidateBits<<<gatherGrid, WB_BLOCK, 0, wbStream>>>(
            dSplits, dClusterBits, dCandidateBits, curBatch, W);

        int gridSize = (curBatch + blockSize - 1) / blockSize;
        #define WB_LAUNCH_TREE_WALK(CAP) do { \
            if (useI128) \
                computeWeightsTreeWalkKernelI128<CAP><<<gridSize, blockSize, 0, wbStream>>>( \
                    dSplits, dCandidateBits, dGeneLgBits, dNodeStream, dTreeNodeOffset, dLeafCount, \
                    curBatch, numTrees, W, totalN, dTwoScores, dProgress); \
            else if (useDouble) \
                computeWeightsTreeWalkKernel<double, false, CAP><<<gridSize, blockSize, 0, wbStream>>>( \
                    dSplits, dCandidateBits, dGeneLgBits, dNodeStream, dTreeNodeOffset, dLeafCount, \
                    curBatch, numTrees, W, totalN, dTwoScores, dProgress); \
            else if (totalN <= WB_SMALL_QI_MAX_N) \
                computeWeightsTreeWalkKernel<long long, true, CAP><<<gridSize, blockSize, 0, wbStream>>>( \
                    dSplits, dCandidateBits, dGeneLgBits, dNodeStream, dTreeNodeOffset, dLeafCount, \
                    curBatch, numTrees, W, totalN, dTwoScores, dProgress); \
            else \
                computeWeightsTreeWalkKernel<long long, false, CAP><<<gridSize, blockSize, 0, wbStream>>>( \
                    dSplits, dCandidateBits, dGeneLgBits, dNodeStream, dTreeNodeOffset, dLeafCount, \
                    curBatch, numTrees, W, totalN, dTwoScores, dProgress); \
        } while (0)

        if      (maxFrontier <= 32)  WB_LAUNCH_TREE_WALK(32);
        else if (maxFrontier <= 64)  WB_LAUNCH_TREE_WALK(64);
        else if (maxFrontier <= 128) WB_LAUNCH_TREE_WALK(128);
        else if (maxFrontier <= 256) WB_LAUNCH_TREE_WALK(256);
        else                         WB_LAUNCH_TREE_WALK(512);
        #undef WB_LAUNCH_TREE_WALK

        char wbLabel[64];
        snprintf(wbLabel, sizeof wbLabel,
                 (numBatches > 1) ? "weight batch %d/%d" : "weight", b + 1, numBatches);
        cudaError_t err = wb_poll_progress(wbStream, pollStream, dProgress, hProgress, curBatch, wbLabel, progressIntervalSec);
        cudaError_t serr = cudaStreamSynchronize(wbStream);
        if (err == cudaErrorNotReady || err == cudaSuccess) err = serr;
        if (err != cudaSuccess) {
            fprintf(stderr, "[ASTRAL-X GPU] kernel error (tree-walk batch %d/%d): %s\n",
                    b + 1, numBatches, cudaGetErrorString(err));
        }

        cudaMemcpy(hTwoScores + (size_t)offset * scoresPerSplit, dTwoScores,
                   (size_t)curBatch * scoresPerSplit * sizeof(long long), cudaMemcpyDeviceToHost);

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
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  100%%  done in %s                    \n",
                    GRN, RST, GRN, bar_buf, RST, numBatches, numBatches, dur_buf);
            } else {
                char eta_buf[32];
                wb_fmt_duration(avg_sec * rem, eta_buf, sizeof(eta_buf));
                fprintf(stderr,
                    "\r  %s[GPU]%s weight  %s[%s]%s  %d/%d  %5.1f%%  %.2fs/batch  ETA: %-8s",
                    GRN, RST, GRN, bar_buf, RST, b + 1, numBatches, pct, avg_sec, eta_buf);
            }
            fflush(stderr);
        }
    }

    jsize outLen = (jsize)((size_t)numSplits * scoresPerSplit);
    jlongArray result = env->NewLongArray(outLen);
    env->SetLongArrayRegion(result, 0, outLen, (jlong*)hTwoScores);

    delete[] hTwoScores;
    cudaFree(dSplits); cudaFree(dTwoScores); cudaFree(dCandidateBits);
    cudaFree(dClusterBits); cudaFree(dGeneLgBits); cudaFree(dNodeStream);
    cudaFree(dTreeNodeOffset); cudaFree(dLeafCount);
    cudaFree(dProgress); cudaFreeHost(hProgress);
    cudaStreamDestroy(wbStream); cudaStreamDestroy(pollStream);

    env->ReleaseIntArrayElements(jSplits, hSplits, JNI_ABORT);
    env->ReleaseLongArrayElements(jClusterBits, hClusterBits, JNI_ABORT);
    env->ReleaseLongArrayElements(jGeneLgBits, hGeneLgBits, JNI_ABORT);
    env->ReleaseIntArrayElements(jNodeStream, hNodeStream, JNI_ABORT);
    env->ReleaseIntArrayElements(jTreeNodeOffset, hTreeNodeOffset, JNI_ABORT);
    env->ReleaseIntArrayElements(jLeafCount, hLeafCount, JNI_ABORT);

    return result;
}

} // extern "C"
