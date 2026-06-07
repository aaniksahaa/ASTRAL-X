package astralx.weight;

import astralx.Config;
import astralx.Logging;
import astralx.cluster.Cluster;
import astralx.cluster.ClusterHash;
import astralx.cluster.ClusterTable;
import astralx.dp.BipartitionSplit;
import astralx.dp.DPTable;
import astralx.gpu.GPUWeightCalculator;
import astralx.partition.Partition;
import astralx.partition.PartitionTable;
import astralx.tree.Tree;
import astralx.util.ProgressBar;
import astralx.util.Threading;

import java.util.*;

/**
 * Precomputed ASTRAL quartet scores for every candidate bipartition split.
 *
 * For a candidate split (A | B) inducing species-tree tripartition (A | B | C)
 * where C = L \ (A ∪ B), and a gene-tree tripartition (M1 | M2 | M3):
 *
 *   Compute 4 core intersections: a0=|A∩M1|, a1=|A∩M2|, b0=|B∩M1|, b1=|B∩M2|
 *
 *   Derive remaining 5 by row/column constraints (valid for complete trees):
 *     a2 = |A| - a0 - a1
 *     b2 = |B| - b0 - b1
 *     c0 = |M1| - a0 - b0
 *     c1 = |M2| - a1 - b1
 *     c2 = |C| - c0 - c1
 *
 *   QI = (1/2) * sum over 6 permutations (i,j,k) of {0,1,2}:
 *          a[i] * b[j] * c[k] * (a[i] + b[j] + c[k] - 3)
 *
 *   score(A|B) = sum over unique gene-tree tripartitions P: P.frequency * QI
 *
 * QI is always a non-negative integer (proven by parity argument),
 * so scores are stored as non-negative longs.
 *
 * Execution path selection:
 *   GPU  — when --gpu flag is set and libastralx_weight.so is loadable.
 *   CPU  — otherwise (multi-threaded via Threading.processRangeParallel).
 */
public class WeightTable {

    private final Map<BipartitionSplit, Long>   scores  = new HashMap<>();   // exact LONG path
    private final Map<BipartitionSplit, Double> scoresD = new HashMap<>();   // DOUBLE path (large n)
    private final int n;   // total taxa

    /**
     * Whether scores are accumulated/stored as 64-bit floating point (double)
     * instead of exact 64-bit integers (long).  Decided once per run from the
     * problem size (see {@link #needsDoubleAccumulation}); true only when the
     * exact integer score would overflow {@code long}.
     */
    private final boolean useDouble;

    // stats (LONG path)
    private long maxScore;
    private long totalScore;
    // stats (DOUBLE path)
    private double maxScoreD = Double.NEGATIVE_INFINITY;
    private double totalScoreD;

    // -------------------------------------------------------------------------

    /**
     * @param clusterTrees  completed gene trees — used for cluster exemplar position lookups
     *                      (Cluster.treeIndex refers into this list)
     * @param partTrees     original (pre-completion) gene trees — used for gene-tree quartet
     *                      scoring (Partition.treeIndex refers into this list).
     *                      When --autocomplete-incomplete-gene-trees is NOT active,
     *                      partTrees == clusterTrees (same reference) and behavior is unchanged.
     */
    public WeightTable(DPTable dpTable, PartitionTable partTable,
                       ClusterTable clusterTable,
                       List<Tree> clusterTrees, List<Tree> partTrees) {
        long t0 = System.nanoTime();
        this.n = clusterTable.getAllTaxaHash().size;

        // ── Numeric-precision decision (LONG vs DOUBLE accumulation) ──────────
        // The exact quartet score grows as O(genes · n^4) and overflows a signed
        // 64-bit integer for very large taxon sets.  When the estimated maximum
        // per-split score would exceed the long-safe range we switch the whole
        // scoring + DP pipeline to double; otherwise we keep exact integers.
        int numGenes   = partTrees.size();
        this.useDouble = needsDoubleAccumulation(n, numGenes);
        logAccumulationDecision(n, numGenes, useDouble);

        // Collect all unique splits from DPTable into an indexed list
        List<BipartitionSplit> splitList = new ArrayList<>();
        for (var entry : dpTable.entries()) splitList.addAll(entry.getValue());
        int numSplits = splitList.size();

        List<PartitionTable.Entry> partList = new ArrayList<>(partTable.entries());
        long[]   scoreArray  = useDouble ? null : new long[numSplits];     // exact integer scores
        double[] scoreArrayD = useDouble ? new double[numSplits] : null;   // floating-point scores

        // When clusterTrees != partTrees (autocomplete active), the GPU path packs both
        // sets of orderings/invIndex into a combined array (slots 0..k-1 = completed,
        // slots k..2k-1 = original) and offsets partition tree indices by k.
        // numGpuTrees reflects the combined size for VRAM budget calculations.
        boolean splitTrees = (clusterTrees != partTrees);
        int numGpuTrees = splitTrees ? clusterTrees.size() * 2 : clusterTrees.size();

        boolean useGPU = (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU)
                         && GPUWeightCalculator.tryLoad();

        if (useGPU) {
            Config cfg = Config.getInstance();
            boolean smallerSide = (cfg.getWeightIntersectionMethod()
                                   == Config.WeightIntersectionMethod.SMALLER_SIDE_TRAVERSAL);

            // Each path keeps its own resident-data representation.  PREFIX_SUM builds
            // the deduplicated node CSR (and the per-block prefix working memory);
            // SMALLER_SIDE_TRAVERSAL builds none of that — it streams the parts and
            // walks the smaller side per intersection, with zero per-thread state.
            NodeCSR csr = smallerSide ? null : buildDedupNodeCSR(partTable, partTrees);

            // Resident data memory (for the vram-control-factor sizing only).
            long orderingMem = (long) numGpuTrees * n * 2 * Integer.BYTES; // orderings + invIndex
            long modeDataMem; String modeDataDesc;
            if (smallerSide) {
                modeDataMem  = (long) partTable.size() * 9 * Integer.BYTES; // parts
                modeDataDesc = "parts";
            } else {
                modeDataMem  = (long) csr.nodeData.length      * Integer.BYTES
                             + (long) csr.nodeFreq.length      * Integer.BYTES
                             + (long) csr.nodeOffset.length    * Integer.BYTES
                             + (long) csr.partLeafCount.length * Integer.BYTES;
                modeDataDesc = "nodeCSR";
            }

            // Resolve batchSizeHint
            //   Priority: no-batch  >  gpu-batches  >  gpu-batch-size
            //           > gpu-vram-control-factor (explicit)  >  auto (gpu-vram-occupancy-factor)
            //   -1  = no batching (single launch)
            //    0  = auto: native queries free VRAM and computes batch size itself
            //   >0  = exact splits-per-batch resolved here; native uses it directly
            int batchSizeHint;
            String batchDesc;
            if (!cfg.isGpuBatch()) {
                batchSizeHint = -1;
                batchDesc = "off (single launch)";
            } else if (cfg.getGpuNumBatches() > 0) {
                int N = cfg.getGpuNumBatches();
                batchSizeHint = (numSplits + N - 1) / N;
                batchDesc = N + " batches → batchSize=" + batchSizeHint;
            } else if (cfg.getGpuBatchSize() > 0) {
                batchSizeHint = cfg.getGpuBatchSize();
                batchDesc = "explicit batchSize=" + batchSizeHint;
            } else if (cfg.isGpuVramControlFactorSet()) {
                // Manual resident-relative sizing:  mem(batch) = F × mem(resident)
                double F           = cfg.getGpuVramControlFactor();
                long   residentMem = modeDataMem + orderingMem;
                long   batchMem    = (long)(F * residentMem);
                long   perSplit    = 10L * Integer.BYTES + Long.BYTES;              // 48 B/split
                batchSizeHint      = (int) Math.max(1, Math.min(numSplits, batchMem / perSplit));
                int numBatches     = (numSplits + batchSizeHint - 1) / batchSizeHint;
                batchDesc = String.format(
                    "vram-control-factor=%.3f  resident=%.1f MB (%s=%.1f orderings=%.1f)  batch=%.1f MB  → %d batches",
                    F, residentMem / 1e6, modeDataDesc, modeDataMem / 1e6, orderingMem / 1e6, batchMem / 1e6, numBatches);
            } else {
                // Default: auto — pass 0 to native; native queries free VRAM after static upload
                // and computes batchSize = floor(freeVRAM * vramFraction / 48 B)
                batchSizeHint = 0;
                batchDesc = String.format("auto (free-VRAM adaptive, occupancy=%.0f%%)",
                    cfg.getGpuVramFraction() * 100);
            }

            boolean ok;
            if (smallerSide) {
                Logging.info("Weight table: GPU path (smaller-side traversal)  splits=%d  uniqueParts=%d  trees=%d  batching=%s",
                    numSplits, partTable.size(), partTrees.size(), batchDesc);
                ok = computeScoresGPUSmallerSide(splitList, partTable, clusterTable,
                                                 clusterTrees, partTrees, numGpuTrees, scoreArray, scoreArrayD,
                                                 batchSizeHint, cfg.getGpuVramFraction());
            } else {
                Logging.info("Weight table: GPU path (prefix-sum tree-DP)  splits=%d  uniqueParts=%d  trees=%d  maxLeaf=%d  batching=%s",
                    numSplits, csr.totalNodes, partTrees.size(), csr.maxLeafCount, batchDesc);
                ok = computeScoresGPUPrefixSum(splitList, csr, clusterTable, clusterTrees, partTrees,
                                               numGpuTrees, scoreArray, scoreArrayD, batchSizeHint, cfg.getGpuVramFraction());
            }
            if (!ok) {
                Logging.info("GPU weight path infeasible (e.g. shared-memory limit), falling back to CPU");
                computeScoresCPU(splitList, partTable.entries(), clusterTable,
                                 clusterTrees, partTrees, scoreArray, scoreArrayD);
            }
        } else {
            if (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU) {
                Logging.info("GPU library not available, falling back to CPU");
            }
            computeScoresCPU(splitList, partTable.entries(), clusterTable,
                             clusterTrees, partTrees, scoreArray, scoreArrayD);
        }

        if (useDouble) {
            for (int i = 0; i < numSplits; i++) {
                double s = scoreArrayD[i];
                scoresD.put(splitList.get(i), s);
                if (s > maxScoreD) maxScoreD = s;
                totalScoreD += s;
            }
            long ms = (System.nanoTime() - t0) / 1_000_000;
            Logging.info("Weight table: %d splits scored [DOUBLE], maxScore=%.6e, totalScore=%.6e in %d ms",
                scoresD.size(), maxScoreD, totalScoreD, ms);
        } else {
            for (int i = 0; i < numSplits; i++) {
                scores.put(splitList.get(i), scoreArray[i]);
                if (scoreArray[i] > maxScore) maxScore = scoreArray[i];
                totalScore += scoreArray[i];
            }
            long ms = (System.nanoTime() - t0) / 1_000_000;
            Logging.info("Weight table: %d splits scored [LONG], maxScore=%d, totalScore=%d in %d ms",
                scores.size(), maxScore, totalScore, ms);
        }
    }

    // -------------------------------------------------------------------------
    // Numeric-precision decision: exact LONG vs floating-point DOUBLE
    // -------------------------------------------------------------------------

    /**
     * Decide whether weight scores must be accumulated as {@code double} to avoid
     * 64-bit integer overflow.
     *
     * <p>For a candidate split scored against {@code numGenes} gene trees, the
     * exact (doubled) quartet score is bounded by roughly
     * {@code numGenes · C(n,4) · 2 ≈ numGenes · n^4 / 12}.  We compare this
     * estimate against {@code Long.MAX_VALUE / 8} (an 8× safety margin that also
     * covers intermediate {@code freq · 2·QI} products and partial sums).  When
     * the estimate exceeds that bound, exact {@code long} arithmetic would wrap
     * around (producing the notorious negative scores), so we switch to
     * {@code double}.
     *
     * <p>Overridable for testing via environment variables
     * {@code ASTRALX_WEIGHT_FORCE_DOUBLE} / {@code ASTRALX_WEIGHT_FORCE_LONG}.
     */
    static boolean needsDoubleAccumulation(int n, int numGenes) {
        if (System.getenv("ASTRALX_WEIGHT_FORCE_DOUBLE") != null) return true;
        if (System.getenv("ASTRALX_WEIGHT_FORCE_LONG")   != null) return false;
        return estimatedMaxTwoScore(n, numGenes) > longSafeBound();
    }

    /** Estimated maximum per-split doubled score ≈ numGenes · n^4 / 12. */
    private static double estimatedMaxTwoScore(int n, int numGenes) {
        double nn = (double) n;
        return (double) numGenes * nn * nn * nn * nn / 12.0;
    }

    /** Long-safe bound with an 8× margin for intermediate products/sums. */
    private static double longSafeBound() {
        return (double) Long.MAX_VALUE / 8.0;   // ≈ 1.153e18
    }

    /** Emit a prominent, human-readable log line stating the chosen score type and why. */
    private static void logAccumulationDecision(int n, int numGenes, boolean useDouble) {
        double est  = estimatedMaxTwoScore(n, numGenes);
        double safe = longSafeBound();
        if (useDouble) {
            Logging.info("Weight accumulation: DOUBLE (64-bit floating point, ~15-16 significant digits)  "
                + "[taxa=%d, genes=%d, est. max 2·score ≈ %.2e exceeds long-safe %.2e]  "
                + "— switched to avoid 64-bit integer overflow; scores are approximate but "
                + "topologically equivalent.", n, numGenes, est, safe);
        } else {
            Logging.info("Weight accumulation: LONG (exact 64-bit integer)  "
                + "[taxa=%d, genes=%d, est. max 2·score ≈ %.2e within long-safe %.2e].",
                n, numGenes, est, safe);
        }
    }

    // -------------------------------------------------------------------------
    // GPU path
    // -------------------------------------------------------------------------

    /**
     * Flatten all data to primitive arrays, call the CUDA kernel via JNI,
     * and write results (score = twoScore/2) into scoreArray.
     */
    /**
     * GPU weight calculation.
     *
     * When clusterTrees == partTrees (no autocomplete), the orderings/invIndex array has
     * numGpuTrees = k slots (indices 0..k-1) and partition treeIndex values are unchanged.
     *
     * When clusterTrees != partTrees (autocomplete active), numGpuTrees = 2k:
     *   slots 0..k-1   → completed tree orderings/invIndex  (for cluster lookups)
     *   slots k..2k-1  → original tree orderings/invIndex   (for gene-tree/partition lookups)
     * Partition treeIndex values are stored as (p.treeIndex + k) so the kernel naturally
     * reads from the original-tree half of the combined array — no kernel changes needed.
     */
    private boolean computeScoresGPUPrefixSum(List<BipartitionSplit> splitList,
                                               NodeCSR csr,
                                               ClusterTable clusterTable,
                                               List<Tree> clusterTrees,
                                               List<Tree> partTrees,
                                               int numGpuTrees,
                                               long[] scoreArray,
                                               double[] scoreArrayD,
                                               int batchSizeHint,
                                               double vramFraction) {
        int numSplits      = splitList.size();
        int numPartTrees   = partTrees.size();
        boolean splitTrees = (clusterTrees != partTrees);
        int partTreeOffset = splitTrees ? clusterTrees.size() : 0;

        int[] splitsData = buildSplitsData(splitList, clusterTable);
        int[][] oi       = buildOrderingsInvIndex(clusterTrees, partTrees, numGpuTrees);
        int[] orderings  = oi[0], invIndex = oi[1];

        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsGPU(
            splitsData, csr.nodeData, csr.nodeFreq, csr.nodeOffset, csr.partLeafCount,
            orderings, invIndex,
            numSplits, numPartTrees, partTreeOffset, csr.maxLeafCount,
            numGpuTrees, n,
            batchSizeHint, vramFraction, useDouble);
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;

        splitsData = null; orderings = null; invIndex = null;   // let GC reclaim
        if (twoScores == null) {
            Logging.info("  GPU kernel returned null after %d ms (infeasible)", gpuMs);
            return false;
        }
        Logging.info("  GPU kernel returned in %d ms", gpuMs);
        unpackTwoScores(twoScores, scoreArray, scoreArrayD, numSplits);
        return true;
    }

    /**
     * Convert the raw per-split doubled-score transport array from the GPU into
     * final per-split scores (score = 2·score / 2).
     *
     * <p>In LONG mode each {@code twoScores[i]} is the exact integer 2·score.
     * In DOUBLE mode the kernel stores the IEEE-754 bit pattern of the
     * floating-point 2·score in the long slot (see {@code __double_as_longlong}
     * in the kernel), which we recover with {@link Double#longBitsToDouble}.
     */
    private void unpackTwoScores(long[] twoScores, long[] scoreArray,
                                 double[] scoreArrayD, int numSplits) {
        if (useDouble) {
            for (int i = 0; i < numSplits; i++)
                scoreArrayD[i] = Double.longBitsToDouble(twoScores[i]) / 2.0;
        } else {
            for (int i = 0; i < numSplits; i++)
                scoreArray[i] = twoScores[i] / 2L;
        }
    }

    /**
     * Legacy GPU path: smaller-side traversal, no prefix sums.
     *
     * Packs the deduplicated tripartitions into the 9-int "parts" layout and calls
     * the one-thread-per-split kernel that counts each intersection by walking the
     * smaller range.  Uses the same splits and orderings/invIndex layout as the
     * prefix-sum path; builds NO node CSR and NO prefix working memory.
     */
    private boolean computeScoresGPUSmallerSide(List<BipartitionSplit> splitList,
                                                 PartitionTable partTable,
                                                 ClusterTable clusterTable,
                                                 List<Tree> clusterTrees,
                                                 List<Tree> partTrees,
                                                 int numGpuTrees,
                                                 long[] scoreArray,
                                                 double[] scoreArrayD,
                                                 int batchSizeHint,
                                                 double vramFraction) {
        int numSplits      = splitList.size();
        boolean splitTrees = (clusterTrees != partTrees);
        int partTreeOffset = splitTrees ? clusterTrees.size() : 0;

        int[] splitsData = buildSplitsData(splitList, clusterTable);

        // --- parts: numParts * 9 ints (deduplicated tripartitions) ---
        // treeIdx stored as (p.treeIndex + partTreeOffset) so the kernel reads the
        // original-tree half of the combined orderings/invIndex.
        // [treeIdx, lo1, hi1, lo2, hi2, sz1, sz2, sz3, frequency]
        int numParts = partTable.size();
        int[] partsData = new int[numParts * 9];
        int j = 0;
        for (PartitionTable.Entry pe : partTable.entries()) {
            Partition p = pe.exemplar;
            int base = j * 9;
            partsData[base + 0] = p.treeIndex + partTreeOffset;
            partsData[base + 1] = p.leftStart;
            partsData[base + 2] = p.leftEnd;
            partsData[base + 3] = p.rightStart;
            partsData[base + 4] = p.rightEnd;
            partsData[base + 5] = p.size1;
            partsData[base + 6] = p.size2;
            partsData[base + 7] = p.size3;
            partsData[base + 8] = pe.frequency;
            j++;
        }

        int[][] oi      = buildOrderingsInvIndex(clusterTrees, partTrees, numGpuTrees);
        int[] orderings = oi[0], invIndex = oi[1];

        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsSmallerSideGPU(
            splitsData, partsData, orderings, invIndex,
            numSplits, numParts, numGpuTrees, n, n,
            batchSizeHint, vramFraction, useDouble);
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;

        splitsData = null; partsData = null; orderings = null; invIndex = null;   // let GC reclaim
        if (twoScores == null) {
            Logging.info("  GPU kernel returned null after %d ms (infeasible)", gpuMs);
            return false;
        }
        Logging.info("  GPU kernel returned in %d ms", gpuMs);
        unpackTwoScores(twoScores, scoreArray, scoreArrayD, numSplits);
        return true;
    }

    // --- shared GPU input packers (identical layout for both kernels) ---

    /**
     * splits: numSplits * 10 ints.  Cluster treeIndex values are 0..k-1 (completed
     * trees, used for membership).
     * [aTree, aLo, aHi, aComp, aSize, bTree, bLo, bHi, bComp, bSize]
     */
    private int[] buildSplitsData(List<BipartitionSplit> splitList, ClusterTable clusterTable) {
        int numSplits = splitList.size();
        int[] splitsData = new int[numSplits * 10];
        for (int i = 0; i < numSplits; i++) {
            BipartitionSplit split = splitList.get(i);
            ClusterTable.Entry eA = clusterTable.get(split.lo);
            ClusterTable.Entry eB = clusterTable.get(split.hi);
            int base = i * 10;
            if (eA != null && eB != null) {
                Cluster cA = eA.exemplar, cB = eB.exemplar;
                splitsData[base + 0] = cA.treeIndex;
                splitsData[base + 1] = cA.left;
                splitsData[base + 2] = cA.right;
                splitsData[base + 3] = cA.complement ? 1 : 0;
                splitsData[base + 4] = cA.size;
                splitsData[base + 5] = cB.treeIndex;
                splitsData[base + 6] = cB.left;
                splitsData[base + 7] = cB.right;
                splitsData[base + 8] = cB.complement ? 1 : 0;
                splitsData[base + 9] = cB.size;
            }
            // else: all zeros → empty clusters → kernel yields score 0 for this split.
        }
        return splitsData;
    }

    /**
     * orderings + invIndex: numGpuTrees * n ints each.
     *   orderings[t*n + pos]   = postorderArray[pos]
     *   invIndex [t*n + taxon] = positionMap[taxon]  (-1 if absent)
     *
     * Layout when splitTrees (autocomplete active):
     *   slots 0..k-1   from clusterTrees (completed)  — cluster membership
     *   slots k..2k-1  from partTrees (original)      — gene-tree leaves
     * Otherwise the single list fills slots 0..k-1.
     *
     * @return int[2][] = {orderings, invIndex}
     */
    private int[][] buildOrderingsInvIndex(List<Tree> clusterTrees, List<Tree> partTrees,
                                            int numGpuTrees) {
        int numClusterTrees = clusterTrees.size();
        boolean splitTrees  = (clusterTrees != partTrees);
        int[] orderings = new int[numGpuTrees * n];
        int[] invIndex  = new int[numGpuTrees * n];
        Arrays.fill(invIndex, -1);
        for (int t = 0; t < numClusterTrees; t++) {
            Tree tree = clusterTrees.get(t);
            int base = t * n;
            for (int pos = 0; pos < tree.leafCount; pos++) orderings[base + pos] = tree.postorderArray[pos];
            for (int taxon = 0; taxon < n; taxon++)        invIndex[base + taxon] = tree.positionMap[taxon];
        }
        if (splitTrees) {
            int numPartTrees = partTrees.size();
            for (int t = 0; t < numPartTrees; t++) {
                Tree tree = partTrees.get(t);
                int base = (numClusterTrees + t) * n;
                for (int pos = 0; pos < tree.leafCount; pos++) orderings[base + pos] = tree.postorderArray[pos];
                for (int taxon = 0; taxon < n; taxon++)        invIndex[base + taxon] = tree.positionMap[taxon];
            }
        }
        return new int[][]{ orderings, invIndex };
    }

    // -------------------------------------------------------------------------
    // Per-tree internal-node CSR (gene-tree tripartitions as leaf intervals)
    // -------------------------------------------------------------------------

    /**
     * Compact, per-exemplar-tree representation of the DEDUPLICATED gene-tree
     * tripartitions.  Each unique tripartition (M1|M2|M3) is stored once, as a
     * contiguous leaf interval (lo, mid, hi) of its exemplar tree (M1 = [lo,mid),
     * M2 = [mid,hi), M3 = Lg \ [lo,hi)), together with its frequency (how many
     * gene-tree nodes realize it).  Entries are bucketed by exemplar tree so the
     * GPU kernel can build each tree's leaf prefix sums once and score only that
     * tree's unique tripartitions.
     *
     * This recovers cross-tree dedup at O(L) working memory (one tree's prefix
     * live at a time) — see DOCS/weight-dedup-by-exemplar-tree-design.md.
     * Scoring is bit-identical to summing QI over every node individually,
     * because Σ_nodes QI ≡ Σ_unique frequency·QI.
     */
    private static final class NodeCSR {
        int[] nodeData;       // numUnique * 3   [lo, mid, hi] of the exemplar
        int[] nodeFreq;       // numUnique       frequency (occurrence count)
        int[] nodeOffset;     // numTrees + 1    CSR row pointers (bucket by exemplar tree)
        int[] partLeafCount;  // numTrees        leaf count L per tree
        int   maxLeafCount;   // max L over trees with ≥1 exemplar (shared-mem sizing)
        int   totalNodes;     // numUnique
    }

    /**
     * Build the deduplicated node CSR from the already-computed PartitionTable,
     * bucketing unique tripartitions by their exemplar tree index.
     */
    private static NodeCSR buildDedupNodeCSR(PartitionTable partTable,
                                              List<Tree> partTrees) {
        int numTrees = partTrees.size();

        // Pass 1: count unique tripartitions per exemplar tree → CSR offsets.
        int[] nodeOffset = new int[numTrees + 1];
        for (PartitionTable.Entry e : partTable.entries()) {
            nodeOffset[e.exemplar.treeIndex + 1]++;
        }
        for (int g = 0; g < numTrees; g++) nodeOffset[g + 1] += nodeOffset[g];
        int total = nodeOffset[numTrees];          // == numUnique == partTable.size()
        if ((long) total * 3 > Integer.MAX_VALUE) {
            throw new IllegalStateException("Too many tripartitions for a single int[] CSR: " + total);
        }

        // Pass 2: scatter each unique tripartition into its exemplar tree's bucket.
        int[] nodeData = new int[total * 3];
        int[] nodeFreq = new int[total];
        int[] cursor   = nodeOffset.clone();       // per-tree write cursor
        for (PartitionTable.Entry e : partTable.entries()) {
            Partition p = e.exemplar;
            int pos = cursor[p.treeIndex]++;
            int b   = pos * 3;
            nodeData[b]     = p.leftStart;          // lo
            nodeData[b + 1] = p.leftEnd;            // mid  (= p.rightStart)
            nodeData[b + 2] = p.rightEnd;           // hi
            nodeFreq[pos]   = e.frequency;
        }

        // Per-tree leaf counts; maxLeaf only over trees that actually need a prefix
        // (an exemplar-empty tree is skipped by the kernel, so its L never matters).
        int[] partLeafCount = new int[numTrees];
        int   maxLeaf = 0;
        for (int g = 0; g < numTrees; g++) {
            int L = partTrees.get(g).leafCount;
            partLeafCount[g] = L;
            if (nodeOffset[g + 1] > nodeOffset[g] && L > maxLeaf) maxLeaf = L;
        }

        NodeCSR csr = new NodeCSR();
        csr.nodeData      = nodeData;
        csr.nodeFreq      = nodeFreq;
        csr.nodeOffset    = nodeOffset;
        csr.partLeafCount = partLeafCount;
        csr.maxLeafCount  = maxLeaf;
        csr.totalNodes    = total;
        return csr;
    }

    // -------------------------------------------------------------------------
    // CPU path (also the fallback when the GPU path is infeasible)
    // -------------------------------------------------------------------------

    private void computeScoresCPU(List<BipartitionSplit> splitList,
                                   Collection<PartitionTable.Entry> partitions,
                                   ClusterTable clusterTable,
                                   List<Tree> clusterTrees, List<Tree> partTrees,
                                   long[] scoreArray, double[] scoreArrayD) {
        int numSplits = splitList.size();
        // CPU: parallel over splits (TRACE: single-threaded for deterministic output)
        if (Logging.isTrace()) {
            for (int idx = 0; idx < numSplits; idx++) {
                BipartitionSplit sp = splitList.get(idx);
                Logging.trace("SPLIT sz=%d|%d  lo=%s  hi=%s",
                    sp.lo.size, sp.hi.size, sp.lo, sp.hi);
                if (useDouble) {
                    scoreArrayD[idx] = computeScoreD(sp, partitions, clusterTable, clusterTrees, partTrees);
                    Logging.trace("  => score=%.6e", scoreArrayD[idx]);
                } else {
                    scoreArray[idx] = computeScore(sp, partitions, clusterTable, clusterTrees, partTrees);
                    Logging.trace("  => score=%d", scoreArray[idx]);
                }
            }
        } else {
            java.util.concurrent.atomic.AtomicInteger wDone = new java.util.concurrent.atomic.AtomicInteger(0);
            ProgressBar wBar = new ProgressBar("Scoring splits (CPU)", numSplits);
            Threading.processRangeParallel(numSplits, idx -> {
                if (useDouble) {
                    scoreArrayD[idx] = computeScoreD(splitList.get(idx), partitions, clusterTable,
                                                     clusterTrees, partTrees);
                } else {
                    scoreArray[idx] = computeScore(splitList.get(idx), partitions, clusterTable,
                                                   clusterTrees, partTrees);
                }
                wBar.update(wDone.incrementAndGet());
            });
            wBar.done();
        }
    }

    // -------------------------------------------------------------------------
    // CPU path
    // -------------------------------------------------------------------------

    private long computeScore(BipartitionSplit split,
                               Collection<PartitionTable.Entry> partitions,
                               ClusterTable clusterTable,
                               List<Tree> clusterTrees, List<Tree> partTrees) {
        // Retrieve exemplars for A (lo half) and B (hi half)
        ClusterTable.Entry eA = clusterTable.get(split.lo);
        ClusterTable.Entry eB = clusterTable.get(split.hi);
        if (eA == null || eB == null) return 0L;

        Cluster cA = eA.exemplar;
        Cluster cB = eB.exemplar;
        // Cluster positions are in completed trees; use clusterTrees for position lookup.
        Tree tA = clusterTrees.get(cA.treeIndex);
        Tree tB = clusterTrees.get(cB.treeIndex);
        int sizeA = cA.size;
        int sizeB = cB.size;
        int sizeC = n - sizeA - sizeB;
        if (sizeC < 0) return 0L;  // sanity

        long twoScore = 0L;

        for (PartitionTable.Entry pe : partitions) {
            Partition p = pe.exemplar;
            // Partition positions are in original trees; use partTrees for gene-tree lookup.
            Tree tGT = partTrees.get(p.treeIndex);

            // M1 = [leftStart, leftEnd), M2 = [rightStart, rightEnd)
            int lo1 = p.leftStart,  hi1 = p.leftEnd;   // M1 range
            int lo2 = p.rightStart, hi2 = p.rightEnd;  // M2 range
            int sz1 = p.size1, sz2 = p.size2, sz3 = p.size3;

            // 4 core intersections
            int a0 = IntersectionCounter.intersect(tGT, lo1, hi1, tA, cA.left, cA.right, cA.complement, sz1);
            int a1 = IntersectionCounter.intersect(tGT, lo2, hi2, tA, cA.left, cA.right, cA.complement, sz2);
            int b0 = IntersectionCounter.intersect(tGT, lo1, hi1, tB, cB.left, cB.right, cB.complement, sz1);
            int b1 = IntersectionCounter.intersect(tGT, lo2, hi2, tB, cB.left, cB.right, cB.complement, sz2);

            // Row sums: for incomplete gene trees, |A∩Lg_GT| < sizeA; must compute explicitly
            int lgA = tGT.isComplete ? sizeA
                    : IntersectionCounter.intersectWithFullTree(tGT, tA, cA.left, cA.right, cA.complement);
            int lgB = tGT.isComplete ? sizeB
                    : IntersectionCounter.intersectWithFullTree(tGT, tB, cB.left, cB.right, cB.complement);

            // Derive remaining 5
            int a2 = lgA - a0 - a1;            // row constraint on A (w.r.t. Lg_GT)
            int b2 = lgB - b0 - b1;            // row constraint on B
            int c0 = sz1 - a0 - b0;            // column constraint M1
            int c1 = sz2 - a1 - b1;            // column constraint M2
            int c2 = sz3 - a2 - b2;            // column constraint M3 (correct formula)

            // All values must be non-negative for a valid intersection matrix
            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) {
                Logging.trace("    SKIP  tGT=%d sz=%d|%d|%d lgA=%d lgB=%d "
                    + "a=[%d,%d,%d] b=[%d,%d,%d] c=[%d,%d,%d]",
                    p.treeIndex, sz1, sz2, sz3, lgA, lgB,
                    a0,a1,a2, b0,b1,b2, c0,c1,c2);
                continue;
            }

            long twoQI = computeTwoQI(a0, a1, a2, b0, b1, b2, c0, c1, c2);
            Logging.trace("    PART  tGT=%d sz=%d|%d|%d lgA=%d lgB=%d "
                + "a=[%d,%d,%d] b=[%d,%d,%d] c=[%d,%d,%d] 2*QI=%d freq=%d",
                p.treeIndex, sz1, sz2, sz3, lgA, lgB,
                a0,a1,a2, b0,b1,b2, c0,c1,c2, twoQI, pe.frequency);
            twoScore += (long) pe.frequency * twoQI;
        }

        // twoScore = 2 * score; score = sum of freq * QI, each QI is integer
        return twoScore / 2;
    }

    /**
     * Compute 2*QI = sum over 6 permutations (i,j,k) of {0,1,2}:
     *   a[i] * b[j] * c[k] * (a[i] + b[j] + c[k] - 3)
     *
     * This is always a non-negative even integer, so QI = result/2 is exact.
     */
    private static long computeTwoQI(int a0, int a1, int a2,
                                      int b0, int b1, int b2,
                                      int c0, int c1, int c2) {
        long[] a = {a0, a1, a2};
        long[] b = {b0, b1, b2};
        long[] c = {c0, c1, c2};

        long sum = 0;
        // All 6 permutations: (i,j,k) with i != j, j != k, i != k
        int[][] perms = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        for (int[] perm : perms) {
            long ai = a[perm[0]], bj = b[perm[1]], ck = c[perm[2]];
            long s = ai + bj + ck - 3;
            if (s > 0) sum += ai * bj * ck * s;
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // CPU path — DOUBLE variant (used when needsDoubleAccumulation() is true)
    //
    // Mirror of computeScore()/computeTwoQI() with floating-point accumulation so
    // very large taxon sets (where the exact integer 2·score overflows long) do
    // not wrap around.  The integer intersection matrix is computed identically;
    // only the QI products and the running total are doubles.
    // -------------------------------------------------------------------------

    private double computeScoreD(BipartitionSplit split,
                                  Collection<PartitionTable.Entry> partitions,
                                  ClusterTable clusterTable,
                                  List<Tree> clusterTrees, List<Tree> partTrees) {
        ClusterTable.Entry eA = clusterTable.get(split.lo);
        ClusterTable.Entry eB = clusterTable.get(split.hi);
        if (eA == null || eB == null) return 0.0;

        Cluster cA = eA.exemplar;
        Cluster cB = eB.exemplar;
        Tree tA = clusterTrees.get(cA.treeIndex);
        Tree tB = clusterTrees.get(cB.treeIndex);
        int sizeA = cA.size;
        int sizeB = cB.size;
        int sizeC = n - sizeA - sizeB;
        if (sizeC < 0) return 0.0;

        double twoScore = 0.0;

        for (PartitionTable.Entry pe : partitions) {
            Partition p = pe.exemplar;
            Tree tGT = partTrees.get(p.treeIndex);

            int lo1 = p.leftStart,  hi1 = p.leftEnd;
            int lo2 = p.rightStart, hi2 = p.rightEnd;
            int sz1 = p.size1, sz2 = p.size2, sz3 = p.size3;

            int a0 = IntersectionCounter.intersect(tGT, lo1, hi1, tA, cA.left, cA.right, cA.complement, sz1);
            int a1 = IntersectionCounter.intersect(tGT, lo2, hi2, tA, cA.left, cA.right, cA.complement, sz2);
            int b0 = IntersectionCounter.intersect(tGT, lo1, hi1, tB, cB.left, cB.right, cB.complement, sz1);
            int b1 = IntersectionCounter.intersect(tGT, lo2, hi2, tB, cB.left, cB.right, cB.complement, sz2);

            int lgA = tGT.isComplete ? sizeA
                    : IntersectionCounter.intersectWithFullTree(tGT, tA, cA.left, cA.right, cA.complement);
            int lgB = tGT.isComplete ? sizeB
                    : IntersectionCounter.intersectWithFullTree(tGT, tB, cB.left, cB.right, cB.complement);

            int a2 = lgA - a0 - a1;
            int b2 = lgB - b0 - b1;
            int c0 = sz1 - a0 - b0;
            int c1 = sz2 - a1 - b1;
            int c2 = sz3 - a2 - b2;

            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

            double twoQI = computeTwoQIDouble(a0, a1, a2, b0, b1, b2, c0, c1, c2);
            twoScore += (double) pe.frequency * twoQI;
        }

        return twoScore / 2.0;
    }

    /** Floating-point mirror of {@link #computeTwoQI}; same formula, double accumulation. */
    private static double computeTwoQIDouble(int a0, int a1, int a2,
                                              int b0, int b1, int b2,
                                              int c0, int c1, int c2) {
        double[] a = {a0, a1, a2};
        double[] b = {b0, b1, b2};
        double[] c = {c0, c1, c2};

        double sum = 0.0;
        int[][] perms = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        for (int[] perm : perms) {
            double ai = a[perm[0]], bj = b[perm[1]], ck = c[perm[2]];
            double s = ai + bj + ck - 3;
            if (s > 0) sum += ai * bj * ck * s;
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    /** Whether scores are floating-point (true) or exact integers (false). */
    public boolean isDouble() { return useDouble; }

    /**
     * Score of a split as a long.  In DOUBLE mode this returns the value rounded
     * to the nearest long (used only by debug/verifier tooling); the DP must use
     * {@link #getScoreD} when {@link #isDouble()} is true.
     */
    public long getScore(BipartitionSplit split) {
        return useDouble ? Math.round(scoresD.getOrDefault(split, 0.0))
                         : scores.getOrDefault(split, 0L);
    }

    /** Score of a split as a double (valid in both modes). */
    public double getScoreD(BipartitionSplit split) {
        return useDouble ? scoresD.getOrDefault(split, 0.0)
                         : (double) scores.getOrDefault(split, 0L);
    }

    public long   getMaxScore()    { return useDouble ? Math.round(maxScoreD)   : maxScore; }
    public long   getTotalScore()  { return useDouble ? Math.round(totalScoreD) : totalScore; }
    public double getMaxScoreD()   { return useDouble ? maxScoreD   : (double) maxScore; }
    public double getTotalScoreD() { return useDouble ? totalScoreD : (double) totalScore; }
    public int    size()           { return useDouble ? scoresD.size() : scores.size(); }

    /** Iterate all (split, score) pairs (LONG mode only; empty in DOUBLE mode). */
    public Set<Map.Entry<BipartitionSplit, Long>> entries() { return scores.entrySet(); }
}
