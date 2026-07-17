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
import astralx.tree.TreeNode;
import astralx.util.Int128;
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

    /**
     * Numeric type used for scores. Decided once per run from the problem size:
     * LONG below the overflow threshold; above it, INT128 (exact, default) or
     * DOUBLE (approximate) per {@link Config#getLargeScoreType()}.
     */
    public enum Mode { LONG, DOUBLE, INT128 }

    private final Map<BipartitionSplit, Long>    scores  = new HashMap<>();   // LONG path
    private final Map<BipartitionSplit, Double>  scoresD = new HashMap<>();   // DOUBLE path
    private final Map<BipartitionSplit, Int128>  scoresI = new HashMap<>();   // INT128 path
    private final int n;   // total taxa

    private final Mode mode;
    private final boolean useDouble;   // mode == DOUBLE
    private final boolean useInt128;   // mode == INT128

    // stats (LONG path)
    private long maxScore;
    private long totalScore;
    // stats (DOUBLE path)
    private double maxScoreD = Double.NEGATIVE_INFINITY;
    private double totalScoreD;
    // stats (INT128 path)
    private Int128 maxScoreI   = null;          // null = -inf sentinel
    private Int128 totalScoreI = Int128.ZERO;

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

        // ── Numeric-precision decision (LONG vs DOUBLE/INT128 accumulation) ───
        // The exact quartet score grows as O(genes · n^4) and overflows a signed
        // 64-bit integer for very large taxon sets.  Below the overflow threshold
        // we keep exact LONG.  Above it we use INT128 (exact, default) or DOUBLE
        // (approximate) per the configured large-score type.
        int numGenes = partTrees.size();
        if (needsDoubleAccumulation(n, numGenes)) {
            this.mode = (Config.getInstance().getLargeScoreType() == Config.LargeScoreType.DOUBLE)
                        ? Mode.DOUBLE : Mode.INT128;
        } else {
            this.mode = Mode.LONG;
        }
        this.useDouble = (mode == Mode.DOUBLE);
        this.useInt128 = (mode == Mode.INT128);
        logAccumulationDecision(n, numGenes, mode);

        // Collect all unique splits from DPTable into an indexed list.  Optionally
        // prune splits whose parent cluster is unreachable from the DP root — the
        // top-down inference DP never scores those, so skipping them is
        // result-preserving (see DPTable.reachableClusters).
        List<BipartitionSplit> splitList = new ArrayList<>();
        if (Config.getInstance().isPruneUnreachableSplits()) {
            Set<ClusterHash> reachable = dpTable.reachableClusters();
            int totalClusters = 0, keptClusters = 0, totalSplits = 0;
            for (var entry : dpTable.entries()) {
                totalClusters++;
                int sz = entry.getValue().size();
                totalSplits += sz;
                if (reachable.contains(entry.getKey())) {
                    splitList.addAll(entry.getValue());
                    keptClusters++;
                }
            }
            int kept = splitList.size();
            Logging.info("DP reachability prune: clusters %d/%d reachable (%.1f%%), "
                + "splits %d/%d need to be scored (%.1f%%) — %d unreachable splits skipped",
                keptClusters, totalClusters, 100.0 * keptClusters / Math.max(1, totalClusters),
                kept, totalSplits, 100.0 * kept / Math.max(1, totalSplits), totalSplits - kept);
        } else {
            for (var entry : dpTable.entries()) splitList.addAll(entry.getValue());
        }
        int numSplits = splitList.size();

        // Per-split score buffers — exactly one is non-null, matching the mode.
        long[]   scoreArray  = (mode == Mode.LONG)   ? new long[numSplits]    : null;  // exact integer
        double[] scoreArrayD = (mode == Mode.DOUBLE) ? new double[numSplits]  : null;  // floating point
        Int128[] scoreArrayI = (mode == Mode.INT128) ? new Int128[numSplits]  : null;  // exact 128-bit

        // When clusterTrees != partTrees, the GPU path packs both sets of
        // orderings/invIndex into a combined array (slots 0..C-1 = cluster trees,
        // slots C..C+P-1 = partition trees, C = clusterTrees.size()) and offsets
        // partition tree indices by C.  numGpuTrees is the combined count.
        //   - autocomplete inference: C == P == k  (completed vs original gene trees)
        //   - score-only mode:        C == 1, P == numGeneTrees  (single species tree)
        // The general C + P form is byte-identical to the old 2k for the C == P case.
        boolean splitTrees = (clusterTrees != partTrees);
        int numGpuTrees = splitTrees ? clusterTrees.size() + partTrees.size() : clusterTrees.size();

        boolean useGPU = (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU)
                         && GPUWeightCalculator.tryLoad();

        // Polytomous gene-tree partitions (d > 3) are scored natively on BOTH GPU paths:
        // prefix-sum via scorePolyNodes (O(1) memory, any degree), and smaller-side via the
        // two-pass-rewalk poly loop (polytomy-design.md §3.8.4).  No CPU fallback is forced
        // by polytomy alone — CPU is used only if the chosen GPU path is otherwise infeasible.

        // Multi-range clusters (consensus emission bridge) are handled fully ON GPU
        // via the two-tier range-CSR: buildSplitRangeData packs each multi-range split
        // side's ranges (single-range sides carry count 0 → the byte-identical fast
        // path), and both kernels sum the intersection over a side's ranges
        // (DOCS/multi-range-cluster-design.md §5.2/§5.3). No CPU correction needed.

        Config.WeightIntersectionMethod method = Config.getInstance().getWeightIntersectionMethod();
        boolean bitset   = (method == Config.WeightIntersectionMethod.BITSET);
        boolean treeWalk = (method == Config.WeightIntersectionMethod.SIMPLE_TREE_WALK);

        if (useGPU && treeWalk) {
            boolean ok = computeScoresGPUTreeWalk(splitList, clusterTable,
                                                  clusterTrees, partTrees,
                                                  scoreArray, scoreArrayD, scoreArrayI);
            if (!ok) {
                Logging.info("GPU tree-walk weight path infeasible, falling back to CPU tree-walk");
                computeScoresCPUTreeWalk(splitList, clusterTable, clusterTrees, partTrees,
                                         scoreArray, scoreArrayD, scoreArrayI);
            }
        } else if (useGPU && bitset) {
            boolean ok = computeScoresGPUBitset(splitList, partTable, clusterTable,
                                                clusterTrees, partTrees,
                                                scoreArray, scoreArrayD, scoreArrayI);
            if (!ok) {
                Logging.info("GPU bitset weight path infeasible, falling back to CPU bitset");
                computeScoresCPUBitset(splitList, partTable, clusterTable,
                                       clusterTrees, partTrees,
                                       scoreArray, scoreArrayD, scoreArrayI);
            }
        } else if (useGPU) {
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
                                                 clusterTrees, partTrees, numGpuTrees,
                                                 scoreArray, scoreArrayD, scoreArrayI,
                                                 batchSizeHint, cfg.getGpuVramFraction());
            } else {
                Logging.info("Weight table: GPU path (prefix-sum tree-DP)  splits=%d  uniqueParts=%d  trees=%d  maxLeaf=%d  batching=%s",
                    numSplits, csr.totalNodes, partTrees.size(), csr.maxLeafCount, batchDesc);
                ok = computeScoresGPUPrefixSum(splitList, csr, clusterTable, clusterTrees, partTrees,
                                               numGpuTrees, scoreArray, scoreArrayD, scoreArrayI,
                                               batchSizeHint, cfg.getGpuVramFraction());
            }
            if (!ok) {
                Logging.info("GPU weight path infeasible (e.g. shared-memory limit), falling back to CPU");
                computeScoresCPU(splitList, partTable.entries(), clusterTable,
                                 clusterTrees, partTrees, scoreArray, scoreArrayD, scoreArrayI);
            }
        } else {
            if (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU) {
                Logging.info("GPU library not available, falling back to CPU");
            }
            if (treeWalk) {
                computeScoresCPUTreeWalk(splitList, clusterTable, clusterTrees, partTrees,
                                         scoreArray, scoreArrayD, scoreArrayI);
            } else if (bitset) {
                computeScoresCPUBitset(splitList, partTable, clusterTable,
                                       clusterTrees, partTrees, scoreArray, scoreArrayD, scoreArrayI);
            } else {
                computeScoresCPU(splitList, partTable.entries(), clusterTable,
                                 clusterTrees, partTrees, scoreArray, scoreArrayD, scoreArrayI);
            }
        }

        long ms;
        if (mode == Mode.INT128) {
            for (int i = 0; i < numSplits; i++) {
                Int128 s = scoreArrayI[i];
                scoresI.put(splitList.get(i), s);
                if (maxScoreI == null || s.compareTo(maxScoreI) > 0) maxScoreI = s;
                totalScoreI = totalScoreI.add(s);
            }
            ms = (System.nanoTime() - t0) / 1_000_000;
            Logging.info("Weight table: %d splits scored [INT128], maxScore=%s, totalScore=%s in %d ms",
                scoresI.size(), maxScoreI, totalScoreI, ms);
        } else if (mode == Mode.DOUBLE) {
            for (int i = 0; i < numSplits; i++) {
                double s = scoreArrayD[i];
                scoresD.put(splitList.get(i), s);
                if (s > maxScoreD) maxScoreD = s;
                totalScoreD += s;
            }
            ms = (System.nanoTime() - t0) / 1_000_000;
            Logging.info("Weight table: %d splits scored [DOUBLE], maxScore=%.6e, totalScore=%.6e in %d ms",
                scoresD.size(), maxScoreD, totalScoreD, ms);
        } else {
            for (int i = 0; i < numSplits; i++) {
                scores.put(splitList.get(i), scoreArray[i]);
                if (scoreArray[i] > maxScore) maxScore = scoreArray[i];
                totalScore += scoreArray[i];
            }
            ms = (System.nanoTime() - t0) / 1_000_000;
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

    /** scoreMode int passed to the native kernel: 0=LONG, 1=DOUBLE, 2=INT128. */
    private int nativeScoreMode() {
        switch (mode) {
            case DOUBLE: return 1;
            case INT128: return 2;
            default:     return 0;
        }
    }

    /** Emit a prominent, human-readable log line stating the chosen score type and why. */
    private static void logAccumulationDecision(int n, int numGenes, Mode mode) {
        double est  = estimatedMaxTwoScore(n, numGenes);
        double safe = longSafeBound();
        switch (mode) {
            case INT128 -> Logging.info(
                "Weight accumulation: INT128 (exact 128-bit integer)  "
                + "[taxa=%d, genes=%d, est. max 2·score ≈ %.2e exceeds long-safe %.2e]  "
                + "— switched to avoid 64-bit integer overflow; scores remain exact "
                + "(full-rate integer math, no FP64 penalty).  Override with "
                + "--large-n-score-type double.", n, numGenes, est, safe);
            case DOUBLE -> Logging.info(
                "Weight accumulation: DOUBLE (64-bit floating point, ~15-16 significant digits)  "
                + "[taxa=%d, genes=%d, est. max 2·score ≈ %.2e exceeds long-safe %.2e]  "
                + "— switched to avoid 64-bit integer overflow; scores are approximate but "
                + "topologically equivalent.", n, numGenes, est, safe);
            default -> Logging.info(
                "Weight accumulation: LONG (exact 64-bit integer)  "
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
                                               Int128[] scoreArrayI,
                                               int batchSizeHint,
                                               double vramFraction) {
        int numSplits      = splitList.size();
        int numPartTrees   = partTrees.size();
        boolean splitTrees = (clusterTrees != partTrees);
        int partTreeOffset = splitTrees ? clusterTrees.size() : 0;

        int[] splitsData = buildSplitsData(splitList, clusterTable);
        int[][] rng      = buildSplitRangeData(splitList, clusterTable);
        int[] splitRangeMeta = rng[0], rangeData = rng[1];
        int[][] oi       = buildOrderingsInvIndex(clusterTrees, partTrees, numGpuTrees);
        int[] orderings  = oi[0], invIndex = oi[1];

        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsGPU(
            splitsData, splitRangeMeta, rangeData,
            csr.nodeData, csr.nodeFreq, csr.nodeOffset, csr.partLeafCount,
            csr.polyTreeOffset, csr.polyBoundOffset, csr.polyBounds, csr.polyFreq,
            orderings, invIndex,
            numSplits, numPartTrees, partTreeOffset, csr.maxLeafCount,
            numGpuTrees, n,
            batchSizeHint, vramFraction, nativeScoreMode(),
            Config.getInstance().getGpuProgressIntervalSec());
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;

        splitsData = null; orderings = null; invIndex = null;   // let GC reclaim
        if (twoScores == null) {
            Logging.info("  GPU kernel returned null after %d ms (infeasible)", gpuMs);
            return false;
        }
        Logging.info("  GPU kernel returned in %d ms", gpuMs);
        unpackTwoScores(twoScores, scoreArray, scoreArrayD, scoreArrayI, numSplits);
        return true;
    }

    /**
     * Convert the raw per-split doubled-score transport array from the GPU into
     * final per-split scores (score = 2·score / 2).
     *
     * <ul>
     *   <li>LONG  — each {@code twoScores[i]} is the exact integer 2·score.</li>
     *   <li>DOUBLE— the kernel stored the IEEE-754 bit pattern of the 2·score in
     *       the long slot ({@code __double_as_longlong}); recover with
     *       {@link Double#longBitsToDouble}.</li>
     *   <li>INT128— two longs per split: {@code [2i]} = low (unsigned),
     *       {@code [2i+1]} = high (signed).</li>
     * </ul>
     */
    private void unpackTwoScores(long[] twoScores, long[] scoreArray,
                                 double[] scoreArrayD, Int128[] scoreArrayI, int numSplits) {
        if (useInt128) {
            for (int i = 0; i < numSplits; i++) {
                long lo = twoScores[2 * i];
                long hi = twoScores[2 * i + 1];
                scoreArrayI[i] = new Int128(hi, lo).halve();   // 2·score → score
            }
        } else if (useDouble) {
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
                                                 Int128[] scoreArrayI,
                                                 int batchSizeHint,
                                                 double vramFraction) {
        int numSplits      = splitList.size();
        boolean splitTrees = (clusterTrees != partTrees);
        int partTreeOffset = splitTrees ? clusterTrees.size() : 0;

        int[] splitsData = buildSplitsData(splitList, clusterTable);
        int[][] rng      = buildSplitRangeData(splitList, clusterTable);
        int[] splitRangeMeta = rng[0], rangeData = rng[1];

        // --- parts (binary d==3): numParts * 9 ints; poly (d>3): separate CSR ---
        // treeIdx stored as (p.treeIndex + partTreeOffset) so the kernel reads the
        // original-tree half of the combined orderings/invIndex.
        // [treeIdx, lo1, hi1, lo2, hi2, sz1, sz2, sz3, frequency]
        int numParts = 0, numPolyParts = 0;
        long polyBoundsLen = 0;
        for (PartitionTable.Entry pe : partTable.entries()) {
            if (pe.exemplar.d == 3) numParts++;
            else { numPolyParts++; polyBoundsLen += pe.exemplar.d; }
        }
        int[] partsData = new int[numParts * 9];
        int[] ssPolyMeta        = new int[numPolyParts * 3];
        int[] ssPolyBoundOffset = new int[numPolyParts + 1];
        int[] ssPolyBounds      = new int[(int) polyBoundsLen];
        int j = 0, pj = 0, boundCur = 0;
        for (PartitionTable.Entry pe : partTable.entries()) {
            Partition p = pe.exemplar;
            if (p.d == 3) {
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
            } else {
                ssPolyMeta[pj * 3 + 0] = p.treeIndex + partTreeOffset;
                ssPolyMeta[pj * 3 + 1] = partTrees.get(p.treeIndex).leafCount;  // L_GT
                ssPolyMeta[pj * 3 + 2] = pe.frequency;
                ssPolyBoundOffset[pj] = boundCur;
                int k = p.d - 1;                         // # child intervals
                for (int i = 0; i < k; i++) ssPolyBounds[boundCur + i] = p.partStarts[i];
                ssPolyBounds[boundCur + k] = p.partEnds[k - 1];   // final boundary = hi
                boundCur += p.d;
                pj++;
            }
        }
        ssPolyBoundOffset[numPolyParts] = boundCur;

        int[][] oi      = buildOrderingsInvIndex(clusterTrees, partTrees, numGpuTrees);
        int[] orderings = oi[0], invIndex = oi[1];

        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsSmallerSideGPU(
            splitsData, splitRangeMeta, rangeData, partsData,
            ssPolyMeta, ssPolyBoundOffset, ssPolyBounds, orderings, invIndex,
            numSplits, numParts, numPolyParts, numGpuTrees, n, n,
            batchSizeHint, vramFraction, nativeScoreMode(),
            Config.getInstance().getGpuProgressIntervalSec());
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;

        splitsData = null; partsData = null; orderings = null; invIndex = null;   // let GC reclaim
        if (twoScores == null) {
            Logging.info("  GPU kernel returned null after %d ms (infeasible)", gpuMs);
            return false;
        }
        Logging.info("  GPU kernel returned in %d ms", gpuMs);
        unpackTwoScores(twoScores, scoreArray, scoreArrayD, scoreArrayI, numSplits);
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
            // Pack both single- and multi-range clusters. For a multi-range side the
            // kernel ignores [lo,hi) (it reads the split's range descriptor instead),
            // but tree/comp/size are still needed — left/right hold the bounding span.
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
     * Per-split range descriptor + resident flat range array for the GPU two-tier
     * multi-range path (DOCS/multi-range-cluster-design.md §5.2/§5.3).
     *   meta[i*4 + {0,1,2,3}] = {aRngOff, aRngCnt, bRngOff, bRngCnt}  (offsets in PAIRS)
     *   rangeData             = concatenated [lo,hi] pairs of every multi-range split side
     * A single-range side has count 0 — the kernel then uses the split's [lo,hi) fast path.
     * For runs with no multi-range clusters, meta is all-zero and rangeData is empty
     * (so the kernel's per-leaf membership is byte-identical to before).
     *
     * @return int[2][] = {meta (numSplits*4), rangeData}
     */
    private int[][] buildSplitRangeData(List<BipartitionSplit> splitList, ClusterTable clusterTable) {
        int numSplits = splitList.size();
        int[] meta = new int[numSplits * 4];
        java.util.ArrayList<Integer> ranges = new java.util.ArrayList<>(); // flat lo,hi pairs
        for (int i = 0; i < numSplits; i++) {
            BipartitionSplit sp = splitList.get(i);
            ClusterTable.Entry eA = clusterTable.get(sp.lo);
            ClusterTable.Entry eB = clusterTable.get(sp.hi);
            if (eA != null && eA.exemplar.isMultiRange()) {
                Cluster c = eA.exemplar;
                meta[i * 4 + 0] = ranges.size() / 2;   // offset in pairs
                meta[i * 4 + 1] = c.los.length;
                for (int j = 0; j < c.los.length; j++) { ranges.add(c.los[j]); ranges.add(c.his[j]); }
            }
            if (eB != null && eB.exemplar.isMultiRange()) {
                Cluster c = eB.exemplar;
                meta[i * 4 + 2] = ranges.size() / 2;
                meta[i * 4 + 3] = c.los.length;
                for (int j = 0; j < c.los.length; j++) { ranges.add(c.los[j]); ranges.add(c.his[j]); }
            }
        }
        int[] rangeData = new int[ranges.size()];
        for (int k = 0; k < ranges.size(); k++) rangeData[k] = ranges.get(k);
        return new int[][]{ meta, rangeData };
    }

    /**
     * orderings + invIndex: numGpuTrees * n ints each.
     *   orderings[t*n + pos]   = postorderArray[pos]
     *   invIndex [t*n + taxon] = positionMap[taxon]  (-1 if absent)
     *
     * Layout when splitTrees (clusterTrees != partTrees):
     *   slots 0..C-1     from clusterTrees  — cluster membership   (C = clusterTrees.size())
     *   slots C..C+P-1   from partTrees      — gene-tree leaves     (P = partTrees.size())
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
        int[] nodeData;       // numBinUnique * 3   [lo, mid, hi] of the exemplar (d==3 only)
        int[] nodeFreq;       // numBinUnique       frequency (occurrence count)
        int[] nodeOffset;     // numTrees + 1       CSR row pointers (binary nodes by exemplar tree)
        int[] partLeafCount;  // numTrees           leaf count L per tree
        int   maxLeafCount;   // max L over trees with ≥1 exemplar (shared-mem sizing)
        int   totalNodes;     // numBinUnique + numPolyUnique  (for logging)
        // Polytomy (d>3) CSR — empty when no polytomous partitions.
        int[] polyTreeOffset;   // numTrees + 1     poly nodes bucketed by exemplar tree
        int[] polyBoundOffset;  // numPoly + 1      range into polyBounds (length d) per poly node
        int[] polyBounds;       // Σ d              concatenated boundary lists b[0..d-1]
        int[] polyFreq;         // numPoly          occurrence count
    }

    /**
     * Build the deduplicated node CSR from the already-computed PartitionTable,
     * bucketing unique tripartitions by their exemplar tree index.
     */
    private static NodeCSR buildDedupNodeCSR(PartitionTable partTable,
                                              List<Tree> partTrees) {
        int numTrees = partTrees.size();

        // Pass 1: count BINARY (d==3) and POLY (d>3) unique partitions per exemplar
        // tree, and the total poly boundary length.
        int[] nodeOffset     = new int[numTrees + 1];   // binary nodes per tree
        int[] polyTreeOffset = new int[numTrees + 1];   // poly nodes per tree
        long  polyBoundsLen  = 0;
        for (PartitionTable.Entry e : partTable.entries()) {
            if (e.exemplar.d == 3) nodeOffset[e.exemplar.treeIndex + 1]++;
            else { polyTreeOffset[e.exemplar.treeIndex + 1]++; polyBoundsLen += e.exemplar.d; }
        }
        for (int g = 0; g < numTrees; g++) {
            nodeOffset[g + 1]     += nodeOffset[g];
            polyTreeOffset[g + 1] += polyTreeOffset[g];
        }
        int total     = nodeOffset[numTrees];           // # binary unique
        int numPoly   = polyTreeOffset[numTrees];       // # poly unique
        if ((long) total * 3 > Integer.MAX_VALUE || polyBoundsLen > Integer.MAX_VALUE) {
            throw new IllegalStateException("Too many partitions for a single int[] CSR");
        }

        // Pass 2: scatter into binary nodeData and poly bound CSR, bucketed by tree.
        int[] nodeData = new int[total * 3];
        int[] nodeFreq = new int[total];
        int[] polyFreq        = new int[numPoly];
        int[] polyBoundOffset = new int[numPoly + 1];
        int[] polyBounds      = new int[(int) polyBoundsLen];
        int[] binCursor       = nodeOffset.clone();     // per-tree binary write cursor
        int[] polyCursor      = polyTreeOffset.clone(); // per-tree poly write cursor

        // Pre-fill polyBoundOffset by walking poly nodes in the SAME scatter order.
        // We fill it incrementally during scatter via a running bound cursor per slot.
        int[] polyDeg = new int[numPoly];               // degree per poly slot (for offsets)
        for (PartitionTable.Entry e : partTable.entries()) {
            Partition p = e.exemplar;
            if (p.d == 3) {
                int pos = binCursor[p.treeIndex]++;
                int b   = pos * 3;
                nodeData[b]     = p.leftStart;          // lo
                nodeData[b + 1] = p.leftEnd;            // mid  (= p.rightStart)
                nodeData[b + 2] = p.rightEnd;           // hi
                nodeFreq[pos]   = e.frequency;
            } else {
                int pos = polyCursor[p.treeIndex]++;
                polyFreq[pos] = e.frequency;
                polyDeg[pos]  = p.d;
            }
        }
        // Build polyBoundOffset (prefix sum of degrees) then scatter boundary lists.
        for (int pn = 0; pn < numPoly; pn++) polyBoundOffset[pn + 1] = polyBoundOffset[pn] + polyDeg[pn];
        int[] polyCursor2 = polyTreeOffset.clone();
        for (PartitionTable.Entry e : partTable.entries()) {
            Partition p = e.exemplar;
            if (p.d == 3) continue;
            int pos  = polyCursor2[p.treeIndex]++;
            int base = polyBoundOffset[pos];
            int k    = p.d - 1;                          // # child intervals
            for (int i = 0; i < k; i++) polyBounds[base + i] = p.partStarts[i];
            polyBounds[base + k] = p.partEnds[k - 1];    // final boundary = hi
        }

        // Per-tree leaf counts; maxLeaf over trees that need a prefix (binary OR poly).
        int[] partLeafCount = new int[numTrees];
        int   maxLeaf = 0;
        for (int g = 0; g < numTrees; g++) {
            int L = partTrees.get(g).leafCount;
            partLeafCount[g] = L;
            boolean hasWork = nodeOffset[g + 1] > nodeOffset[g] || polyTreeOffset[g + 1] > polyTreeOffset[g];
            if (hasWork && L > maxLeaf) maxLeaf = L;
        }

        NodeCSR csr = new NodeCSR();
        csr.nodeData        = nodeData;
        csr.nodeFreq        = nodeFreq;
        csr.nodeOffset      = nodeOffset;
        csr.partLeafCount   = partLeafCount;
        csr.maxLeafCount    = maxLeaf;
        csr.totalNodes      = total + numPoly;
        csr.polyTreeOffset  = polyTreeOffset;
        csr.polyBoundOffset = polyBoundOffset;
        csr.polyBounds      = polyBounds;
        csr.polyFreq        = polyFreq;
        return csr;
    }

    // -------------------------------------------------------------------------
    // BITSET path (low-taxa fast option; CPU + GPU).
    //
    // Every cluster and every gene-tree part is materialized ONCE as a global-taxon
    // bitset of W = ceil(n/64) 64-bit words.  Each core intersection |X ∩ Y| is then
    // popcount(X & Y) over W words — O(1) for small n, and independent of which tree
    // either set came from (both are keyed by global taxon id, so there is no
    // cross-tree coordinate walk and no orderings/invIndex in the score loop).  The
    // 3×3 derivation, poly formula, and QI math are the EXACT same helpers the other
    // methods use, so scores are bit-identical.  Best when n is small and the gene
    // count is large; the per-intersection cost grows with W as n grows.
    // -------------------------------------------------------------------------

    /**
     * Precomputed global-taxon bitsets for the whole weight computation.
     *
     * A "cluster pool" holds one deduplicated bitset per distinct candidate-cluster
     * (cid 0 is a reserved all-zero, size-0 empty cluster used for splits whose side
     * is absent from the ClusterTable — those score 0, matching the range paths).
     * Parts are split into binary (M1,M2 bitsets) and polytomous (a CSR of child
     * bitsets); each gene tree carries a "present taxa" (Lg) bitset for the
     * incomplete-tree row sums.
     */
    private static final class BitsetData {
        int      W;                 // 64-bit words per set = ceil(n/64)
        int      numClusters;
        long[]   clusterBits;       // numClusters * W
        int[]    splitCid;          // numSplits * 4  [aCid, bCid, aSize, bSize]
        int      numBin;
        long[]   partM1, partM2;    // numBin * W  each
        int[]    partMeta;          // numBin * 5  [lgTree, sz1, sz2, sz3, freq]
        int      numPartTrees;
        long[]   geneLgBits;        // numPartTrees * W  (present-taxa mask per gene tree)
        int      numPoly;
        int[]    polyMeta;          // numPoly * 5  [lgTree, d, lastSize, freq, L_GT]
        int[]    polyChildOffset;   // numPoly + 1  CSR row pointers
        long[]   polyChildBits;     // totalChildren * W
        int[]    polyChildSize;     // totalChildren
    }

    /** Set bits for the taxa in postorder range [lo,hi) of tree t into arr[base..base+W). */
    private static void setRangeBits(long[] arr, int base, Tree t, int lo, int hi) {
        int[] po = t.postorderArray;
        for (int pos = lo; pos < hi; pos++) {
            int tax = po[pos];
            arr[base + (tax >>> 6)] |= 1L << (tax & 63);
        }
    }

    /** Write one cluster's global-taxon bitset into arr[base..base+W) (complement-aware). */
    private void buildClusterBitsInto(long[] arr, int base, Cluster c, List<Tree> clusterTrees, int W) {
        Tree t = clusterTrees.get(c.treeIndex);
        if (c.isMultiRange()) {
            for (int j = 0; j < c.los.length; j++) setRangeBits(arr, base, t, c.los[j], c.his[j]);
        } else {
            setRangeBits(arr, base, t, c.left, c.right);
        }
        if (c.complement) {                       // A = S \ range, complement over ALL n taxa
            for (int k = 0; k < W; k++) arr[base + k] = ~arr[base + k];
            int rem = n & 63;
            if (rem != 0) arr[base + W - 1] &= (1L << rem) - 1;   // clear bits ≥ n in the top word
        }
    }

    /**
     * Materialize all cluster/part/Lg bitsets once; shared by the CPU and GPU bitset paths.
     *
     * Each bitset occupies a disjoint slice of its output array, so the three fills are
     * done in parallel across threads.  Only the cluster-id assignment (a HashMap dedup)
     * is sequential; the actual cluster bitset construction is then parallel.
     */
    private BitsetData buildBitsetData(List<BipartitionSplit> splitList,
                                       PartitionTable partTable, ClusterTable clusterTable,
                                       List<Tree> clusterTrees, List<Tree> partTrees) {
        int W = (n + 63) >>> 6;
        int numSplits = splitList.size();
        int numPartTrees = partTrees.size();

        // --- Cluster pool: phase 1 sequential id assignment (dedup by ClusterHash) ---
        // cid 0 = empty cluster (all-zero, size 0): splits whose side is absent score 0.
        Map<ClusterHash, Integer> cidMap = new HashMap<>();
        List<Cluster> clusterExemplars = new ArrayList<>();
        clusterExemplars.add(null);               // cid 0 → empty
        int[] splitCid = new int[numSplits * 4];
        ClusterHash[] side = new ClusterHash[2];
        for (int i = 0; i < numSplits; i++) {
            BipartitionSplit sp = splitList.get(i);
            side[0] = sp.lo; side[1] = sp.hi;
            for (int s = 0; s < 2; s++) {
                ClusterTable.Entry e = clusterTable.get(side[s]);
                int cid = 0, sz = 0;
                if (e != null) {
                    Integer id = cidMap.get(side[s]);
                    if (id == null) {
                        id = clusterExemplars.size();
                        clusterExemplars.add(e.exemplar);
                        cidMap.put(side[s], id);
                    }
                    cid = id; sz = e.exemplar.size;
                }
                splitCid[i * 4 + s]     = cid;
                splitCid[i * 4 + 2 + s] = sz;
            }
        }
        int numClusters = clusterExemplars.size();
        if ((long) numClusters * W > Integer.MAX_VALUE)
            throw new IllegalStateException("Bitset cluster pool too large for a single long[]");
        // Phase 2: parallel fill of the deduplicated cluster bitsets (disjoint slices).
        long[] clusterBits = new long[numClusters * W];
        Threading.processRangeParallel(numClusters, cid -> {
            if (cid == 0) return;                 // cid 0 stays all-zero
            buildClusterBitsInto(clusterBits, cid * W, clusterExemplars.get(cid), clusterTrees, W);
        });

        // --- Per gene-tree present-taxa (Lg) bitsets (parallel over trees) ---
        if ((long) numPartTrees * W > Integer.MAX_VALUE)
            throw new IllegalStateException("Bitset gene-tree Lg pool too large for a single long[]");
        long[] geneLgBits = new long[numPartTrees * W];
        Threading.processRangeParallel(numPartTrees, g -> {
            Tree t = partTrees.get(g);
            setRangeBits(geneLgBits, g * W, t, 0, t.leafCount);
        });

        // --- Part bitsets: split entries into binary / poly, prefix the poly child cursor,
        //     then fill each group in parallel (each part owns a disjoint slice). ---
        List<PartitionTable.Entry> binEntries  = new ArrayList<>();
        List<PartitionTable.Entry> polyEntries = new ArrayList<>();
        for (PartitionTable.Entry e : partTable.entries()) {
            if (e.exemplar.d == 3) binEntries.add(e);
            else                   polyEntries.add(e);
        }
        int numBin = binEntries.size(), numPoly = polyEntries.size();
        int[] polyChildOffset = new int[numPoly + 1];
        for (int pi = 0; pi < numPoly; pi++)
            polyChildOffset[pi + 1] = polyChildOffset[pi] + (polyEntries.get(pi).exemplar.d - 1);
        long childTotal = polyChildOffset[numPoly];
        if ((long) numBin * W > Integer.MAX_VALUE || childTotal * W > Integer.MAX_VALUE)
            throw new IllegalStateException("Bitset part pool too large for a single long[]");

        long[] partM1 = new long[numBin * W];
        long[] partM2 = new long[numBin * W];
        int[]  partMeta = new int[numBin * 5];
        int[]  polyMeta = new int[numPoly * 5];
        long[] polyChildBits = new long[(int) childTotal * W];
        int[]  polyChildSize = new int[(int) childTotal];

        Threading.processRangeParallel(numBin, bi -> {
            PartitionTable.Entry e = binEntries.get(bi);
            Partition p = e.exemplar;
            Tree tGT = partTrees.get(p.treeIndex);
            setRangeBits(partM1, bi * W, tGT, p.leftStart,  p.leftEnd);
            setRangeBits(partM2, bi * W, tGT, p.rightStart, p.rightEnd);
            int m = bi * 5;
            partMeta[m]     = p.treeIndex;
            partMeta[m + 1] = p.size1;
            partMeta[m + 2] = p.size2;
            partMeta[m + 3] = p.size3;
            partMeta[m + 4] = e.frequency;
        });
        Threading.processRangeParallel(numPoly, pi -> {
            PartitionTable.Entry e = polyEntries.get(pi);
            Partition p = e.exemplar;
            Tree tGT = partTrees.get(p.treeIndex);
            int d = p.d, childCur = polyChildOffset[pi];
            for (int c = 0; c < d - 1; c++) {
                setRangeBits(polyChildBits, (childCur + c) * W, tGT, p.partStarts[c], p.partEnds[c]);
                polyChildSize[childCur + c] = p.sizes[c];
            }
            int m = pi * 5;
            polyMeta[m]     = p.treeIndex;
            polyMeta[m + 1] = d;
            polyMeta[m + 2] = p.sizes[d - 1];      // complement (last) part size
            polyMeta[m + 3] = e.frequency;
            polyMeta[m + 4] = tGT.leafCount;       // L_GT
        });

        BitsetData bd = new BitsetData();
        bd.W = W;
        bd.numClusters = numClusters;
        bd.clusterBits = clusterBits;
        bd.splitCid = splitCid;
        bd.numBin = numBin;
        bd.partM1 = partM1;
        bd.partM2 = partM2;
        bd.partMeta = partMeta;
        bd.numPartTrees = numPartTrees;
        bd.geneLgBits = geneLgBits;
        bd.numPoly = numPoly;
        bd.polyMeta = polyMeta;
        bd.polyChildOffset = polyChildOffset;
        bd.polyChildBits = polyChildBits;
        bd.polyChildSize = polyChildSize;
        return bd;
    }

    /** popcount( X[xBase..] & Y[yBase..] ) over W words. */
    private static int popAnd(long[] x, int xBase, long[] y, int yBase, int W) {
        int c = 0;
        for (int k = 0; k < W; k++) c += Long.bitCount(x[xBase + k] & y[yBase + k]);
        return c;
    }

    // --- CPU bitset scorers (mirror computeScore / D / I; reuse the same QI helpers) ---

    private long computeScoreBitset(int i, BitsetData bd, int totalN) {
        int W = bd.W;
        int aCid = bd.splitCid[i * 4], bCid = bd.splitCid[i * 4 + 1];
        int aSize = bd.splitCid[i * 4 + 2], bSize = bd.splitCid[i * 4 + 3];
        if (totalN - aSize - bSize < 0) return 0L;
        int aB = aCid * W, bB = bCid * W;
        long twoScore = 0L;

        for (int j = 0; j < bd.numBin; j++) {
            int mW = j * W, m = j * 5;
            int a0 = popAnd(bd.clusterBits, aB, bd.partM1, mW, W);
            int a1 = popAnd(bd.clusterBits, aB, bd.partM2, mW, W);
            int b0 = popAnd(bd.clusterBits, bB, bd.partM1, mW, W);
            int b1 = popAnd(bd.clusterBits, bB, bd.partM2, mW, W);
            int lgTree = bd.partMeta[m], sz1 = bd.partMeta[m + 1], sz2 = bd.partMeta[m + 2],
                sz3 = bd.partMeta[m + 3], freq = bd.partMeta[m + 4];
            int L_GT = sz1 + sz2 + sz3, lgA, lgB;
            if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
            else { int lb = lgTree * W;
                   lgA = popAnd(bd.clusterBits, aB, bd.geneLgBits, lb, W);
                   lgB = popAnd(bd.clusterBits, bB, bd.geneLgBits, lb, W); }
            int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1;
            int c0 = sz1 - a0 - b0, c1 = sz2 - a1 - b1, c2 = sz3 - a2 - b2;
            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;
            twoScore += (long) freq * computeTwoQI(a0, a1, a2, b0, b1, b2, c0, c1, c2);
        }
        for (int pn = 0; pn < bd.numPoly; pn++) {
            int m = pn * 5;
            int lgTree = bd.polyMeta[m], d = bd.polyMeta[m + 1], lastSize = bd.polyMeta[m + 2],
                freq = bd.polyMeta[m + 3], L_GT = bd.polyMeta[m + 4];
            int[][] parts = polyPartsBitset(bd, pn, d, lastSize, L_GT, totalN, aB, bB, aSize, bSize);
            if (parts == null) continue;
            twoScore += (long) freq * polyTwoQILong(parts[0], parts[1], parts[2], d);
        }
        return twoScore / 2;
    }

    private double computeScoreBitsetD(int i, BitsetData bd, int totalN) {
        int W = bd.W;
        int aCid = bd.splitCid[i * 4], bCid = bd.splitCid[i * 4 + 1];
        int aSize = bd.splitCid[i * 4 + 2], bSize = bd.splitCid[i * 4 + 3];
        if (totalN - aSize - bSize < 0) return 0.0;
        int aB = aCid * W, bB = bCid * W;
        double twoScore = 0.0;

        for (int j = 0; j < bd.numBin; j++) {
            int mW = j * W, m = j * 5;
            int a0 = popAnd(bd.clusterBits, aB, bd.partM1, mW, W);
            int a1 = popAnd(bd.clusterBits, aB, bd.partM2, mW, W);
            int b0 = popAnd(bd.clusterBits, bB, bd.partM1, mW, W);
            int b1 = popAnd(bd.clusterBits, bB, bd.partM2, mW, W);
            int lgTree = bd.partMeta[m], sz1 = bd.partMeta[m + 1], sz2 = bd.partMeta[m + 2],
                sz3 = bd.partMeta[m + 3], freq = bd.partMeta[m + 4];
            int L_GT = sz1 + sz2 + sz3, lgA, lgB;
            if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
            else { int lb = lgTree * W;
                   lgA = popAnd(bd.clusterBits, aB, bd.geneLgBits, lb, W);
                   lgB = popAnd(bd.clusterBits, bB, bd.geneLgBits, lb, W); }
            int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1;
            int c0 = sz1 - a0 - b0, c1 = sz2 - a1 - b1, c2 = sz3 - a2 - b2;
            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;
            twoScore += (double) freq * computeTwoQIDouble(a0, a1, a2, b0, b1, b2, c0, c1, c2);
        }
        for (int pn = 0; pn < bd.numPoly; pn++) {
            int m = pn * 5;
            int d = bd.polyMeta[m + 1], lastSize = bd.polyMeta[m + 2],
                freq = bd.polyMeta[m + 3], L_GT = bd.polyMeta[m + 4];
            int[][] parts = polyPartsBitset(bd, pn, d, lastSize, L_GT, totalN, aB, bB, aSize, bSize);
            if (parts == null) continue;
            twoScore += (double) freq * polyTwoQIDouble(parts[0], parts[1], parts[2], d);
        }
        return twoScore / 2.0;
    }

    private Int128 computeScoreBitsetI(int i, BitsetData bd, int totalN) {
        int W = bd.W;
        int aCid = bd.splitCid[i * 4], bCid = bd.splitCid[i * 4 + 1];
        int aSize = bd.splitCid[i * 4 + 2], bSize = bd.splitCid[i * 4 + 3];
        if (totalN - aSize - bSize < 0) return Int128.ZERO;
        int aB = aCid * W, bB = bCid * W;
        Int128 twoScore = Int128.ZERO;

        for (int j = 0; j < bd.numBin; j++) {
            int mW = j * W, m = j * 5;
            int a0 = popAnd(bd.clusterBits, aB, bd.partM1, mW, W);
            int a1 = popAnd(bd.clusterBits, aB, bd.partM2, mW, W);
            int b0 = popAnd(bd.clusterBits, bB, bd.partM1, mW, W);
            int b1 = popAnd(bd.clusterBits, bB, bd.partM2, mW, W);
            int lgTree = bd.partMeta[m], sz1 = bd.partMeta[m + 1], sz2 = bd.partMeta[m + 2],
                sz3 = bd.partMeta[m + 3], freq = bd.partMeta[m + 4];
            int L_GT = sz1 + sz2 + sz3, lgA, lgB;
            if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
            else { int lb = lgTree * W;
                   lgA = popAnd(bd.clusterBits, aB, bd.geneLgBits, lb, W);
                   lgB = popAnd(bd.clusterBits, bB, bd.geneLgBits, lb, W); }
            int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1;
            int c0 = sz1 - a0 - b0, c1 = sz2 - a1 - b1, c2 = sz3 - a2 - b2;
            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;
            twoScore = twoScore.add(computeTwoQIInt128(a0, a1, a2, b0, b1, b2, c0, c1, c2).mulScalar(freq));
        }
        for (int pn = 0; pn < bd.numPoly; pn++) {
            int m = pn * 5;
            int d = bd.polyMeta[m + 1], lastSize = bd.polyMeta[m + 2],
                freq = bd.polyMeta[m + 3], L_GT = bd.polyMeta[m + 4];
            int[][] parts = polyPartsBitset(bd, pn, d, lastSize, L_GT, totalN, aB, bB, aSize, bSize);
            if (parts == null) continue;
            twoScore = twoScore.add(polyTwoQIInt128(parts[0], parts[1], parts[2], d).mulScalar(freq));
        }
        return twoScore.halve();
    }

    /**
     * Bitset analogue of {@link #polyParts}: build the d×3 matrix for a polytomous
     * part from precomputed child bitsets.  Returns null on an incomplete-tree row
     * mismatch (negative complement part), matching the range path.
     */
    private int[][] polyPartsBitset(BitsetData bd, int pn, int d, int lastSize, int L_GT,
                                    int totalN, int aB, int bB, int aSize, int bSize) {
        int W = bd.W;
        int cbeg = bd.polyChildOffset[pn];
        int[] a = new int[d], b = new int[d], c = new int[d];
        int sumA = 0, sumB = 0;
        for (int ci = 0; ci < d - 1; ci++) {
            int mW = (cbeg + ci) * W, szi = bd.polyChildSize[cbeg + ci];
            int ai = popAnd(bd.clusterBits, aB, bd.polyChildBits, mW, W);
            int bi = popAnd(bd.clusterBits, bB, bd.polyChildBits, mW, W);
            int cci = szi - ai - bi;
            if (cci < 0) return null;
            a[ci] = ai; b[ci] = bi; c[ci] = cci;
            sumA += ai; sumB += bi;
        }
        int lgTree = bd.polyMeta[pn * 5];
        int lgA, lgB;
        if (L_GT == totalN) { lgA = aSize; lgB = bSize; }
        else { int lb = lgTree * W;
               lgA = popAnd(bd.clusterBits, aB, bd.geneLgBits, lb, W);
               lgB = popAnd(bd.clusterBits, bB, bd.geneLgBits, lb, W); }
        int aC = lgA - sumA, bC = lgB - sumB, cC = lastSize - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) return null;
        a[d - 1] = aC; b[d - 1] = bC; c[d - 1] = cC;
        return new int[][]{ a, b, c };
    }

    /** CPU bitset path: build bitsets once, then score splits in parallel. */
    private void computeScoresCPUBitset(List<BipartitionSplit> splitList,
                                        PartitionTable partTable, ClusterTable clusterTable,
                                        List<Tree> clusterTrees, List<Tree> partTrees,
                                        long[] scoreArray, double[] scoreArrayD, Int128[] scoreArrayI) {
        BitsetData bd = buildBitsetData(splitList, partTable, clusterTable, clusterTrees, partTrees);
        int numSplits = splitList.size();
        int totalN = n;
        Logging.info("Weight table: CPU path (bitset)  splits=%d  W=%d  clusters=%d  bin=%d  poly=%d  trees=%d",
            numSplits, bd.W, bd.numClusters, bd.numBin, bd.numPoly, bd.numPartTrees);
        java.util.concurrent.atomic.AtomicInteger wDone = new java.util.concurrent.atomic.AtomicInteger(0);
        ProgressBar wBar = new ProgressBar("Scoring splits (CPU bitset)", numSplits);
        Threading.processRangeParallel(numSplits, idx -> {
            if (useInt128)      scoreArrayI[idx] = computeScoreBitsetI(idx, bd, totalN);
            else if (useDouble) scoreArrayD[idx] = computeScoreBitsetD(idx, bd, totalN);
            else                scoreArray[idx]  = computeScoreBitset(idx, bd, totalN);
            wBar.update(wDone.incrementAndGet());
        });
        wBar.done();
    }

    /** GPU bitset path: build bitsets once, resolve batch size, call the native kernel. */
    private boolean computeScoresGPUBitset(List<BipartitionSplit> splitList,
                                           PartitionTable partTable, ClusterTable clusterTable,
                                           List<Tree> clusterTrees, List<Tree> partTrees,
                                           long[] scoreArray, double[] scoreArrayD, Int128[] scoreArrayI) {
        Config cfg = Config.getInstance();
        BitsetData bd = buildBitsetData(splitList, partTable, clusterTable, clusterTrees, partTrees);
        int numSplits = splitList.size();

        // Resident device memory (for vram-control-factor sizing + logging only).
        long residentMem = 8L * (bd.clusterBits.length + bd.partM1.length + bd.partM2.length
                                 + bd.geneLgBits.length + bd.polyChildBits.length)
                         + 4L * (bd.partMeta.length + bd.polyMeta.length
                                 + bd.polyChildOffset.length + bd.polyChildSize.length);
        long perSplit = 4L * Integer.BYTES + (useInt128 ? 2L : 1L) * Long.BYTES;   // 4 ints in + score out

        // Batch-size hint — same priority as the other GPU paths.
        int batchSizeHint; String batchDesc;
        if (!cfg.isGpuBatch()) {
            batchSizeHint = -1; batchDesc = "off (single launch)";
        } else if (cfg.getGpuNumBatches() > 0) {
            int N = cfg.getGpuNumBatches();
            batchSizeHint = (numSplits + N - 1) / N;
            batchDesc = N + " batches → batchSize=" + batchSizeHint;
        } else if (cfg.getGpuBatchSize() > 0) {
            batchSizeHint = cfg.getGpuBatchSize();
            batchDesc = "explicit batchSize=" + batchSizeHint;
        } else if (cfg.isGpuVramControlFactorSet()) {
            double F = cfg.getGpuVramControlFactor();
            long batchMem = (long) (F * residentMem);
            batchSizeHint = (int) Math.max(1, Math.min(numSplits, batchMem / perSplit));
            batchDesc = String.format("vram-control-factor=%.3f  resident=%.1f MB  batch→%d",
                F, residentMem / 1e6, (numSplits + batchSizeHint - 1) / Math.max(1, batchSizeHint));
        } else {
            batchSizeHint = 0;
            batchDesc = String.format("auto (free-VRAM adaptive, occupancy=%.0f%%)",
                cfg.getGpuVramFraction() * 100);
        }

        Logging.info("Weight table: GPU path (bitset)  splits=%d  W=%d  clusters=%d  bin=%d  poly=%d  trees=%d  resident=%.1f MB  batching=%s",
            numSplits, bd.W, bd.numClusters, bd.numBin, bd.numPoly, bd.numPartTrees,
            residentMem / 1e6, batchDesc);

        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsBitsetGPU(
            bd.splitCid, bd.clusterBits, bd.partM1, bd.partM2, bd.partMeta, bd.geneLgBits,
            bd.polyMeta, bd.polyChildOffset, bd.polyChildBits, bd.polyChildSize,
            numSplits, bd.numClusters, bd.numBin, bd.numPoly, bd.numPartTrees, bd.W, n,
            batchSizeHint, cfg.getGpuVramFraction(), nativeScoreMode(),
            cfg.getGpuProgressIntervalSec());
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;

        if (twoScores == null) {
            Logging.info("  GPU bitset kernel returned null after %d ms (infeasible)", gpuMs);
            return false;
        }
        Logging.info("  GPU bitset kernel returned in %d ms", gpuMs);
        unpackTwoScores(twoScores, scoreArray, scoreArrayD, scoreArrayI, numSplits);
        return true;
    }

    // -------------------------------------------------------------------------
    // SIMPLE-TREE-WALK path (many-candidate fast option; CPU + GPU).
    //
    // One thread per split walks a resident flat postorder token stream of all gene
    // trees sequentially, maintaining a small O(n) per-thread stack of
    // (nA,nB,nS) = (|node∩A|, |node∩B|, |node|) triples.  Every non-root internal
    // node's tripartition is scored in O(1) from its children — the SAME node set
    // (all non-root internal nodes) and SAME QI helpers as the other methods, with
    // no dedup, so the raw Σ over nodes is bit-identical to the deduped Σ freq·QI.
    // -------------------------------------------------------------------------

    /** GPU per-thread postorder stack cap — mirrors WB_TW_STACK_CAP in astralx_weight.cu. */
    private static final int TW_GPU_STACK_CAP = 512;

    private static final class TreeWalkData {
        int      W;
        int      numClusters;
        long[]   clusterBits;     // numClusters * W
        int[]    splitCid;        // numSplits * 4  [aCid, bCid, aSize, bSize]
        int      numTrees;
        long[]   geneLgBits;      // numTrees * W
        int[]    nodeStream;      // flat postorder tokens (leaf=taxon≥0, internal=-childCount)
        int[]    treeNodeOffset;  // numTrees + 1  CSR row pointers
        int[]    leafCount;       // numTrees  (L per gene tree = LgSize)
        int      maxLeaf;         // max leaf count over trees (CPU stack sizing)
    }

    /** Count postorder tokens a tree emits: one per leaf, one per NON-root internal node. */
    private int countTreeTokens(TreeNode node) {
        if (node.isLeaf()) return 1;
        int c = 0;
        if (node.isPolytomous()) for (TreeNode ch : node.children) c += countTreeTokens(ch);
        else { c += countTreeTokens(node.left); c += countTreeTokens(node.right); }
        if (!node.isRoot()) c += 1;   // this internal node emits a token; root is skipped
        return c;
    }

    /** Emit postorder tokens into stream[cursor..]; leaf → taxonId (≥0), non-root internal → -childCount. */
    private int fillTreeTokens(TreeNode node, int[] stream, int cursor) {
        if (node.isLeaf()) { stream[cursor++] = node.taxonId; return cursor; }
        int k;
        if (node.isPolytomous()) {
            k = node.children.length;
            for (TreeNode ch : node.children) cursor = fillTreeTokens(ch, stream, cursor);
        } else {
            k = 2;
            cursor = fillTreeTokens(node.left,  stream, cursor);
            cursor = fillTreeTokens(node.right, stream, cursor);
        }
        if (!node.isRoot()) stream[cursor++] = -k;   // score this node; root emits nothing
        return cursor;
    }

    /** Materialize the cluster pool, Lg bitsets, and postorder token stream (parallel fills). */
    private TreeWalkData buildTreeWalkData(List<BipartitionSplit> splitList, ClusterTable clusterTable,
                                           List<Tree> clusterTrees, List<Tree> partTrees) {
        int W = (n + 63) >>> 6;
        int numSplits = splitList.size();
        int numTrees = partTrees.size();

        // --- Cluster pool (dedup by ClusterHash); cid 0 = empty (all-zero, size 0) ---
        Map<ClusterHash, Integer> cidMap = new HashMap<>();
        List<Cluster> clusterExemplars = new ArrayList<>();
        clusterExemplars.add(null);               // cid 0
        int[] splitCid = new int[numSplits * 4];
        ClusterHash[] side = new ClusterHash[2];
        for (int i = 0; i < numSplits; i++) {
            BipartitionSplit sp = splitList.get(i);
            side[0] = sp.lo; side[1] = sp.hi;
            for (int s = 0; s < 2; s++) {
                ClusterTable.Entry e = clusterTable.get(side[s]);
                int cid = 0, sz = 0;
                if (e != null) {
                    Integer id = cidMap.get(side[s]);
                    if (id == null) {
                        id = clusterExemplars.size();
                        clusterExemplars.add(e.exemplar);
                        cidMap.put(side[s], id);
                    }
                    cid = id; sz = e.exemplar.size;
                }
                splitCid[i * 4 + s]     = cid;
                splitCid[i * 4 + 2 + s] = sz;
            }
        }
        int numClusters = clusterExemplars.size();
        if ((long) numClusters * W > Integer.MAX_VALUE)
            throw new IllegalStateException("Tree-walk cluster pool too large for a single long[]");
        long[] clusterBits = new long[numClusters * W];
        Threading.processRangeParallel(numClusters, cid -> {
            if (cid == 0) return;
            buildClusterBitsInto(clusterBits, cid * W, clusterExemplars.get(cid), clusterTrees, W);
        });

        // --- Per gene-tree present-taxa (Lg) bitsets ---
        if ((long) numTrees * W > Integer.MAX_VALUE)
            throw new IllegalStateException("Tree-walk gene-tree Lg pool too large for a single long[]");
        long[] geneLgBits = new long[numTrees * W];
        Threading.processRangeParallel(numTrees, g -> {
            Tree t = partTrees.get(g);
            setRangeBits(geneLgBits, g * W, t, 0, t.leafCount);
        });

        // --- Postorder token stream: count → offsets → parallel fill (disjoint segments) ---
        int[] treeNodeOffset = new int[numTrees + 1];
        int[] leafCount = new int[numTrees];
        int maxLeaf = 0;
        for (int g = 0; g < numTrees; g++) {
            Tree t = partTrees.get(g);
            leafCount[g] = t.leafCount;
            if (t.leafCount > maxLeaf) maxLeaf = t.leafCount;
            long tok = (long) treeNodeOffset[g] + countTreeTokens(t.root);
            if (tok > Integer.MAX_VALUE)
                throw new IllegalStateException("Tree-walk node stream too large for a single int[]");
            treeNodeOffset[g + 1] = (int) tok;
        }
        int[] nodeStream = new int[treeNodeOffset[numTrees]];
        Threading.processRangeParallel(numTrees, g ->
            fillTreeTokens(partTrees.get(g).root, nodeStream, treeNodeOffset[g]));

        TreeWalkData d = new TreeWalkData();
        d.W = W; d.numClusters = numClusters; d.clusterBits = clusterBits; d.splitCid = splitCid;
        d.numTrees = numTrees; d.geneLgBits = geneLgBits; d.nodeStream = nodeStream;
        d.treeNodeOffset = treeNodeOffset; d.leafCount = leafCount; d.maxLeaf = Math.max(1, maxLeaf);
        return d;
    }

    // --- CPU tree-walk scorers (mirror computeScore / D / I; reuse the same QI helpers) ---

    private long computeScoreTreeWalk(int idx, TreeWalkData d, int totalN, int[] stack) {
        int W = d.W;
        int aCid = d.splitCid[idx * 4], bCid = d.splitCid[idx * 4 + 1];
        int aSize = d.splitCid[idx * 4 + 2], bSize = d.splitCid[idx * 4 + 3];
        if (totalN - aSize - bSize < 0) return 0L;
        int aB = aCid * W, bB = bCid * W;
        long[] cb = d.clusterBits, lg = d.geneLgBits;
        int[] ns = d.nodeStream, off = d.treeNodeOffset, lc = d.leafCount;
        long twoScore = 0L;

        for (int g = 0; g < d.numTrees; g++) {
            int segBeg = off[g], segEnd = off[g + 1];
            if (segBeg == segEnd) continue;
            int LgSize = lc[g], lgA, lgB;
            if (LgSize == totalN) { lgA = aSize; lgB = bSize; }
            else { int lb = g * W; lgA = popAnd(cb, aB, lg, lb, W); lgB = popAnd(cb, bB, lg, lb, W); }

            int top = 0;
            for (int i = segBeg; i < segEnd; i++) {
                int tok = ns[i];
                if (tok >= 0) {
                    int inA = (int) ((cb[aB + (tok >>> 6)] >>> (tok & 63)) & 1L);
                    int inB = (int) ((cb[bB + (tok >>> 6)] >>> (tok & 63)) & 1L);
                    int e = top * 3; stack[e] = inA; stack[e + 1] = inB; stack[e + 2] = 1; top++;
                } else {
                    int k = -tok, cbase = top - k;
                    if (k == 2) {
                        int e0 = cbase * 3;
                        int a0 = stack[e0], b0 = stack[e0 + 1], s0 = stack[e0 + 2];
                        int a1 = stack[e0 + 3], b1 = stack[e0 + 4], s1 = stack[e0 + 5];
                        int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1, sz3 = LgSize - s0 - s1;
                        int c0 = s0 - a0 - b0, c1 = s1 - a1 - b1, c2 = sz3 - a2 - b2;
                        if (!(a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0))
                            twoScore += computeTwoQI(a0, a1, a2, b0, b1, b2, c0, c1, c2);
                        stack[e0] = a0 + a1; stack[e0 + 1] = b0 + b1; stack[e0 + 2] = s0 + s1;
                        top = cbase + 1;
                    } else {
                        int dd = k + 1;
                        int[] pa = new int[dd], pb = new int[dd], pc = new int[dd];
                        int sumA = 0, sumB = 0, sumS = 0;
                        for (int j = 0; j < k; j++) {
                            int e = (cbase + j) * 3;
                            int aj = stack[e], bj = stack[e + 1], sj = stack[e + 2];
                            pa[j] = aj; pb[j] = bj; pc[j] = sj - aj - bj;
                            sumA += aj; sumB += bj; sumS += sj;
                        }
                        int aC = lgA - sumA, bC = lgB - sumB, cC = (LgSize - sumS) - aC - bC;
                        if (!(aC < 0 || bC < 0 || cC < 0)) {
                            pa[k] = aC; pb[k] = bC; pc[k] = cC;
                            twoScore += polyTwoQILong(pa, pb, pc, dd);
                        }
                        stack[cbase * 3] = sumA; stack[cbase * 3 + 1] = sumB; stack[cbase * 3 + 2] = sumS;
                        top = cbase + 1;
                    }
                }
            }
        }
        return twoScore / 2;
    }

    private double computeScoreTreeWalkD(int idx, TreeWalkData d, int totalN, int[] stack) {
        int W = d.W;
        int aCid = d.splitCid[idx * 4], bCid = d.splitCid[idx * 4 + 1];
        int aSize = d.splitCid[idx * 4 + 2], bSize = d.splitCid[idx * 4 + 3];
        if (totalN - aSize - bSize < 0) return 0.0;
        int aB = aCid * W, bB = bCid * W;
        long[] cb = d.clusterBits, lg = d.geneLgBits;
        int[] ns = d.nodeStream, off = d.treeNodeOffset, lc = d.leafCount;
        double twoScore = 0.0;

        for (int g = 0; g < d.numTrees; g++) {
            int segBeg = off[g], segEnd = off[g + 1];
            if (segBeg == segEnd) continue;
            int LgSize = lc[g], lgA, lgB;
            if (LgSize == totalN) { lgA = aSize; lgB = bSize; }
            else { int lb = g * W; lgA = popAnd(cb, aB, lg, lb, W); lgB = popAnd(cb, bB, lg, lb, W); }

            int top = 0;
            for (int i = segBeg; i < segEnd; i++) {
                int tok = ns[i];
                if (tok >= 0) {
                    int inA = (int) ((cb[aB + (tok >>> 6)] >>> (tok & 63)) & 1L);
                    int inB = (int) ((cb[bB + (tok >>> 6)] >>> (tok & 63)) & 1L);
                    int e = top * 3; stack[e] = inA; stack[e + 1] = inB; stack[e + 2] = 1; top++;
                } else {
                    int k = -tok, cbase = top - k;
                    if (k == 2) {
                        int e0 = cbase * 3;
                        int a0 = stack[e0], b0 = stack[e0 + 1], s0 = stack[e0 + 2];
                        int a1 = stack[e0 + 3], b1 = stack[e0 + 4], s1 = stack[e0 + 5];
                        int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1, sz3 = LgSize - s0 - s1;
                        int c0 = s0 - a0 - b0, c1 = s1 - a1 - b1, c2 = sz3 - a2 - b2;
                        if (!(a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0))
                            twoScore += computeTwoQIDouble(a0, a1, a2, b0, b1, b2, c0, c1, c2);
                        stack[e0] = a0 + a1; stack[e0 + 1] = b0 + b1; stack[e0 + 2] = s0 + s1;
                        top = cbase + 1;
                    } else {
                        int dd = k + 1;
                        int[] pa = new int[dd], pb = new int[dd], pc = new int[dd];
                        int sumA = 0, sumB = 0, sumS = 0;
                        for (int j = 0; j < k; j++) {
                            int e = (cbase + j) * 3;
                            int aj = stack[e], bj = stack[e + 1], sj = stack[e + 2];
                            pa[j] = aj; pb[j] = bj; pc[j] = sj - aj - bj;
                            sumA += aj; sumB += bj; sumS += sj;
                        }
                        int aC = lgA - sumA, bC = lgB - sumB, cC = (LgSize - sumS) - aC - bC;
                        if (!(aC < 0 || bC < 0 || cC < 0)) {
                            pa[k] = aC; pb[k] = bC; pc[k] = cC;
                            twoScore += polyTwoQIDouble(pa, pb, pc, dd);
                        }
                        stack[cbase * 3] = sumA; stack[cbase * 3 + 1] = sumB; stack[cbase * 3 + 2] = sumS;
                        top = cbase + 1;
                    }
                }
            }
        }
        return twoScore / 2.0;
    }

    private Int128 computeScoreTreeWalkI(int idx, TreeWalkData d, int totalN, int[] stack) {
        int W = d.W;
        int aCid = d.splitCid[idx * 4], bCid = d.splitCid[idx * 4 + 1];
        int aSize = d.splitCid[idx * 4 + 2], bSize = d.splitCid[idx * 4 + 3];
        if (totalN - aSize - bSize < 0) return Int128.ZERO;
        int aB = aCid * W, bB = bCid * W;
        long[] cb = d.clusterBits, lg = d.geneLgBits;
        int[] ns = d.nodeStream, off = d.treeNodeOffset, lc = d.leafCount;
        Int128 twoScore = Int128.ZERO;

        for (int g = 0; g < d.numTrees; g++) {
            int segBeg = off[g], segEnd = off[g + 1];
            if (segBeg == segEnd) continue;
            int LgSize = lc[g], lgA, lgB;
            if (LgSize == totalN) { lgA = aSize; lgB = bSize; }
            else { int lb = g * W; lgA = popAnd(cb, aB, lg, lb, W); lgB = popAnd(cb, bB, lg, lb, W); }

            int top = 0;
            for (int i = segBeg; i < segEnd; i++) {
                int tok = ns[i];
                if (tok >= 0) {
                    int inA = (int) ((cb[aB + (tok >>> 6)] >>> (tok & 63)) & 1L);
                    int inB = (int) ((cb[bB + (tok >>> 6)] >>> (tok & 63)) & 1L);
                    int e = top * 3; stack[e] = inA; stack[e + 1] = inB; stack[e + 2] = 1; top++;
                } else {
                    int k = -tok, cbase = top - k;
                    if (k == 2) {
                        int e0 = cbase * 3;
                        int a0 = stack[e0], b0 = stack[e0 + 1], s0 = stack[e0 + 2];
                        int a1 = stack[e0 + 3], b1 = stack[e0 + 4], s1 = stack[e0 + 5];
                        int a2 = lgA - a0 - a1, b2 = lgB - b0 - b1, sz3 = LgSize - s0 - s1;
                        int c0 = s0 - a0 - b0, c1 = s1 - a1 - b1, c2 = sz3 - a2 - b2;
                        if (!(a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0))
                            twoScore = twoScore.add(computeTwoQIInt128(a0, a1, a2, b0, b1, b2, c0, c1, c2));
                        stack[e0] = a0 + a1; stack[e0 + 1] = b0 + b1; stack[e0 + 2] = s0 + s1;
                        top = cbase + 1;
                    } else {
                        int dd = k + 1;
                        int[] pa = new int[dd], pb = new int[dd], pc = new int[dd];
                        int sumA = 0, sumB = 0, sumS = 0;
                        for (int j = 0; j < k; j++) {
                            int e = (cbase + j) * 3;
                            int aj = stack[e], bj = stack[e + 1], sj = stack[e + 2];
                            pa[j] = aj; pb[j] = bj; pc[j] = sj - aj - bj;
                            sumA += aj; sumB += bj; sumS += sj;
                        }
                        int aC = lgA - sumA, bC = lgB - sumB, cC = (LgSize - sumS) - aC - bC;
                        if (!(aC < 0 || bC < 0 || cC < 0)) {
                            pa[k] = aC; pb[k] = bC; pc[k] = cC;
                            twoScore = twoScore.add(polyTwoQIInt128(pa, pb, pc, dd));
                        }
                        stack[cbase * 3] = sumA; stack[cbase * 3 + 1] = sumB; stack[cbase * 3 + 2] = sumS;
                        top = cbase + 1;
                    }
                }
            }
        }
        return twoScore.halve();
    }

    /** CPU tree-walk path: build resident data once, then walk per split in parallel. */
    private void computeScoresCPUTreeWalk(List<BipartitionSplit> splitList, ClusterTable clusterTable,
                                          List<Tree> clusterTrees, List<Tree> partTrees,
                                          long[] scoreArray, double[] scoreArrayD, Int128[] scoreArrayI) {
        TreeWalkData d = buildTreeWalkData(splitList, clusterTable, clusterTrees, partTrees);
        int numSplits = splitList.size();
        int totalN = n;
        int cap = d.maxLeaf * 3;
        ThreadLocal<int[]> stackTL = ThreadLocal.withInitial(() -> new int[cap]);
        Logging.info("Weight table: CPU path (simple-tree-walk)  splits=%d  W=%d  clusters=%d  trees=%d  tokens=%d",
            numSplits, d.W, d.numClusters, d.numTrees, d.nodeStream.length);
        java.util.concurrent.atomic.AtomicInteger wDone = new java.util.concurrent.atomic.AtomicInteger(0);
        ProgressBar wBar = new ProgressBar("Scoring splits (CPU tree-walk)", numSplits);
        Threading.processRangeParallel(numSplits, idx -> {
            int[] stack = stackTL.get();
            if (useInt128)      scoreArrayI[idx] = computeScoreTreeWalkI(idx, d, totalN, stack);
            else if (useDouble) scoreArrayD[idx] = computeScoreTreeWalkD(idx, d, totalN, stack);
            else                scoreArray[idx]  = computeScoreTreeWalk(idx, d, totalN, stack);
            wBar.update(wDone.incrementAndGet());
        });
        wBar.done();
    }

    /** GPU tree-walk path: build resident data once, resolve batch size, call the native kernel. */
    private boolean computeScoresGPUTreeWalk(List<BipartitionSplit> splitList, ClusterTable clusterTable,
                                             List<Tree> clusterTrees, List<Tree> partTrees,
                                             long[] scoreArray, double[] scoreArrayD, Int128[] scoreArrayI) {
        // Early feasibility check: the GPU kernel's per-thread stack is capped at
        // TW_GPU_STACK_CAP taxa.  Larger sets go straight to the CPU tree walk (no
        // wasted resident-data build).
        if (n > TW_GPU_STACK_CAP) {
            Logging.info("GPU tree-walk: numTaxa=%d exceeds stack cap %d — using CPU tree-walk", n, TW_GPU_STACK_CAP);
            return false;
        }

        Config cfg = Config.getInstance();
        TreeWalkData d = buildTreeWalkData(splitList, clusterTable, clusterTrees, partTrees);
        int numSplits = splitList.size();

        long residentMem = 8L * (d.clusterBits.length + d.geneLgBits.length)
                         + 4L * ((long) d.nodeStream.length + d.treeNodeOffset.length + d.leafCount.length);
        long perSplit = 4L * Integer.BYTES + (useInt128 ? 2L : 1L) * Long.BYTES;

        int batchSizeHint; String batchDesc;
        if (!cfg.isGpuBatch()) {
            batchSizeHint = -1; batchDesc = "off (single launch)";
        } else if (cfg.getGpuNumBatches() > 0) {
            int N = cfg.getGpuNumBatches();
            batchSizeHint = (numSplits + N - 1) / N;
            batchDesc = N + " batches → batchSize=" + batchSizeHint;
        } else if (cfg.getGpuBatchSize() > 0) {
            batchSizeHint = cfg.getGpuBatchSize();
            batchDesc = "explicit batchSize=" + batchSizeHint;
        } else if (cfg.isGpuVramControlFactorSet()) {
            double F = cfg.getGpuVramControlFactor();
            long batchMem = (long) (F * residentMem);
            batchSizeHint = (int) Math.max(1, Math.min(numSplits, batchMem / perSplit));
            batchDesc = String.format("vram-control-factor=%.3f  resident=%.1f MB  batch→%d",
                F, residentMem / 1e6, (numSplits + batchSizeHint - 1) / Math.max(1, batchSizeHint));
        } else {
            batchSizeHint = 0;
            batchDesc = String.format("auto (free-VRAM adaptive, occupancy=%.0f%%)", cfg.getGpuVramFraction() * 100);
        }

        Logging.info("Weight table: GPU path (simple-tree-walk)  splits=%d  W=%d  clusters=%d  trees=%d  tokens=%d  resident=%.1f MB  batching=%s",
            numSplits, d.W, d.numClusters, d.numTrees, d.nodeStream.length, residentMem / 1e6, batchDesc);

        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsTreeWalkGPU(
            d.splitCid, d.clusterBits, d.geneLgBits, d.nodeStream, d.treeNodeOffset, d.leafCount,
            numSplits, d.numClusters, d.numTrees, d.W, n,
            batchSizeHint, cfg.getGpuVramFraction(), nativeScoreMode(),
            cfg.getGpuProgressIntervalSec());
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;

        if (twoScores == null) {
            Logging.info("  GPU tree-walk kernel returned null after %d ms (infeasible)", gpuMs);
            return false;
        }
        Logging.info("  GPU tree-walk kernel returned in %d ms", gpuMs);
        unpackTwoScores(twoScores, scoreArray, scoreArrayD, scoreArrayI, numSplits);
        return true;
    }

    // -------------------------------------------------------------------------
    // CPU path (also the fallback when the GPU path is infeasible)
    // -------------------------------------------------------------------------

    private void computeScoresCPU(List<BipartitionSplit> splitList,
                                   Collection<PartitionTable.Entry> partitions,
                                   ClusterTable clusterTable,
                                   List<Tree> clusterTrees, List<Tree> partTrees,
                                   long[] scoreArray, double[] scoreArrayD, Int128[] scoreArrayI) {
        int numSplits = splitList.size();
        // CPU: parallel over splits (TRACE: single-threaded for deterministic output)
        if (Logging.isTrace()) {
            for (int idx = 0; idx < numSplits; idx++) {
                BipartitionSplit sp = splitList.get(idx);
                Logging.trace("SPLIT sz=%d|%d  lo=%s  hi=%s",
                    sp.lo.size, sp.hi.size, sp.lo, sp.hi);
                if (useInt128) {
                    scoreArrayI[idx] = computeScoreI(sp, partitions, clusterTable, clusterTrees, partTrees);
                    Logging.trace("  => score=%s", scoreArrayI[idx]);
                } else if (useDouble) {
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
                if (useInt128) {
                    scoreArrayI[idx] = computeScoreI(splitList.get(idx), partitions, clusterTable,
                                                     clusterTrees, partTrees);
                } else if (useDouble) {
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
    // Cluster-side intersection dispatch (single-range fast path / multi-range).
    //
    // For a single-range cluster (los == null) these are byte-identical to the
    // original IntersectionCounter calls.  For a multi-range cluster they sum the
    // intersection over the cluster's disjoint ranges (see multi-range-cluster-design.md
    // §5.1).  Centralizing here keeps all three numeric modes consistent.
    // -------------------------------------------------------------------------

    /** |M_range ∩ cluster| where cluster c (in tree tC) may be single- or multi-range. */
    private static int clusterIntersect(Tree tGT, int loGT, int hiGT,
                                        Tree tC, Cluster c, int sizeGTRange) {
        if (c.los != null)
            return IntersectionCounter.intersectMulti(tGT, loGT, hiGT, tC, c.los, c.his,
                                                      c.complement, sizeGTRange);
        return IntersectionCounter.intersect(tGT, loGT, hiGT, tC, c.left, c.right,
                                             c.complement, sizeGTRange);
    }

    /** |cluster ∩ Lg_GT| row sum for incomplete gene trees; single- or multi-range. */
    private static int clusterFullTree(Tree tGT, Tree tC, Cluster c) {
        if (c.los != null)
            return IntersectionCounter.intersectWithFullTreeMulti(tGT, tC, c.los, c.his, c.complement);
        return IntersectionCounter.intersectWithFullTree(tGT, tC, c.left, c.right, c.complement);
    }

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

            // Polytomous partition (d > 3): O(d) QI formula (polytomy-design.md §3.8.2).
            if (p.d > 3) {
                int lgAp = tGT.isComplete ? sizeA : clusterFullTree(tGT, tA, cA);
                int lgBp = tGT.isComplete ? sizeB : clusterFullTree(tGT, tB, cB);
                int[][] parts = polyParts(p, tGT, tA, cA, tB, cB, lgAp, lgBp);
                if (parts == null) continue;
                long twoQI = polyTwoQILong(parts[0], parts[1], parts[2], p.d);
                twoScore += (long) pe.frequency * twoQI;
                continue;
            }

            // M1 = [leftStart, leftEnd), M2 = [rightStart, rightEnd)
            int lo1 = p.leftStart,  hi1 = p.leftEnd;   // M1 range
            int lo2 = p.rightStart, hi2 = p.rightEnd;  // M2 range
            int sz1 = p.size1, sz2 = p.size2, sz3 = p.size3;

            // 4 core intersections
            int a0 = clusterIntersect(tGT, lo1, hi1, tA, cA, sz1);
            int a1 = clusterIntersect(tGT, lo2, hi2, tA, cA, sz2);
            int b0 = clusterIntersect(tGT, lo1, hi1, tB, cB, sz1);
            int b1 = clusterIntersect(tGT, lo2, hi2, tB, cB, sz2);

            // Row sums: for incomplete gene trees, |A∩Lg_GT| < sizeA; must compute explicitly
            int lgA = tGT.isComplete ? sizeA : clusterFullTree(tGT, tA, cA);
            int lgB = tGT.isComplete ? sizeB : clusterFullTree(tGT, tB, cB);

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

            if (p.d > 3) {   // polytomous: O(d) QI (double accumulation)
                int lgAp = tGT.isComplete ? sizeA : clusterFullTree(tGT, tA, cA);
                int lgBp = tGT.isComplete ? sizeB : clusterFullTree(tGT, tB, cB);
                int[][] parts = polyParts(p, tGT, tA, cA, tB, cB, lgAp, lgBp);
                if (parts == null) continue;
                double twoQI = polyTwoQIDouble(parts[0], parts[1], parts[2], p.d);
                twoScore += (double) pe.frequency * twoQI;
                continue;
            }

            int lo1 = p.leftStart,  hi1 = p.leftEnd;
            int lo2 = p.rightStart, hi2 = p.rightEnd;
            int sz1 = p.size1, sz2 = p.size2, sz3 = p.size3;

            int a0 = clusterIntersect(tGT, lo1, hi1, tA, cA, sz1);
            int a1 = clusterIntersect(tGT, lo2, hi2, tA, cA, sz2);
            int b0 = clusterIntersect(tGT, lo1, hi1, tB, cB, sz1);
            int b1 = clusterIntersect(tGT, lo2, hi2, tB, cB, sz2);

            int lgA = tGT.isComplete ? sizeA : clusterFullTree(tGT, tA, cA);
            int lgB = tGT.isComplete ? sizeB : clusterFullTree(tGT, tB, cB);

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
    // CPU path — INT128 variant (exact, overflow-free; used when the configured
    // large-score type is INT128).  Same intersection matrix as computeScore();
    // only the QI products and the running total are 128-bit.
    // -------------------------------------------------------------------------

    private Int128 computeScoreI(BipartitionSplit split,
                                  Collection<PartitionTable.Entry> partitions,
                                  ClusterTable clusterTable,
                                  List<Tree> clusterTrees, List<Tree> partTrees) {
        ClusterTable.Entry eA = clusterTable.get(split.lo);
        ClusterTable.Entry eB = clusterTable.get(split.hi);
        if (eA == null || eB == null) return Int128.ZERO;

        Cluster cA = eA.exemplar;
        Cluster cB = eB.exemplar;
        Tree tA = clusterTrees.get(cA.treeIndex);
        Tree tB = clusterTrees.get(cB.treeIndex);
        int sizeA = cA.size;
        int sizeB = cB.size;
        int sizeC = n - sizeA - sizeB;
        if (sizeC < 0) return Int128.ZERO;

        Int128 twoScore = Int128.ZERO;

        for (PartitionTable.Entry pe : partitions) {
            Partition p = pe.exemplar;
            Tree tGT = partTrees.get(p.treeIndex);

            if (p.d > 3) {   // polytomous: O(d) QI (exact 128-bit)
                int lgAp = tGT.isComplete ? sizeA : clusterFullTree(tGT, tA, cA);
                int lgBp = tGT.isComplete ? sizeB : clusterFullTree(tGT, tB, cB);
                int[][] parts = polyParts(p, tGT, tA, cA, tB, cB, lgAp, lgBp);
                if (parts == null) continue;
                Int128 twoQI = polyTwoQIInt128(parts[0], parts[1], parts[2], p.d);
                twoScore = twoScore.add(twoQI.mulScalar(pe.frequency));
                continue;
            }

            int lo1 = p.leftStart,  hi1 = p.leftEnd;
            int lo2 = p.rightStart, hi2 = p.rightEnd;
            int sz1 = p.size1, sz2 = p.size2, sz3 = p.size3;

            int a0 = clusterIntersect(tGT, lo1, hi1, tA, cA, sz1);
            int a1 = clusterIntersect(tGT, lo2, hi2, tA, cA, sz2);
            int b0 = clusterIntersect(tGT, lo1, hi1, tB, cB, sz1);
            int b1 = clusterIntersect(tGT, lo2, hi2, tB, cB, sz2);

            int lgA = tGT.isComplete ? sizeA : clusterFullTree(tGT, tA, cA);
            int lgB = tGT.isComplete ? sizeB : clusterFullTree(tGT, tB, cB);

            int a2 = lgA - a0 - a1;
            int b2 = lgB - b0 - b1;
            int c0 = sz1 - a0 - b0;
            int c1 = sz2 - a1 - b1;
            int c2 = sz3 - a2 - b2;

            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

            Int128 twoQI = computeTwoQIInt128(a0, a1, a2, b0, b1, b2, c0, c1, c2);
            twoScore = twoScore.add(twoQI.mulScalar(pe.frequency));
        }

        return twoScore.halve();
    }

    /** Exact 128-bit mirror of {@link #computeTwoQI}; per-term ai·bj·ck·su promoted to 128-bit. */
    private static Int128 computeTwoQIInt128(int a0, int a1, int a2,
                                              int b0, int b1, int b2,
                                              int c0, int c1, int c2) {
        long[] a = {a0, a1, a2};
        long[] b = {b0, b1, b2};
        long[] c = {c0, c1, c2};

        Int128 sum = Int128.ZERO;
        int[][] perms = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        for (int[] perm : perms) {
            long ai = a[perm[0]], bj = b[perm[1]], ck = c[perm[2]];
            long s = ai + bj + ck - 3;
            if (s > 0) {
                long abc = ai * bj * ck;                 // ≤ ~2^51, fits long
                sum = sum.add(Int128.mulLong(abc, s));   // (≤2^70) exact 128-bit
            }
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // Polytomy (d > 3) QI — the O(d) ASTRAL-III formula (polytomy-design.md §3.8.2):
    //   2·QI = Σᵢ aᵢ(aᵢ-1)·[(Sb-bᵢ)(Sc-cᵢ) - Sbc + bᵢcᵢ]
    //        + Σᵢ bᵢ(bᵢ-1)·[(Sa-aᵢ)(Sc-cᵢ) - Sac + aᵢcᵢ]
    //        + Σᵢ cᵢ(cᵢ-1)·[(Sa-aᵢ)(Sb-bᵢ) - Sab + aᵢbᵢ]
    // Every bracket is a sum of products of non-negative parts ⇒ ≥ 0 (no cancellation),
    // and the total ≤ O(n⁴) — the same magnitude budget as the binary formula.
    // -------------------------------------------------------------------------

    /**
     * Build the d×3 intersection matrix (aᵢ,bᵢ,cᵢ for i=0..d-1) for a polytomous
     * partition.  The d-1 child parts are intersected directly; the complement part
     * (index d-1) is recovered from the row constraints (lgA, lgB).  Returns null if
     * any complement-derived part is negative (incomplete-tree row mismatch → skip),
     * mirroring the binary {@code a2/b2/c0/c1/c2 < 0} guard.
     *
     * @return {a[], b[], c[]} each length d, or null if invalid.
     */
    private static int[][] polyParts(Partition p, Tree tGT,
                                     Tree tA, Cluster cA, Tree tB, Cluster cB,
                                     int lgA, int lgB) {
        int d = p.d;
        int[] a = new int[d], b = new int[d], c = new int[d];
        int sumA = 0, sumB = 0;
        for (int i = 0; i < d - 1; i++) {
            int lo = p.partStarts[i], hi = p.partEnds[i], szi = p.sizes[i];
            int ai = clusterIntersect(tGT, lo, hi, tA, cA, szi);
            int bi = clusterIntersect(tGT, lo, hi, tB, cB, szi);
            int ci = szi - ai - bi;                 // ≥ 0: A,B disjoint ⇒ aᵢ+bᵢ ≤ |Mᵢ|
            if (ci < 0) return null;                // defensive
            a[i] = ai; b[i] = bi; c[i] = ci;
            sumA += ai; sumB += bi;
        }
        int aC = lgA - sumA;                        // complement part via row constraint
        int bC = lgB - sumB;
        int cC = p.sizes[d - 1] - aC - bC;
        if (aC < 0 || bC < 0 || cC < 0) return null;
        a[d - 1] = aC; b[d - 1] = bC; c[d - 1] = cC;
        return new int[][]{ a, b, c };
    }

    private static long polyTwoQILong(int[] a, int[] b, int[] c, int d) {
        long Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        for (int i = 0; i < d; i++) {
            Sa += a[i]; Sb += b[i]; Sc += c[i];
            Sab += (long) a[i] * b[i]; Sac += (long) a[i] * c[i]; Sbc += (long) b[i] * c[i];
        }
        long two = 0;
        for (int i = 0; i < d; i++) {
            long ai = a[i], bi = b[i], ci = c[i];
            two += ai * (ai - 1) * ((Sb - bi) * (Sc - ci) - Sbc + bi * ci);
            two += bi * (bi - 1) * ((Sa - ai) * (Sc - ci) - Sac + ai * ci);
            two += ci * (ci - 1) * ((Sa - ai) * (Sb - bi) - Sab + ai * bi);
        }
        return two;
    }

    private static double polyTwoQIDouble(int[] a, int[] b, int[] c, int d) {
        double Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        for (int i = 0; i < d; i++) {
            Sa += a[i]; Sb += b[i]; Sc += c[i];
            Sab += (double) a[i] * b[i]; Sac += (double) a[i] * c[i]; Sbc += (double) b[i] * c[i];
        }
        double two = 0;
        for (int i = 0; i < d; i++) {
            double ai = a[i], bi = b[i], ci = c[i];
            two += ai * (ai - 1) * ((Sb - bi) * (Sc - ci) - Sbc + bi * ci);
            two += bi * (bi - 1) * ((Sa - ai) * (Sc - ci) - Sac + ai * ci);
            two += ci * (ci - 1) * ((Sa - ai) * (Sb - bi) - Sab + ai * bi);
        }
        return two;
    }

    private static Int128 polyTwoQIInt128(int[] a, int[] b, int[] c, int d) {
        long Sa = 0, Sb = 0, Sc = 0, Sab = 0, Sac = 0, Sbc = 0;
        for (int i = 0; i < d; i++) {
            Sa += a[i]; Sb += b[i]; Sc += c[i];
            Sab += (long) a[i] * b[i]; Sac += (long) a[i] * c[i]; Sbc += (long) b[i] * c[i];
        }
        // Each weight aᵢ(aᵢ-1) ≤ n² and each bracket ≤ n² fit in a signed long; their
        // product (≤ n⁴) is formed exactly in 128-bit via Int128.mulLong.
        Int128 two = Int128.ZERO;
        for (int i = 0; i < d; i++) {
            long ai = a[i], bi = b[i], ci = c[i];
            long bracketA = (Sb - bi) * (Sc - ci) - Sbc + bi * ci;
            long bracketB = (Sa - ai) * (Sc - ci) - Sac + ai * ci;
            long bracketC = (Sa - ai) * (Sb - bi) - Sab + ai * bi;
            if (ai >= 2 && bracketA != 0) two = two.add(Int128.mulLong(ai * (ai - 1), bracketA));
            if (bi >= 2 && bracketB != 0) two = two.add(Int128.mulLong(bi * (bi - 1), bracketB));
            if (ci >= 2 && bracketC != 0) two = two.add(Int128.mulLong(ci * (ci - 1), bracketC));
        }
        return two;
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    /** The active score numeric type. */
    public Mode getMode()      { return mode; }
    public boolean isDouble()  { return useDouble; }
    public boolean isInt128()  { return useInt128; }

    /**
     * Score of a split as a long.  In DOUBLE/INT128 mode this returns the value
     * rounded/clamped to long (debug/verifier tooling only); the DP must use the
     * mode-matching accessor ({@link #getScoreD} / {@link #getScoreI}).
     */
    public long getScore(BipartitionSplit split) {
        if (useInt128) { Int128 v = scoresI.get(split); return v == null ? 0L : Math.round(v.toDouble()); }
        if (useDouble) return Math.round(scoresD.getOrDefault(split, 0.0));
        return scores.getOrDefault(split, 0L);
    }

    /** Score of a split as a double (valid in all modes; approximate for INT128). */
    public double getScoreD(BipartitionSplit split) {
        if (useInt128) { Int128 v = scoresI.get(split); return v == null ? 0.0 : v.toDouble(); }
        if (useDouble) return scoresD.getOrDefault(split, 0.0);
        return (double) scores.getOrDefault(split, 0L);
    }

    /** Score of a split as an exact Int128 (valid in all modes). */
    public Int128 getScoreI(BipartitionSplit split) {
        if (useInt128) return scoresI.getOrDefault(split, Int128.ZERO);
        if (useDouble) return Int128.ofLong(Math.round(scoresD.getOrDefault(split, 0.0)));
        return Int128.ofLong(scores.getOrDefault(split, 0L));
    }

    public long   getMaxScore()    { return useInt128 ? Math.round(getMaxScoreD())
                                          : useDouble ? Math.round(maxScoreD)   : maxScore; }
    public long   getTotalScore()  { return useInt128 ? Math.round(getTotalScoreD())
                                          : useDouble ? Math.round(totalScoreD) : totalScore; }
    public double getMaxScoreD()   { return useInt128 ? (maxScoreI == null ? 0.0 : maxScoreI.toDouble())
                                          : useDouble ? maxScoreD   : (double) maxScore; }
    public double getTotalScoreD() { return useInt128 ? totalScoreI.toDouble()
                                          : useDouble ? totalScoreD : (double) totalScore; }
    public int    size()           { return useInt128 ? scoresI.size()
                                          : useDouble ? scoresD.size() : scores.size(); }

    /** Iterate all (split, score) pairs (LONG mode only; empty in DOUBLE/INT128 mode). */
    public Set<Map.Entry<BipartitionSplit, Long>> entries() { return scores.entrySet(); }
}
