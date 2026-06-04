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

    private final Map<BipartitionSplit, Long> scores = new HashMap<>();
    private final int n;   // total taxa

    // stats
    private long maxScore;
    private long totalScore;

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

        // Collect all unique splits from DPTable into an indexed list
        List<BipartitionSplit> splitList = new ArrayList<>();
        for (var entry : dpTable.entries()) splitList.addAll(entry.getValue());
        int numSplits = splitList.size();

        List<PartitionTable.Entry> partList = new ArrayList<>(partTable.entries());
        long[] scoreArray = new long[numSplits];

        // When clusterTrees != partTrees (autocomplete active), the GPU path packs both
        // sets of orderings/invIndex into a combined array (slots 0..k-1 = completed,
        // slots k..2k-1 = original) and offsets partition tree indices by k.
        // numGpuTrees reflects the combined size for VRAM budget calculations.
        boolean splitTrees = (clusterTrees != partTrees);
        int numGpuTrees = splitTrees ? clusterTrees.size() * 2 : clusterTrees.size();

        boolean useGPU = (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU)
                         && GPUWeightCalculator.tryLoad();

        if (useGPU) {
            // Build per-tree internal-node CSR over the gene trees (partTrees).
            // Each non-root internal node contributes one tripartition as a
            // contiguous leaf interval (lo, mid, hi); no global dedup.
            NodeCSR csr = buildNodeCSR(partTrees);

            // Resolve batchSizeHint
            //   Priority: no-batch  >  gpu-batches  >  gpu-batch-size
            //           > gpu-vram-control-factor (explicit)  >  auto (gpu-vram-occupancy-factor)
            //   -1  = no batching (single launch)
            //    0  = auto: native queries free VRAM and computes batch size itself
            //   >0  = exact splits-per-batch resolved here; native uses it directly
            Config cfg = Config.getInstance();
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
                long   nodeMem     = (long) csr.nodeData.length      * Integer.BYTES
                                   + (long) csr.nodeOffset.length    * Integer.BYTES
                                   + (long) csr.partLeafCount.length * Integer.BYTES;
                long   orderingMem = (long) numGpuTrees * n  *  2 * Integer.BYTES; // orderings + invIndex
                long   residentMem = nodeMem + orderingMem;
                long   batchMem    = (long)(F * residentMem);
                long   perSplit    = 10L * Integer.BYTES + Long.BYTES;              // 48 B/split
                batchSizeHint      = (int) Math.max(1, Math.min(numSplits, batchMem / perSplit));
                int numBatches     = (numSplits + batchSizeHint - 1) / batchSizeHint;
                batchDesc = String.format(
                    "vram-control-factor=%.3f  resident=%.1f MB (nodeCSR=%.1f orderings=%.1f)  batch=%.1f MB  → %d batches",
                    F, residentMem / 1e6, nodeMem / 1e6, orderingMem / 1e6, batchMem / 1e6, numBatches);
            } else {
                // Default: auto — pass 0 to native; native queries free VRAM after static upload
                // and computes batchSize = floor(freeVRAM * vramFraction / 48 B)
                batchSizeHint = 0;
                batchDesc = String.format("auto (free-VRAM adaptive, occupancy=%.0f%%)",
                    cfg.getGpuVramFraction() * 100);
            }
            Logging.info("Weight table: GPU path  splits=%d  internalNodes=%d  trees=%d  maxLeaf=%d  batching=%s",
                numSplits, csr.totalNodes, partTrees.size(), csr.maxLeafCount, batchDesc);

            boolean ok = computeScoresGPU(splitList, csr, clusterTable, clusterTrees, partTrees,
                                          numGpuTrees, scoreArray, batchSizeHint, cfg.getGpuVramFraction());
            if (!ok) {
                Logging.info("GPU weight path infeasible (e.g. shared-memory limit), falling back to CPU");
                computeScoresCPU(splitList, partTable.entries(), clusterTable,
                                 clusterTrees, partTrees, scoreArray);
            }
        } else {
            if (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU) {
                Logging.info("GPU library not available, falling back to CPU");
            }
            computeScoresCPU(splitList, partTable.entries(), clusterTable,
                             clusterTrees, partTrees, scoreArray);
        }

        for (int i = 0; i < numSplits; i++) {
            scores.put(splitList.get(i), scoreArray[i]);
            if (scoreArray[i] > maxScore) maxScore = scoreArray[i];
            totalScore += scoreArray[i];
        }

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Weight table: %d splits scored, maxScore=%d, totalScore=%d in %d ms",
            scores.size(), maxScore, totalScore, ms);
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
    private boolean computeScoresGPU(List<BipartitionSplit> splitList,
                                      NodeCSR csr,
                                      ClusterTable clusterTable,
                                      List<Tree> clusterTrees,
                                      List<Tree> partTrees,
                                      int numGpuTrees,
                                      long[] scoreArray,
                                      int batchSizeHint,
                                      double vramFraction) {
        int numSplits       = splitList.size();
        int numClusterTrees = clusterTrees.size();
        int numPartTrees    = partTrees.size();
        boolean splitTrees  = (clusterTrees != partTrees);
        // partTreeOffset: orderings/invIndex slot offset so node-CSR leaf lookups
        // index into the original-tree half of the combined array.
        int partTreeOffset = splitTrees ? numClusterTrees : 0;

        // --- splits: numSplits * 10 ints ---
        // Cluster treeIndex values are 0..k-1 (completed trees, used for membership).
        // [aTree, aLo, aHi, aComp, aSize, bTree, bLo, bHi, bComp, bSize]
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

        // --- orderings + invIndex: numGpuTrees * n ints each ---
        // orderings[t*n + pos]   = postorderArray[pos]
        // invIndex [t*n + taxon] = positionMap[taxon]  (-1 if absent)
        //
        // Layout when splitTrees:
        //   slots 0..k-1   filled from clusterTrees (completed)  — cluster membership
        //   slots k..2k-1  filled from partTrees (original)      — gene-tree leaves
        // Layout when !splitTrees: same list fills slots 0..k-1.
        int[] orderings = new int[numGpuTrees * n];
        int[] invIndex  = new int[numGpuTrees * n];
        Arrays.fill(invIndex, -1);
        for (int t = 0; t < numClusterTrees; t++) {
            Tree tree = clusterTrees.get(t);
            int base = t * n;
            for (int pos = 0; pos < tree.leafCount; pos++) {
                orderings[base + pos] = tree.postorderArray[pos];
            }
            for (int taxon = 0; taxon < n; taxon++) {
                invIndex[base + taxon] = tree.positionMap[taxon];
            }
        }
        if (splitTrees) {
            for (int t = 0; t < numPartTrees; t++) {
                Tree tree = partTrees.get(t);
                int base = (numClusterTrees + t) * n;
                for (int pos = 0; pos < tree.leafCount; pos++) {
                    orderings[base + pos] = tree.postorderArray[pos];
                }
                for (int taxon = 0; taxon < n; taxon++) {
                    invIndex[base + taxon] = tree.positionMap[taxon];
                }
            }
        }

        // --- call GPU ---
        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsGPU(
            splitsData, csr.nodeData, csr.nodeOffset, csr.partLeafCount,
            orderings, invIndex,
            numSplits, numPartTrees, partTreeOffset, csr.maxLeafCount,
            numGpuTrees, n,
            batchSizeHint, vramFraction);
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;

        // Input arrays are no longer needed after the kernel returns; free them
        // before the twoScores loop so GC can reclaim memory while we fill scoreArray.
        splitsData = null;
        orderings  = null;
        invIndex   = null;

        if (twoScores == null) {
            Logging.info("  GPU kernel returned null after %d ms (infeasible)", gpuMs);
            return false;
        }
        Logging.info("  GPU kernel returned in %d ms", gpuMs);

        // twoScores[i] = 2 * score; divide by 2
        for (int i = 0; i < numSplits; i++) {
            scoreArray[i] = twoScores[i] / 2L;
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Per-tree internal-node CSR (gene-tree tripartitions as leaf intervals)
    // -------------------------------------------------------------------------

    /**
     * Compact, per-tree representation of every non-root internal node of the
     * gene trees.  Each such node is one tripartition (M1|M2|M3), stored as a
     * contiguous leaf interval (lo, mid, hi) where M1 = [lo,mid), M2 = [mid,hi),
     * M3 = Lg \ [lo,hi).  Mirrors the gene-tree tripartitions extracted by
     * PartitionTable, but grouped by tree (no global dedup) for the prefix-sum
     * GPU kernel.
     */
    private static final class NodeCSR {
        int[] nodeData;       // totalNodes * 3   [lo, mid, hi]
        int[] nodeOffset;     // numTrees + 1     CSR row pointers
        int[] partLeafCount;  // numTrees         leaf count L per tree
        int   maxLeafCount;   // max L over trees (shared-memory sizing)
        int   totalNodes;     // sum of contributing internal nodes
    }

    private static NodeCSR buildNodeCSR(List<Tree> partTrees) {
        int numTrees = partTrees.size();
        int[] nodeOffset    = new int[numTrees + 1];
        int[] partLeafCount = new int[numTrees];
        int   maxLeaf = 0;
        long  total   = 0;
        for (int g = 0; g < numTrees; g++) {
            Tree tr = partTrees.get(g);
            partLeafCount[g] = tr.leafCount;
            if (tr.leafCount > maxLeaf) maxLeaf = tr.leafCount;
            int c = countContribNodes(tr.root);
            nodeOffset[g + 1] = nodeOffset[g] + c;
            total += c;
        }
        if (total > Integer.MAX_VALUE / 3) {
            throw new IllegalStateException("Too many internal nodes for a single int[] CSR: " + total);
        }
        int[] nodeData = new int[(int) total * 3];
        for (int g = 0; g < numTrees; g++) {
            int end = fillNodes(partTrees.get(g).root, nodeData, nodeOffset[g]);
            assert end == nodeOffset[g + 1] : "CSR fill/count mismatch for tree " + g;
        }

        NodeCSR csr = new NodeCSR();
        csr.nodeData      = nodeData;
        csr.nodeOffset    = nodeOffset;
        csr.partLeafCount = partLeafCount;
        csr.maxLeafCount  = maxLeaf;
        csr.totalNodes    = (int) total;
        return csr;
    }

    /** Count non-root internal nodes (each yields one tripartition). */
    private static int countContribNodes(TreeNode node) {
        if (node.isLeaf()) return 0;
        int c = countContribNodes(node.left) + countContribNodes(node.right);
        if (!node.isRoot()) c++;
        return c;
    }

    /**
     * Post-order fill of (lo, mid, hi) for every non-root internal node, starting
     * at node-index {@code pos}; returns the next free node-index.  Order matches
     * PartitionTable's extraction (left, right, self).
     */
    private static int fillNodes(TreeNode node, int[] nodeData, int pos) {
        if (node.isLeaf()) return pos;
        pos = fillNodes(node.left,  nodeData, pos);
        pos = fillNodes(node.right, nodeData, pos);
        if (!node.isRoot()) {
            int b = pos * 3;
            nodeData[b]     = node.rangeStart;     // lo
            nodeData[b + 1] = node.left.rangeEnd;  // mid  (= node.right.rangeStart)
            nodeData[b + 2] = node.rangeEnd;       // hi
            pos++;
        }
        return pos;
    }

    // -------------------------------------------------------------------------
    // CPU path (also the fallback when the GPU path is infeasible)
    // -------------------------------------------------------------------------

    private void computeScoresCPU(List<BipartitionSplit> splitList,
                                   Collection<PartitionTable.Entry> partitions,
                                   ClusterTable clusterTable,
                                   List<Tree> clusterTrees, List<Tree> partTrees,
                                   long[] scoreArray) {
        int numSplits = splitList.size();
        // CPU: parallel over splits (TRACE: single-threaded for deterministic output)
        if (Logging.isTrace()) {
            for (int idx = 0; idx < numSplits; idx++) {
                BipartitionSplit sp = splitList.get(idx);
                Logging.trace("SPLIT sz=%d|%d  lo=%s  hi=%s",
                    sp.lo.size, sp.hi.size, sp.lo, sp.hi);
                scoreArray[idx] = computeScore(sp, partitions, clusterTable, clusterTrees, partTrees);
                Logging.trace("  => score=%d", scoreArray[idx]);
            }
        } else {
            java.util.concurrent.atomic.AtomicInteger wDone = new java.util.concurrent.atomic.AtomicInteger(0);
            ProgressBar wBar = new ProgressBar("Scoring splits (CPU)", numSplits);
            Threading.processRangeParallel(numSplits, idx -> {
                scoreArray[idx] = computeScore(splitList.get(idx), partitions, clusterTable,
                                               clusterTrees, partTrees);
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
    // Queries
    // -------------------------------------------------------------------------

    public long getScore(BipartitionSplit split) {
        return scores.getOrDefault(split, 0L);
    }

    public long getMaxScore()   { return maxScore; }
    public long getTotalScore() { return totalScore; }
    public int  size()          { return scores.size(); }

    /** Iterate all (split, score) pairs. */
    public Set<Map.Entry<BipartitionSplit, Long>> entries() { return scores.entrySet(); }
}
