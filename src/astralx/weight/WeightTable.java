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

    public WeightTable(DPTable dpTable, PartitionTable partTable,
                       ClusterTable clusterTable, List<Tree> trees) {
        long t0 = System.nanoTime();
        this.n = clusterTable.getAllTaxaHash().size;

        // Collect all unique splits from DPTable into an indexed list
        List<BipartitionSplit> splitList = new ArrayList<>();
        for (var entry : dpTable.entries()) splitList.addAll(entry.getValue());
        int numSplits = splitList.size();

        List<PartitionTable.Entry> partList = new ArrayList<>(partTable.entries());
        long[] scoreArray = new long[numSplits];

        boolean useGPU = (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU)
                         && GPUWeightCalculator.tryLoad();

        if (useGPU) {
            Logging.info("Weight table: using GPU path (%d splits, %d partitions)",
                numSplits, partList.size());
            computeScoresGPU(splitList, partList, clusterTable, trees, scoreArray);
        } else {
            if (Config.getInstance().getComputeMode() == Config.ComputeMode.GPU) {
                Logging.info("GPU library not available, falling back to CPU");
            }
            // CPU: parallel over splits
            Collection<PartitionTable.Entry> partitions = partTable.entries();
            Threading.processRangeParallel(numSplits, idx -> {
                scoreArray[idx] = computeScore(splitList.get(idx), partitions, clusterTable, trees);
            });
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
    private void computeScoresGPU(List<BipartitionSplit> splitList,
                                   List<PartitionTable.Entry> partList,
                                   ClusterTable clusterTable,
                                   List<Tree> trees,
                                   long[] scoreArray) {
        int numSplits = splitList.size();
        int numParts  = partList.size();
        int numTrees  = trees.size();

        // --- splits: numSplits * 10 ints ---
        // [loTree, loLeft, loRight, loComp, loSize, hiTree, hiLeft, hiRight, hiComp, hiSize]
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
            // else: all zeros → kernel will compute sizeC = n - 0 - 0, but the
            // first valid partition check will likely skip; score stays 0.
        }

        // --- partitions: numParts * 9 ints ---
        // [treeIdx, lo1, hi1, lo2, hi2, sz1, sz2, sz3, frequency]
        int[] partsData = new int[numParts * 9];
        for (int j = 0; j < numParts; j++) {
            PartitionTable.Entry pe = partList.get(j);
            Partition p = pe.exemplar;
            int base = j * 9;
            partsData[base + 0] = p.treeIndex;
            partsData[base + 1] = p.leftStart;
            partsData[base + 2] = p.leftEnd;
            partsData[base + 3] = p.rightStart;
            partsData[base + 4] = p.rightEnd;
            partsData[base + 5] = p.size1;
            partsData[base + 6] = p.size2;
            partsData[base + 7] = p.size3;
            partsData[base + 8] = pe.frequency;
        }

        // --- orderings + invIndex: numTrees * n ints each ---
        // orderings[t*n + pos]   = postorderArray[pos]
        // invIndex [t*n + taxon] = positionMap[taxon]  (-1 if absent)
        int[] orderings = new int[numTrees * n];
        int[] invIndex  = new int[numTrees * n];
        Arrays.fill(invIndex, -1);
        for (int t = 0; t < numTrees; t++) {
            Tree tree = trees.get(t);
            int base = t * n;
            for (int pos = 0; pos < tree.leafCount; pos++) {
                orderings[base + pos] = tree.postorderArray[pos];
            }
            for (int taxon = 0; taxon < n; taxon++) {
                invIndex[base + taxon] = tree.positionMap[taxon];
            }
        }

        // --- call GPU ---
        long t1 = System.nanoTime();
        long[] twoScores = GPUWeightCalculator.computeWeightsGPU(
            splitsData, partsData, orderings, invIndex,
            numSplits, numParts, numTrees, n, n);
        long gpuMs = (System.nanoTime() - t1) / 1_000_000;
        Logging.info("  GPU kernel returned in %d ms", gpuMs);

        // twoScores[i] = 2 * score; divide by 2
        for (int i = 0; i < numSplits; i++) {
            scoreArray[i] = twoScores[i] / 2L;
        }
    }

    // -------------------------------------------------------------------------
    // CPU path
    // -------------------------------------------------------------------------

    private long computeScore(BipartitionSplit split,
                               Collection<PartitionTable.Entry> partitions,
                               ClusterTable clusterTable, List<Tree> trees) {
        // Retrieve exemplars for A (lo half) and B (hi half)
        ClusterTable.Entry eA = clusterTable.get(split.lo);
        ClusterTable.Entry eB = clusterTable.get(split.hi);
        if (eA == null || eB == null) return 0L;

        Cluster cA = eA.exemplar;
        Cluster cB = eB.exemplar;
        Tree tA = trees.get(cA.treeIndex);
        Tree tB = trees.get(cB.treeIndex);
        int sizeA = cA.size;
        int sizeB = cB.size;
        int sizeC = n - sizeA - sizeB;
        if (sizeC < 0) return 0L;  // sanity

        long twoScore = 0L;

        for (PartitionTable.Entry pe : partitions) {
            Partition p = pe.exemplar;
            Tree tGT = trees.get(p.treeIndex);

            // M1 = [leftStart, leftEnd), M2 = [rightStart, rightEnd)
            int lo1 = p.leftStart,  hi1 = p.leftEnd;   // M1 range
            int lo2 = p.rightStart, hi2 = p.rightEnd;  // M2 range
            int sz1 = p.size1, sz2 = p.size2, sz3 = p.size3;

            // 4 core intersections
            int a0 = IntersectionCounter.intersect(tGT, lo1, hi1, tA, cA.left, cA.right, cA.complement, sz1);
            int a1 = IntersectionCounter.intersect(tGT, lo2, hi2, tA, cA.left, cA.right, cA.complement, sz2);
            int b0 = IntersectionCounter.intersect(tGT, lo1, hi1, tB, cB.left, cB.right, cB.complement, sz1);
            int b1 = IntersectionCounter.intersect(tGT, lo2, hi2, tB, cB.left, cB.right, cB.complement, sz2);

            // Derive remaining 5
            int a2 = sizeA - a0 - a1;          // row constraint on A
            int b2 = sizeB - b0 - b1;          // row constraint on B
            int c0 = sz1 - a0 - b0;            // column constraint (complete trees)
            int c1 = sz2 - a1 - b1;
            int c2 = sz3 - c0 - c1;            // row constraint on C (via M3)

            // All values must be non-negative for a valid intersection matrix
            if (a2 < 0 || b2 < 0 || c0 < 0 || c1 < 0 || c2 < 0) continue;

            long twoQI = computeTwoQI(a0, a1, a2, b0, b1, b2, c0, c1, c2);
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
