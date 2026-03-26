package astralx.weight;

import astralx.Logging;
import astralx.cluster.Cluster;
import astralx.cluster.ClusterHash;
import astralx.cluster.ClusterTable;
import astralx.dp.BipartitionSplit;
import astralx.dp.DPTable;
import astralx.partition.Partition;
import astralx.partition.PartitionTable;
import astralx.tree.Tree;

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

        // Collect all unique splits from DPTable
        Set<BipartitionSplit> allSplits = new LinkedHashSet<>();
        for (var entry : dpTable.entries()) {
            allSplits.addAll(entry.getValue());
        }

        // Precompute score for each split
        Collection<PartitionTable.Entry> partitions = partTable.entries();
        for (BipartitionSplit split : allSplits) {
            long score = computeScore(split, partitions, clusterTable, trees);
            scores.put(split, score);
            if (score > maxScore) maxScore = score;
            totalScore += score;
        }

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Weight table: %d splits scored, maxScore=%d, totalScore=%d in %d ms",
            scores.size(), maxScore, totalScore, ms);
    }

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
