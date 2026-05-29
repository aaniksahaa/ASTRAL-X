package astralx.greedy;

import astralx.tree.Tree;
import astralx.tree.TreeNode;

/**
 * Sequential UPGMA on a small {@code g × g} similarity matrix, used by the
 * polytomy resolution phase.
 *
 * Why not reuse {@link astralx.completion.UPGMAClusterer}?  That class uses
 * {@code Threading.processRangeParallel} internally.  If we call it from
 * inside an outer polytomy-pool worker (which is also submitted to the
 * shared {@code Threading} executor), the inner {@code processRangeParallel}
 * queues sub-tasks AND blocks on a latch — but every worker thread is busy
 * holding its polytomy task, so the sub-tasks never start: deadlock.
 *
 * For polytomy sizes ({@code g ≤ √(50+25n)}, typically ≤ 31 in practice),
 * the sequential UPGMA is also faster than the parallel one — internal
 * threading overhead dominates the actual work.
 *
 * Output format matches {@link astralx.completion.UPGMAClusterer#build}:
 *   - {@link TreeNode#left}/{@link TreeNode#right} for the dendrogram structure
 *   - {@link Tree#postorderArray} listing the original cluster indices in
 *     left-to-right post-order
 *   - per-node {@code rangeStart}/{@code rangeEnd} stamps into postorderArray
 */
public final class MiniUPGMA {

    private MiniUPGMA() {}

    /**
     * Run UPGMA on the flat {@code n × n} similarity matrix and return the
     * dendrogram.  Time: O(n³); for n ≤ 31 this is ≤ 30k ops, trivial.
     */
    public static Tree build(double[] sim, int n, int treeIndex) {
        if (n <= 0) throw new IllegalArgumentException("MiniUPGMA: n <= 0");
        if (n == 1) {
            TreeNode leaf = new TreeNode();
            leaf.taxonId = 0;
            leaf.rangeStart = 0;
            leaf.rangeEnd   = 1;
            return new Tree(treeIndex, leaf, new int[]{0}, new int[]{0}, 1, 1);
        }

        TreeNode[] clusterRoot = new TreeNode[n];
        double[]    weight     = new double[n];
        boolean[]   active     = new boolean[n];
        for (int i = 0; i < n; i++) {
            TreeNode leaf = new TreeNode();
            leaf.taxonId   = i;
            clusterRoot[i] = leaf;
            weight[i]      = 1.0;
            active[i]      = true;
        }

        // Materialize a mutable copy so we can update rows in place
        double[][] mat = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) mat[i][j] = sim[i * n + j];
        }

        for (int iter = 0; iter < n - 1; iter++) {
            // Find max-similarity active pair
            int bestI = -1, bestJ = -1;
            double bestS = -Double.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                if (!active[i]) continue;
                for (int j = i + 1; j < n; j++) {
                    if (!active[j]) continue;
                    if (mat[i][j] > bestS) { bestS = mat[i][j]; bestI = i; bestJ = j; }
                }
            }
            if (bestI < 0) break;          // disconnected — shouldn't happen on a full matrix

            // Merge J into I
            TreeNode newNode = new TreeNode();
            newNode.left  = clusterRoot[bestI];
            newNode.right = clusterRoot[bestJ];
            clusterRoot[bestI].parent = newNode;
            clusterRoot[bestJ].parent = newNode;
            clusterRoot[bestI] = newNode;

            double wI = weight[bestI], wJ = weight[bestJ];
            weight[bestI] = wI + wJ;
            active[bestJ] = false;

            // Weighted-average update of row I
            for (int k = 0; k < n; k++) {
                if (k == bestI || !active[k]) continue;
                double newIK = (mat[bestI][k] * wI + mat[bestJ][k] * wJ) / (wI + wJ);
                mat[bestI][k] = newIK;
                mat[k][bestI] = newIK;
            }
        }

        TreeNode root = null;
        for (int i = 0; i < n; i++) if (active[i]) { root = clusterRoot[i]; break; }
        if (root == null) root = clusterRoot[0];

        int[] postArr = new int[n];
        int[] posMap  = new int[n];
        int[] pos = {0};
        stampRanges(root, postArr, pos);
        for (int i = 0; i < n; i++) posMap[postArr[i]] = i;

        return new Tree(treeIndex, root, postArr, posMap, n, n);
    }

    private static void stampRanges(TreeNode node, int[] postArr, int[] pos) {
        int lo = pos[0];
        if (node.left == null) {           // leaf
            postArr[pos[0]++] = node.taxonId;
        } else {
            stampRanges(node.left,  postArr, pos);
            stampRanges(node.right, postArr, pos);
        }
        node.rangeStart = lo;
        node.rangeEnd   = pos[0];
    }
}
