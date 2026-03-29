package astralx.completion;

import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.ProgressBar;

import java.util.Arrays;
import java.util.List;

/**
 * Builds the average pairwise topological distance matrix from a list of gene trees (CPU).
 *
 * Algorithm (O(k × n²) total):
 *   For each gene tree T:
 *     1. Compute leaf depths via DFS.
 *     2. For each internal node u (depth d_u), enumerate all pairs (a ∈ left(u), b ∈ right(u)):
 *        D(a,b) += depth(a) + depth(b) - 2*d_u
 *        cooccurrence(a,b) += 1
 *
 * Every pair of leaves has exactly one node where they "split" (their LCA), so each pair
 * is processed exactly once per tree that contains both.
 */
public class DistanceMatrixBuilder {

    public static DistanceMatrix buildCPU(List<Tree> trees, int n) {
        DistanceMatrix dm = new DistanceMatrix(n);

        ProgressBar bar = new ProgressBar("Building distance matrix (CPU)", trees.size());
        int done = 0;
        for (Tree tree : trees) {
            accumulateTree(tree, dm);
            bar.update(++done);
        }
        bar.done();

        dm.normalize();
        return dm;
    }

    // -------------------------------------------------------------------------

    private static void accumulateTree(Tree tree, DistanceMatrix dm) {
        int n = dm.n;

        // Step 1: compute depth of each leaf (taxonId → depth; -1 if absent)
        int[] leafDepths = new int[n];
        Arrays.fill(leafDepths, -1);
        computeLeafDepths(tree.root, 0, leafDepths);

        // Step 2: for each internal node, accumulate across-subtree pairs
        accumulatePairs(tree.root, 0, leafDepths, tree, dm);
    }

    // ── DFS helpers ──────────────────────────────────────────────────────────

    private static void computeLeafDepths(TreeNode node, int depth, int[] leafDepths) {
        if (node.isLeaf()) {
            leafDepths[node.taxonId] = depth;
            return;
        }
        computeLeafDepths(node.left,  depth + 1, leafDepths);
        computeLeafDepths(node.right, depth + 1, leafDepths);
    }

    /**
     * Post-order accumulation: for each internal node u at depth d_u,
     * add (depth[a] + depth[b] - 2*d_u) for every pair (a in left, b in right).
     *
     * Uses tree.postorderArray to enumerate leaves in each half-open range.
     */
    private static void accumulatePairs(TreeNode node, int depth,
                                         int[] leafDepths, Tree tree, DistanceMatrix dm) {
        if (node.isLeaf()) return;

        accumulatePairs(node.left,  depth + 1, leafDepths, tree, dm);
        accumulatePairs(node.right, depth + 1, leafDepths, tree, dm);

        int n   = dm.n;
        int loL = node.left.rangeStart,  hiL = node.left.rangeEnd;
        int loR = node.right.rangeStart, hiR = node.right.rangeEnd;
        double twoDu = 2.0 * depth;

        for (int pi = loL; pi < hiL; pi++) {
            int    ta = tree.postorderArray[pi];
            double da = leafDepths[ta];
            for (int pj = loR; pj < hiR; pj++) {
                int    tb = tree.postorderArray[pj];
                double d  = da + leafDepths[tb] - twoDu;
                dm.distSum[ta * n + tb] += d;
                dm.distSum[tb * n + ta] += d;
                dm.cooccurrence[ta * n + tb]++;
                dm.cooccurrence[tb * n + ta]++;
            }
        }
    }
}
