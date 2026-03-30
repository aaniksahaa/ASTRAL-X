package astralx.completion;

import astralx.tree.Tree;
import astralx.tree.TreeNode;

/**
 * Builds the Euler tour and sparse-table RMQ structure for one gene tree.
 *
 * Euler tour definition (DFS, recording depths):
 *   - Append depth(node) when first entering any node.
 *   - Append depth(node) when returning from its LEFT child.
 *   - Do NOT append when returning from the right child.
 *
 * Tour length for L leaves = 3L − 2  (every internal node appears exactly
 * twice; each leaf appears exactly once).
 *
 * LCA property: for any two leaves a, b with first occurrences at positions
 * fa, fb in the tour:
 *   depth(LCA(a, b)) = min(eulerDepths[min(fa,fb) .. max(fa,fb)])
 *
 * Sparse table:
 *   sparseMin[lvl][pos] = minimum depth in eulerDepths[pos .. pos + 2^lvl - 1]
 *   Levels: 0 .. LOG-1 where LOG = ceil(log2(tourLen))
 *
 * All depths are stored as short (int16) — sufficient for any realistic tree
 * (max depth < 32767).
 */
public class EulerTourBuilder {

    /** Result of building Euler tour + sparse table for one tree. */
    public static final class TourData {
        /** Euler tour depths, length = tourLen. */
        public final short[] depths;
        /**
         * Sparse table: sparseMin[lvl][pos] = min depth in depths[pos..pos+2^lvl).
         * Dimensions: [LOG][tourLen].
         */
        public final short[][] sparseMin;
        /**
         * firstOcc[taxonId] = first position of taxon in Euler tour; -1 if absent.
         * Length = n (total taxa count).
         */
        public final int[] firstOcc;
        /** Actual Euler tour length (= 3*leafCount - 2). */
        public final int tourLen;
        /** Number of sparse-table levels (= ceil(log2(tourLen))). */
        public final int log;

        TourData(short[] depths, short[][] sparseMin, int[] firstOcc, int tourLen, int log) {
            this.depths    = depths;
            this.sparseMin = sparseMin;
            this.firstOcc  = firstOcc;
            this.tourLen   = tourLen;
            this.log       = log;
        }
    }

    /**
     * Build Euler tour and sparse table for one tree.
     *
     * @param tree  the gene tree
     * @param n     total taxon count (size of firstOcc array)
     */
    public static TourData build(Tree tree, int n) {
        int L = tree.leafCount;
        int tourLen = Math.max(1, 3 * L - 2);  // 3L-2 for L≥2; 1 for single-leaf degenerate

        short[] depths  = new short[tourLen];
        int[]   firstOcc = new int[n];
        java.util.Arrays.fill(firstOcc, -1);

        // DFS to fill depths and firstOcc
        int[] cursor = {0};
        buildDFS(tree.root, 0, depths, firstOcc, cursor);

        // tourLen is exactly cursor[0] after DFS
        int actualLen = cursor[0];

        // Build sparse table over depths[0..actualLen)
        int log = 1;
        while ((1 << log) < actualLen) log++;

        short[][] sparse = new short[log][actualLen];
        // Level 0: copy depths directly
        for (int i = 0; i < actualLen; i++) sparse[0][i] = depths[i];
        // Higher levels: sparse[lvl][i] = min(sparse[lvl-1][i], sparse[lvl-1][i + 2^(lvl-1)])
        for (int lvl = 1; lvl < log; lvl++) {
            int half = 1 << (lvl - 1);
            int end  = actualLen - (1 << lvl) + 1;
            for (int i = 0; i < end; i++) {
                sparse[lvl][i] = (short) Math.min(sparse[lvl-1][i], sparse[lvl-1][i + half]);
            }
            // Positions beyond end are never queried (range would exceed tour), leave as 0
        }

        return new TourData(depths, sparse, firstOcc, actualLen, log);
    }

    // ── DFS ─────────────────────────────────────────────────────────────────

    /**
     * Recursive DFS building the Euler tour:
     *   1. Append depth.
     *   2. If internal: recurse left; append depth; recurse right. (No append after right.)
     *   3. If leaf: record first occurrence in firstOcc.
     */
    private static void buildDFS(TreeNode node, int depth,
                                   short[] depths, int[] firstOcc, int[] cursor) {
        int pos = cursor[0]++;
        depths[pos] = (short) depth;

        if (node.isLeaf()) {
            firstOcc[node.taxonId] = pos;
        } else {
            buildDFS(node.left, depth + 1, depths, firstOcc, cursor);
            // Append parent depth on return from left
            int retPos = cursor[0]++;
            depths[retPos] = (short) depth;
            buildDFS(node.right, depth + 1, depths, firstOcc, cursor);
            // No append on return from right
        }
    }
}
