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
 * Sparse table (depths / min):
 *   sparseMin[lvl][pos] = minimum depth in eulerDepths[pos .. pos + 2^lvl - 1]
 *   Levels: 0 .. LOG-1 where LOG = ceil(log2(tourLen))
 *
 * Sparse table (SubLC payload / left-biased argmin):
 *   sparseSubLC[lvl][pos] = sub[LCA] in depths[pos .. pos + 2^lvl - 1]
 *   where sub[v] = number of leaves in subtree(v).
 *   "Left-biased": when two equal-depth positions tie, we keep the LEFT one.
 *   This ensures the same LCA is always selected as the min-depth RMQ.
 *
 * All depths and sub-leaf-counts stored as short (int16) — sufficient for
 * any realistic tree (max depth < 32767, max leaves < 32767).
 */
public class EulerTourBuilder {

    /** Result of building Euler tour + sparse table for one tree. */
    public static class TourData {
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
     * Extended tour data for GPU similarity-matrix computation.
     *
     * Per-position sub-leaf-count payload:
     *   eulerSubLC[pos] = number of leaves in subtree of the node at this
     *                     Euler tour position (1 for leaves, sub[v] for internals).
     *
     * Payload sparse table (left-biased argmin of sparseMin):
     *   sparseSubLC[lvl][pos] = sub[LCA] for the range [pos, pos+2^lvl)
     *   where LCA is identified by the leftmost minimum depth in that range.
     *
     * GPU query for pair (a,b):
     *   l = min(firstOcc[a], firstOcc[b])
     *   r = max(firstOcc[a], firstOcc[b])
     *   k_lvl = floor(log2(r−l+1)),  l2 = r − 2^k_lvl + 1
     *   dL = sparseMin[k_lvl][l],  dR = sparseMin[k_lvl][l2]
     *   sub_lca = (dL <= dR) ? sparseSubLC[k_lvl][l] : sparseSubLC[k_lvl][l2]
     *   num_T(a,b) = C2(kt − sub_lca)   (= 0 when sub_lca ≥ kt−1)
     *   den_T(a,b) = C2(kt − 2)
     */
    public static final class FullTourData extends TourData {
        /** Sub-leaf-count at each Euler position (int16, length = tourLen). */
        public final short[] eulerSubLC;
        /** Left-biased argmin carries sub[LCA]. Dimensions [LOG][tourLen]. */
        public final short[][] sparseSubLC;
        /** Leaf count of the tree (k_t). */
        public final int leafCount;

        FullTourData(TourData base,
                     short[] eulerSubLC, short[][] sparseSubLC,
                     int leafCount) {
            super(base.depths, base.sparseMin, base.firstOcc, base.tourLen, base.log);
            this.eulerSubLC  = eulerSubLC;
            this.sparseSubLC = sparseSubLC;
            this.leafCount   = leafCount;
        }
    }

    // ── C2 helper ────────────────────────────────────────────────────────────────

    /** C2(x) = x*(x-1)/2. Returns 0 for x < 2. */
    static long c2(long x) { return (x < 2) ? 0L : x * (x - 1) / 2; }

    /** C2(x) = x*(x-1)/2. Returns 0 for x < 2. (int overload for convenience) */
    static long c2(int x) { return (x < 2) ? 0L : (long)x * (x - 1) / 2; }

    // ── Lite build (distance matrix) ─────────────────────────────────────────────

    /**
     * Build Euler tour and sparse table for one tree.
     *
     * @param tree  the gene tree
     * @param n     total taxon count (size of firstOcc array)
     */
    public static TourData build(Tree tree, int n) {
        int L = tree.leafCount;
        int tourLen = Math.max(1, 3 * L - 2);  // 3L-2 for L≥2; 1 for single-leaf degenerate

        short[] depths   = new short[tourLen];
        int[]   firstOcc = new int[n];
        java.util.Arrays.fill(firstOcc, -1);

        // DFS to fill depths and firstOcc
        int[] cursor = {0};
        buildDFS(tree.root, 0, depths, firstOcc, cursor);

        int actualLen = cursor[0];

        // Build sparse table over depths[0..actualLen)
        int log = 1;
        while ((1 << log) < actualLen) log++;

        short[][] sparse = new short[log][actualLen];
        for (int i = 0; i < actualLen; i++) sparse[0][i] = depths[i];
        for (int lvl = 1; lvl < log; lvl++) {
            int half = 1 << (lvl - 1);
            int end  = actualLen - (1 << lvl) + 1;
            for (int i = 0; i < end; i++) {
                sparse[lvl][i] = (short) Math.min(sparse[lvl-1][i], sparse[lvl-1][i + half]);
            }
        }

        return new TourData(depths, sparse, firstOcc, actualLen, log);
    }

    // ── Full build (similarity matrix, SubLC payload) ─────────────────────────

    /**
     * Build the full tour data needed for similarity-matrix computation.
     *
     * Computes per-position sub-leaf-count (eulerSubLC) via a DFS, then
     * builds a left-biased payload sparse table (sparseSubLC) that carries
     * sub[LCA] for O(1) queries.
     *
     * Query: num_T(a,b) = C2(kt − sub_lca)
     * where sub_lca = sparseSubLC at the leftmost-minimum position in
     * [min(firstOcc[a],firstOcc[b]), max(firstOcc[a],firstOcc[b])].
     *
     * @param tree  the gene tree (binary, as parsed by ASTRAL-X)
     * @param n     total taxon count
     */
    public static FullTourData buildFull(Tree tree, int n) {
        // ── Step 1: base tour (depths, sparseMin, firstOcc) ─────────────────
        TourData base = build(tree, n);
        int kt        = tree.leafCount;
        int actualLen = base.tourLen;

        // ── Step 2: sub-leaf-count at each Euler position ────────────────────
        short[] eulerSubLC = new short[actualLen];

        int[] cursor2 = {0};
        buildSubLCDFS(tree.root, eulerSubLC, cursor2);

        // ── Step 3: left-biased payload sparse table (sparseSubLC) ───────────
        int log = base.log;
        short[][] sparseSubLC = new short[log][actualLen];

        for (int i = 0; i < actualLen; i++) {
            sparseSubLC[0][i] = eulerSubLC[i];
        }

        short[][] baseMin = base.sparseMin;
        for (int lvl = 1; lvl < log; lvl++) {
            int half = 1 << (lvl - 1);
            int end  = actualLen - (1 << lvl) + 1;
            for (int i = 0; i < end; i++) {
                short dL = baseMin[lvl - 1][i];
                short dR = baseMin[lvl - 1][i + half];
                // Left-biased: prefer left half on tie (same as sparseMin behavior)
                sparseSubLC[lvl][i] = (dL <= dR)
                        ? sparseSubLC[lvl - 1][i]
                        : sparseSubLC[lvl - 1][i + half];
            }
        }

        return new FullTourData(base, eulerSubLC, sparseSubLC, kt);
    }

    // ── DFS helpers ──────────────────────────────────────────────────────────────

    /**
     * Recursive DFS building the Euler tour (depths + firstOcc).
     */
    private static void buildDFS(TreeNode node, int depth,
                                  short[] depths, int[] firstOcc, int[] cursor) {
        int pos = cursor[0]++;
        depths[pos] = (short) depth;

        if (node.isLeaf()) {
            firstOcc[node.taxonId] = pos;
        } else {
            buildDFS(node.left, depth + 1, depths, firstOcc, cursor);
            int retPos = cursor[0]++;
            depths[retPos] = (short) depth;
            buildDFS(node.right, depth + 1, depths, firstOcc, cursor);
        }
    }

    /**
     * DFS building eulerSubLC: the sub-leaf-count of the node at each
     * Euler tour position.
     *
     * For a leaf:    sub = 1
     * For internal v: sub = v.rangeEnd − v.rangeStart
     */
    private static void buildSubLCDFS(TreeNode node, short[] eulerSubLC, int[] cursor) {
        int pos = cursor[0]++;

        if (node.isLeaf()) {
            eulerSubLC[pos] = 1;
        } else {
            int sub = node.rangeEnd - node.rangeStart;
            eulerSubLC[pos] = (short) sub;

            buildSubLCDFS(node.left, eulerSubLC, cursor);

            // Return from left child: record this node's sub again
            int retPos = cursor[0]++;
            eulerSubLC[retPos] = (short) sub;

            buildSubLCDFS(node.right, eulerSubLC, cursor);
        }
    }
}
