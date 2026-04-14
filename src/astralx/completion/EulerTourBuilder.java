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
     * Extended tour data for similarity-matrix computation.
     *
     * Adds three payload-tracking sparse tables that allow the GPU kernel to
     * recover subLeafCount[c_a], subLeafCount[c_b], and S[u] for the LCA node
     * u = LCA(a,b) in a single RMQ overlap query — with no per-node ID arrays.
     *
     * At each Euler tour position pos:
     *   prevChildSubLC[pos] — subLeafCount of the child whose subtree JUST ended
     *                         before pos (i.e. the child to the left of this visit).
     *                         Zero for first-visit and leaf positions.
     *   nextChildSubLC[pos] — subLeafCount of the child whose subtree STARTS
     *                         right after pos (i.e. the child to the right of this visit).
     *                         Zero for leaf positions and the last intermediate visit.
     *   eulerS[pos]         — S[u] value for the node at this position:
     *                         S[u] = C2(kt − subLC[u]) + Σ_children C2(subLC[c])
     *                         Zero for leaf positions.
     *
     * Then for a pair (a,b) with l = min(fa,fb), r = max(fa,fb):
     *   leftmost argmin in [l,r]  → prevChildSubLC  = subLC of child containing
     *                                                  the LEFT leaf (min firstOcc)
     *   rightmost argmin in [l,r] → nextChildSubLC  = subLC of child containing
     *                                                  the RIGHT leaf (max firstOcc)
     *   leftmost argmin in [l,r]  → eulerS          = S[u]
     *
     * Sparse tables:
     *   sparseSubLCLeft[lvl][pos]  — left-biased argmin carries prevChildSubLC
     *   sparseSubLCRight[lvl][pos] — right-biased argmin carries nextChildSubLC
     *   sparseSLeft[lvl][pos]      — left-biased argmin carries eulerS (int32)
     */
    public static final class FullTourData extends TourData {
        /** prevChildSubLC at each Euler position (short, length = tourLen). */
        public final short[] prevChildSubLC;
        /** nextChildSubLC at each Euler position (short, length = tourLen). */
        public final short[] nextChildSubLC;
        /** S[u] at each Euler position (int32, length = tourLen). */
        public final int[] eulerS;
        /** Left-biased argmin carries prevChildSubLC. Dimensions [LOG][tourLen]. */
        public final short[][] sparseSubLCLeft;
        /** Right-biased argmin carries nextChildSubLC. Dimensions [LOG][tourLen]. */
        public final short[][] sparseSubLCRight;
        /** Left-biased argmin carries eulerS. Dimensions [LOG][tourLen]. */
        public final int[][] sparseSLeft;
        /** Leaf count of the tree (k_t), used to compute S[u] and C2(k_t−2). */
        public final int leafCount;

        FullTourData(TourData base,
                     short[] prevChildSubLC, short[] nextChildSubLC, int[] eulerS,
                     short[][] sparseSubLCLeft, short[][] sparseSubLCRight, int[][] sparseSLeft,
                     int leafCount) {
            super(base.depths, base.sparseMin, base.firstOcc, base.tourLen, base.log);
            this.prevChildSubLC  = prevChildSubLC;
            this.nextChildSubLC  = nextChildSubLC;
            this.eulerS          = eulerS;
            this.sparseSubLCLeft  = sparseSubLCLeft;
            this.sparseSubLCRight = sparseSubLCRight;
            this.sparseSLeft      = sparseSLeft;
            this.leafCount        = leafCount;
        }
    }

    // ── C2 helper ────────────────────────────────────────────────────────────────

    /** C2(x) = x*(x-1)/2. Returns 0 for x < 2. */
    static int c2(int x) { return (x < 2) ? 0 : x * (x - 1) / 2; }

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

    // ── Full build (similarity matrix) ──────────────────────────────────────────

    /**
     * Build the full tour data needed for GPU similarity-matrix computation.
     * Extends the base TourData with three payload-tracking sparse tables.
     *
     * @param tree  the gene tree (binary)
     * @param n     total taxon count (size of firstOcc / leafDepth arrays)
     */
    public static FullTourData buildFull(Tree tree, int n) {
        // ── Step 1: build the base tour (depths, sparseMin, firstOcc) ────────
        TourData base = build(tree, n);
        int kt       = tree.leafCount;
        int actualLen = base.tourLen;

        // ── Step 2: build per-position payload arrays via full DFS ───────────
        short[] prevSubLC = new short[actualLen];
        short[] nextSubLC = new short[actualLen];
        int[]   eulerS    = new int  [actualLen];

        int[] cursor2 = {0};
        buildFullDFS(tree.root, kt, prevSubLC, nextSubLC, eulerS, cursor2);

        // ── Step 3: build payload-tracking sparse tables ─────────────────────
        int log = base.log;
        short[][] sparseSubLCLeft  = new short[log][actualLen];
        short[][] sparseSubLCRight = new short[log][actualLen];
        int[][]   sparseSLeft      = new int  [log][actualLen];

        // Level 0: each position is its own argmin
        for (int i = 0; i < actualLen; i++) {
            sparseSubLCLeft [0][i] = prevSubLC[i];
            sparseSubLCRight[0][i] = nextSubLC[i];
            sparseSLeft     [0][i] = eulerS   [i];
        }

        // Higher levels: propagate following left-biased or right-biased argmin
        short[][] baseMin = base.sparseMin;
        for (int lvl = 1; lvl < log; lvl++) {
            int half = 1 << (lvl - 1);
            int end  = actualLen - (1 << lvl) + 1;
            for (int i = 0; i < end; i++) {
                short dL = baseMin[lvl - 1][i];
                short dR = baseMin[lvl - 1][i + half];

                // Left-biased: prefer left half on tie
                if (dL <= dR) {
                    sparseSubLCLeft[lvl][i] = sparseSubLCLeft [lvl - 1][i];
                    sparseSLeft    [lvl][i] = sparseSLeft     [lvl - 1][i];
                } else {
                    sparseSubLCLeft[lvl][i] = sparseSubLCLeft [lvl - 1][i + half];
                    sparseSLeft    [lvl][i] = sparseSLeft     [lvl - 1][i + half];
                }

                // Right-biased: prefer right half on tie
                if (dR <= dL) {
                    sparseSubLCRight[lvl][i] = sparseSubLCRight[lvl - 1][i + half];
                } else {
                    sparseSubLCRight[lvl][i] = sparseSubLCRight[lvl - 1][i];
                }
            }
        }

        return new FullTourData(base, prevSubLC, nextSubLC, eulerS,
                                sparseSubLCLeft, sparseSubLCRight, sparseSLeft, kt);
    }

    // ── DFS ─────────────────────────────────────────────────────────────────────

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

    /**
     * Extended DFS that records prevChildSubLC, nextChildSubLC, and eulerS at
     * each Euler tour position.
     *
     * For an internal node u with left child ca and right child cb:
     *   First-visit position p0:
     *     prevSubLC[p0] = 0   (no previous child yet)
     *     nextSubLC[p0] = subLC[ca]
     *     eulerS   [p0] = S[u]
     *   Intermediate position p1 (between ca and cb):
     *     prevSubLC[p1] = subLC[ca]
     *     nextSubLC[p1] = subLC[cb]
     *     eulerS   [p1] = S[u]
     * For a leaf:
     *     all three = 0
     *
     * Note: the first-visit position p0 of any internal node u is always
     * BEFORE firstOcc[any leaf in sub(u)], so it is never the argmin in any
     * [fa,fb] query where u = LCA(a,b).  Only the intermediate position p1
     * appears in such ranges — and it carries the correct payloads.
     */
    private static void buildFullDFS(TreeNode node, int kt,
                                      short[] prevSubLC, short[] nextSubLC, int[] eulerS,
                                      int[] cursor) {
        int pos = cursor[0];  // position already filled by the base buildDFS
        cursor[0]++;

        if (node.isLeaf()) {
            prevSubLC[pos] = 0;
            nextSubLC[pos] = 0;
            eulerS   [pos] = 0;
        } else {
            int subLC_ca = node.left.rangeEnd  - node.left.rangeStart;
            int subLC_cb = node.right.rangeEnd - node.right.rangeStart;
            int subLC_u  = node.rangeEnd - node.rangeStart;
            int s_u      = c2(kt - subLC_u) + c2(subLC_ca) + c2(subLC_cb);

            // First visit to u
            prevSubLC[pos] = 0;
            nextSubLC[pos] = (short) subLC_ca;
            eulerS   [pos] = s_u;

            buildFullDFS(node.left, kt, prevSubLC, nextSubLC, eulerS, cursor);

            // Intermediate position: return from left, before entering right
            int retPos = cursor[0]++;
            prevSubLC[retPos] = (short) subLC_ca;
            nextSubLC[retPos] = (short) subLC_cb;
            eulerS   [retPos] = s_u;

            buildFullDFS(node.right, kt, prevSubLC, nextSubLC, eulerS, cursor);
        }
    }
}
