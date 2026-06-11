package astralx.completion;

import astralx.tree.Tree;
import astralx.tree.TreeNode;

/**
 * Builds the Euler tour and sparse-table RMQ structure for one gene tree.
 *
 * Euler tour definition (DFS, recording depths) for a binary tree:
 *   - Append depth(node) when first entering any node (ENTER position)
 *   - Append depth(node) when returning from its LEFT child (INTERMEDIATE)
 *   - Do NOT append when returning from the right child
 *
 * Tour length for L leaves = 3L − 2.
 *
 * LCA property (binary trees):
 *   For two leaves a, b with first occurrences fa, fb in the tour:
 *     depth(LCA(a, b)) = min(eulerDepths[min(fa,fb) .. max(fa,fb)])
 *   The leftmost min-depth position is the *INTERMEDIATE* visit of LCA(a,b)
 *   (the position between the two subtree tours of the LCA).
 *
 * Sparse table:
 *   sparseMin[lvl][pos] = min depth in depths[pos .. pos + 2^lvl − 1]
 *
 * ────────────────────────────────────────────────────────────────────────────
 * NEW (similarity-matrix bridge formula): per-position payloads (s, F) of the
 * LCA's children, populated at INTERMEDIATE visits.
 *
 * For each node v in the gene tree:
 *   s(v) = number of descendant leaves
 *   f(v) = (s(v) + n − s(parent(v)) − 2) · (s(parent(v)) − s(v))
 *   F(v) = F(parent(v)) + f(v),    F(root) = 0
 *
 * (n here = kt = number of leaves in this gene tree.) F(v) is the running
 * "path-summand" used by the validated O(1) QD formula:
 *   QD(x,y) = ½·[ (F(x) − F(cx)) + (F(y) − F(cy)) + (cxS−1)·Z + (cyS−1)·Z ]
 * where cx, cy are the children of LCA(x,y) on the x and y sides.
 *
 * At each Euler position p the per-position arrays hold:
 *   eulerF[p]              = F(node at p)         (used at LEAF positions
 *                                                  to recover F(x), F(y))
 *   eulerLeftChildS[p]     = s(node.left)         INTERMEDIATE positions only
 *   eulerLeftChildF[p]     = F(node.left)         INTERMEDIATE positions only
 *   eulerRightChildS[p]    = s(node.right)        INTERMEDIATE positions only
 *   eulerRightChildF[p]    = F(node.right)        INTERMEDIATE positions only
 *
 * Sparse tables for the child-of-LCA payloads use the SAME left-biased argmin
 * as sparseMin (left index wins on tie). The argmin selects the INTERMEDIATE
 * visit of the LCA, where the child payloads are meaningful.
 *
 * HINT for n-ary extension: with arbitrary-degree internal nodes, an LCA u
 * has multiple intermediate positions (one between each pair of consecutive
 * children). The leftmost min-depth in [l,r] is between the child holding the
 * leftmost leaf (smaller firstOcc) and the next child; the rightmost min-depth
 * is between the prior child and the child holding the rightmost leaf.
 * Then one would need BOTH a left-biased and a right-biased sparse table.
 * The binary path above coincidentally requires only one because there is
 * exactly one intermediate position per LCA query.
 */
public class EulerTourBuilder {

    /** Result of building Euler tour + sparse table for one tree. */
    public static class TourData {
        public final short[]   depths;
        public final short[][] sparseMin;
        public final int[]     firstOcc;
        public final int       tourLen;
        public final int       log;

        TourData(short[] depths, short[][] sparseMin, int[] firstOcc, int tourLen, int log) {
            this.depths    = depths;
            this.sparseMin = sparseMin;
            this.firstOcc  = firstOcc;
            this.tourLen   = tourLen;
            this.log       = log;
        }
    }

    /**
     * Extended tour data for GPU similarity-matrix computation, carrying the
     * (s, F) child-of-LCA payloads needed by the bridge-formula kernel.
     */
    public static final class FullTourData extends TourData {
        public final double[]   eulerF;
        public final short[]    eulerLeftChildS;
        public final double[]   eulerLeftChildF;
        public final short[]    eulerRightChildS;
        public final double[]   eulerRightChildF;

        public final short[][]  sparseLeftChildS;
        public final double[][] sparseLeftChildF;
        public final short[][]  sparseRightChildS;
        public final double[][] sparseRightChildF;

        public final int leafCount;

        FullTourData(TourData base,
                     double[] eulerF,
                     short[]  eulerLeftChildS,  double[] eulerLeftChildF,
                     short[]  eulerRightChildS, double[] eulerRightChildF,
                     short[][]  sparseLeftChildS,  double[][] sparseLeftChildF,
                     short[][]  sparseRightChildS, double[][] sparseRightChildF,
                     int leafCount) {
            super(base.depths, base.sparseMin, base.firstOcc, base.tourLen, base.log);
            this.eulerF             = eulerF;
            this.eulerLeftChildS    = eulerLeftChildS;
            this.eulerLeftChildF    = eulerLeftChildF;
            this.eulerRightChildS   = eulerRightChildS;
            this.eulerRightChildF   = eulerRightChildF;
            this.sparseLeftChildS   = sparseLeftChildS;
            this.sparseLeftChildF   = sparseLeftChildF;
            this.sparseRightChildS  = sparseRightChildS;
            this.sparseRightChildF  = sparseRightChildF;
            this.leafCount          = leafCount;
        }
    }

    // ── C2 helper ────────────────────────────────────────────────────────────

    /** C2(x) = x*(x-1)/2. Returns 0 for x < 2. */
    static long c2(long x) { return (x < 2) ? 0L : x * (x - 1) / 2; }

    /** C2(x) overload for int. */
    static long c2(int x) { return (x < 2) ? 0L : (long)x * (x - 1) / 2; }

    // ── Lite build (distance matrix) ─────────────────────────────────────────

    public static TourData build(Tree tree, int n) {
        int L = tree.leafCount;
        int tourLen = Math.max(1, 3 * L - 2);

        short[] depths   = new short[tourLen];
        int[]   firstOcc = new int[n];
        java.util.Arrays.fill(firstOcc, -1);

        int[] cursor = {0};
        buildDFS(tree.root, 0, depths, firstOcc, cursor);

        int actualLen = cursor[0];

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

    // ── Full build (similarity matrix, bridge-formula payloads) ──────────────

    /**
     * Build the full tour data for the bridge-formula similarity kernel.
     *
     * Three DFS passes (each O(n)):
     *   1. base tour (depths, firstOcc, sparseMin)        — via build()
     *   2. compute s(v), F(v) for every node              — once, bottom-up + top-down
     *   3. emit per-position payloads via Euler-tour DFS
     */
    public static FullTourData buildFull(Tree tree, int n) {
        TourData base = build(tree, n);
        int kt        = tree.leafCount;
        int len       = base.tourLen;
        int log       = base.log;

        // ── Step A: compute s(v) bottom-up, F(v) top-down ────────────────────
        // For a degenerate single-leaf tree (kt == 1) there is exactly one
        // node (a leaf) with s=1, F=0; the rest of the build still runs cleanly.
        java.util.IdentityHashMap<TreeNode, double[]> sf = new java.util.IdentityHashMap<>();
        computeS(tree.root, sf);
        computeF(tree.root, kt, /*parentS*/ -1, /*parentF*/ 0.0, sf);

        // ── Step B: Euler-tour DFS emitting per-position payloads ────────────
        double[] eulerF            = new double[len];
        short[]  eulerLeftChildS   = new short [len];
        double[] eulerLeftChildF   = new double[len];
        short[]  eulerRightChildS  = new short [len];
        double[] eulerRightChildF  = new double[len];

        int[] cursor = {0};
        emitPayloads(tree.root, sf, eulerF,
                     eulerLeftChildS, eulerLeftChildF,
                     eulerRightChildS, eulerRightChildF,
                     cursor);

        // ── Step C: build payload sparse tables (left-biased argmin) ─────────
        short[][]  sparseLeftChildS   = new short [log][len];
        double[][] sparseLeftChildF   = new double[log][len];
        short[][]  sparseRightChildS  = new short [log][len];
        double[][] sparseRightChildF  = new double[log][len];

        for (int i = 0; i < len; i++) {
            sparseLeftChildS [0][i] = eulerLeftChildS [i];
            sparseLeftChildF [0][i] = eulerLeftChildF [i];
            sparseRightChildS[0][i] = eulerRightChildS[i];
            sparseRightChildF[0][i] = eulerRightChildF[i];
        }

        short[][] baseMin = base.sparseMin;
        for (int lvl = 1; lvl < log; lvl++) {
            int half = 1 << (lvl - 1);
            int end  = len - (1 << lvl) + 1;
            for (int i = 0; i < end; i++) {
                short dL = baseMin[lvl - 1][i];
                short dR = baseMin[lvl - 1][i + half];
                boolean pickLeft = (dL <= dR);   // left-biased on ties
                int srcIdx = pickLeft ? i : (i + half);
                sparseLeftChildS [lvl][i] = sparseLeftChildS [lvl - 1][srcIdx];
                sparseLeftChildF [lvl][i] = sparseLeftChildF [lvl - 1][srcIdx];
                sparseRightChildS[lvl][i] = sparseRightChildS[lvl - 1][srcIdx];
                sparseRightChildF[lvl][i] = sparseRightChildF[lvl - 1][srcIdx];
            }
        }

        return new FullTourData(
            base, eulerF,
            eulerLeftChildS, eulerLeftChildF,
            eulerRightChildS, eulerRightChildF,
            sparseLeftChildS, sparseLeftChildF,
            sparseRightChildS, sparseRightChildF,
            kt);
    }

    // ── DFS helpers ──────────────────────────────────────────────────────────

    /**
     * Recursive DFS building the Euler tour (depths + firstOcc).
     */
    private static void buildDFS(TreeNode node, int depth,
                                  short[] depths, int[] firstOcc, int[] cursor) {
        int pos = cursor[0]++;
        depths[pos] = (short) depth;

        if (node.isLeaf()) {
            firstOcc[node.taxonId] = pos;
        } else if (node.isPolytomous()) {
            // n-ary: enter, recurse child[0], then INTERMEDIATE + recurse child[i] for i≥1.
            // The intermediate (depth of this node) between consecutive children preserves
            // the RMQ-LCA property for any degree (polytomy-design.md / §HINT above).
            TreeNode[] ch = node.children;
            buildDFS(ch[0], depth + 1, depths, firstOcc, cursor);
            for (int i = 1; i < ch.length; i++) {
                int retPos = cursor[0]++;
                depths[retPos] = (short) depth;
                buildDFS(ch[i], depth + 1, depths, firstOcc, cursor);
            }
        } else {
            buildDFS(node.left, depth + 1, depths, firstOcc, cursor);
            int retPos = cursor[0]++;
            depths[retPos] = (short) depth;
            buildDFS(node.right, depth + 1, depths, firstOcc, cursor);
        }
    }

    /**
     * Pass 1: compute s(v) bottom-up. Stores [s, 0.0] in the map; F is filled
     * later by computeF.
     */
    private static int computeS(TreeNode node,
                                 java.util.IdentityHashMap<TreeNode, double[]> sf) {
        int s;
        if (node.isLeaf()) {
            s = 1;
        } else if (node.isPolytomous()) {
            s = 0;
            for (TreeNode c : node.children) s += computeS(c, sf);
        } else {
            s = computeS(node.left, sf) + computeS(node.right, sf);
        }
        sf.put(node, new double[] { (double) s, 0.0 });
        return s;
    }

    /**
     * Pass 2: top-down F propagation. parentS = −1 marks the root call.
     *   F(root) = 0;  F(v) = F(parent) + f(v),
     *   f(v) = (s(v) + kt − s(parent) − 2) · (s(parent) − s(v))
     */
    private static void computeF(TreeNode node, int kt,
                                  int parentS, double parentF,
                                  java.util.IdentityHashMap<TreeNode, double[]> sf) {
        double[] entry = sf.get(node);
        int s = (int) entry[0];
        double F;
        if (parentS < 0) {
            F = 0.0;                         // root
        } else {
            double fVal = ((double)s + kt - parentS - 2) * (parentS - s);
            F = parentF + fVal;
        }
        entry[1] = F;
        if (!node.isLeaf()) {
            if (node.isPolytomous()) {
                for (TreeNode c : node.children) computeF(c, kt, s, F, sf);
            } else {
                computeF(node.left,  kt, s, F, sf);
                computeF(node.right, kt, s, F, sf);
            }
        }
    }

    /**
     * Euler-tour DFS that emits per-position payloads.
     *
     * At each position p we record eulerF[p] = F(node at p). At INTERMEDIATE
     * positions of an internal node v we also record (s,F) of v.left and
     * v.right; for ENTER positions and leaf positions these payloads are 0
     * (they are never selected by the RMQ in a valid leaf-pair query).
     *
     * HINT for n-ary extension: the INTERMEDIATE between v's child c_i and
     * c_{i+1} would carry (leftChildS,F) = (s(c_i), F(c_i)) and
     * (rightChildS,F) = (s(c_{i+1}), F(c_{i+1})). With one intermediate per
     * (consecutive-child-pair), a left-biased argmin selects between the
     * "x-side neighbor" and a right-biased argmin selects between the
     * "y-side neighbor". For binary trees these coincide.
     */
    private static void emitPayloads(TreeNode node,
                                      java.util.IdentityHashMap<TreeNode, double[]> sf,
                                      double[] eulerF,
                                      short[]  eulerLeftChildS,  double[] eulerLeftChildF,
                                      short[]  eulerRightChildS, double[] eulerRightChildF,
                                      int[] cursor) {
        int pos = cursor[0]++;
        double[] entry = sf.get(node);
        eulerF[pos] = entry[1];

        if (node.isLeaf()) {
            // Payloads at leaf positions are placeholders (never selected).
            return;
        }

        if (node.isPolytomous()) {
            // n-ary: enter, recurse child[0]; for each i≥1, an INTERMEDIATE between
            // child[i-1] and child[i] carries their (s,F), then recurse child[i].
            // (The similarity bridge query is left-biased, so for n-ary nodes it
            // selects one intermediate — an approximation, never a crash.)
            TreeNode[] ch = node.children;
            emitPayloads(ch[0], sf, eulerF,
                         eulerLeftChildS, eulerLeftChildF,
                         eulerRightChildS, eulerRightChildF, cursor);
            for (int i = 1; i < ch.length; i++) {
                int retPos = cursor[0]++;
                eulerF[retPos] = entry[1];
                double[] Lc = sf.get(ch[i - 1]);
                double[] Rc = sf.get(ch[i]);
                eulerLeftChildS [retPos] = (short) (int) Lc[0];
                eulerLeftChildF [retPos] = Lc[1];
                eulerRightChildS[retPos] = (short) (int) Rc[0];
                eulerRightChildF[retPos] = Rc[1];
                emitPayloads(ch[i], sf, eulerF,
                             eulerLeftChildS, eulerLeftChildF,
                             eulerRightChildS, eulerRightChildF, cursor);
            }
            return;
        }

        // Recurse left subtree
        emitPayloads(node.left,  sf, eulerF,
                     eulerLeftChildS, eulerLeftChildF,
                     eulerRightChildS, eulerRightChildF, cursor);

        // INTERMEDIATE position for `node`
        int retPos = cursor[0]++;
        eulerF[retPos] = entry[1];
        double[] L = sf.get(node.left);
        double[] R = sf.get(node.right);
        eulerLeftChildS [retPos] = (short) (int) L[0];
        eulerLeftChildF [retPos] = L[1];
        eulerRightChildS[retPos] = (short) (int) R[0];
        eulerRightChildF[retPos] = R[1];

        // Recurse right subtree
        emitPayloads(node.right, sf, eulerF,
                     eulerLeftChildS, eulerLeftChildF,
                     eulerRightChildS, eulerRightChildF, cursor);
    }
}
