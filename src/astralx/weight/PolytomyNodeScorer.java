package astralx.weight;

import astralx.tree.Tree;
import astralx.tree.TreeNode;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Exact CPU scoring of unresolved (polytomous) species-tree nodes.
 *
 * <p>The precomputed {@link WeightTable} scores binary tripartitions
 * (A | B | rest) only.  A species-tree node with d >= 3 rooted children induces
 * the d-partition (or (d+1)-partition, with the complement) P = (P_1 | ... | P_D).
 * The gene-tree quartets it resolves are exactly those with two taxa in one part
 * and one taxon in each of two other parts; such a quartet is counted once here
 * and once more at the node where the other pair diverges, so summing this
 * contribution with the binary-node weights keeps the whole-tree invariant
 * "objective total = 2 x quartet score" that the reported score relies on.
 * This mirrors ASTRAL's handling (sum of the weights of every tripartition
 * formed by three distinct parts of the node), evaluated directly.
 *
 * <p>For a gene-tree node with partition M = (M_1 | ... | M_E) (children plus the
 * complement within that gene tree) and the intersection matrix
 * n[i][p] = |P_i ∩ M_p|, the number of agreeing quartets is
 * <pre>
 *   sum over (i,p) of  C(n[i][p], 2) * U(i,p)
 *   U(i,p) = number of unordered pairs of cells {(j,q),(k,r)} with
 *            j != k, q != r, all of j,k != i and q,r != p, weighted n[j][q]*n[k][r]
 * </pre>
 * U is obtained in O(1) per cell from row/column moments, so a gene node costs
 * O(D * E).  Per species polytomy node the whole pass is O(D * total gene-tree
 * size).  Binary species trees never reach this class.
 */
public final class PolytomyNodeScorer {

    private PolytomyNodeScorer() {}

    /** Total contribution of all unresolved nodes, and how many there were. */
    public static final class Result {
        public final BigInteger total;
        public final int polytomyNodes;
        Result(BigInteger total, int polytomyNodes) {
            this.total = total;
            this.polytomyNodes = polytomyNodes;
        }
    }

    /**
     * Sum, over every polytomous node of {@code species}, of the number of
     * gene-tree quartets resolved at that node that agree with the species tree.
     *
     * @param species   complete species tree over {@code numTaxa} taxa
     * @param geneTrees native gene trees (binary or polytomous, complete or not)
     * @param numTaxa   taxon universe size (species tree must contain all of them)
     */
    public static Result score(Tree species, List<Tree> geneTrees, int numTaxa) {
        if (species.leafCount != numTaxa) {
            throw new IllegalArgumentException("Species tree has " + species.leafCount
                + " leaves but the taxon universe has " + numTaxa);
        }
        List<TreeNode> polyNodes = new ArrayList<>();
        collectPolytomies(species.root, polyNodes);
        BigInteger total = BigInteger.ZERO;
        for (TreeNode u : polyNodes) {
            total = total.add(scoreNode(species, u, geneTrees, numTaxa));
        }
        return new Result(total, polyNodes.size());
    }

    private static void collectPolytomies(TreeNode u, List<TreeNode> out) {
        if (u.isLeaf()) return;
        if (u.isPolytomous()) {
            out.add(u);
            for (TreeNode c : u.children) collectPolytomies(c, out);
        } else {
            collectPolytomies(u.left, out);
            collectPolytomies(u.right, out);
        }
    }

    /** Contribution of one polytomous species node across all gene trees. */
    static BigInteger scoreNode(Tree species, TreeNode u, List<Tree> geneTrees,
                                int numTaxa) {
        TreeNode[] kids = u.children;
        int d = kids.length;
        boolean hasComplement = u.rangeSize() < species.leafCount;
        int numParts = hasComplement ? d + 1 : d;

        // partOf[taxon] = index of the species part containing it (complement = d).
        int[] partOf = new int[numTaxa];
        Arrays.fill(partOf, hasComplement ? d : -1);
        for (int c = 0; c < d; c++) {
            for (int pos = kids[c].rangeStart; pos < kids[c].rangeEnd; pos++) {
                partOf[species.postorderArray[pos]] = c;
            }
        }
        for (int t = 0; t < numTaxa; t++) {
            if (partOf[t] < 0) {
                throw new IllegalStateException("Species tree root does not cover taxon " + t);
            }
        }

        Accumulator acc = new Accumulator();
        for (Tree g : geneTrees) {
            int[] totalInTree = new int[numParts];
            for (int pos = 0; pos < g.leafCount; pos++) {
                totalInTree[partOf[g.postorderArray[pos]]]++;
            }
            walk(g.root, partOf, numParts, totalInTree, acc);
        }
        return acc.value();
    }

    /**
     * Post-order walk of a gene tree.  Returns the per-species-part leaf counts
     * of the subtree under {@code v}; adds the contribution of {@code v}'s own
     * partition to {@code acc}.
     */
    private static int[] walk(TreeNode v, int[] partOf, int numParts,
                              int[] totalInTree, Accumulator acc) {
        int[] cnt = new int[numParts];
        if (v.isLeaf()) {
            cnt[partOf[v.taxonId]]++;
            return cnt;
        }
        TreeNode[] kids = v.isPolytomous() ? v.children : new TreeNode[] {v.left, v.right};
        int e = kids.length;
        int[][] childCnt = new int[e][];
        for (int q = 0; q < e; q++) {
            childCnt[q] = walk(kids[q], partOf, numParts, totalInTree, acc);
            for (int i = 0; i < numParts; i++) cnt[i] += childCnt[q][i];
        }
        int[] comp = new int[numParts];
        int compSize = 0;
        for (int i = 0; i < numParts; i++) {
            comp[i] = totalInTree[i] - cnt[i];
            compSize += comp[i];
        }
        int numGeneParts = compSize > 0 ? e + 1 : e;
        if (numGeneParts >= 3 && numParts >= 3) {
            // n[i][p]: species part i x gene part p
            long[][] n = new long[numParts][numGeneParts];
            for (int q = 0; q < e; q++) {
                for (int i = 0; i < numParts; i++) n[i][q] = childCnt[q][i];
            }
            if (compSize > 0) {
                for (int i = 0; i < numParts; i++) n[i][e] = comp[i];
            }
            countAgreeing(n, numParts, numGeneParts, acc);
        }
        return cnt;
    }

    /**
     * Number of quartets with two taxa in the same cell (i,p) and one taxon in
     * each of two further cells lying in distinct other rows and distinct other
     * columns.  Exact; O(D*E).
     */
    static void countAgreeing(long[][] n, int D, int E, Accumulator acc) {
        long[] R = new long[D], RS = new long[D];   // row sums, row sums of squares
        long[] C = new long[E], CS = new long[E];   // column sums, column sums of squares
        long G = 0, Q = 0;
        for (int i = 0; i < D; i++) {
            for (int p = 0; p < E; p++) {
                long v = n[i][p];
                R[i] += v; C[p] += v; G += v;
                RS[i] += v * v; CS[p] += v * v; Q += v * v;
            }
        }
        // A[p] = sum_j (R_j - n_jp)^2 ;  B[i] = sum_q (C_q - n_iq)^2
        long[] A = new long[E], B = new long[D];
        for (int p = 0; p < E; p++) {
            long s = 0;
            for (int j = 0; j < D; j++) { long x = R[j] - n[j][p]; s += x * x; }
            A[p] = s;
        }
        for (int i = 0; i < D; i++) {
            long s = 0;
            for (int q = 0; q < E; q++) { long x = C[q] - n[i][q]; s += x * x; }
            B[i] = s;
        }
        for (int i = 0; i < D; i++) {
            for (int p = 0; p < E; p++) {
                long v = n[i][p];
                if (v < 2) continue;
                long S    = G - R[i] - C[p] + v;                    // sum over j!=i, q!=p
                long rows = A[p] - (R[i] - v) * (R[i] - v);          // sum_{j!=i} (sum_{q!=p} n_jq)^2
                long cols = B[i] - (C[p] - v) * (C[p] - v);          // sum_{q!=p} (sum_{j!=i} n_jq)^2
                long sq   = Q - RS[i] - CS[p] + v * v;               // sum_{j!=i,q!=p} n_jq^2
                long T    = S * S - rows - cols + sq;                // ordered cell pairs
                if (T < 0 || (T & 1L) != 0) {
                    throw new IllegalStateException("Polytomy pair count " + T
                        + " is not a non-negative even integer");
                }
                long U = T >> 1;
                long c2 = v * (v - 1) / 2;
                acc.addProduct(c2, U);
            }
        }
    }

    /** Overflow-safe non-negative accumulator (long fast path, BigInteger spill). */
    static final class Accumulator {
        private long small = 0;
        private BigInteger big = BigInteger.ZERO;

        void add(long v) {
            try {
                small = Math.addExact(small, v);
            } catch (ArithmeticException overflow) {
                big = big.add(BigInteger.valueOf(small));
                small = v;
            }
        }

        void addProduct(long a, long b) {
            if (a == 0 || b == 0) return;
            if (a <= Long.MAX_VALUE / b) {
                add(a * b);
            } else {
                big = big.add(BigInteger.valueOf(a).multiply(BigInteger.valueOf(b)));
            }
        }

        BigInteger value() { return big.add(BigInteger.valueOf(small)); }
    }
}
