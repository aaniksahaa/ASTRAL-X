package astralx.dp;

import astralx.Logging;
import astralx.cluster.ClusterHash;
import astralx.hash.PrefixHashArrays;
import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.Int128;
import astralx.weight.PolytomyNodeScorer;
import astralx.weight.WeightTable;

import java.math.BigInteger;
import java.util.List;

/**
 * Quartet score of a fixed (supplied or inferred) species tree.
 *
 * <p>The score is the number of gene-tree quartets whose topology agrees with
 * the species tree, exactly as ASTRAL reports it.  It is obtained by walking
 * the species tree itself rather than by running the DP over its transition
 * table, so unresolved (polytomous) species-tree nodes are scored too:
 * <ul>
 *   <li>a binary internal node contributes the precomputed weight of its
 *       tripartition (left | right | rest), looked up in the {@link WeightTable};</li>
 *   <li>a polytomous node contributes its exact d-partition weight from
 *       {@link PolytomyNodeScorer};</li>
 *   <li>the root (empty complement) contributes nothing.</li>
 * </ul>
 * Every agreeing quartet is counted at exactly two nodes, so the walk total is
 * twice the quartet score; the halving happens once, in
 * {@link Inference#quartetScoreFromDpTotal}.
 *
 * <p>For a binary species tree the walk total is also compared with the DP
 * total over the same table (they must agree exactly in the integer modes), so
 * the previously verified DP path keeps guarding the result.
 */
public final class FixedTreeScorer {

    private FixedTreeScorer() {}

    public static String score(DPTable dpTable, WeightTable weightTable,
                               Tree species, PrefixHashArrays speciesPref,
                               List<Tree> geneTrees, int numTaxa, String logLabel) {
        long t0 = System.nanoTime();
        int m = speciesPref.numSeeds();
        Walk walk = new Walk(weightTable, speciesPref, species.treeIndex, m, numTaxa);
        walk.visit(species.root);

        BigInteger poly = BigInteger.ZERO;
        int polyNodes = 0;
        if (species.hasPolytomy) {
            PolytomyNodeScorer.Result r = PolytomyNodeScorer.score(species, geneTrees, numTaxa);
            poly = r.total;
            polyNodes = r.polytomyNodes;
        }

        String score;
        String totalText;
        if (weightTable.isInt128()) {
            Int128 total = walk.totalI.add(int128Of(poly));
            score = Inference.quartetScoreFromDpTotal(total);
            totalText = total + "  [int128]";
        } else if (weightTable.isDouble()) {
            double total = walk.totalD + poly.doubleValue();
            score = Inference.quartetScoreFromDpTotal(total);
            totalText = String.format(java.util.Locale.ROOT, "%.0f  [double]", total);
        } else {
            long total = Math.addExact(walk.totalL, poly.longValueExact());
            score = Inference.quartetScoreFromDpTotal(total);
            totalText = total + "  [long]";
        }
        long ms = (System.nanoTime() - t0) / 1_000_000;
        if (polyNodes > 0) {
            Logging.info("%s: %d unresolved species-tree node(s) scored exactly "
                + "(contribution %s to the objective total)", logLabel, polyNodes, poly);
        }
        Logging.info("%s: objective total = %s  (= 2 x quartet score); quartet score = %s  "
            + "(%d binary node(s), %d unresolved; %d ms)",
            logLabel, totalText, score, walk.binaryNodes, polyNodes, ms);

        if (!species.hasPolytomy) {
            crossCheckAgainstDp(dpTable, weightTable, score, logLabel);
        }
        return score;
    }

    /** The DP over the fixed tree's own transitions must reproduce the walk. */
    private static void crossCheckAgainstDp(DPTable dpTable, WeightTable weightTable,
                                            String walkScore, String logLabel) {
        String dpScore = new Inference().dpQuartetScore(dpTable, weightTable);
        if (dpScore.equals(walkScore)) return;
        if (weightTable.isDouble()) {
            // Floating-point summation order differs between the two paths.
            double a = Double.parseDouble(walkScore), b = Double.parseDouble(dpScore);
            double rel = Math.abs(a - b) / Math.max(1.0, Math.max(Math.abs(a), Math.abs(b)));
            if (rel <= 1e-9) return;
            Logging.warn("%s: double-mode walk score %s and DP score %s differ "
                + "(relative %.3g)", logLabel, walkScore, dpScore, rel);
            return;
        }
        throw new IllegalStateException(logLabel + ": fixed-tree walk score " + walkScore
            + " disagrees with the DP score " + dpScore
            + " over the same transition table");
    }

    private static Int128 int128Of(BigInteger v) {
        if (v.signum() < 0 || v.bitLength() > 126) {
            throw new ArithmeticException("Polytomy contribution out of Int128 range: " + v);
        }
        long hi = v.shiftRight(64).longValueExact();
        long lo = v.longValue();          // low 64 bits (two's complement truncation)
        return new Int128(hi, lo);
    }

    /** Post-order accumulation of binary-node weights. */
    private static final class Walk {
        private final WeightTable wt;
        private final PrefixHashArrays pref;
        private final int ti, m, n;
        long totalL = 0;
        double totalD = 0.0;
        Int128 totalI = Int128.ZERO;
        int binaryNodes = 0;

        Walk(WeightTable wt, PrefixHashArrays pref, int ti, int m, int n) {
            this.wt = wt; this.pref = pref; this.ti = ti; this.m = m; this.n = n;
        }

        void visit(TreeNode u) {
            if (u.isLeaf()) return;
            if (u.isPolytomous()) {
                for (TreeNode c : u.children) visit(c);
                return;               // scored by PolytomyNodeScorer
            }
            visit(u.left);
            visit(u.right);
            if (u.isRoot()) return;   // (left | right | empty) carries no quartets
            binaryNodes++;
            BipartitionSplit split = findSplit(u);
            if (wt.isInt128())      totalI = totalI.add(wt.getScoreI(split));
            else if (wt.isDouble()) totalD += wt.getScoreD(split);
            else                    totalL = Math.addExact(totalL, wt.getScore(split));
        }

        /**
         * The tripartition {left, right, rest} of u is stored under one of three
         * equivalent split keys depending on how the DP table was built (plain
         * rooted split, or an unrooted rotation in anchored-root mode).  Any
         * present key carries the same weight.
         */
        private BipartitionSplit findSplit(TreeNode u) {
            ClusterHash hL = hash(u.left, false);
            ClusterHash hR = hash(u.right, false);
            BipartitionSplit s = new BipartitionSplit(hL, hR);
            if (wt.hasScore(s)) return s;
            ClusterHash hC = hash(u, true);
            s = new BipartitionSplit(hR, hC);
            if (wt.hasScore(s)) return s;
            s = new BipartitionSplit(hL, hC);
            if (wt.hasScore(s)) return s;
            throw new IllegalStateException("Weight table holds no score for the "
                + "species-tree node covering " + u.rangeSize() + " taxa (range ["
                + u.rangeStart + "," + u.rangeEnd + "))");
        }

        private ClusterHash hash(TreeNode node, boolean complement) {
            int lo = node.rangeStart, hi = node.rangeEnd;
            long[] sums = new long[m], xors = new long[m];
            for (int s = 0; s < m; s++) {
                sums[s] = complement ? pref.superCompSum(ti, s, lo, hi) : pref.rangeSum(ti, s, lo, hi);
                xors[s] = complement ? pref.superCompXor(ti, s, lo, hi) : pref.rangeXor(ti, s, lo, hi);
            }
            int sz = complement ? (n - (hi - lo)) : (hi - lo);
            return new ClusterHash(sums, xors, sz, m);
        }
    }
}
