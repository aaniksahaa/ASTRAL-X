package astralx.greedy;

import astralx.Logging;
import astralx.cluster.ClusterTable;
import astralx.hash.PrefixHashArrays;
import astralx.tree.Tree;

import java.util.List;

/**
 * Part I driver: single-pass laminar build with 7-threshold snapshotting.
 *
 * Replaces the legacy "build 7 trees in parallel from 7 prefixes of the
 * frequency-sorted list" with a single incremental pass that snapshots the
 * partial laminar forest each time the current bipartition's frequency
 * drops below the next threshold boundary.  The set of clusters contained
 * in T[ti] is exactly the prefix with ratio ≥ THRESHOLDS[ti] — bit-identical
 * to ASTRAL-MP's allGreedies[ti].
 */
public final class GreedyConsensus {

    /** Ascending threshold list — matches ASTRAL-MP's GREEDY_ADDITION_THRESHOLDS. */
    public static final double[] THRESHOLDS =
        { 0.0, 1.0/100, 1.0/50, 1.0/20, 1.0/10, 1.0/5, 1.0/3 };

    /** Result bundle. */
    public static final class Result {
        public final ConsensusTree[] snapshots;   // length == THRESHOLDS.length
        public final int numBipartitions;         // unique bipartitions consumed
        public final int numAccepted;             // INSERT outcomes
        public final int numSkippedTrivial;
        public final int numSkippedRedundant;
        public final int numRejectedCrossCut;

        Result(ConsensusTree[] snapshots,
               int numBipartitions, int numAccepted,
               int numSkippedTrivial, int numSkippedRedundant,
               int numRejectedCrossCut) {
            this.snapshots           = snapshots;
            this.numBipartitions     = numBipartitions;
            this.numAccepted         = numAccepted;
            this.numSkippedTrivial   = numSkippedTrivial;
            this.numSkippedRedundant = numSkippedRedundant;
            this.numRejectedCrossCut = numRejectedCrossCut;
        }
    }

    /**
     * Run greedy consensus.
     *
     * @param clusterTable  source of the all-taxa cluster signature (used for
     *                      computing complement hashes during canonicalization)
     * @param geneTrees     gene trees ONLY (no UPGMA guide tree).  Used both to
     *                      count bipartition frequencies (excluding any guide
     *                      tree contribution) AND for the laminar builder's
     *                      taxa-enumeration step (since exemplar Cluster
     *                      objects index into this list via {@link Tree#treeIndex}).
     * @param pref          prefix hash arrays covering {@code geneTrees}
     * @param numTaxa       n
     */
    public static Result build(ClusterTable clusterTable, List<Tree> geneTrees,
                                PrefixHashArrays pref, int numTaxa) {
        long t0 = System.nanoTime();
        int k = geneTrees.size();
        if (k <= 0) {
            throw new IllegalStateException("greedyConsensus: gene-tree list is empty");
        }

        // ── Phase A: walk gene trees → per-bipartition counts, sort desc ──
        List<Bipartition> bps = BipartitionCounter.collectFromGeneTrees(
            geneTrees, pref, clusterTable.getAllTaxaHash(), numTaxa);
        Logging.debug("GreedyConsensus: %d unique bipartitions (from %d gene trees)",
            bps.size(), k);

        // ── Phase B: single-pass laminar build + snapshot at each threshold ──
        LaminarForest forest = new LaminarForest(numTaxa);
        LaminarBuilder builder = new LaminarBuilder(forest, geneTrees, numTaxa);

        ConsensusTree[] snapshots = new ConsensusTree[THRESHOLDS.length];
        int ti = THRESHOLDS.length - 1;        // start at highest (1/3)
        double currentThreshold = THRESHOLDS[ti];

        int accepted = 0, skippedTriv = 0, skippedRed = 0, rejected = 0;

        for (Bipartition b : bps) {
            double ratio = (double) b.frequency / (double) k;

            // Snapshot all thresholds we are about to drop below.
            while (ti >= 0 && currentThreshold > ratio) {
                snapshots[ti] = ConsensusTree.snapshot(forest);
                ti--;
                if (ti < 0) break;
                currentThreshold = THRESHOLDS[ti];
            }
            // We still INSERT this bipartition — it contributes to the next
            // lower-threshold snapshot.

            LaminarBuilder.Outcome o = builder.insert(b);
            switch (o) {
                case ACCEPT            -> accepted++;
                case SKIP_TRIVIAL      -> skippedTriv++;
                case SKIP_REDUNDANT    -> skippedRed++;
                case REJECT_CROSS_CUT  -> rejected++;
            }
        }

        // Drain remaining lower thresholds — all clusters consumed by now.
        while (ti >= 0) {
            snapshots[ti] = ConsensusTree.snapshot(forest);
            ti--;
        }

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Greedy consensus: %d bps, INSERT: %d accept / %d redundant / %d cross-cut / %d trivial (%d ms)",
            bps.size(), accepted, skippedRed, rejected, skippedTriv, ms);
        if (Logging.isDebug()) {
            for (int i = 0; i < snapshots.length; i++) {
                ConsensusTree s = snapshots[i];
                Logging.debug("  T[%d] threshold=%.4f  internal=%d  polytomies=%d",
                    i, THRESHOLDS[i], s.numInternalNodes(), s.numPolytomies());
            }
        }

        return new Result(snapshots, bps.size(),
            accepted, skippedTriv, skippedRed, rejected);
    }

    private GreedyConsensus() {}
}
