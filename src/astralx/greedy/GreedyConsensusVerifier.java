package astralx.greedy;

import astralx.cluster.ClusterTable;
import astralx.hash.PrefixHashArrays;
import astralx.taxon.TaxonRegistry;
import astralx.tree.Tree;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;

/**
 * Verifier for the greedy-consensus phase (Part I).
 *
 * Walks every unique bipartition in frequency-descending order, applying it
 * to both the fast {@link LaminarForest}+{@link LaminarBuilder} path and the
 * brute-force {@link LaminarOracle}.  Asserts:
 *
 *   1. Per-INSERT outcome agreement (ACCEPT / SKIP / REJECT).
 *   2. At every threshold boundary AND at the very end, the fast snapshot
 *      and the oracle's current state share the same canonical leaf-set
 *      representation — i.e. the two trees encode the same laminar family.
 *
 * Also dumps the 7 snapshot trees in Newick form to {@code outFile} (or stdout)
 * so they can be diffed against ASTRAL-MP's {@code allGreedies[i]} output.
 */
public final class GreedyConsensusVerifier {

    /**
     * @param geneTrees     gene trees only (no UPGMA), per the ASTRAL-MP-faithful
     *                      counting path.  Used for both bipartition counting
     *                      and exemplar taxa enumeration.
     */
    public static void dump(List<Tree> geneTrees, TaxonRegistry registry,
                            ClusterTable clusterTable, PrefixHashArrays pref,
                            String outFile) throws IOException {
        PrintStream out = (outFile != null)
            ? new PrintStream(new FileOutputStream(outFile)) : System.out;

        int n = registry.size();
        int k = geneTrees.size();
        out.printf("=== Phase GC (Greedy Consensus) Verification ===%n");
        out.printf("Taxa: %d  Gene trees: %d  (UPGMA excluded)%n", n, k);
        out.printf("Cluster-side entries in X (incl. UPGMA): %d%n%n", clusterTable.size());

        List<Bipartition> bps = BipartitionCounter.collectFromGeneTrees(
            geneTrees, pref, clusterTable.getAllTaxaHash(), n);
        out.printf("Unique bipartitions: %d%n", bps.size());

        // ── Lockstep run of fast path + oracle ──
        LaminarForest forest    = new LaminarForest(n);
        LaminarBuilder fast     = new LaminarBuilder(forest, geneTrees, n);
        LaminarOracle oracle    = new LaminarOracle(n, geneTrees);

        ConsensusTree[] fastSnaps   = new ConsensusTree[GreedyConsensus.THRESHOLDS.length];
        String[]        oracleSnaps = new String[GreedyConsensus.THRESHOLDS.length];

        int ti = GreedyConsensus.THRESHOLDS.length - 1;
        double threshold = GreedyConsensus.THRESHOLDS[ti];

        int outcomeMismatches = 0;
        int snapshotMismatches = 0;
        int processed = 0;

        for (Bipartition b : bps) {
            double ratio = (double) b.frequency / (double) k;
            while (ti >= 0 && threshold > ratio) {
                fastSnaps[ti]   = ConsensusTree.snapshot(forest);
                oracleSnaps[ti] = oracle.canonicalLeafSets();
                String fastSets = fastSnaps[ti].canonicalLeafSets();
                if (!fastSets.equals(oracleSnaps[ti])) {
                    snapshotMismatches++;
                    out.printf("FAIL snapshot ti=%d threshold=%.4f%n", ti, threshold);
                    printDiff(out, fastSets, oracleSnaps[ti]);
                }
                ti--;
                if (ti < 0) break;
                threshold = GreedyConsensus.THRESHOLDS[ti];
            }

            LaminarBuilder.Outcome fo = fast.insert(b);
            LaminarBuilder.Outcome oo = oracle.insert(b);
            if (fo != oo) {
                outcomeMismatches++;
                if (outcomeMismatches <= 10) {
                    out.printf("FAIL outcome  bp#%d  size=%d freq=%d  fast=%s oracle=%s%n",
                        processed, b.size, b.frequency, fo, oo);
                }
            }
            processed++;
        }

        // Drain remaining lower thresholds
        while (ti >= 0) {
            fastSnaps[ti]   = ConsensusTree.snapshot(forest);
            oracleSnaps[ti] = oracle.canonicalLeafSets();
            String fastSets = fastSnaps[ti].canonicalLeafSets();
            if (!fastSets.equals(oracleSnaps[ti])) {
                snapshotMismatches++;
                out.printf("FAIL snapshot ti=%d threshold=%.4f (drain)%n",
                    ti, GreedyConsensus.THRESHOLDS[ti]);
                printDiff(out, fastSets, oracleSnaps[ti]);
            }
            ti--;
        }

        // ── Per-threshold stats + Newick dump ──
        out.printf("%n--- Per-threshold snapshots ---%n");
        for (int i = 0; i < fastSnaps.length; i++) {
            ConsensusTree s = fastSnaps[i];
            out.printf("T[%d] threshold=%.4f  internal=%d  polytomies=%d%n",
                i, GreedyConsensus.THRESHOLDS[i],
                s.numInternalNodes(), s.numPolytomies());
        }

        out.printf("%n--- Newick (for ASTRAL-MP head-to-head) ---%n");
        for (int i = 0; i < fastSnaps.length; i++) {
            out.printf("T[%d]_threshold_%.4f: %s%n",
                i, GreedyConsensus.THRESHOLDS[i],
                fastSnaps[i].toNewick(registry));
        }

        // ── Summary ──
        out.printf("%n--- Summary ---%n");
        out.printf("Bipartitions processed: %d%n", processed);
        out.printf("INSERT outcome mismatches:  %d%n", outcomeMismatches);
        out.printf("Snapshot leaf-set mismatches: %d%n", snapshotMismatches);
        if (outcomeMismatches == 0 && snapshotMismatches == 0) {
            out.println("ALL ASSERTIONS PASSED (fast == oracle)");
        } else {
            out.println("FAILURES PRESENT — see above");
        }

        if (outFile != null) out.close();
    }

    /** Print up to a few diff lines so a mismatch isn't a wall of text. */
    private static void printDiff(PrintStream out, String fast, String oracle) {
        String[] fLines = fast.split("\n");
        String[] oLines = oracle.split("\n");
        java.util.Set<String> fSet = new java.util.HashSet<>(java.util.Arrays.asList(fLines));
        java.util.Set<String> oSet = new java.util.HashSet<>(java.util.Arrays.asList(oLines));

        int shown = 0;
        for (String s : fLines) {
            if (!oSet.contains(s) && shown < 10) { out.printf("  + (fast only)   %s%n", s); shown++; }
        }
        for (String s : oLines) {
            if (!fSet.contains(s) && shown < 20) { out.printf("  - (oracle only) %s%n", s); shown++; }
        }
    }

    private GreedyConsensusVerifier() {}
}
