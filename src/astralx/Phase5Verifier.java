package astralx;

import astralx.cluster.Cluster;
import astralx.cluster.ClusterHash;
import astralx.cluster.ClusterTable;
import astralx.dp.BipartitionSplit;
import astralx.dp.DPTable;
import astralx.hash.PrefixHashArrays;
import astralx.taxon.TaxonRegistry;
import astralx.tree.Tree;

import java.io.*;
import java.util.*;

/**
 * Verifies Phase-5 DP search space (tree-local transitions).
 *
 * Checks:
 *   1. Root hash has at least one split.
 *   2. For every split (A → B | C): size(B) + size(C) == size(A).
 *   3. For every split, both halves are in ClusterTable (or one of them is a
 *      root-child subtree cluster, which is always valid).
 *   4. Expected transition counts match theory (Type1 count = # internal nodes
 *      incl. root; Type2 count = # non-root nodes whose parent is not root).
 *   5. Small input: print all splits with taxon names.
 */
public class Phase5Verifier {

    public static void dump(List<Tree> trees, TaxonRegistry registry,
                            PrefixHashArrays pref,
                            ClusterTable clusterTable, DPTable dpTable,
                            String outFile) throws IOException {
        PrintStream out = (outFile != null)
            ? new PrintStream(new FileOutputStream(outFile)) : System.out;

        int n = registry.size();
        int k = trees.size();

        out.printf("=== Phase 5 DP Search Space Verification ===%n");
        out.printf("Taxa: %d  Trees: %d  Clusters in X: %d%n", n, k, clusterTable.size());
        out.printf("Clusters with splits: %d%n", dpTable.numClusters());
        out.printf("Unique splits total:  %d%n", dpTable.numUniqueSplits());
        out.printf("Transitions emitted:  %d (before dedup)%n%n", dpTable.numEmitted());

        int fails = 0;

        // ── Check 1: Root has splits ─────────────────────────────────────────
        ClusterHash root = dpTable.getRootHash();
        if (!dpTable.hasSplits(root)) {
            out.println("FAIL: root cluster has no splits");
            fails++;
        } else {
            out.printf("Root splits: %d%n", dpTable.getSplits(root).size());
        }

        // ── Check 2: Size consistency for every split ────────────────────────
        int sizeFailCount = 0;
        for (var entry : dpTable.entries()) {
            ClusterHash parent = entry.getKey();
            for (BipartitionSplit split : entry.getValue()) {
                int expected = parent.size;
                int actual   = split.lo.size + split.hi.size;
                if (actual != expected) {
                    out.printf("FAIL size: parent.size=%d but lo.size=%d + hi.size=%d = %d  in %s%n",
                        expected, split.lo.size, split.hi.size, actual, parent);
                    sizeFailCount++;
                    fails++;
                }
            }
        }
        if (sizeFailCount == 0) out.println("Check 2 (size consistency): PASSED");
        else                    out.printf("Check 2 (size consistency): %d FAILURES%n", sizeFailCount);

        // ── Check 3: Both halves in ClusterTable (warn on misses) ────────────
        // Halves should always be in X; if not, something is wrong with extraction.
        int membershipFails = 0;
        for (var entry : dpTable.entries()) {
            for (BipartitionSplit split : entry.getValue()) {
                boolean loInX = clusterTable.contains(split.lo);
                boolean hiInX = clusterTable.contains(split.hi);
                if (!loInX || !hiInX) {
                    out.printf("FAIL membership: lo=%s(%s) hi=%s(%s)%n",
                        split.lo, loInX ? "OK" : "MISSING",
                        split.hi, hiInX ? "OK" : "MISSING");
                    membershipFails++;
                    fails++;
                }
            }
        }
        if (membershipFails == 0) out.println("Check 3 (cluster membership): PASSED");
        else out.printf("Check 3 (cluster membership): %d FAILURES%n", membershipFails);

        // ── Check 4: Expected transition counts (tree-structural) ────────────
        // Type 1 per tree: # internal nodes (incl. root) = leafCount - 1
        // Type 2 per tree: # non-root internal nodes with non-root parent
        //   For a balanced 5-leaf tree: root has 2 children (each internal),
        //   those have leaf children -- Type 2 applies only at depth > 1.
        int expType1 = 0;
        int expType2 = 0;
        for (Tree t : trees) {
            int internal = t.leafCount - 1;      // internal nodes incl. root
            expType1 += internal;
            // Count Type 2 eligible nodes via recursion
            expType2 += countType2(t.root);
        }
        out.printf("%nExpected Type1 transitions (incl. root): %d%n", expType1);
        out.printf("Expected Type2 transitions: %d%n", expType2);
        out.printf("Total emitted (Type1+Type2): %d  (expected %d)%n",
            dpTable.numEmitted(), expType1 + expType2);
        if (dpTable.numEmitted() != expType1 + expType2) {
            out.println("FAIL: emitted count mismatch");
            fails++;
        } else {
            out.println("Check 4 (transition count): PASSED");
        }

        // ── Summary ──────────────────────────────────────────────────────────
        out.printf("%n--- Summary ---%n");
        if (fails == 0) out.println("ALL ASSERTIONS PASSED");
        else            out.printf("%d FAILURES%n", fails);

        // ── Split count distribution ─────────────────────────────────────────
        out.printf("%n--- Splits-per-cluster distribution ---%n");
        Map<Integer, Integer> dist = new TreeMap<>();
        for (var entry : dpTable.entries()) {
            dist.merge(entry.getValue().size(), 1, Integer::sum);
        }
        dist.forEach((cnt, num) -> out.printf("  splits=%2d : %d clusters%n", cnt, num));

        // ── Small input: print all splits with taxa names ─────────────────────
        if (n <= 8) {
            out.printf("%n--- All DP transitions (small input) ---%n");
            // Sort by parent size then parent hash
            var allEntries = new ArrayList<>(dpTable.entries());
            allEntries.sort(Comparator.comparingInt(e -> e.getKey().size));

            for (var entry : allEntries) {
                ClusterHash parent = entry.getKey();
                String parentName = clusterName(parent, clusterTable, trees, registry);
                for (BipartitionSplit split : entry.getValue()) {
                    String loName = clusterName(split.lo, clusterTable, trees, registry);
                    String hiName = clusterName(split.hi, clusterTable, trees, registry);
                    out.printf("  {%s} -> {%s} | {%s}%n", parentName, loName, hiName);
                }
            }
        }

        if (outFile != null) out.close();
    }

    // -------------------------------------------------------------------------

    /** Count Type 2 eligible nodes under subtree rooted at u. */
    private static int countType2(astralx.tree.TreeNode u) {
        if (u.isLeaf()) return 0;
        int count = 0;
        // u is non-root AND parent is non-root → eligible
        if (!u.isRoot() && !u.parent.isRoot()) count = 1;
        return count + countType2(u.left) + countType2(u.right);
    }

    /**
     * Return a comma-separated taxon name string for the given cluster hash.
     * Looks up the exemplar in ClusterTable; falls back to "(root)" for the
     * all-taxa cluster.
     */
    private static String clusterName(ClusterHash hash, ClusterTable ct,
                                       List<Tree> trees, TaxonRegistry registry) {
        ClusterTable.Entry entry = ct.get(hash);
        if (entry == null) return "(root)";
        Cluster ex = entry.exemplar;
        Tree t = trees.get(ex.treeIndex);
        StringBuilder sb = new StringBuilder();
        if (!ex.complement) {
            for (int i = ex.left; i < ex.right; i++) {
                if (sb.length() > 0) sb.append(',');
                sb.append(registry.getName(t.postorderArray[i]));
            }
        } else {
            for (int i = 0; i < t.leafCount; i++) {
                if (i >= ex.left && i < ex.right) continue;
                if (sb.length() > 0) sb.append(',');
                sb.append(registry.getName(t.postorderArray[i]));
            }
        }
        return sb.toString();
    }
}
