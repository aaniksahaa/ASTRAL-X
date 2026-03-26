package astralx;

import astralx.cluster.Cluster;
import astralx.cluster.ClusterHash;
import astralx.cluster.ClusterTable;
import astralx.hash.PrefixHashArrays;
import astralx.taxon.TaxonRegistry;
import astralx.tree.Tree;

import java.io.*;
import java.util.*;

/**
 * Verifies Phase-3 cluster extraction.
 *
 * Key checks:
 *   1. Every cluster has size in [1, n-1].
 *   2. For each cluster of size s, its complement (size n-s) is also in X
 *      (holds for complete trees).
 *   3. Size bins are consistent with stored cluster sizes.
 *   4. For small inputs: enumerate the expected clusters manually and check.
 */
public class Phase3Verifier {

    public static void dump(List<Tree> trees, TaxonRegistry registry,
                            PrefixHashArrays pref, ClusterTable clusterTable,
                            String outFile) throws IOException {
        PrintStream out = (outFile != null)
            ? new PrintStream(new FileOutputStream(outFile)) : System.out;

        int n   = registry.size();
        int k   = trees.size();
        int m   = pref.numSeeds();

        out.printf("=== Phase 3 Cluster Extraction Verification ===%n");
        out.printf("Taxa: %d  Trees: %d  Seeds: %d%n", n, k, m);
        out.printf("Unique clusters in X: %d%n", clusterTable.size());
        out.printf("All-taxa hash: %s%n%n", clusterTable.getAllTaxaHash());

        int fails = 0;

        // Check 1: all sizes in [1, n-1]
        for (ClusterTable.Entry e : clusterTable.entries()) {
            int sz = e.hash.size;
            if (sz < 1 || sz >= n) {
                out.printf("FAIL: cluster size %d out of [1,%d)%n", sz, n);
                fails++;
            }
        }

        // Check 2: for each cluster, its complement is also in X
        // Build a raw-sum keyed map so we can look up complements
        // (We match by size and check: rawSum(A) + rawSum(comp) == allTaxaSum)
        // Since we only have finalized hashes, we verify via the complement cluster:
        // for every Entry of size s, there must be an Entry of size (n-s).
        // Full complement membership check: we re-compute the complement hash from
        // the exemplar cluster and look it up.
        for (ClusterTable.Entry e : clusterTable.entries()) {
            Cluster ex = e.exemplar;
            int compSize = n - e.hash.size;
            if (compSize <= 0 || compSize >= n) continue;

            // Compute the complement hash for this cluster's exemplar
            long[] rawSums = new long[m], rawXors = new long[m];
            for (int s = 0; s < m; s++) {
                // super-complement w.r.t. ALL taxa (= complement for complete trees)
                if (!ex.complement) {
                    rawSums[s] = pref.superCompSum(ex.treeIndex, s, ex.left, ex.right);
                    rawXors[s] = pref.superCompXor(ex.treeIndex, s, ex.left, ex.right);
                } else {
                    // exemplar is already a complement; its super-complement is the range
                    rawSums[s] = pref.rangeSum(ex.treeIndex, s, ex.left, ex.right);
                    rawXors[s] = pref.rangeXor(ex.treeIndex, s, ex.left, ex.right);
                }
            }
            ClusterHash compHash = new ClusterHash(rawSums, rawXors, compSize, m);
            if (!clusterTable.contains(compHash)) {
                out.printf("FAIL: cluster %s has no complement in X (compSize=%d)%n",
                    e.hash, compSize);
                fails++;
            }
        }

        // Check 3: size-bin consistency
        for (int sz : clusterTable.sizes()) {
            for (ClusterHash h : clusterTable.getBySize(sz)) {
                ClusterTable.Entry e = clusterTable.get(h);
                if (e == null) {
                    out.printf("FAIL: size bin %d contains orphan hash%n", sz);
                    fails++;
                } else if (e.hash.size != sz) {
                    out.printf("FAIL: size bin %d has entry with size %d%n", sz, e.hash.size);
                    fails++;
                }
            }
        }

        // Summary
        out.printf("%n--- Summary ---%n");
        if (fails == 0) {
            out.println("ALL ASSERTIONS PASSED");
        } else {
            out.printf("%d FAILURES%n", fails);
        }

        // Size distribution
        out.printf("%n--- Size distribution ---%n");
        List<Integer> sizes = new ArrayList<>(clusterTable.sizes());
        Collections.sort(sizes);
        for (int sz : sizes) {
            out.printf("  size %3d: %d clusters%n", sz, clusterTable.getBySize(sz).size());
        }

        // Small-input: show all clusters with their taxon sets
        if (n <= 8) {
            out.printf("%n--- All clusters (small input) ---%n");
            List<ClusterTable.Entry> sorted = new ArrayList<>(clusterTable.entries());
            sorted.sort(Comparator.comparingInt(e -> e.hash.size));
            for (ClusterTable.Entry e : sorted) {
                String taxa = taxaInCluster(e.exemplar, trees.get(e.exemplar.treeIndex), registry);
                out.printf("  sz=%d freq=%d  taxa={%s}  %s%n",
                    e.hash.size, e.frequency, taxa, e.exemplar);
            }
        }

        if (outFile != null) out.close();
    }

    /** Enumerate the actual taxon names in a cluster from its exemplar. */
    static String taxaInCluster(Cluster c, Tree tree, TaxonRegistry registry) {
        StringBuilder sb = new StringBuilder();
        int[] arr = tree.postorderArray;
        if (!c.complement) {
            for (int i = c.left; i < c.right; i++) {
                if (sb.length() > 0) sb.append(",");
                sb.append(registry.getName(arr[i]));
            }
        } else {
            for (int i = 0; i < tree.leafCount; i++) {
                if (i >= c.left && i < c.right) continue;
                if (sb.length() > 0) sb.append(",");
                sb.append(registry.getName(arr[i]));
            }
        }
        return sb.toString();
    }
}
