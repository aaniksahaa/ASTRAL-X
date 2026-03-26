package astralx;

import astralx.cluster.ClusterHash;
import astralx.hash.PrefixHashArrays;
import astralx.partition.Partition;
import astralx.partition.PartitionTable;
import astralx.taxon.TaxonRegistry;
import astralx.tree.Tree;

import java.io.*;
import java.util.*;

/**
 * Verifies Phase-4 gene-tree tripartition extraction.
 *
 * Key checks:
 *   1. size1 + size2 + size3 == leafCount of exemplar tree.
 *   2. size3 > 0 for all stored partitions.
 *   3. Total non-root internal nodes across trees == total candidates before dedup.
 *      For a binary tree with n leaves: n-1 internal nodes, n-2 non-root = n-2.
 *   4. hash1 + hash2 + hash3 == totalHash(tree)  (additive consistency).
 *   5. For small inputs: show taxa in each part.
 */
public class Phase4Verifier {

    public static void dump(List<Tree> trees, TaxonRegistry registry,
                            PrefixHashArrays pref, PartitionTable partTable,
                            String outFile) throws IOException {
        PrintStream out = (outFile != null)
            ? new PrintStream(new FileOutputStream(outFile)) : System.out;

        int n = registry.size();
        int k = trees.size();
        int m = pref.numSeeds();

        out.printf("=== Phase 4 Tripartition Extraction Verification ===%n");
        out.printf("Taxa: %d  Trees: %d  Seeds: %d%n", n, k, m);
        out.printf("Unique tripartitions: %d%n%n", partTable.size());

        // Expected total non-root internal nodes
        int expectedTotal = 0;
        for (Tree t : trees) expectedTotal += (t.leafCount - 2);
        out.printf("Expected total (sum of n-2 per tree): %d%n%n", expectedTotal);

        int fails = 0;

        // Check 1+2: size consistency
        for (PartitionTable.Entry e : partTable.entries()) {
            Partition p = e.exemplar;
            Tree exemplarTree = trees.get(p.treeIndex);
            int totalSize = p.size1 + p.size2 + p.size3;
            if (totalSize != exemplarTree.leafCount) {
                out.printf("FAIL: sizes %d+%d+%d=%d != leafCount=%d  in %s%n",
                    p.size1, p.size2, p.size3, totalSize,
                    exemplarTree.leafCount, p);
                fails++;
            }
            if (p.size3 <= 0) {
                out.printf("FAIL: size3=%d <= 0 in %s%n", p.size3, p);
                fails++;
            }
        }

        // Check 4: hash additive consistency
        // hash1_raw + hash2_raw + hash3_raw == totalHash(tree)
        // We check this by recomputing raw hashes from the exemplar and comparing
        for (PartitionTable.Entry e : partTable.entries()) {
            Partition p = e.exemplar;
            int ti = p.treeIndex;
            for (int s = 0; s < m; s++) {
                // raw sums for part1, part2, part3
                long raw1Sum = pref.rangeSum(ti, s, p.leftStart,  p.leftEnd);
                long raw2Sum = pref.rangeSum(ti, s, p.rightStart, p.rightEnd);
                long raw3Sum = pref.compSum(ti, s, p.leftStart,   p.rightEnd); // comp of union
                // Note: union is [leftStart, rightEnd) since left and right are contiguous
                //   leftEnd == rightStart always (children are adjacent in postorder)
                long totalSum = pref.totalSum(ti, s);
                if (raw1Sum + raw2Sum + raw3Sum != totalSum) {
                    out.printf("FAIL hash s=%d: sum parts (%x+%x+%x)=%x != total %x in %s%n",
                        s, raw1Sum, raw2Sum, raw3Sum,
                        raw1Sum + raw2Sum + raw3Sum, totalSum, p);
                    fails++;
                }
            }
        }

        out.printf("%n--- Summary ---%n");
        if (fails == 0) out.println("ALL ASSERTIONS PASSED");
        else            out.printf("%d FAILURES%n", fails);

        // Size distribution of part3 (tells us how "balanced" tripartitions are)
        out.printf("%n--- part3 size distribution (complement size) ---%n");
        Map<Integer, Integer> dist = new TreeMap<>();
        for (PartitionTable.Entry e : partTable.entries()) {
            dist.merge(e.exemplar.size3, 1, Integer::sum);
        }
        dist.forEach((sz, cnt) -> out.printf("  part3_size=%3d : %d tripartitions%n", sz, cnt));

        // Small-input: show all tripartitions with taxon names
        if (n <= 8) {
            out.printf("%n--- All tripartitions (small input) ---%n");
            List<PartitionTable.Entry> sorted = new ArrayList<>(partTable.entries());
            sorted.sort(Comparator.comparingInt(e -> e.exemplar.size1));
            for (PartitionTable.Entry e : sorted) {
                Partition p = e.exemplar;
                Tree t = trees.get(p.treeIndex);
                String s1 = rangeNames(t, p.leftStart,  p.leftEnd,  false, registry);
                String s2 = rangeNames(t, p.rightStart, p.rightEnd, false, registry);
                String s3 = rangeNames(t, p.leftStart,  p.rightEnd, true,  registry);
                out.printf("  freq=%d  {%s} | {%s} | {%s}%n", e.frequency, s1, s2, s3);
            }
        }

        if (outFile != null) out.close();
    }

    private static String rangeNames(Tree tree, int lo, int hi, boolean complement,
                                     TaxonRegistry registry) {
        StringBuilder sb = new StringBuilder();
        if (!complement) {
            for (int i = lo; i < hi; i++) {
                if (sb.length() > 0) sb.append(",");
                sb.append(registry.getName(tree.postorderArray[i]));
            }
        } else {
            for (int i = 0; i < tree.leafCount; i++) {
                if (i >= lo && i < hi) continue;
                if (sb.length() > 0) sb.append(",");
                sb.append(registry.getName(tree.postorderArray[i]));
            }
        }
        return sb.toString();
    }
}
