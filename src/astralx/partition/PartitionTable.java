package astralx.partition;

import astralx.Logging;
import astralx.cluster.ClusterHash;
import astralx.util.ProgressBar;
import astralx.hash.PrefixHashArrays;
import astralx.tree.Tree;
import astralx.tree.TreeNode;

import java.util.*;

/**
 * Table of unique gene-tree tripartitions with their frequencies.
 *
 * For each non-root internal node u of each gene tree g we extract:
 *   part1 = sub(left(u))    range [L.start, L.end)   -- left subtree
 *   part2 = sub(right(u))   range [R.start, R.end)   -- right subtree
 *   part3 = Lg \ sub(u)     complement of [u.start, u.end) w.r.t. Lg
 *
 * The root node's two children define a bipartition with part3 = empty set,
 * which contributes 0 weight in ASTRAL -- so we skip the root.
 *
 * Deduplication: PartitionHash is order-invariant over (part1, part2).
 */
public class PartitionTable {

    public static final class Entry {
        public final PartitionHash hash;
        public final Partition     exemplar;
        public int                 frequency;

        Entry(PartitionHash h, Partition p) { this.hash = h; this.exemplar = p; this.frequency = 1; }
    }

    private final Map<PartitionHash, Entry> table = new HashMap<>();
    private final int m;

    // -------------------------------------------------------------------------

    public PartitionTable(List<Tree> trees, PrefixHashArrays pref) {
        long t0 = System.nanoTime();
        this.m = pref.numSeeds();

        int totalCandidates = 0;
        int treesDone = 0;
        ProgressBar bar = new ProgressBar("Tripartition extraction", trees.size());
        for (Tree tree : trees) {
            totalCandidates += extractFromTree(tree, pref);
            bar.update(++treesDone);
        }
        bar.done();

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Partition extraction: %d candidates -> %d unique tripartitions in %d ms",
            totalCandidates, table.size(), ms);
    }

    private int extractFromTree(Tree tree, PrefixHashArrays pref) {
        int ti = tree.treeIndex;
        int L  = tree.leafCount;
        int[] count = {0};
        extractNode(tree.root, ti, L, pref, count);
        return count[0];
    }

    /**
     * Recurse post-order. For each non-root internal node u, register the
     * tripartition (left | right | complement_of_parent_range).
     */
    private void extractNode(TreeNode node, int ti, int L,
                              PrefixHashArrays pref, int[] count) {
        if (node.isLeaf()) return;
        extractNode(node.left,  ti, L, pref, count);
        extractNode(node.right, ti, L, pref, count);

        if (node.isRoot()) return;  // root gives bipartition with empty part3 -- skip

        int lStart = node.left.rangeStart,  lEnd = node.left.rangeEnd;
        int rStart = node.right.rangeStart, rEnd = node.right.rangeEnd;
        // part3 is the complement of node's full range [node.rangeStart, node.rangeEnd)
        int pStart = node.rangeStart, pEnd = node.rangeEnd;

        int sz1 = lEnd - lStart;
        int sz2 = rEnd - rStart;
        int sz3 = L - (pEnd - pStart);   // complement size

        // sz3 == 0 can only happen when node is root, which we already skip
        if (sz3 == 0) return;

        // Build ClusterHash for each part
        ClusterHash h1 = buildHash(ti, lStart, lEnd, false, sz1, pref);
        ClusterHash h2 = buildHash(ti, rStart, rEnd, false, sz2, pref);
        ClusterHash h3 = buildHash(ti, pStart, pEnd, true,  sz3, pref); // complement

        PartitionHash ph = new PartitionHash(h1, h2, h3);

        Entry existing = table.get(ph);
        if (existing != null) {
            existing.frequency++;
        } else {
            Partition p = new Partition(h1, h2, h3, sz1, sz2, sz3,
                                        ti, lStart, lEnd, rStart, rEnd);
            table.put(ph, new Entry(ph, p));
        }
        count[0]++;
    }

    /** Compute a ClusterHash for a subtree range or its complement. */
    private ClusterHash buildHash(int ti, int lo, int hi, boolean complement,
                                   int size, PrefixHashArrays pref) {
        long[] rawSums = new long[m], rawXors = new long[m];
        for (int s = 0; s < m; s++) {
            rawSums[s] = complement ? pref.compSum(ti, s, lo, hi)
                                    : pref.rangeSum(ti, s, lo, hi);
            rawXors[s] = complement ? pref.compXor(ti, s, lo, hi)
                                    : pref.rangeXor(ti, s, lo, hi);
        }
        return new ClusterHash(rawSums, rawXors, size, m);
    }

    // -------------------------------------------------------------------------

    public Entry get(PartitionHash ph) { return table.get(ph); }
    public int size()                  { return table.size(); }
    public Collection<Entry> entries() { return table.values(); }
}
