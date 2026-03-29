package astralx.cluster;

import astralx.Logging;
import astralx.hash.PrefixHashArrays;
import astralx.util.ProgressBar;
import astralx.tree.Tree;
import astralx.tree.TreeNode;

import java.util.*;

/**
 * The cluster set X: all unique clusters extracted from gene trees.
 *
 * For each gene tree (treated as unrooted), we walk every internal node u
 * (excluding root) and register two clusters:
 *   1. sub(u)      -- the subtree range [u.left, u.right)
 *   2. Lg \ sub(u) -- complement w.r.t. that tree's taxa (if size > 0)
 *
 * Also registers the all-taxa cluster (DP root) separately.
 * Singleton clusters (size 1) are included -- they are DP base cases.
 * Empty clusters (size 0) are excluded.
 *
 * Deduplication is done by ClusterHash. One exemplar Cluster is kept per unique hash.
 * The table also maintains size-binned lists for DP space construction.
 */
public class ClusterTable {

    /** Entry in the cluster hash table. */
    public static final class Entry {
        public final ClusterHash hash;
        public final Cluster     exemplar;  // any one cluster with this taxa set
        public int               frequency; // how many times this exact taxa set appeared

        Entry(ClusterHash h, Cluster c) { this.hash = h; this.exemplar = c; this.frequency = 1; }
    }

    // Main table: ClusterHash -> Entry
    private final Map<ClusterHash, Entry> table = new HashMap<>();

    // Size bins: size -> list of ClusterHash objects of that size
    private final Map<Integer, List<ClusterHash>> sizeBins = new HashMap<>();

    // Special: the all-taxa cluster hash (DP root)
    private ClusterHash allTaxaHash;

    private final int m; // number of hash seeds

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    public ClusterTable(List<Tree> trees, PrefixHashArrays pref, int numTaxa) {
        long t0 = System.nanoTime();
        this.m = pref.numSeeds();

        // Build all-taxa hash from prefix arrays (using any complete tree)
        long[] atSums = new long[m], atXors = new long[m];
        for (int s = 0; s < m; s++) {
            atSums[s] = pref.allTaxaSum(s);
            atXors[s] = pref.allTaxaXor(s);
        }
        allTaxaHash = new ClusterHash(atSums, atXors, numTaxa, m);

        int totalCandidates = 0;
        int treesDone = 0;
        ProgressBar bar = new ProgressBar("Cluster extraction", trees.size());
        for (Tree tree : trees) {
            totalCandidates += extractFromTree(tree, pref, numTaxa);
            bar.update(++treesDone);
        }
        bar.done();

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Cluster extraction: %d candidates -> %d unique clusters in %d ms",
            totalCandidates, table.size(), ms);
        Logging.debug("  all-taxa cluster (DP root): %s", allTaxaHash);
        if (Logging.isDebug()) {
            logSizeSummary();
        }
    }

    /**
     * Extract clusters from one tree: subtree ranges + complements for every
     * non-root internal node. Leaves are also included (size-1 clusters).
     * Returns the number of candidates generated (before dedup).
     */
    private int extractFromTree(Tree tree, PrefixHashArrays pref, int numTaxa) {
        int ti = tree.treeIndex;
        int L  = tree.leafCount;
        int[] count = {0};

        walkNodes(tree.root, ti, L, pref, numTaxa, count);
        return count[0];
    }

    /**
     * Post-order walk. For every node (including leaves, but excluding root)
     * register the subtree cluster and its super-complement S\[lo,hi).
     */
    private void walkNodes(TreeNode node, int ti, int L,
                           PrefixHashArrays pref, int numTaxa, int[] count) {
        if (!node.isLeaf()) {
            walkNodes(node.left,  ti, L, pref, numTaxa, count);
            walkNodes(node.right, ti, L, pref, numTaxa, count);
        }

        if (node.isRoot()) return;  // skip root -- it is the all-taxa cluster

        int lo = node.rangeStart, hi = node.rangeEnd;
        int rangeSize = hi - lo;

        // ── 1. Subtree cluster [lo, hi) ──────────────────────────────────────
        registerCluster(ti, lo, hi, false, rangeSize, L, pref, numTaxa);
        count[0]++;

        // ── 2. Super-complement: S \ [lo, hi) (w.r.t. ALL taxa, not just Lg) ──
        int superCompSize = numTaxa - rangeSize;
        if (superCompSize > 0) {  // skip empty (only if rangeSize == numTaxa, impossible here)
            registerCluster(ti, lo, hi, true, superCompSize, numTaxa, pref, numTaxa);
            count[0]++;
        }
    }

    /**
     * Compute hash for a cluster and insert (or increment frequency) in the table.
     */
    private void registerCluster(int ti, int lo, int hi, boolean complement,
                                  int size, int leafCount,
                                  PrefixHashArrays pref, int numTaxa) {
        long[] rawSums = new long[m], rawXors = new long[m];
        for (int s = 0; s < m; s++) {
            if (!complement) {
                rawSums[s] = pref.rangeSum(ti, s, lo, hi);
                rawXors[s] = pref.rangeXor(ti, s, lo, hi);
            } else {
                // Super-complement w.r.t. ALL taxa (S \ [lo,hi))
                rawSums[s] = pref.superCompSum(ti, s, lo, hi);
                rawXors[s] = pref.superCompXor(ti, s, lo, hi);
            }
        }

        ClusterHash hash = new ClusterHash(rawSums, rawXors, size, m);

        // Skip the all-taxa cluster (it's the DP root, not in X)
        if (hash.equals(allTaxaHash)) return;

        Entry existing = table.get(hash);
        if (existing != null) {
            existing.frequency++;
        } else {
            Cluster exemplar = new Cluster(ti, lo, hi, complement, leafCount);
            Entry entry = new Entry(hash, exemplar);
            table.put(hash, entry);
            sizeBins.computeIfAbsent(size, k -> new ArrayList<>()).add(hash);
        }
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    public Entry get(ClusterHash hash)      { return table.get(hash); }
    public boolean contains(ClusterHash h)  { return table.containsKey(h); }
    public int size()                       { return table.size(); }
    public ClusterHash getAllTaxaHash()     { return allTaxaHash; }
    public Collection<Entry> entries()     { return table.values(); }
    public int numSeeds()                  { return m; }

    /** All cluster hashes of a given size. */
    public List<ClusterHash> getBySize(int sz) {
        return sizeBins.getOrDefault(sz, Collections.emptyList());
    }

    /** All sizes present in X. */
    public Set<Integer> sizes() { return sizeBins.keySet(); }

    // -------------------------------------------------------------------------

    private void logSizeSummary() {
        if (sizeBins.isEmpty()) return;
        int minSz = Integer.MAX_VALUE, maxSz = 0;
        for (int sz : sizeBins.keySet()) {
            minSz = Math.min(minSz, sz);
            maxSz = Math.max(maxSz, sz);
        }
        Logging.debug("  cluster size range: [%d, %d]", minSz, maxSz);
        Logging.debug("  distinct sizes: %d", sizeBins.size());
    }
}
