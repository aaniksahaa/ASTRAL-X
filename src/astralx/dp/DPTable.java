package astralx.dp;

import astralx.Logging;
import astralx.cluster.ClusterHash;
import astralx.cluster.ClusterTable;
import astralx.hash.PrefixHashArrays;
import astralx.hash.TaxonHasher;
import astralx.tree.Tree;
import astralx.tree.TreeNode;

import java.util.*;

/**
 * DP search space: maps each cluster hash to its set of candidate bipartition splits.
 *
 * Built via Mode 1 (tree-local transitions only), O(nk):
 *
 *   Type 1  -- for every internal node u (incl. root):
 *              sub(u) → sub(left(u)) | sub(right(u))
 *
 *   Type 2  -- for every non-root internal node u whose parent is also non-root:
 *              [Lg \ sub(u)] → sub(sibling(u)) | [Lg \ sub(parent(u))]
 *
 * For complete trees (Lg == S), Types 3a/3b add nothing new and are skipped.
 *
 * The root of the DP is the all-taxa cluster (from ClusterTable.getAllTaxaHash()).
 * Its Type 1 transition(s) are stored at that hash key.
 */
public class DPTable {

    // transitions[parentHash] = set of distinct BipartitionSplits for that cluster
    private final Map<ClusterHash, Set<BipartitionSplit>> transitions = new HashMap<>();

    private final ClusterHash rootHash; // allTaxaHash -- starting point of the DP
    private final int m;

    // stats
    private int totalEmitted = 0;   // total emitted (with duplicates across trees)
    private int uniqueSplits = 0;   // sum of set sizes after dedup

    // -------------------------------------------------------------------------

    public DPTable(List<Tree> trees, PrefixHashArrays pref, ClusterTable clusterTable) {
        long t0 = System.nanoTime();
        this.m        = pref.numSeeds();
        this.rootHash = clusterTable.getAllTaxaHash();

        for (Tree tree : trees) {
            extractFromTree(tree, pref);
        }

        // Count total unique splits
        for (Set<BipartitionSplit> s : transitions.values()) uniqueSplits += s.size();

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("DP table: %d clusters with splits, %d unique splits (%d emitted) in %d ms",
            transitions.size(), uniqueSplits, totalEmitted, ms);
    }

    // -------------------------------------------------------------------------
    // Tree traversal
    // -------------------------------------------------------------------------

    private void extractFromTree(Tree tree, PrefixHashArrays pref) {
        int ti = tree.treeIndex;
        int L  = tree.leafCount;
        emit(tree.root, ti, L, pref);
    }

    /** Post-order recursion: emit transitions for this node, then children. */
    private void emit(TreeNode u, int ti, int L, PrefixHashArrays pref) {
        if (u.isLeaf()) return;
        emit(u.left,  ti, L, pref);
        emit(u.right, ti, L, pref);

        // ── Type 1: sub(u) → sub(left) | sub(right) ─────────────────────────
        ClusterHash hU     = hashRange(ti, u.rangeStart,       u.rangeEnd,       false, L, pref);
        ClusterHash hLeft  = hashRange(ti, u.left.rangeStart,  u.left.rangeEnd,  false, L, pref);
        ClusterHash hRight = hashRange(ti, u.right.rangeStart, u.right.rangeEnd, false, L, pref);
        addTransition(hU, hLeft, hRight);

        // ── Type 2: comp(u) → sub(sibling) | comp(parent) ───────────────────
        // Applies only if u is not root AND u's parent is not root.
        // (If parent is root, comp(parent) = empty -- degenerate, skip.)
        if (!u.isRoot() && !u.parent.isRoot()) {
            TreeNode sib    = u.getSibling();
            TreeNode parent = u.parent;

            ClusterHash hCompU      = hashRange(ti, u.rangeStart,      u.rangeEnd,      true, L, pref);
            ClusterHash hSib        = hashRange(ti, sib.rangeStart,    sib.rangeEnd,    false, L, pref);
            ClusterHash hCompParent = hashRange(ti, parent.rangeStart, parent.rangeEnd, true,  L, pref);
            addTransition(hCompU, hSib, hCompParent);
        }
    }

    // -------------------------------------------------------------------------

    private void addTransition(ClusterHash parent, ClusterHash a, ClusterHash b) {
        totalEmitted++;
        BipartitionSplit split = new BipartitionSplit(a, b);
        transitions.computeIfAbsent(parent, k -> new LinkedHashSet<>()).add(split);
    }

    /**
     * Compute a finalized ClusterHash for the range [lo,hi) in tree ti,
     * optionally complement w.r.t. Lg (the L leaves of this tree).
     */
    private ClusterHash hashRange(int ti, int lo, int hi, boolean complement, int L,
                                   PrefixHashArrays pref) {
        long[] rawSums = new long[m], rawXors = new long[m];
        for (int s = 0; s < m; s++) {
            rawSums[s] = complement ? pref.compSum(ti, s, lo, hi) : pref.rangeSum(ti, s, lo, hi);
            rawXors[s] = complement ? pref.compXor(ti, s, lo, hi) : pref.rangeXor(ti, s, lo, hi);
        }
        int sz = complement ? (L - (hi - lo)) : (hi - lo);
        return new ClusterHash(rawSums, rawXors, sz, m);
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    public ClusterHash getRootHash()                       { return rootHash; }
    public Set<BipartitionSplit> getSplits(ClusterHash h)  { return transitions.getOrDefault(h, Collections.emptySet()); }
    public boolean hasSplits(ClusterHash h)                { return transitions.containsKey(h); }
    public int numClusters()                               { return transitions.size(); }
    public int numUniqueSplits()                           { return uniqueSplits; }
    public int numEmitted()                                { return totalEmitted; }
    public Set<Map.Entry<ClusterHash, Set<BipartitionSplit>>> entries() { return transitions.entrySet(); }
}
