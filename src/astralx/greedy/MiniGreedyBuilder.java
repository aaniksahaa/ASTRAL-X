package astralx.greedy;

/**
 * Mini greedy laminar builder over {@code d ≤ 31} virtual leaves, used by
 * Step B's {@code resolveLinearly} (design §8.4).  Each cluster is an int
 * bitmap; the tree maintains parent/children arrays + per-node subtree
 * bitmaps so {@link #tryInsert} exactly mirrors ASTRAL-MP's
 * {@code Utils.buildTreeFromClusters} semantics:
 *
 *   1. Find LCA of the new cluster's reps in the current tree.
 *   2. For every child of LCA, classify by whole-node test:
 *        - {@code child ⊆ cluster}     → move
 *        - {@code child ∩ cluster == ∅} → ignore (disjoint sibling)
 *        - else                          → REJECT (cross-cut)
 *   3. After scanning all children, require {@code accountedFor == |cluster|}
 *      (no leaves left over) AND {@code moved.size() ≥ 2} (refines structure)
 *      AND {@code moved != allChildren} (creates a NEW internal split).
 *   4. On ACCEPT: create new internal node with the moved children, attach to LCA.
 *
 * Skipped clusters are still skipped by the calling loop; only successful
 * inserts here cause an emission.
 *
 * Memory: O(d) — at most 2d nodes (d leaves + d-1 internals + virtual root).
 */
final class MiniGreedyBuilder {

    private final int d;
    private final int allBits;

    private final int[] parent;
    private final int[] bitmap;          // bits in subtree
    private int[][]    children;         // per-node child list (resized on insert)
    private final int[] childCount;
    private final int   virtualRoot;
    private final int[] repToLeafNode;   // repToLeafNode[r] = node id of leaf-r
    private int nextId;

    MiniGreedyBuilder(int d) {
        if (d < 1 || d > 31) throw new IllegalArgumentException("MiniGreedyBuilder: d ∈ [1, 31]");
        this.d        = d;
        this.allBits  = (d == 32) ? -1 : ((1 << d) - 1);
        int cap       = 2 * d + 2;

        this.parent     = new int[cap];
        this.bitmap     = new int[cap];
        this.children   = new int[cap][];
        this.childCount = new int[cap];

        this.virtualRoot = 0;
        parent[0]   = -1;
        bitmap[0]   = allBits;
        children[0] = new int[d];
        childCount[0] = d;
        for (int r = 0; r < d; r++) {
            int leaf = 1 + r;
            parent[leaf]     = virtualRoot;
            bitmap[leaf]     = (1 << r);
            children[leaf]   = new int[0];
            childCount[leaf] = 0;
            children[0][r]   = leaf;
        }
        this.repToLeafNode = new int[d];
        for (int r = 0; r < d; r++) repToLeafNode[r] = 1 + r;
        this.nextId = 1 + d;
    }

    /**
     * Attempt to insert cluster bitmap {@code bm}.  Returns true if accepted
     * (the tree is now refined); false if rejected or skipped.
     */
    boolean tryInsert(int bm) {
        int sz = Integer.bitCount(bm);
        if (sz < 2 || sz > d - 1) return false;          // trivial

        int lca = findLCA(bm);
        int kc  = childCount[lca];
        int[] kids = children[lca];

        int[] movedIds = new int[kc];
        int movedN     = 0;
        int accounted  = 0;

        for (int i = 0; i < kc; i++) {
            int c   = kids[i];
            int cbm = bitmap[c];
            int and = cbm & bm;
            if (and == 0) continue;                        // disjoint
            if (and == cbm) {                              // child wholly in bm
                movedIds[movedN++] = c;
                accounted += Integer.bitCount(cbm);
                continue;
            }
            return false;                                  // cross-cut
        }
        if (accounted != sz) return false;                 // some leaves unaccounted
        if (movedN < 2)      return false;                 // single-child / no-move
        if (movedN == kc)    return false;                 // bm == LCA (redundant)

        // ── Accept: create new internal node ─────────────────────────────
        int nNode = nextId++;
        parent[nNode]     = lca;
        bitmap[nNode]     = bm;
        int[] newKids     = new int[movedN];
        System.arraycopy(movedIds, 0, newKids, 0, movedN);
        children[nNode]   = newKids;
        childCount[nNode] = movedN;
        for (int j = 0; j < movedN; j++) parent[movedIds[j]] = nNode;

        // Remove moved children from LCA; append the new internal node
        int newLcaKc = kc - movedN + 1;
        int[] newLcaKids = new int[newLcaKc];
        int p = 0;
        outer:
        for (int i = 0; i < kc; i++) {
            int c = kids[i];
            for (int j = 0; j < movedN; j++) if (movedIds[j] == c) continue outer;
            newLcaKids[p++] = c;
        }
        newLcaKids[p++] = nNode;
        children[lca]   = newLcaKids;
        childCount[lca] = p;
        return true;
    }

    /** Smallest currently-accepted node whose subtree bitmap ⊇ bm. */
    private int findLCA(int bm) {
        int anyRep = Integer.numberOfTrailingZeros(bm);
        int node   = repToLeafNode[anyRep];
        while (true) {
            // node's bitmap ⊇ bm  ⇔  (bm & ~bitmap[node]) == 0
            if ((bm & ~bitmap[node]) == 0) return node;
            int p = parent[node];
            if (p < 0) return virtualRoot;
            node = p;
        }
    }

    /**
     * Iterate every accepted internal node (excluding the virtual root and
     * the d singleton leaves) and feed its subtree bitmap to {@code visitor}.
     */
    void forEachAcceptedInternal(java.util.function.IntConsumer visitor) {
        for (int id = 1 + d; id < nextId; id++) visitor.accept(bitmap[id]);
    }
}
