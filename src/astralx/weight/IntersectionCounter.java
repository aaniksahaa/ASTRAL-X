package astralx.weight;

import astralx.tree.Tree;

/**
 * O(min(|P|, |Q|)) intersection counts between postorder ranges.
 *
 * For complete gene trees (Lg = L), complement intersections reduce to:
 *   |comp(P) ∩ Q| = |Q| - |P ∩ Q|
 *   |P ∩ comp(Q)| = |P| - |P ∩ Q|
 *   |comp(P) ∩ comp(Q)| = n - |P| - |Q| + |P ∩ Q|
 *
 * So we only ever need the one "core" non-complement count.
 */
public final class IntersectionCounter {

    private IntersectionCounter() {}

    /**
     * |range_in_treeA ∩ range_in_treeB| -- both ranges non-complement.
     * Iterates over the smaller range; looks up each taxon in the other tree's
     * positionMap to check membership.
     */
    public static int coreIntersect(Tree tA, int loA, int hiA,
                                     Tree tB, int loB, int hiB) {
        int count = 0;
        if (hiA - loA <= hiB - loB) {
            for (int pos = loA; pos < hiA; pos++) {
                int taxon = tA.postorderArray[pos];
                int posB  = tB.positionMap[taxon];
                if (posB >= loB && posB < hiB) count++;
            }
        } else {
            for (int pos = loB; pos < hiB; pos++) {
                int taxon = tB.postorderArray[pos];
                int posA  = tA.positionMap[taxon];
                if (posA >= loA && posA < hiA) count++;
            }
        }
        return count;
    }

    /**
     * |M_range ∩ cluster| where M_range is non-complement (a gene-tree subtree
     * range) and cluster may be complement (a species-tree candidate part).
     *
     * For complete trees: |comp(cluster_range) ∩ M_range| = |M_range| - |cluster_range ∩ M_range|
     *
     * @param tGT       gene-tree tree
     * @param loGT, hiGT  range in gene-tree postorder array (non-complement)
     * @param tC        exemplar tree of the candidate cluster
     * @param loC, hiC  range in candidate cluster's exemplar tree
     * @param cComp     whether the candidate cluster is complement w.r.t. its tree
     * @param sizeGTRange  = hiGT - loGT (passed in to avoid recomputation)
     */
    public static int intersect(Tree tGT, int loGT, int hiGT,
                                 Tree tC, int loC, int hiC, boolean cComp,
                                 int sizeGTRange) {
        int core = coreIntersect(tGT, loGT, hiGT, tC, loC, hiC);
        return cComp ? (sizeGTRange - core) : core;
    }
}
