package astralx.greedy;

import astralx.cluster.Cluster;
import astralx.cluster.ClusterTable;
import astralx.tree.Tree;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Bridges polytomy-resolution emissions into the DP cluster set X.
 *
 * Each {@link EmittedBipartition} carries a {@code ClusterHash} signature (computed
 * via the consensus tree's prefix scans with the SAME {@code TaxonHasher} as the
 * gene-tree clusters, so it is directly comparable) and a {@link MultiRange}
 * descriptor of its smaller side in some consensus snapshot's {@code aCons}.
 *
 * Two tiers (DOCS/consensus-emission-and-restriction-optimization.md §2):
 *   Tier 1 — the set is already in X (some gene tree realizes it contiguously):
 *            the signature is present, so nothing to synthesize — it is already a
 *            scorable single-range cluster.
 *   Tier 2 — the set is contiguous in no tree: synthesize a multi-range
 *            {@link Cluster} anchored on the consensus snapshot tree (a COMPLETE
 *            exemplar over all n taxa) and insert it into X.
 *
 * Consensus snapshot trees that contribute a synthesized exemplar are wrapped as
 * lightweight {@link Tree}s (postorderArray = {@code aCons}, inverse positionMap,
 * {@code root == null} since they are membership-only) and APPENDED to a
 * weight-calc exemplar list — NOT to the gene-tree list the DP mines for local
 * transitions. The synthesized cluster's {@code treeIndex} points into that
 * appended region. Mode 2 (cross-tree) then discovers the transitions that make
 * the cluster usable in the DP (it is hash-based, so it incorporates the new
 * clusters automatically).
 */
public final class EmissionBridge {

    private EmissionBridge() {}

    /**
     * Insert all emissions into {@code clusterTable}, appending any needed
     * consensus exemplar trees to {@code clusterTreesMutable}.
     *
     * @param clusterTreesMutable  the WEIGHT-calc exemplar list (a copy of the
     *                             gene/completed-tree list); this method appends
     *                             consensus exemplar trees to it. The DP's own
     *                             gene-tree list must NOT be this list.
     * @return {@code int[]{tier1, tier2}} — counts of already-present vs synthesized.
     */
    public static int[] bridge(EmissionBuffer buffer, ClusterTable clusterTable,
                               List<Tree> clusterTreesMutable, int numTaxa) {
        int tier1 = 0, tier2 = 0;
        // One exemplar Tree per distinct consensus snapshot that needs synthesis.
        IdentityHashMap<ConsensusTree, Integer> exemplarIndex = new IdentityHashMap<>();

        for (EmittedBipartition e : buffer.all()) {
            if (clusterTable.contains(e.signature)) {   // Tier 1: already a scorable cluster
                tier1++;
                continue;
            }
            // Tier 2: synthesize a consensus-anchored multi-range exemplar.
            MultiRange mr = e.canonicalSide;
            ConsensusTree ct = mr.tree;
            Integer ti = exemplarIndex.get(ct);
            if (ti == null) {
                ti = clusterTreesMutable.size();
                clusterTreesMutable.add(buildExemplar(ct, ti, numTaxa));
                exemplarIndex.put(ct, ti);
            }
            Cluster c = new Cluster(ti, mr.los, mr.his, /*complement=*/false, e.size);
            if (clusterTable.addCluster(e.signature, c)) tier2++;
        }
        return new int[]{ tier1, tier2 };
    }

    /**
     * Wrap a consensus snapshot as a membership-only {@link Tree}: postorderArray
     * = {@code aCons}, inverse positionMap, {@code root == null}. The consensus
     * tree spans all n taxa, so {@code isComplete} is true and complement-free
     * intersection works directly.
     */
    private static Tree buildExemplar(ConsensusTree ct, int treeIndex, int numTaxa) {
        int[] aCons = ct.aCons();
        int[] posMap = new int[numTaxa];
        Arrays.fill(posMap, -1);
        for (int p = 0; p < aCons.length; p++) posMap[aCons[p]] = p;
        return new Tree(treeIndex, /*root=*/null, aCons, posMap, aCons.length, numTaxa);
    }
}
