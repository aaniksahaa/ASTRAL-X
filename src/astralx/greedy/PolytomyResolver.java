package astralx.greedy;

import astralx.cluster.ClusterHash;
import astralx.completion.SimilarityMatrix;
import astralx.completion.UPGMAClusterer;
import astralx.tree.Tree;
import astralx.tree.TreeNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Polytomy resolution: per-polytomy Step A (UPGMA on the group similarity
 * matrix → bipartitions) and Step B (sampleAndResolve, in a follow-up).
 *
 * Step A — design §8.3:
 *   1. Build a g×g group-similarity matrix where entry (i,j) =
 *      average sim[x][y] over x ∈ group_i, y ∈ group_j (id-indexed lookups).
 *      Self-similarity (i,i) is left at 0.  The g×g fill dominates (O(|v|²)).
 *   2. Run UPGMA on that matrix.  Every non-root internal dendrogram node
 *      defines a bipartition (groups in its subtree | the rest).
 *   3. For each emission, compute the side's multi-range (union of selected
 *      groups' consensus-tree ranges, plus the rest group's split sub-ranges
 *      when the rest is selected), pick the smaller side, compute the
 *      double-hash signature via the consensus-tree prefix arrays, and add
 *      to the emission buffer (dedup by signature).
 */
public final class PolytomyResolver {

    private PolytomyResolver() {}

    /**
     * Run Step A for one polytomy.  Returns the number of new bipartitions
     * added to the buffer (signatures not previously seen).
     */
    public static int stepA(PolytomyTask task, SimilarityMatrix sim,
                            EmissionBuffer buffer, int numTaxa) {
        int g = task.numGroups;
        if (g < 3) return 0;  // need ≥ 3 groups for a non-trivial bipartition

        ConsensusTree ct = task.tree;
        int[] aCons = ct.aCons();

        // ── (1) Build g×g group similarity matrix ───────────────────────────
        double[] groupSim = new double[g * g];
        for (int i = 0; i < g; i++) {
            int iLo = task.groupLos[i], iHi = task.groupHis[i];
            int iLo2 = -1, iHi2 = -1;
            if (i == g - 1 && task.numGroups > task.degree && task.restSplit) {
                iLo2 = task.restLos2; iHi2 = task.restHis2;
            }
            int iSize = (iHi - iLo) + (iLo2 < 0 ? 0 : (iHi2 - iLo2));

            for (int j = i + 1; j < g; j++) {
                int jLo = task.groupLos[j], jHi = task.groupHis[j];
                int jLo2 = -1, jHi2 = -1;
                if (j == g - 1 && task.numGroups > task.degree && task.restSplit) {
                    jLo2 = task.restLos2; jHi2 = task.restHis2;
                }
                int jSize = (jHi - jLo) + (jLo2 < 0 ? 0 : (jHi2 - jLo2));

                double s = avgSim(sim, aCons, iLo, iHi, iLo2, iHi2,
                                  jLo, jHi, jLo2, jHi2);
                // Average over (iSize × jSize) pairs
                if (iSize > 0 && jSize > 0) s /= ((double) iSize * (double) jSize);
                groupSim[i * g + j] = s;
                groupSim[j * g + i] = s;
            }
        }

        // ── (2) Run UPGMA on the g×g matrix ─────────────────────────────────
        Tree dendrogram = UPGMAClusterer.build(groupSim, g, /*treeIndex*/0);

        // ── (3) Walk dendrogram; emit each non-root internal node ──────────
        int[] addedNew = {0};
        walkDendrogram(dendrogram.root, dendrogram.postorderArray,
                       task, buffer, numTaxa, addedNew);
        return addedNew[0];
    }

    // ── Group similarity: sum over pairs (without averaging — caller divides) ──

    private static double avgSim(SimilarityMatrix sim, int[] aCons,
                                  int iLo, int iHi, int iLo2, int iHi2,
                                  int jLo, int jHi, int jLo2, int jHi2) {
        double s = 0;
        s += pairSum(sim, aCons, iLo, iHi, jLo, jHi);
        if (jLo2 >= 0) s += pairSum(sim, aCons, iLo, iHi, jLo2, jHi2);
        if (iLo2 >= 0) {
            s += pairSum(sim, aCons, iLo2, iHi2, jLo, jHi);
            if (jLo2 >= 0) s += pairSum(sim, aCons, iLo2, iHi2, jLo2, jHi2);
        }
        return s;
    }

    private static double pairSum(SimilarityMatrix sim, int[] aCons,
                                   int iLo, int iHi, int jLo, int jHi) {
        double s = 0;
        for (int ai = iLo; ai < iHi; ai++) {
            int x = aCons[ai];
            for (int aj = jLo; aj < jHi; aj++) {
                int y = aCons[aj];
                s += sim.getSim(x, y);
            }
        }
        return s;
    }

    // ── Dendrogram walk: every non-root internal node = one bipartition ──

    private static void walkDendrogram(TreeNode dn, int[] postArr,
                                        PolytomyTask task, EmissionBuffer buffer,
                                        int numTaxa, int[] addedNew) {
        if (dn.isLeaf()) return;
        walkDendrogram(dn.left,  postArr, task, buffer, numTaxa, addedNew);
        walkDendrogram(dn.right, postArr, task, buffer, numTaxa, addedNew);
        if (dn.isRoot()) return;

        int nGroups = task.numGroups;
        boolean[] selected = new boolean[nGroups];
        int selectedSize = 0;
        for (int p = dn.rangeStart; p < dn.rangeEnd; p++) {
            int gi = postArr[p];
            if (!selected[gi]) {
                selected[gi] = true;
                selectedSize += groupSize(task, gi);
            }
        }
        int complementSize = numTaxa - selectedSize;
        if (selectedSize <= 1 || complementSize <= 1) return;        // trivial
        if (selectedSize == numTaxa) return;                          // whole tree

        // Pick smaller side; build its multi-range
        MultiRange canonical;
        int canonicalSize;
        if (selectedSize <= complementSize) {
            canonical = buildSideMultiRange(task, selected, /*inverted=*/false);
            canonicalSize = selectedSize;
        } else {
            canonical = buildSideMultiRange(task, selected, /*inverted=*/true);
            canonicalSize = complementSize;
        }

        // Compute double-hash signature via consensus prefix scan
        int m = task.tree.numSeeds();
        long[] sums = new long[m];
        long[] xors = new long[m];
        for (int s = 0; s < m; s++) {
            sums[s] = task.tree.combineDisjointSigma1(s, canonical.los, canonical.his);
            xors[s] = task.tree.combineDisjointSigma2(s, canonical.los, canonical.his);
        }
        ClusterHash sig = new ClusterHash(sums, xors, canonicalSize, m);

        if (buffer.add(new EmittedBipartition(
                sig, canonical, canonicalSize, 'A', task.thresholdIndex))) {
            addedNew[0]++;
        }
    }

    /** Number of taxa in group {@code gi}, accounting for the split-rest case. */
    private static int groupSize(PolytomyTask task, int gi) {
        int sz = task.groupHis[gi] - task.groupLos[gi];
        boolean isRest = (gi == task.numGroups - 1) && (task.numGroups > task.degree);
        if (isRest && task.restSplit) sz += task.restHis2 - task.restLos2;
        return sz;
    }

    /** Build a MultiRange of the chosen groups (or their complement when inverted). */
    private static MultiRange buildSideMultiRange(PolytomyTask task,
                                                   boolean[] selected,
                                                   boolean inverted) {
        // Collect (lo, hi) pairs, splitting rest into two when applicable.
        List<int[]> ranges = new ArrayList<>();
        for (int i = 0; i < task.numGroups; i++) {
            boolean want = inverted ? !selected[i] : selected[i];
            if (!want) continue;
            int lo = task.groupLos[i], hi = task.groupHis[i];
            if (lo < hi) ranges.add(new int[]{lo, hi});
            boolean isRest = (i == task.numGroups - 1) && (task.numGroups > task.degree);
            if (isRest && task.restSplit) {
                ranges.add(new int[]{task.restLos2, task.restHis2});
            }
        }
        int[] los = new int[ranges.size()];
        int[] his = new int[ranges.size()];
        for (int k = 0; k < ranges.size(); k++) {
            los[k] = ranges.get(k)[0];
            his[k] = ranges.get(k)[1];
        }
        return new MultiRange(task.tree, los, his);
    }
}
