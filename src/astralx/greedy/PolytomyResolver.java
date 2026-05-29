package astralx.greedy;

import astralx.cluster.ClusterHash;
import astralx.completion.SimilarityMatrix;
import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.Threading;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

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
     * Drive Step A + Step B for every polytomy in the pool, in parallel.
     *
     * Dispatch is LPT-first: tasks are sorted by {@link PolytomyTask#estimatedCost}
     * descending and submitted individually to the {@link Threading} executor.
     * The fixed-thread-pool's internal queue picks them up dynamically, so the
     * costly polytomies start first and short ones fill the gaps — effectively
     * "work-stealing-lite" via the standard executor without any custom queue.
     *
     * Per-task RNG is seeded deterministically from
     * {@code baseSeed XOR (thresholdIndex << 32) XOR node.id} so the emission
     * set is reproducible across runs and threading configurations.
     *
     * NO nested {@code Threading.processRangeParallel}: Step A's UPGMA and
     * Step B's resolveByDistance use {@link MiniUPGMA}, which is sequential.
     * This is what makes the outer parallelism safe — otherwise blocked
     * polytomy workers would prevent inner sub-tasks from ever starting.
     *
     * The shared {@link EmissionBuffer} is a {@code ConcurrentHashMap}; the
     * Step B adaptive-bonus check uses the {@code putIfAbsent} return value
     * (one true per newly-accepted signature) so it is unaffected by races
     * from other threads emitting overlapping signatures — local novelty
     * (design §10.4 recommendation).
     */
    public static int runAllParallel(List<PolytomyTask> tasks,
                                      List<Tree> geneTrees, SimilarityMatrix sim,
                                      EmissionBuffer buffer, int numTaxa,
                                      long baseSeed) {
        if (tasks.isEmpty()) return 0;

        List<PolytomyTask> sorted = new ArrayList<>(tasks);
        sorted.sort((a, b) -> Long.compare(b.estimatedCost(), a.estimatedCost()));

        List<Future<int[]>> futures = new ArrayList<>(sorted.size());
        for (PolytomyTask task : sorted) {
            final PolytomyTask t = task;
            long seed = baseSeed ^ ((long) t.thresholdIndex << 32) ^ (long) t.node.id;
            futures.add(Threading.submit(() -> {
                Random rng = new Random(seed);
                int aCount = (sim != null) ? stepA(t, sim, buffer, numTaxa) : 0;
                int bCount = stepB(t, geneTrees, buffer, numTaxa, rng, sim);
                return new int[]{aCount, bCount};
            }));
        }

        int totalA = 0, totalB = 0;
        for (Future<int[]> f : futures) {
            try {
                int[] c = f.get();
                totalA += c[0]; totalB += c[1];
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } catch (ExecutionException e) {
                throw new RuntimeException(e.getCause());
            }
        }
        return totalA + totalB;
    }

    /** Stable insertion-order accessor for the polytomy list — used by callers
     *  that want a fully deterministic single-thread pass for diffing. */
    public static List<PolytomyTask> sortLPT(List<PolytomyTask> tasks) {
        List<PolytomyTask> s = new ArrayList<>(tasks);
        Collections.sort(s, (a, b) -> Long.compare(b.estimatedCost(), a.estimatedCost()));
        return s;
    }

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
        Tree dendrogram = MiniUPGMA.build(groupSim, g, /*treeIndex*/0);

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

    // ── §8.4 Step B: sampleAndResolve with d-rep restriction ─────────────

    /** Legacy tuning constants — matches WQDataCollection.java lines 61-67. */
    public static final int STEPB_DEFAULT_RUNS      = 10;
    public static final int STEPB_MAX               = 100;
    public static final int STEPB_IMPROVEMENT_REWARD = 2;
    public static final int STEPB_MIN_FREQ          = 5;

    /**
     * Step B — sampleAndResolve.  Runs {@link #STEPB_DEFAULT_RUNS} base rounds
     * and up to {@link #STEPB_MAX} adaptive bonus rounds (each productive round
     * adds {@link #STEPB_IMPROVEMENT_REWARD} more), per the legacy adaptive
     * scheme.  A round is "productive" if it adds ≥ {@link #STEPB_MIN_FREQ}
     * new signatures to the buffer (local novelty per §10.4).
     *
     * Each round:
     *   1. Pick one random representative taxon per group.
     *   2. For each gene tree, walk it postorder; propagate a per-node int
     *      bitmap of "present reps in subtree".  When a node's popcount is in
     *      [2, presentCount - 1], it defines an induced split — map the rep
     *      bitmap back to group indices, build the multi-range, compute the
     *      smaller-side signature, emit.
     *
     * Per-tree walk is O(n) — fine for typical inputs.  An O(d log n) variant
     * via marked-ancestor walks or precomputed LCA is a future optimization.
     *
     * Limitation: assumes {@code numGroups ≤ 31} so the rep-membership bitmap
     * fits in an int.  The legacy size-limit budget already caps polytomy
     * degree near √(50 + 25n) ≪ 31 for typical n; we assert anyway.
     */
    public static int stepB(PolytomyTask task, List<Tree> geneTrees,
                            EmissionBuffer buffer, int numTaxa, Random rng,
                            SimilarityMatrix sim) {
        int d = task.numGroups;
        if (d < 4) return 0;                // need ≥ 4 groups for non-trivial induced split
        if (d > 31) return 0;               // int-bitmap limit; future: use long[]

        int totalNewSignatures = 0;
        int adaptBonus = 0;
        int j = 0;
        while (j < STEPB_DEFAULT_RUNS + adaptBonus) {
            int beforeSize = buffer.size();
            stepBRound(task, geneTrees, buffer, numTaxa, rng, sim);
            int newThisRound = buffer.size() - beforeSize;
            totalNewSignatures += newThisRound;
            if (newThisRound >= STEPB_MIN_FREQ && adaptBonus < STEPB_MAX) {
                adaptBonus += STEPB_IMPROVEMENT_REWARD;
            }
            j++;
        }
        return totalNewSignatures;
    }

    /**
     * Run one Step B round, matching the legacy {@code sampleAndResolve} →
     * {@code resolveLinearly} flow:
     *   1. Pick one rep per group.
     *   2. For each gene tree, walk postorder and emit each non-root internal
     *      node's rep-bitmap if it passes the size filter (cnt ∈ [2, d-2]).
     *      This mirrors {@code Utils.getBitsets}.
     *   3. Aggregate bitmaps into a frequency map; each bipartition is counted
     *      under whichever side was encountered first (complementary dedupe
     *      against the existing key — same shape as
     *      {@code returnBitSetCounts}).
     *   4. Sort by frequency descending (with deterministic tie-break).
     *   5. Run a mini-greedy laminar build on the d reps: accept a bipartition
     *      iff it is pairwise nested-or-disjoint with every previously
     *      accepted bipartition.  ASTRAL-MP's exact buildTreeFromClusters
     *      check additionally requires the new cluster to MOVE ≥ 2 of the
     *      LCA's children — but exact duplicates (single-child matches) are
     *      already caught by the global signature dedup in {@link EmissionBuffer},
     *      so pairwise laminar gives an equivalent emission set here.
     *   6. Each accepted bipartition → emit full-taxa bipartition via
     *      smaller-side selection, hashed via the consensus prefix-scan.
     */
    private static void stepBRound(PolytomyTask task, List<Tree> geneTrees,
                                    EmissionBuffer buffer, int numTaxa, Random rng,
                                    SimilarityMatrix sim) {
        int d = task.numGroups;
        int[] aCons = task.tree.aCons();
        int allBits = (1 << d) - 1;

        int[] reps = new int[d];
        for (int gi = 0; gi < d; gi++) {
            int firstLo = task.groupLos[gi], firstHi = task.groupHis[gi];
            int firstSize = firstHi - firstLo;
            int totalSize = firstSize;
            boolean isRest = (gi == d - 1) && (task.numGroups > task.degree);
            int restExtraSize = 0;
            if (isRest && task.restSplit) {
                restExtraSize = task.restHis2 - task.restLos2;
                totalSize += restExtraSize;
            }
            if (totalSize <= 0) { reps[gi] = -1; continue; }
            int idx = rng.nextInt(totalSize);
            int pos = (idx < firstSize)
                ? firstLo + idx
                : task.restLos2 + (idx - firstSize);
            reps[gi] = aCons[pos];
        }

        // ── Step (2)+(3): collect induced bipartition counts ────────────────
        java.util.HashMap<Integer, Integer> counts = new java.util.HashMap<>();
        java.util.ArrayList<Integer> perTreeBitmaps = new java.util.ArrayList<>(8);
        for (Tree gt : geneTrees) {
            collectGeneTreeBitmaps(gt, task, reps, d, perTreeBitmaps);
            for (int bm : perTreeBitmaps) {
                Integer cur = counts.get(bm);
                if (cur != null) {
                    counts.put(bm, cur + 1);
                    continue;
                }
                int comp = allBits ^ bm;
                Integer compCur = counts.get(comp);
                if (compCur != null) {
                    counts.put(comp, compCur + 1);
                    continue;
                }
                counts.put(bm, 1);
            }
        }
        if (counts.isEmpty()) return;

        // ── Step (4): sort by freq desc; deterministic tie-break by bitmap ──
        List<int[]> sorted = new ArrayList<>(counts.size());
        for (var e : counts.entrySet()) sorted.add(new int[]{e.getKey(), e.getValue()});
        sorted.sort((a, b) -> {
            int c = Integer.compare(b[1], a[1]);
            return (c != 0) ? c : Integer.compare(a[0], b[0]);
        });

        // ── Step (5): mini-greedy laminar build with buildTreeFromClusters
        //              semantics (LCA + ≥ 2 children moved).
        MiniGreedyBuilder mg = new MiniGreedyBuilder(d);
        for (int[] entry : sorted) {
            mg.tryInsert(entry[0]);
        }

        // ── Step (6): emit each accepted internal cluster bitmap as a full-taxa
        //              bipartition (smaller side, hashed via consensus prefix scan).
        mg.forEachAcceptedInternal(bm -> emitInducedSplit(bm, task, numTaxa, buffer));

        // ── Step (7): resolveByDistance — UPGMA on the d×d induced similarity
        //              matrix (per-round, on the sampled reps).  Each non-root
        //              internal node of the dendrogram → one emission, mapped
        //              back via group ranges.  This matches ASTRAL-MP's
        //              {@code resolveByDistance} call from sampleAndResolve.
        if (sim != null) {
            stepBResolveByDistance(task, reps, sim, numTaxa, buffer);
        }
    }

    /** UPGMA on the d×d sampled-rep similarity matrix; emit dendrogram bipartitions. */
    private static void stepBResolveByDistance(PolytomyTask task, int[] reps,
                                                 SimilarityMatrix sim,
                                                 int numTaxa, EmissionBuffer buffer) {
        int d = task.numGroups;
        double[] inducedSim = new double[d * d];
        for (int i = 0; i < d; i++) {
            int ri = reps[i];
            if (ri < 0) continue;
            for (int j = i + 1; j < d; j++) {
                int rj = reps[j];
                if (rj < 0) continue;
                double s = sim.getSim(ri, rj);
                inducedSim[i * d + j] = s;
                inducedSim[j * d + i] = s;
            }
        }
        Tree dendro = MiniUPGMA.build(inducedSim, d, /*treeIndex*/0);
        walkDendroAsRepBitmap(dendro.root, dendro.postorderArray, task, numTaxa, buffer);
    }

    /** Walk dendrogram; for each non-root internal node, emit the rep-bitmap as a bipartition. */
    private static void walkDendroAsRepBitmap(TreeNode dn, int[] postArr,
                                               PolytomyTask task, int numTaxa,
                                               EmissionBuffer buffer) {
        if (dn.isLeaf()) return;
        walkDendroAsRepBitmap(dn.left,  postArr, task, numTaxa, buffer);
        walkDendroAsRepBitmap(dn.right, postArr, task, numTaxa, buffer);
        if (dn.isRoot()) return;
        int bm = 0;
        for (int p = dn.rangeStart; p < dn.rangeEnd; p++) bm |= (1 << postArr[p]);
        int sz = Integer.bitCount(bm);
        int d = task.numGroups;
        if (sz < 2 || sz > d - 1) return;
        emitInducedSplit(bm, task, numTaxa, buffer);
    }

    /** Postorder walk that fills {@code out} with rep-bitmaps for each qualifying
     *  non-root internal node (mirrors {@code Utils.getBitsets}). */
    private static void collectGeneTreeBitmaps(Tree gt, PolytomyTask task, int[] reps,
                                                int d, java.util.ArrayList<Integer> out) {
        out.clear();
        int[] repAtPos = new int[gt.leafCount];
        Arrays.fill(repAtPos, -1);
        int presentCount = 0;
        for (int gi = 0; gi < d; gi++) {
            int r = reps[gi];
            if (r < 0) continue;
            int p = gt.positionMap[r];
            if (p < 0) continue;
            repAtPos[p] = gi;
            presentCount++;
        }
        if (presentCount < 2) return;
        walkCollect(gt.root, /*isRoot=*/true, repAtPos, d, out);
    }

    private static int walkCollect(TreeNode node, boolean isRoot, int[] repAtPos,
                                    int d, java.util.ArrayList<Integer> out) {
        if (node.isLeaf()) {
            int gi = repAtPos[node.rangeStart];
            return (gi >= 0) ? (1 << gi) : 0;
        }
        int leftBM  = walkCollect(node.left,  false, repAtPos, d, out);
        int rightBM = walkCollect(node.right, false, repAtPos, d, out);
        int bm = leftBM | rightBM;
        // Skip binary root (matches the `isRoot && childCount == 2` skip in Utils.getBitsets)
        if (isRoot) return bm;
        int legit = (leftBM != 0 ? 1 : 0) + (rightBM != 0 ? 1 : 0);
        if (legit < 2) return bm;
        int sz = Integer.bitCount(bm);
        if (sz < 2 || sz >= d - 1) return bm;
        out.add(bm);
        return bm;
    }

    /** Convert a rep-bitmap into a group-bipartition and emit (smaller side). */
    private static void emitInducedSplit(int repBitmap, PolytomyTask task,
                                          int numTaxa, EmissionBuffer buffer) {
        int d = task.numGroups;
        boolean[] selected = new boolean[d];
        int selectedSize = 0;
        for (int gi = 0; gi < d; gi++) {
            if ((repBitmap & (1 << gi)) != 0) {
                selected[gi] = true;
                selectedSize += groupSize(task, gi);
            }
        }
        int complementSize = numTaxa - selectedSize;
        if (selectedSize <= 1 || complementSize <= 1) return;
        if (selectedSize == numTaxa) return;

        MultiRange canonical;
        int canonicalSize;
        if (selectedSize <= complementSize) {
            canonical = buildSideMultiRange(task, selected, /*inverted=*/false);
            canonicalSize = selectedSize;
        } else {
            canonical = buildSideMultiRange(task, selected, /*inverted=*/true);
            canonicalSize = complementSize;
        }

        int m = task.tree.numSeeds();
        long[] sums = new long[m];
        long[] xors = new long[m];
        for (int s = 0; s < m; s++) {
            sums[s] = task.tree.combineDisjointSigma1(s, canonical.los, canonical.his);
            xors[s] = task.tree.combineDisjointSigma2(s, canonical.los, canonical.his);
        }
        ClusterHash sig = new ClusterHash(sums, xors, canonicalSize, m);
        buffer.add(new EmittedBipartition(
            sig, canonical, canonicalSize, 'B', task.thresholdIndex));
    }

    // ─────────────────────────────────────────────────────────────────────

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
