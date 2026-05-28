package astralx.completion;

import astralx.Config;
import astralx.Logging;
import astralx.gpu.GPUSimilarityMatrix;
import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.ProgressBar;
import astralx.util.Threading;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Builds the quartet-based taxon similarity matrix from gene trees.
 *
 * Reproduces ASTRAL-MP's SimilarityMatrix.populateByQuartetDistance exactly:
 * for each tree T with kt leaves, and for each pair (a, b) both present in T,
 *   num_T(a, b) = same_side_T(a, b)
 *              = number of quartets {a, b, x, y} resolved by T where a and b
 *                fall on the same side of the bipartition
 *   den_T(a, b) = C2(kt − 2)
 * The final similarity is sim(a,b) = Σ num_T / Σ den_T.
 *
 * ── CPU path (buildCPU) ──────────────────────────────────────────────────────
 * Per-tree scatter, identical to ASTRAL-MP:
 *   For every internal node u, form three "components":
 *     left  = leaves of left  child of u
 *     right = leaves of right child of u
 *     others = leaves in T but not in subtree(u)
 *   totalPairs = C2(|left|) + C2(|right|) + C2(|others|)
 *   For each component-pair (X, Y), the per-(l ∈ X, r ∈ Y) scatter is
 *     sim = totalPairs − C2(|X|) − C2(|Y|)
 *
 * This sums contributions across every internal node on the unrooted path
 * between a and b — exactly the same-side quartet count. The earlier
 * implementation only scattered at the LCA, dropping the (path × others)
 * contributions; that was the source of the systematic mismatch.
 *
 * ── GPU path (buildGPU) ──────────────────────────────────────────────────────
 * Uses the validated bridge identity
 *   same_side_T(a, b)  =  C2(kt − 2)  −  QD_gt(a, b)
 * with an O(1) closed form for QD_gt via Euler tour + sparse-table RMQ
 * carrying (s, F) child-of-LCA payloads. See
 *   DOCS/similarity-matrix-design.md
 * for the derivation. For now the GPU path still uses the legacy LCA-only
 * formula; this file's CPU path is the byte-for-byte-correct reference.
 */
public class SimilarityMatrixBuilder {

    // ── Public entry points ───────────────────────────────────────────────────

    public static SimilarityMatrix buildCPU(List<Tree> trees, int n) {
        SimilarityMatrix sm = new SimilarityMatrix(n);

        ProgressBar bar = new ProgressBar("Building similarity matrix (CPU)", trees.size());
        int done = 0;
        for (Tree tree : trees) {
            accumulateCPU(tree, sm);
            bar.update(++done);
        }
        bar.done();

        sm.normalize();
        return sm;
    }

    public static SimilarityMatrix buildGPU(List<Tree> trees, int n) {
        int k = trees.size();
        Logging.info("Building FullTourData for %d trees (parallel CPU)", k);

        // ── Step 1: Build per-tree full tour data in parallel ─────────────────
        EulerTourBuilder.FullTourData[] tours = new EulerTourBuilder.FullTourData[k];
        ProgressBar eulerBar    = new ProgressBar("FullTour + RMQ build", k);
        AtomicInteger eulerDone = new AtomicInteger(0);

        Threading.processRangeParallel(k, i -> {
            tours[i] = EulerTourBuilder.buildFull(trees.get(i), n);
            eulerBar.update(eulerDone.incrementAndGet());
        });
        eulerBar.done();

        // ── Step 2: Compute flat-array layout constants ───────────────────────
        int eMaxRaw = 0, logMaxRaw = 0;
        for (EulerTourBuilder.FullTourData td : tours) {
            if (td.tourLen > eMaxRaw)   eMaxRaw   = td.tourLen;
            if (td.log     > logMaxRaw) logMaxRaw = td.log;
        }
        int ePadded = 1;
        while (ePadded < eMaxRaw) ePadded <<= 1;
        final int E_max   = ePadded;
        final int LOG_max = logMaxRaw;

        // Per-position arrays:
        //   eulerDepths (short, 2B), eulerLeftChildS/RightChildS (short, 2B each),
        //   eulerF, eulerLeftChildF, eulerRightChildF (double, 8B each)
        // → 2 + 2 + 2 + 8 + 8 + 8 = 30 bytes/pos
        double euler_mb  = (double)k * E_max * 30 / 1e6;
        // Sparse tables: sparseMin (2B), sparseLeftChildS/RightChildS (2B each),
        // sparseLeftChildF/RightChildF (8B each) → 2 + 2 + 2 + 8 + 8 = 22 bytes/cell
        double sparse_mb = (double)k * LOG_max * E_max * 22 / 1e6;
        double leaf_mb   = (double)k * n * 4 / 1e6;
        Logging.info("  E_max=%d  LOG=%d  euler %.1f MB  sparse %.1f MB  leaf %.1f MB",
            E_max, LOG_max, euler_mb, sparse_mb, leaf_mb);

        // ── Step 3: Flatten into contiguous Java arrays ───────────────────────
        long edSize = (long)k * E_max;
        long spSize = (long)k * LOG_max * E_max;
        long ldSize = (long)k * n;

        short[]  eulerDepths       = new short [(int)edSize];
        double[] eulerF            = new double[(int)edSize];
        short[]  eulerLeftChildS   = new short [(int)edSize];
        double[] eulerLeftChildF   = new double[(int)edSize];
        short[]  eulerRightChildS  = new short [(int)edSize];
        double[] eulerRightChildF  = new double[(int)edSize];

        short[]  sparseMin         = new short [(int)spSize];
        short[]  sparseLeftChildS  = new short [(int)spSize];
        double[] sparseLeftChildF  = new double[(int)spSize];
        short[]  sparseRightChildS = new short [(int)spSize];
        double[] sparseRightChildF = new double[(int)spSize];

        int[]    firstOcc          = new int   [(int)ldSize];
        int[]    eulerLen          = new int   [k];
        int[]    leafCount         = new int   [k];

        Arrays.fill(firstOcc, -1);

        ProgressBar flatBar    = new ProgressBar("Flattening similarity tour data", k);
        AtomicInteger flatDone = new AtomicInteger(0);

        Threading.processRangeParallel(k, i -> {
            EulerTourBuilder.FullTourData td = tours[i];
            int len   = td.tourLen;
            int treeN = td.firstOcc.length;

            long edOff = (long)i * E_max;
            long spOff = (long)i * LOG_max * E_max;
            long ldOff = (long)i * n;

            for (int p = 0; p < len; p++) {
                int dst = (int)(edOff + p);
                eulerDepths      [dst] = td.depths[p];
                eulerF           [dst] = td.eulerF[p];
                eulerLeftChildS  [dst] = td.eulerLeftChildS[p];
                eulerLeftChildF  [dst] = td.eulerLeftChildF[p];
                eulerRightChildS [dst] = td.eulerRightChildS[p];
                eulerRightChildF [dst] = td.eulerRightChildF[p];
            }

            for (int lvl = 0; lvl < td.log; lvl++) {
                int rowLen = Math.max(0, len - (1 << lvl) + 1);
                long dst   = spOff + (long)lvl * E_max;
                for (int p = 0; p < rowLen; p++) {
                    int idx = (int)(dst + p);
                    sparseMin         [idx] = td.sparseMin        [lvl][p];
                    sparseLeftChildS  [idx] = td.sparseLeftChildS [lvl][p];
                    sparseLeftChildF  [idx] = td.sparseLeftChildF [lvl][p];
                    sparseRightChildS [idx] = td.sparseRightChildS[lvl][p];
                    sparseRightChildF [idx] = td.sparseRightChildF[lvl][p];
                }
            }

            for (int a = 0; a < treeN; a++) {
                int fo = td.firstOcc[a];
                if (fo >= 0) firstOcc[(int)(ldOff + a)] = fo;
            }

            eulerLen [i] = len;
            leafCount[i] = td.leafCount;
            flatBar.update(flatDone.incrementAndGet());
        });
        flatBar.done();

        // ── Step 4: Call GPU kernel ───────────────────────────────────────────
        Config cfg = Config.getInstance();
        int    tileSizeB        = cfg.getGpuDistTileSizeB();
        double progressInterval = cfg.getGpuDpProgressInterval();
        int    progressMaxSteps = cfg.getGpuDpProgressMaxSteps();

        SimilarityMatrix sm = new SimilarityMatrix(n);
        GPUSimilarityMatrix.computeSimilarityGPU(
            eulerDepths,
            eulerF,
            eulerLeftChildS,  eulerLeftChildF,
            eulerRightChildS, eulerRightChildF,
            sparseMin,
            sparseLeftChildS,  sparseLeftChildF,
            sparseRightChildS, sparseRightChildF,
            firstOcc, eulerLen, leafCount,
            k, n, E_max, LOG_max,
            tileSizeB, progressInterval, progressMaxSteps,
            sm.numSum, sm.denSum
        );

        sm.normalize();
        return sm;
    }

    // ── CPU accumulation (per-tree, scatter mirroring ASTRAL-MP) ─────────────

    /**
     * Scatter same-side quartet counts to all (l, r) pairs at every internal
     * node, plus accumulate the per-pair denominator C2(kt − 2).
     */
    private static void accumulateCPU(Tree tree, SimilarityMatrix sm) {
        int kt = tree.leafCount;
        if (kt < 4) {
            // C2(kt-2) = 0 → tree contributes nothing to numerator or denominator
            // (matches ASTRAL-MP, which skips by virtue of sim and den both being 0).
            return;
        }

        long denPerPair = EulerTourBuilder.c2(kt - 2);
        int n = sm.n;
        int[] postArr = tree.postorderArray;

        // ── Numerator: scatter across all internal nodes ─────────────────────
        scatterAtNode(tree.root, tree, kt, sm);

        // ── Denominator: every pair (a, b) co-occurring in T gets += C2(kt-2) ─
        // ASTRAL-MP accumulates 2·C2(kt-2) in dn[l][r] (doubled by the
        // mirror-write pattern), then normalizes by dn/2. Equivalent to a
        // single sum here; the constant factor 2 cancels.
        for (int i = 0; i < kt; i++) {
            int a = postArr[i];
            long rowOff = (long) a * n;
            for (int j = 0; j < kt; j++) {
                if (i == j) continue;
                int b = postArr[j];
                sm.denSum[(int)(rowOff + b)] += denPerPair;
            }
        }
    }

    /**
     * Post-order scatter at one internal node.
     *
     * Components at u:
     *   left   = subtree of u.left   (rangeStart_L .. rangeEnd_L)
     *   right  = subtree of u.right  (rangeStart_R .. rangeEnd_R)
     *   others = leaves of T outside subtree(u)
     *            (i.e. positions [0, u.rangeStart) ∪ [u.rangeEnd, kt) in postArr)
     *
     * For each ordered component-pair, scatter
     *   sim = totalPairs − C2(|X|) − C2(|Y|)
     * to every (l ∈ X, r ∈ Y) leaf pair, symmetrically.
     */
    private static void scatterAtNode(TreeNode node, Tree tree, int kt, SimilarityMatrix sm) {
        if (node.isLeaf()) return;
        scatterAtNode(node.left,  tree, kt, sm);
        scatterAtNode(node.right, tree, kt, sm);

        int subL = node.left.rangeEnd  - node.left.rangeStart;
        int subR = node.right.rangeEnd - node.right.rangeStart;
        int subU = node.rangeEnd       - node.rangeStart;
        int subO = kt - subU;                                 // "others" size

        long cL = EulerTourBuilder.c2(subL);
        long cR = EulerTourBuilder.c2(subR);
        long cO = EulerTourBuilder.c2(subO);
        long totalPairs = cL + cR + cO;                       // only positive comps survive

        int n = sm.n;
        int[] postArr = tree.postorderArray;

        // ── (left × right): always present ───────────────────────────────────
        long simLR = totalPairs - cL - cR;
        if (simLR != 0) {
            scatterRangeRange(
                postArr,
                node.left.rangeStart,  node.left.rangeEnd,
                node.right.rangeStart, node.right.rangeEnd,
                simLR, sm.numSum, n);
        }

        // ── (left × others) and (right × others): only when u is non-root ────
        if (subO > 0) {
            long simLO = totalPairs - cL - cO;
            if (simLO != 0) {
                scatterRangeOthers(
                    postArr, kt,
                    node.left.rangeStart, node.left.rangeEnd,
                    node.rangeStart,      node.rangeEnd,
                    simLO, sm.numSum, n);
            }
            long simRO = totalPairs - cR - cO;
            if (simRO != 0) {
                scatterRangeOthers(
                    postArr, kt,
                    node.right.rangeStart, node.right.rangeEnd,
                    node.rangeStart,       node.rangeEnd,
                    simRO, sm.numSum, n);
            }
        }
    }

    /** Scatter `sim` to every (a ∈ [aLo,aHi)) × (b ∈ [bLo,bHi)) leaf-pair, both directions. */
    private static void scatterRangeRange(int[] postArr,
                                           int aLo, int aHi,
                                           int bLo, int bHi,
                                           long sim, double[] numSum, int n) {
        double sd = (double) sim;
        for (int pi = aLo; pi < aHi; pi++) {
            int a = postArr[pi];
            long rowA = (long) a * n;
            for (int pj = bLo; pj < bHi; pj++) {
                int b = postArr[pj];
                numSum[(int)(rowA + b)] += sd;
                numSum[b * n + a]       += sd;
            }
        }
    }

    /**
     * Scatter `sim` to every (a ∈ [aLo,aHi)) × (b ∈ others) leaf-pair, where
     * others = leaves in T outside [subLo, subHi).
     */
    private static void scatterRangeOthers(int[] postArr, int kt,
                                            int aLo, int aHi,
                                            int subLo, int subHi,
                                            long sim, double[] numSum, int n) {
        double sd = (double) sim;
        for (int pi = aLo; pi < aHi; pi++) {
            int a = postArr[pi];
            long rowA = (long) a * n;
            for (int pj = 0; pj < subLo; pj++) {
                int b = postArr[pj];
                numSum[(int)(rowA + b)] += sd;
                numSum[b * n + a]       += sd;
            }
            for (int pj = subHi; pj < kt; pj++) {
                int b = postArr[pj];
                numSum[(int)(rowA + b)] += sd;
                numSum[b * n + a]       += sd;
            }
        }
    }
}
