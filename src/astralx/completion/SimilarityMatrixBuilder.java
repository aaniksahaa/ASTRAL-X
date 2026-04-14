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
 * For each pair (a, b) the similarity is:
 *   sim(a,b) = sum_T num_T(a,b) / sum_T den_T(a,b)
 *
 * where the sum is over all trees T that contain both a and b with kt ≥ 4,
 * and:
 *   num_T(a,b) = C2(kt − sub[LCA_T(a,b)])
 *   den_T(a,b) = C2(kt − 2)
 *
 * Pairs with num_T = 0 contribute nothing to either numerator or denominator.
 *
 * ── CPU path (buildCPU) ──────────────────────────────────────────────────────
 * Scatter: for each internal node u, assign num = C2(kt − sub[u]) to all
 * (left-subtree leaf × right-subtree leaf) pairs whose LCA IS u.
 * O(k·n²) time, O(n²) space.
 *
 * ── GPU path (buildGPU) ──────────────────────────────────────────────────────
 * Uses Euler tour + O(1) sparse-table RMQ:
 *   sub_lca = sparseSubLC at the leftmost-min-depth position in
 *             [min(firstOcc[a], firstOcc[b]),  max(firstOcc[a], firstOcc[b])]
 *   num_T(a,b) = C2(kt − sub_lca)
 *
 * Architecture:
 *   1. Build FullTourData per tree in parallel (eulerSubLC + sparseSubLC).
 *   2. Flatten into contiguous arrays padded to E_max.
 *   3. Call native CUDA kernel (Δ-tree × B×B tile).
 *   4. Normalize.
 */
public class SimilarityMatrixBuilder {

    // ── Public entry points ───────────────────────────────────────────────────

    public static SimilarityMatrix buildCPU(List<Tree> trees, int n) {
        SimilarityMatrix sm = new SimilarityMatrix(n);

        ProgressBar bar  = new ProgressBar("Building similarity matrix (CPU)", trees.size());
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
        ProgressBar eulerBar   = new ProgressBar("FullTour + RMQ build", k);
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

        // Memory estimate (in MB)
        double euler_mb  = (double)k * E_max * 4 / 1e6;           // depths(2) + subLC(2)
        double sparse_mb = (double)k * LOG_max * E_max * 4 / 1e6; // min(2) + subLC(2)
        double leaf_mb   = (double)k * n * 4 / 1e6;               // firstOcc(4)
        Logging.info("  E_max=%d  LOG=%d  euler %.1f MB  sparse %.1f MB  leaf %.1f MB",
            E_max, LOG_max, euler_mb, sparse_mb, leaf_mb);

        // ── Step 3: Flatten into contiguous Java arrays ───────────────────────
        // eulerDepths  [k × E_max]           short   depth at each tour position
        // eulerSubLC   [k × E_max]           short   sub[v] at each tour position
        // sparseMin    [k × LOG_max × E_max] short   min-depth sparse table
        // sparseSubLC  [k × LOG_max × E_max] short   left-biased sub[LCA] payload
        // firstOcc     [k × n]               int     first tour pos of each leaf (-1 absent)
        // eulerLen     [k]                   int
        // leafCount    [k]                   int

        long edSize = (long)k * E_max;
        long spSize = (long)k * LOG_max * E_max;
        long ldSize = (long)k * n;

        short[] eulerDepths = new short[(int)edSize];
        short[] eulerSubLC  = new short[(int)edSize];
        short[] sparseMin   = new short[(int)spSize];
        short[] sparseSubLC = new short[(int)spSize];
        int[]   firstOcc    = new int  [(int)ldSize];
        int[]   eulerLen    = new int  [k];
        int[]   leafCount   = new int  [k];

        Arrays.fill(firstOcc, -1);

        ProgressBar flatBar    = new ProgressBar("Flattening similarity tour data", k);
        AtomicInteger flatDone = new AtomicInteger(0);

        Threading.processRangeParallel(k, i -> {
            EulerTourBuilder.FullTourData td = tours[i];
            int len   = td.tourLen;
            int treeN = td.firstOcc.length;   // = n

            long edOff = (long)i * E_max;
            long spOff = (long)i * LOG_max * E_max;
            long ldOff = (long)i * n;

            // Euler arrays
            for (int p = 0; p < len; p++) {
                eulerDepths[(int)(edOff + p)] = td.depths[p];
                eulerSubLC [(int)(edOff + p)] = td.eulerSubLC[p];
            }

            // Sparse tables
            for (int lvl = 0; lvl < td.log; lvl++) {
                int rowLen = Math.max(0, len - (1 << lvl) + 1);
                long dst   = spOff + (long)lvl * E_max;
                for (int p = 0; p < rowLen; p++) {
                    sparseMin  [(int)(dst + p)] = td.sparseMin  [lvl][p];
                    sparseSubLC[(int)(dst + p)] = td.sparseSubLC[lvl][p];
                }
            }

            // Leaf first-occurrence map
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
            eulerDepths, eulerSubLC,
            sparseMin, sparseSubLC,
            firstOcc, eulerLen, leafCount,
            k, n, E_max, LOG_max,
            tileSizeB, progressInterval, progressMaxSteps,
            sm.numSum, sm.denSum
        );

        sm.normalize();
        return sm;
    }

    // ── CPU accumulation (per-tree) ───────────────────────────────────────────

    /**
     * Scatter C2(out[u]) = C2(kt − sub[u]) to every (left-leaf × right-leaf) pair.
     * For those pairs, u is the LCA, so num = C2(kt − sub[LCA]).
     */
    private static void accumulateCPU(Tree tree, SimilarityMatrix sm) {
        int kt = tree.leafCount;
        if (kt < 4) return;   // C2(kt-2) = 0 for kt ≤ 3
        long den = EulerTourBuilder.c2(kt - 2);
        accumulateNodeCPU(tree.root, tree, kt, den, sm);
    }

    private static void accumulateNodeCPU(TreeNode node, Tree tree, int kt,
                                           long den, SimilarityMatrix sm) {
        if (node.isLeaf()) return;
        accumulateNodeCPU(node.left,  tree, kt, den, sm);
        accumulateNodeCPU(node.right, tree, kt, den, sm);

        int subU  = node.rangeEnd   - node.rangeStart;
        long num  = EulerTourBuilder.c2(kt - subU);   // C2(out[u]) = C2(kt − sub[u])
        if (num == 0) return;

        int n   = sm.n;
        int loL = node.left.rangeStart,  hiL = node.left.rangeEnd;
        int loR = node.right.rangeStart, hiR = node.right.rangeEnd;

        for (int pi = loL; pi < hiL; pi++) {
            int ta = tree.postorderArray[pi];
            for (int pj = loR; pj < hiR; pj++) {
                int tb = tree.postorderArray[pj];
                sm.numSum[ta * n + tb] += num;
                sm.numSum[tb * n + ta] += num;
                sm.denSum[ta * n + tb] += den;
                sm.denSum[tb * n + ta] += den;
            }
        }
    }
}
