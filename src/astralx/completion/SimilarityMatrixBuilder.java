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
 * CPU path (buildCPU):
 *   For each gene tree T (leaf count k_t ≥ 4), for each internal node u:
 *     subLC_u  = u.rangeEnd − u.rangeStart
 *     subLC_ca = left-child subLeafCount
 *     subLC_cb = right-child subLeafCount
 *     S_u      = C2(k_t − subLC_u) + C2(subLC_ca) + C2(subLC_cb)
 *     num      = S_u − C2(subLC_ca) − C2(subLC_cb)  =  C2(k_t − subLC_u)
 *     den      = C2(k_t − 2)
 *     Accumulate (num, den) for every pair (a ∈ left-sub, b ∈ right-sub).
 *   O(k × n²) time, O(n²) space.
 *
 * GPU path (buildGPU):
 *   1. Build FullTourData per tree in parallel.
 *   2. Flatten into contiguous arrays padded to E_max.
 *   3. Call native CUDA kernel (same Δ-tree × B×B tile architecture as dist).
 *      GPU VRAM = O(B² + Δ·n·log n).
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
        ProgressBar eulerBar  = new ProgressBar("FullTour + RMQ build", k);
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

        Logging.info("  E_max=%d  LOG=%d  euler %.1f MB  sparseMin %.1f MB  total-sim %.1f MB",
            E_max, LOG_max,
            (double)k * E_max * 2 / 1e6,
            (double)k * LOG_max * E_max * 2 / 1e6,
            (double)k * (E_max * (2+2+2+4) + (long)LOG_max * E_max * (2+2+2+4)) / 1e6);

        // ── Step 3: Flatten into contiguous Java arrays ───────────────────────
        // eulerDepths       [k × E_max]             short
        // eulerPrevSubLC    [k × E_max]             short
        // eulerNextSubLC    [k × E_max]             short
        // eulerS            [k × E_max]             int
        // sparseMin         [k × LOG_max × E_max]   short
        // sparseSubLCLeft   [k × LOG_max × E_max]   short
        // sparseSubLCRight  [k × LOG_max × E_max]   short
        // sparseSLeft       [k × LOG_max × E_max]   int
        // firstOcc          [k × n]                 int
        // leafDepth         [k × n]                 short   (-1 if absent)
        // eulerLen          [k]                     int
        // leafCount         [k]                     int

        long edSize = (long)k * E_max;
        long spSize = (long)k * LOG_max * E_max;
        long ldSize = (long)k * n;

        short[] eulerDepths      = new short[(int)edSize];
        short[] eulerPrevSubLC   = new short[(int)edSize];
        short[] eulerNextSubLC   = new short[(int)edSize];
        int[]   eulerS           = new int  [(int)edSize];
        short[] sparseMin        = new short[(int)spSize];
        short[] sparseSubLCLeft  = new short[(int)spSize];
        short[] sparseSubLCRight = new short[(int)spSize];
        int[]   sparseSLeft      = new int  [(int)spSize];
        int[]   firstOcc         = new int  [(int)ldSize];
        short[] leafDepth        = new short[(int)ldSize];
        int[]   eulerLen         = new int  [k];
        int[]   leafCount        = new int  [k];

        Arrays.fill(leafDepth, (short)-1);
        Arrays.fill(firstOcc,  -1);

        ProgressBar flatBar  = new ProgressBar("Flattening similarity tour data", k);
        AtomicInteger flatDone = new AtomicInteger(0);

        Threading.processRangeParallel(k, i -> {
            EulerTourBuilder.FullTourData td = tours[i];
            int len = td.tourLen;
            int treeN = td.firstOcc.length;  // = n

            long edOff = (long)i * E_max;
            long spOff = (long)i * LOG_max * E_max;
            long ldOff = (long)i * n;

            // Euler arrays
            for (int p = 0; p < len; p++) {
                eulerDepths   [(int)(edOff + p)] = td.depths        [p];
                eulerPrevSubLC[(int)(edOff + p)] = td.prevChildSubLC[p];
                eulerNextSubLC[(int)(edOff + p)] = td.nextChildSubLC[p];
                eulerS        [(int)(edOff + p)] = td.eulerS        [p];
            }

            // Sparse tables
            for (int lvl = 0; lvl < td.log; lvl++) {
                int rowLen = Math.max(0, len - (1 << lvl) + 1);
                long dst = spOff + (long)lvl * E_max;
                for (int p = 0; p < rowLen; p++) {
                    sparseMin       [(int)(dst + p)] = td.sparseMin       [lvl][p];
                    sparseSubLCLeft [(int)(dst + p)] = td.sparseSubLCLeft [lvl][p];
                    sparseSubLCRight[(int)(dst + p)] = td.sparseSubLCRight[lvl][p];
                    sparseSLeft     [(int)(dst + p)] = td.sparseSLeft     [lvl][p];
                }
            }

            // Leaf maps
            for (int a = 0; a < treeN; a++) {
                int fo = td.firstOcc[a];
                if (fo >= 0) {
                    firstOcc [(int)(ldOff + a)] = fo;
                    leafDepth[(int)(ldOff + a)] = td.depths[fo];
                }
            }

            eulerLen [i] = len;
            leafCount[i] = td.leafCount;
            flatBar.update(flatDone.incrementAndGet());
        });
        flatBar.done();

        // ── Step 4: Call GPU kernel ───────────────────────────────────────────
        Config cfg = Config.getInstance();
        int    tileSizeB         = cfg.getGpuDistTileSizeB();
        double progressInterval  = cfg.getGpuDpProgressInterval();
        int    progressMaxSteps  = cfg.getGpuDpProgressMaxSteps();

        SimilarityMatrix sm = new SimilarityMatrix(n);
        GPUSimilarityMatrix.computeSimilarityGPU(
            eulerDepths, eulerPrevSubLC, eulerNextSubLC, eulerS,
            sparseMin, sparseSubLCLeft, sparseSubLCRight, sparseSLeft,
            firstOcc, leafDepth, eulerLen, leafCount,
            k, n, E_max, LOG_max,
            tileSizeB, progressInterval, progressMaxSteps,
            sm.numSum, sm.denSum
        );

        sm.normalize();
        return sm;
    }

    // ── CPU accumulation (per-tree) ───────────────────────────────────────────

    private static void accumulateCPU(Tree tree, SimilarityMatrix sm) {
        int kt = tree.leafCount;
        if (kt < 4) return;   // C2(kt-2) = 0 for kt ≤ 3, no contribution
        long den = EulerTourBuilder.c2(kt - 2);
        accumulateNodeCPU(tree.root, tree, kt, den, sm);
    }

    private static void accumulateNodeCPU(TreeNode node, Tree tree, int kt,
                                           long den, SimilarityMatrix sm) {
        if (node.isLeaf()) return;
        accumulateNodeCPU(node.left,  tree, kt, den, sm);
        accumulateNodeCPU(node.right, tree, kt, den, sm);

        int subLC_u  = node.rangeEnd        - node.rangeStart;
        long num = EulerTourBuilder.c2(kt - subLC_u);   // simplified binary formula
        if (num == 0) return;

        int n    = sm.n;
        int loL  = node.left.rangeStart,  hiL = node.left.rangeEnd;
        int loR  = node.right.rangeStart, hiR = node.right.rangeEnd;

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
