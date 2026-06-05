package astralx.gpu;

/**
 * JNI bridge to the CUDA weight-calculation kernel.
 *
 * The native library (libastralx_weight.so) must be on java.library.path
 * or the directory passed via -Djava.library.path=native/.
 *
 * Call tryLoad() once at startup; it returns false if the .so is missing.
 * After a successful load, computeWeightsGPU() offloads all score computation
 * to the GPU with adaptive split-batching to bound peak VRAM usage.
 *
 * batchSizeHint controls the batching:
 *    0  — auto: the C layer queries cudaMemGetInfo after uploading static data
 *               and picks the largest batch that fits in 75% of free VRAM.
 *   -1  — no batching: all splits in one launch (original single-kernel behaviour).
 *   >0  — manual override: use exactly this many splits per launch.
 */
public class GPUWeightCalculator {

    private static volatile boolean loaded = false;
    private static volatile boolean loadAttempted = false;

    /** Try to load the native library; returns true on success. */
    public static synchronized boolean tryLoad() {
        if (loadAttempted) return loaded;
        loadAttempted = true;
        try {
            System.loadLibrary("astralx_weight");
            loaded = true;
        } catch (UnsatisfiedLinkError e) {
            // Not a fatal error — caller falls back to CPU path
        }
        return loaded;
    }

    public static boolean isLoaded() { return loaded; }

    /**
     * Compute 2*score for every split on the GPU (prefix-sum tree-DP) with
     * adaptive split-batching.
     *
     * One thread block per split; the block loops over every gene tree, builds
     * per-tree leaf prefix sums for both sides of the split in shared memory,
     * then derives each internal node's 3×3 intersection matrix in O(1).
     *
     * Static data (orderings, invIndex, node CSR) is uploaded to the GPU once.
     * Splits are streamed in adaptive batches; scores are accumulated into the
     * host result array.  This bounds peak VRAM at:
     *
     *   O(numGpuTrees × numTaxa)   [orderings + invIndex, permanent]
     * + O(totalNodes)              [node CSR, permanent]
     * + batchSize × 48 B           [current split batch + score slice]
     *
     * Per-block transient working set is 2·(maxLeafCount+1) ints of shared memory.
     *
     * @param splits         flat int array, numSplits × 10
     *                       [aTree,aLo,aHi,aComp,aSize, bTree,bLo,bHi,bComp,bSize]
     * @param nodeData       flat int array, numUnique × 3  [lo, mid, hi] (exemplar interval)
     * @param nodeFreq       flat int array, numUnique  (frequency of each unique tripartition)
     * @param nodeOffset     flat int array, numPartTrees + 1  (CSR row pointers, bucket by exemplar)
     * @param partLeafCount  flat int array, numPartTrees  (leaf count L per gene tree)
     * @param orderings      flat int array, numGpuTrees × numTaxa
     * @param invIndex       flat int array, numGpuTrees × numTaxa
     * @param numSplits      number of candidate splits
     * @param numPartTrees   number of gene trees contributing tripartitions
     * @param partTreeOffset orderings/invIndex slot offset for gene trees
     *                       (0, or numClusterTrees when autocomplete is active)
     * @param maxLeafCount   max leaf count over the gene trees (shared-mem sizing)
     * @param numGpuTrees    total orderings/invIndex slots (for VRAM accounting)
     * @param numTaxa        total taxon count (registry size)
     * @param batchSizeHint  0=auto, -1=no batching, >0=exact batch size
     * @param vramFraction   fraction of free VRAM to use when batchSizeHint==0
     * @return long[numSplits] where result[i] = 2 * score(split i),
     *         or null if the GPU path is infeasible (caller falls back to CPU)
     */
    public static native long[] computeWeightsGPU(
        int[] splits,
        int[] nodeData,
        int[] nodeFreq,
        int[] nodeOffset,
        int[] partLeafCount,
        int[] orderings,
        int[] invIndex,
        int numSplits,
        int numPartTrees,
        int partTreeOffset,
        int maxLeafCount,
        int numGpuTrees,
        int numTaxa,
        int batchSizeHint,
        double vramFraction
    );

    /**
     * Query GPU free and total VRAM via cudaMemGetInfo.
     * Returns long[2] = {freeMiB, totalMiB}, or null if unavailable.
     */
    public static native long[] queryVRAMMiB();
}
