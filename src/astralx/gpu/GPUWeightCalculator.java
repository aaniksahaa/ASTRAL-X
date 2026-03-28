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
     * Compute 2*score for every split on the GPU with adaptive split-batching.
     *
     * Static data (orderings, invIndex, parts) is uploaded to the GPU once.
     * Splits are streamed in adaptive batches; scores are accumulated into the
     * host result array.  This bounds peak VRAM at:
     *
     *   O(numTrees × numTaxa)   [orderings + invIndex, permanent]
     * + O(numParts)              [parts, permanent]
     * + batchSize × 48 B         [current split batch + score slice]
     *
     * @param splits         flat int array, numSplits × 10
     * @param parts          flat int array, numParts × 9
     * @param orderings      flat int array, numTrees × numTaxa
     * @param invIndex        flat int array, numTrees × numTaxa
     * @param numSplits      number of candidate splits
     * @param numParts       number of gene-tree tripartitions
     * @param numTrees       number of gene trees
     * @param numTaxa        total taxon count (registry size)
     * @param totalN         total taxon count (same as numTaxa, passed to kernel)
     * @param batchSizeHint  0=auto, -1=no batching, >0=exact batch size
     * @param vramFraction   fraction of free VRAM to use when batchSizeHint==0
     *                       (e.g. 0.75 means use 75%, reserve 25% as headroom)
     * @return long[numSplits] where result[i] = 2 * score(split i)
     */
    public static native long[] computeWeightsGPU(
        int[] splits,
        int[] parts,
        int[] orderings,
        int[] invIndex,
        int numSplits,
        int numParts,
        int numTrees,
        int numTaxa,
        int totalN,
        int batchSizeHint,
        double vramFraction
    );

    /**
     * Query GPU free and total VRAM via cudaMemGetInfo.
     * Returns long[2] = {freeMiB, totalMiB}, or null if unavailable.
     */
    public static native long[] queryVRAMMiB();
}
