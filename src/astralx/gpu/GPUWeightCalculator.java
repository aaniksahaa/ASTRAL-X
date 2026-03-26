package astralx.gpu;

/**
 * JNI bridge to the CUDA weight-calculation kernel.
 *
 * The native library (libastralx_weight.so) must be on java.library.path
 * or the directory passed via -Djava.library.path=native/.
 *
 * Call tryLoad() once at startup; it returns false if the .so is missing.
 * After a successful load, computeWeightsGPU() offloads all score computation
 * to the GPU in a single kernel launch.
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
     * Compute 2*score for every split on the GPU.
     *
     * @param splits     flat int array, numSplits × 10:
     *                   [loTreeIdx, loLeft, loRight, loComplement, loSize,
     *                    hiTreeIdx, hiLeft, hiRight, hiComplement, hiSize]
     * @param parts      flat int array, numParts × 9:
     *                   [treeIdx, lo1, hi1, lo2, hi2, sz1, sz2, sz3, frequency]
     * @param orderings  flat int array, numTrees × numTaxa:
     *                   orderings[t*numTaxa + pos] = taxon id
     * @param invIndex   flat int array, numTrees × numTaxa:
     *                   invIndex[t*numTaxa + taxon] = postorder position (-1 if absent)
     * @param numSplits  number of candidate splits
     * @param numParts   number of gene-tree tripartitions
     * @param numTrees   number of gene trees
     * @param numTaxa    total taxon count (registry size)
     * @param totalN     total taxon count (same as numTaxa, passed to kernel)
     * @return long[numSplits] where result[i] = 2 * score(split i);
     *         divide each element by 2 for the final ASTRAL quartet score
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
        int totalN
    );
}
