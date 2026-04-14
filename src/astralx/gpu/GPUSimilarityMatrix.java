package astralx.gpu;

/**
 * JNI bridge to the CUDA similarity-matrix computation kernel.
 *
 * The native implementation uses:
 *   - Euler tour + O(1) sparse-table RMQ for LCA queries
 *   - Three payload-tracking sparse tables for subLC[c_a], subLC[c_b], S[u]
 *   - Δ-tree batching: tree data occupies O(Δ × n log n) GPU VRAM
 *   - B×B pair tiling: output buffers occupy O(B²) GPU VRAM (B ≈ sqrt(n·k))
 *
 * For each tile, the kernel accumulates into numTile[B²] and denTile[B²]
 * (double precision), with no atomics (each thread owns a unique pair cell).
 *
 * GPU VRAM = 2·B²·8  +  Δ·(E_max·(2+2+2+4) + LOG·E_max·(2+2+2+4) + n·6) bytes
 */
public class GPUSimilarityMatrix {

    private static volatile boolean loaded    = false;
    private static volatile boolean attempted = false;

    public static synchronized boolean tryLoad() {
        if (attempted) return loaded;
        attempted = true;
        try {
            System.loadLibrary("astralx_sim");
            loaded = true;
        } catch (UnsatisfiedLinkError e) {
            loaded = false;
        }
        return loaded;
    }

    public static boolean isLoaded() { return loaded; }

    /**
     * Compute the pairwise similarity matrix on GPU.
     *
     * @param eulerDepths      flat [numTrees × E_max] — Euler tour depths (short)
     * @param eulerPrevSubLC   flat [numTrees × E_max] — prevChildSubLC per position (short)
     * @param eulerNextSubLC   flat [numTrees × E_max] — nextChildSubLC per position (short)
     * @param eulerS           flat [numTrees × E_max] — S[u] per position (int)
     * @param sparseMin        flat [numTrees × LOG × E_max] — min-depth sparse table (short)
     * @param sparseSubLCLeft  flat [numTrees × LOG × E_max] — left-biased prevChildSubLC (short)
     * @param sparseSubLCRight flat [numTrees × LOG × E_max] — right-biased nextChildSubLC (short)
     * @param sparseSLeft      flat [numTrees × LOG × E_max] — left-biased S[u] (int)
     * @param firstOcc         flat [numTrees × n] — first Euler position of each leaf (-1 absent)
     * @param leafDepth        flat [numTrees × n] — depth of each leaf (short, -1 absent)
     * @param eulerLen         [numTrees] — actual tour length per tree
     * @param leafCount        [numTrees] — k_t (leaf count) per tree
     * @param numTrees         k
     * @param n                total taxon count
     * @param E_max            padded Euler tour length
     * @param LOG              number of sparse-table levels
     * @param tileSizeB        B — GPU pair tile side (0 = auto)
     * @param progressInterval seconds between progress updates
     * @param progressMaxSteps max progress prints (0 = time-interval mode)
     * @param numSumOut        pre-zeroed [n × n] double[] — native fills numerator sums
     * @param denSumOut        pre-zeroed [n × n] double[] — native fills denominator sums
     */
    public static native void computeSimilarityGPU(
        short[] eulerDepths,
        short[] eulerPrevSubLC,
        short[] eulerNextSubLC,
        int[]   eulerS,
        short[] sparseMin,
        short[] sparseSubLCLeft,
        short[] sparseSubLCRight,
        int[]   sparseSLeft,
        int[]   firstOcc,
        short[] leafDepth,
        int[]   eulerLen,
        int[]   leafCount,
        int     numTrees,
        int     n,
        int     E_max,
        int     LOG,
        int     tileSizeB,
        double  progressInterval,
        int     progressMaxSteps,
        double[] numSumOut,
        double[] denSumOut
    );
}
