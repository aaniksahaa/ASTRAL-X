package astralx.gpu;

/**
 * JNI bridge to the CUDA similarity-matrix computation kernel.
 *
 * GPU query for pair (a,b) in tree T:
 *   l = min(firstOcc[a], firstOcc[b])
 *   r = max(firstOcc[a], firstOcc[b])
 *   k_lvl = floor(log2(r−l+1)),  l2 = r − 2^k_lvl + 1
 *   dL = sparseMin[k_lvl][l],  dR = sparseMin[k_lvl][l2]
 *   sub_lca = (dL <= dR) ? sparseSubLC[k_lvl][l] : sparseSubLC[k_lvl][l2]
 *   num_T(a,b) = C2(kt − sub_lca)
 *   den_T(a,b) = C2(kt − 2)
 *
 * Architecture:
 *   - Δ-tree batching: tree data O(Δ · n · log n) GPU VRAM
 *   - B×B pair tiling: output tile O(B²) GPU VRAM  (B ≈ sqrt(n·k))
 *   - No atomics: thread (da,db) owns pair (a0+da, b0+db) uniquely
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
     * @param eulerDepths   flat [numTrees × E_max] — Euler tour depths (short)
     * @param eulerSubLC    flat [numTrees × E_max] — sub[v] at each Euler pos (short)
     * @param sparseMin     flat [numTrees × LOG × E_max] — min-depth sparse table (short)
     * @param sparseSubLC   flat [numTrees × LOG × E_max] — left-biased sub[LCA] payload (short)
     * @param firstOcc      flat [numTrees × n] — first Euler pos of each leaf (-1 absent)
     * @param eulerLen      [numTrees] — actual Euler tour length per tree
     * @param leafCount     [numTrees] — kt (leaf count) per tree
     * @param numTrees      k
     * @param n             total taxon count
     * @param E_max         padded Euler tour length
     * @param LOG           number of sparse-table levels
     * @param tileSizeB     B — GPU pair tile side (0 = auto)
     * @param progressInterval  seconds between progress updates
     * @param progressMaxSteps  max progress prints (0 = time-interval mode)
     * @param numSumOut     pre-zeroed [n × n] double[] — native fills numerator sums
     * @param denSumOut     pre-zeroed [n × n] double[] — native fills denominator sums
     */
    public static native void computeSimilarityGPU(
        short[]  eulerDepths,
        short[]  eulerSubLC,
        short[]  sparseMin,
        short[]  sparseSubLC,
        int[]    firstOcc,
        int[]    eulerLen,
        int[]    leafCount,
        int      numTrees,
        int      n,
        int      E_max,
        int      LOG,
        int      tileSizeB,
        double   progressInterval,
        int      progressMaxSteps,
        double[] numSumOut,
        double[] denSumOut
    );
}
