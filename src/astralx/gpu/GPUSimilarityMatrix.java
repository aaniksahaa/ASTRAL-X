package astralx.gpu;

/**
 * JNI bridge to the CUDA similarity-matrix computation kernel.
 *
 * Implements the validated bridge formula:
 *
 *   same_side_T(a,b)  =  C2(kt − 2)  −  QD_T(a,b)
 *
 *   QD_T(x,y) = ½ · [ (F(x) − F(cx)) + (F(y) − F(cy)) + (cxS−1)·Z + (cyS−1)·Z ]
 *
 * where:
 *   w  = LCA(x, y) in tree T
 *   cx = child of w on the x-side, cy = child of w on the y-side
 *   cxS = s(cx),  cyS = s(cy),   Z = kt − cxS − cyS
 *   F(v) is the path-prefix sum along the root→v path described in
 *   EulerTourBuilder.
 *
 * GPU per-pair query for tree T:
 *   l = min(firstOcc[x], firstOcc[y]),  r = max(...)
 *   k_lvl = floor(log2(r − l + 1)),     l2 = r − 2^k_lvl + 1
 *   pick left-biased min-depth position (LCA's INTERMEDIATE visit), read
 *   (leftChildS, leftChildF, rightChildS, rightChildF) at that position.
 *   If firstOcc[x] ≤ firstOcc[y]:  x's child = left,  y's child = right.
 *   Else: swap.
 *
 * Architecture:
 *   - Δ-tree batching: tree data O(Δ · n · log n) GPU VRAM
 *   - B×B pair tiling: output tile O(B²) GPU VRAM (B ≈ √(n·k))
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
     * Compute the pairwise similarity matrix on GPU using the bridge formula.
     *
     * @param eulerDepths       flat [numTrees × E_max]            (short) tour depths
     * @param eulerF            flat [numTrees × E_max]            (double) F(node) per pos
     * @param eulerLeftChildS   flat [numTrees × E_max]            (short)  s(leftChild) at intermediates
     * @param eulerLeftChildF   flat [numTrees × E_max]            (double) F(leftChild) at intermediates
     * @param eulerRightChildS  flat [numTrees × E_max]            (short)  s(rightChild) at intermediates
     * @param eulerRightChildF  flat [numTrees × E_max]            (double) F(rightChild) at intermediates
     * @param sparseMin         flat [numTrees × LOG × E_max]      (short)  left-biased min-depth
     * @param sparseLeftChildS  flat [numTrees × LOG × E_max]      (short)  argmin payload
     * @param sparseLeftChildF  flat [numTrees × LOG × E_max]      (double) argmin payload
     * @param sparseRightChildS flat [numTrees × LOG × E_max]      (short)  argmin payload
     * @param sparseRightChildF flat [numTrees × LOG × E_max]      (double) argmin payload
     * @param firstOcc          flat [numTrees × n]                (int)    first tour pos, −1 absent
     * @param eulerLen          [numTrees]                         (int)    actual tour length
     * @param leafCount         [numTrees]                         (int)    kt per tree
     * @param numTrees          k
     * @param n                 total taxon count
     * @param E_max             padded Euler tour length
     * @param LOG               number of sparse-table levels
     * @param tileSizeB         B — GPU pair tile side (0 = auto)
     * @param progressInterval  seconds between progress updates
     * @param progressMaxSteps  max progress prints (0 = time-interval mode)
     * @param numSumOut         pre-zeroed [n × n] — native fills numerator sums
     * @param denSumOut         pre-zeroed [n × n] — native fills denominator sums
     */
    public static native void computeSimilarityGPU(
        short[]  eulerDepths,
        double[] eulerF,
        short[]  eulerLeftChildS,
        double[] eulerLeftChildF,
        short[]  eulerRightChildS,
        double[] eulerRightChildF,
        short[]  sparseMin,
        short[]  sparseLeftChildS,
        double[] sparseLeftChildF,
        short[]  sparseRightChildS,
        double[] sparseRightChildF,
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
