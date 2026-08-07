package astralx.completion;

/**
 * Quartet-based taxon similarity matrix.
 *
 * For taxa a and b:
 *   M[a][b] = Σ_{t: a,b present} num_t(a,b)
 *             ──────────────────────────────
 *             Σ_{t: a,b present} den_t(a,b)
 *
 * where:
 *   num_t(a,b) = S[LCA_t(a,b)] − C2(subLC[c_a]) − C2(subLC[c_b])
 *   den_t(a,b) = C2(k_t − 2)
 *   k_t        = leaf count of tree t
 *
 * After normalize():
 *   sim[a][b]  ∈ [0,1], diagonal = 1, symmetric
 *   dist[a][b] = 1 − sim[a][b]   (used by TreeCompleter)
 *
 * Pairs that never co-occur in any tree have sim = 0, dist = 1.
 */
public class SimilarityMatrix {
    public final int n;

    /** Accumulated numerator: Σ num_t(a,b). Flat n×n double. */
    final double[] numSum;

    /** Accumulated denominator: Σ den_t(a,b). Flat n×n double. */
    final double[] denSum;

    /** Finalized similarity: numSum / denSum. Populated by normalize(). */
    public final double[] sim;

    /**
     * Finalized distance: 1 − sim.
     * Passed directly to TreeCompleter.completeAll() as the dist array.
     */
    public final double[] dist;

    public SimilarityMatrix(int n) {
        long cellsLong = (long)n * n;
        if (cellsLong > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("Similarity matrix for " + n + " taxa requires "
                + cellsLong + " cells per array; Java arrays support at most "
                + (Integer.MAX_VALUE - 8));
        }
        int cells = (int)cellsLong;
        this.n      = n;
        this.numSum = new double[cells];
        this.denSum = new double[cells];
        this.sim    = new double[cells];
        this.dist   = new double[cells];
    }

    /**
     * Finalize: compute sim[a][b] = numSum / denSum, then dist = 1 − sim.
     * Pairs that never co-occur get sim = 0, dist = 1.
     * Diagonal is set to sim = 1, dist = 0.
     */
    public void normalize() {
        for (int i = 0; i < sim.length; i++) {
            sim [i] = (denSum[i] > 0.0) ? numSum[i] / denSum[i] : 0.0;
            dist[i] = 1.0 - sim[i];
        }
        // Diagonal: self-similarity = 1, self-distance = 0
        for (int a = 0; a < n; a++) {
            sim [a * n + a] = 1.0;
            dist[a * n + a] = 0.0;
        }
    }

    /** Returns M[a][b]. Call after normalize(). */
    public double getSim(int a, int b)  { return sim [a * n + b]; }

    /** Returns (1 − M[a][b]). Call after normalize(). */
    public double getDist(int a, int b) { return dist[a * n + b]; }
}
