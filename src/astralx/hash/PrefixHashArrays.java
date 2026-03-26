package astralx.hash;

import astralx.Logging;
import astralx.tree.Tree;

import java.util.List;

/**
 * Prefix sum and prefix XOR arrays over taxon hashes for every gene tree.
 *
 * For tree t, seed s, position i:
 *   prefSum[t][s][i]  = sum  of hashes[s][postorder[t][j]] for j in [0, i)
 *   prefXor[t][s][i]  = XOR  of hashes[s][postorder[t][j]] for j in [0, i)
 *
 * Both arrays have length (leafCount + 1) with index 0 = identity (0).
 * Missing taxa contribute 0 (identity for both sum and XOR).
 *
 * This enables O(1) hash of any range [l, r):
 *   rangeSum(t, s, l, r) = prefSum[t][s][r] - prefSum[t][s][l]
 *   rangeXor(t, s, l, r) = prefXor[t][s][r] ^ prefXor[t][s][l]
 *
 * And O(1) complement hash (w.r.t. tree's taxa):
 *   compSum(t, s, l, r) = totalSum[t][s] - rangeSum(t, s, l, r)
 *   compXor(t, s, l, r) = totalXor[t][s] ^ rangeXor(t, s, l, r)
 *
 * All arithmetic is unsigned 64-bit (Java long wraps mod 2^64).
 */
public class PrefixHashArrays {

    private final int k;   // number of gene trees
    private final int m;   // number of hash seeds

    // prefSum[t][s][0..leafCount] -- length leafCount+1, index 0 = 0
    private final long[][][] prefSum;
    private final long[][][] prefXor;

    // Totals for each tree (= last entry of prefix arrays)
    private final long[][] totalSum;  // [t][s]
    private final long[][] totalXor;  // [t][s]

    // Hash of ALL taxa (union across all trees; same as any complete tree's total)
    private final long[] allTaxaSum;  // [s]
    private final long[] allTaxaXor;  // [s]

    public PrefixHashArrays(List<Tree> trees, TaxonHasher hasher) {
        long t0 = System.nanoTime();

        k = trees.size();
        m = hasher.numSeeds();

        prefSum  = new long[k][][];
        prefXor  = new long[k][][];
        totalSum = new long[k][m];
        totalXor = new long[k][m];

        for (int ti = 0; ti < k; ti++) {
            Tree tree = trees.get(ti);
            int L = tree.leafCount;
            prefSum[ti] = new long[m][L + 1];
            prefXor[ti] = new long[m][L + 1];

            for (int s = 0; s < m; s++) {
                for (int pos = 0; pos < L; pos++) {
                    int taxId = tree.postorderArray[pos];
                    long h = hasher.get(s, taxId);
                    prefSum[ti][s][pos + 1] = prefSum[ti][s][pos] + h;
                    prefXor[ti][s][pos + 1] = prefXor[ti][s][pos] ^ h;
                }
                totalSum[ti][s] = prefSum[ti][s][L];
                totalXor[ti][s] = prefXor[ti][s][L];
            }
        }

        // allTaxa: use the first complete tree (all complete trees have same total)
        allTaxaSum = new long[m];
        allTaxaXor = new long[m];
        for (Tree tree : trees) {
            if (tree.isComplete) {
                for (int s = 0; s < m; s++) {
                    allTaxaSum[s] = totalSum[tree.treeIndex][s];
                    allTaxaXor[s] = totalXor[tree.treeIndex][s];
                }
                break;
            }
        }

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Built prefix hash arrays: k=%d trees, m=%d seeds in %d ms", k, m, ms);
    }

    // -------------------------------------------------------------------------
    // Range hash queries -- all O(1)
    // -------------------------------------------------------------------------

    /** Sum hash of taxa in range [l, r) of tree t under seed s. */
    public long rangeSum(int t, int s, int l, int r) {
        return prefSum[t][s][r] - prefSum[t][s][l];
    }

    /** XOR hash of taxa in range [l, r) of tree t under seed s. */
    public long rangeXor(int t, int s, int l, int r) {
        return prefXor[t][s][r] ^ prefXor[t][s][l];
    }

    /** Sum hash of complement of [l, r) w.r.t. tree t's taxa. */
    public long compSum(int t, int s, int l, int r) {
        return totalSum[t][s] - rangeSum(t, s, l, r);
    }

    /** XOR hash of complement of [l, r) w.r.t. tree t's taxa. */
    public long compXor(int t, int s, int l, int r) {
        return totalXor[t][s] ^ rangeXor(t, s, l, r);
    }

    /** Sum hash of complement of [l, r) w.r.t. ALL taxa (super-complement). */
    public long superCompSum(int t, int s, int l, int r) {
        return allTaxaSum[s] - rangeSum(t, s, l, r);
    }

    /** XOR hash of complement of [l, r) w.r.t. ALL taxa (super-complement). */
    public long superCompXor(int t, int s, int l, int r) {
        return allTaxaXor[s] ^ rangeXor(t, s, l, r);
    }

    /** Total sum hash of all taxa in tree t under seed s. */
    public long totalSum(int t, int s) { return totalSum[t][s]; }

    /** Total XOR hash of all taxa in tree t under seed s. */
    public long totalXor(int t, int s) { return totalXor[t][s]; }

    /** Sum hash of ALL taxa (over entire taxon set) under seed s. */
    public long allTaxaSum(int s) { return allTaxaSum[s]; }

    /** XOR hash of ALL taxa under seed s. */
    public long allTaxaXor(int s) { return allTaxaXor[s]; }

    public int numSeeds() { return m; }
    public int numTrees() { return k; }
}
