package astralx.hash;

import astralx.preprocess.PreprocessedGeneTrees;
import astralx.preprocess.TreePreprocessInfo;

import java.util.ArrayList;
import java.util.List;

public final class PrefixHashIndex {
    public static final class TreePrefix {
        public final long[][] prefixSum; // [replicate][n]
        public final long[][] prefixXor; // [replicate][n]
        public final ClusterHashVector treeTotal;

        public TreePrefix(long[][] prefixSum, long[][] prefixXor, ClusterHashVector treeTotal) {
            this.prefixSum = prefixSum;
            this.prefixXor = prefixXor;
            this.treeTotal = treeTotal;
        }
    }

    private final List<TreePrefix> perTree;
    private final ClusterHashVector allTaxaTotal;

    public PrefixHashIndex(PreprocessedGeneTrees prep, SeededTaxonHashes taxonHashes) {
        int k = prep.treeInfos.size();
        int n = prep.totalTaxa;
        int m = taxonHashes.replicates;

        long[] globalSum = new long[m];
        long[] globalXor = new long[m];
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                long h = taxonHashes.hashesByTaxon[t][r];
                globalSum[r] += h;
                globalXor[r] ^= h;
            }
        }
        this.allTaxaTotal = new ClusterHashVector(globalSum, globalXor);

        this.perTree = new ArrayList<>(k);
        for (TreePreprocessInfo info : prep.treeInfos) {
            long[][] prefixSum = new long[m][n];
            long[][] prefixXor = new long[m][n];

            for (int pos = 0; pos < n; pos++) {
                int taxon = info.taxaByPostorderLeaf[pos];
                for (int r = 0; r < m; r++) {
                    long prevS = pos == 0 ? 0L : prefixSum[r][pos - 1];
                    long prevX = pos == 0 ? 0L : prefixXor[r][pos - 1];
                    if (taxon >= 0) {
                        long h = taxonHashes.hashesByTaxon[taxon][r];
                        prefixSum[r][pos] = prevS + h;
                        prefixXor[r][pos] = prevX ^ h;
                    } else {
                        prefixSum[r][pos] = prevS;
                        prefixXor[r][pos] = prevX;
                    }
                }
            }

            long[] totalSum = new long[m];
            long[] totalXor = new long[m];
            int last = Math.max(0, info.presentTaxaCount - 1);
            for (int r = 0; r < m; r++) {
                totalSum[r] = info.presentTaxaCount == 0 ? 0L : prefixSum[r][last];
                totalXor[r] = info.presentTaxaCount == 0 ? 0L : prefixXor[r][last];
            }
            perTree.add(new TreePrefix(prefixSum, prefixXor, new ClusterHashVector(totalSum, totalXor)));
        }
    }

    public TreePrefix treePrefix(int treeIndex) {
        return perTree.get(treeIndex);
    }

    public ClusterHashVector allTaxaTotal() {
        return allTaxaTotal;
    }

    public ClusterHashVector rangeHash(int treeIndex, int left, int right) {
        TreePrefix tp = perTree.get(treeIndex);
        int m = tp.prefixSum.length;
        long[] sum = new long[m];
        long[] xor = new long[m];
        for (int r = 0; r < m; r++) {
            long s = tp.prefixSum[r][right];
            long x = tp.prefixXor[r][right];
            if (left > 0) {
                s -= tp.prefixSum[r][left - 1];
                x ^= tp.prefixXor[r][left - 1];
            }
            sum[r] = s;
            xor[r] = x;
        }
        return new ClusterHashVector(sum, xor);
    }
}
