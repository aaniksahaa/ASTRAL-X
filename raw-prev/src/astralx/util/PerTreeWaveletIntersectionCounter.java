package astralx.util;

import astralx.cluster.Cluster;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.preprocess.TreePreprocessInfo;

import java.util.HashMap;
import java.util.Map;

/**
 * Per-tree wavelet index with CPU fallback.
 * Keeps wavelet matrices for one anchor tree i to all j at a time: O(k * n log n) memory.
 */
public final class PerTreeWaveletIntersectionCounter implements IntersectionCounter {
    private final PreprocessedGeneTrees prep;
    private final CpuIntersectionCounter fallback;

    private int anchorTree = -1;
    private WaveletMatrix[] anchorToTree; // indexed by tree j
    private final Map<Long, Integer> sharedUniverseCache = new HashMap<>();

    public PerTreeWaveletIntersectionCounter(PreprocessedGeneTrees prep) {
        this.prep = prep;
        this.fallback = new CpuIntersectionCounter(prep);
    }

    @Override
    public int intersectionSize(Cluster a, Cluster b) {
        if (a.allTaxa) return clusterSize(b);
        if (b.allTaxa) return clusterSize(a);

        int localA = localSize(a);
        int localB = localSize(b);
        int localAB = localIntersection(a, b);

        if (!a.globalComplement && !b.globalComplement) {
            return localAB;
        }
        if (a.globalComplement && !b.globalComplement) {
            return localB - localAB;
        }
        if (!a.globalComplement && b.globalComplement) {
            return localA - localAB;
        }
        return prep.totalTaxa - localA - localB + localAB;
    }

    @Override
    public int clusterSize(Cluster c) {
        if (c.allTaxa) {
            return prep.totalTaxa;
        }
        int local = localSize(c);
        return c.globalComplement ? prep.totalTaxa - local : local;
    }

    private int localSize(Cluster c) {
        int range = c.right - c.left + 1;
        if (!c.localComplement) {
            return range;
        }
        return prep.treeInfos.get(c.sourceTreeIndex).presentTaxaCount - range;
    }

    private int localIntersection(Cluster a, Cluster b) {
        if (!a.localComplement && !b.localComplement) {
            return rangeRange(a.sourceTreeIndex, a.left, a.right, b.sourceTreeIndex, b.left, b.right);
        }

        if (a.localComplement && !b.localComplement) {
            int uiRj = universeRangeIntersection(a.sourceTreeIndex, b);
            int rr = rangeRange(a.sourceTreeIndex, a.left, a.right, b.sourceTreeIndex, b.left, b.right);
            return uiRj - rr;
        }

        if (!a.localComplement && b.localComplement) {
            int riUj = rangeUniverseIntersection(a, b.sourceTreeIndex);
            int rr = rangeRange(a.sourceTreeIndex, a.left, a.right, b.sourceTreeIndex, b.left, b.right);
            return riUj - rr;
        }

        int uiUj = universeUniverseIntersection(a.sourceTreeIndex, b.sourceTreeIndex);
        int riUj = rangeUniverseIntersection(a, b.sourceTreeIndex);
        int uiRj = universeRangeIntersection(a.sourceTreeIndex, b);
        int rr = rangeRange(a.sourceTreeIndex, a.left, a.right, b.sourceTreeIndex, b.left, b.right);
        return uiUj - riUj - uiRj + rr;
    }

    private int universeRangeIntersection(int treeI, Cluster rangeJ) {
        TreePreprocessInfo ti = prep.treeInfos.get(treeI);
        return rangeRange(treeI, 0, ti.presentTaxaCount - 1, rangeJ.sourceTreeIndex, rangeJ.left, rangeJ.right);
    }

    private int rangeUniverseIntersection(Cluster rangeI, int treeJ) {
        TreePreprocessInfo tj = prep.treeInfos.get(treeJ);
        return rangeRange(rangeI.sourceTreeIndex, rangeI.left, rangeI.right, treeJ, 0, tj.presentTaxaCount - 1);
    }

    private int universeUniverseIntersection(int i, int j) {
        if (i == j) {
            return prep.treeInfos.get(i).presentTaxaCount;
        }
        long k = pairKey(i, j);
        Integer cached = sharedUniverseCache.get(k);
        if (cached != null) {
            return cached;
        }
        TreePreprocessInfo ti = prep.treeInfos.get(i);
        TreePreprocessInfo tj = prep.treeInfos.get(j);
        int cnt = rangeRange(i, 0, ti.presentTaxaCount - 1, j, 0, tj.presentTaxaCount - 1);
        sharedUniverseCache.put(k, cnt);
        return cnt;
    }

    private int rangeRange(int i, int l1, int r1, int j, int l2, int r2) {
        if (l1 > r1 || l2 > r2) {
            return 0;
        }

        if (i == j) {
            int lo = Math.max(l1, l2);
            int hi = Math.min(r1, r2);
            return Math.max(0, hi - lo + 1);
        }

        // If this range is tiny, CPU direct scan is often faster and avoids cache churn.
        if (r1 - l1 + 1 <= 8) {
            return cpuRangeRange(i, l1, r1, j, l2, r2);
        }

        try {
            ensureAnchor(i);
            WaveletMatrix wm = anchorToTree[j];
            if (wm == null) {
                return cpuRangeRange(i, l1, r1, j, l2, r2);
            }
            return wm.rangeFreq(l1, r1 + 1, l2 + 1, r2 + 2);
        } catch (RuntimeException ex) {
            return cpuRangeRange(i, l1, r1, j, l2, r2);
        }
    }

    private int cpuRangeRange(int i, int l1, int r1, int j, int l2, int r2) {
        int cnt = 0;
        TreePreprocessInfo ti = prep.treeInfos.get(i);
        TreePreprocessInfo tj = prep.treeInfos.get(j);
        for (int p = l1; p <= r1; p++) {
            int taxon = ti.taxaByPostorderLeaf[p];
            if (taxon < 0) continue;
            int q = tj.positionByTaxon[taxon];
            if (q >= l2 && q <= r2) cnt++;
        }
        return cnt;
    }

    private void ensureAnchor(int treeIndex) {
        if (anchorTree == treeIndex && anchorToTree != null) {
            return;
        }

        int k = prep.treeInfos.size();
        anchorToTree = new WaveletMatrix[k];
        anchorTree = treeIndex;

        TreePreprocessInfo anchorInfo = prep.treeInfos.get(treeIndex);
        int anchorLen = anchorInfo.presentTaxaCount;

        for (int j = 0; j < k; j++) {
            if (j == treeIndex) {
                continue;
            }
            TreePreprocessInfo tj = prep.treeInfos.get(j);
            int[] mapped = new int[anchorLen];
            for (int p = 0; p < anchorLen; p++) {
                int taxon = anchorInfo.taxaByPostorderLeaf[p];
                int posJ = tj.positionByTaxon[taxon];
                mapped[p] = posJ + 1; // missing -> 0
            }
            // Value range: 0..presentTaxaCount(j)+1
            int maxValueExclusive = tj.presentTaxaCount + 2;
            anchorToTree[j] = new WaveletMatrix(mapped, maxValueExclusive);
        }
    }

    private static long pairKey(int a, int b) {
        int x = Math.min(a, b);
        int y = Math.max(a, b);
        return (((long) x) << 32) ^ (long) y;
    }
}
