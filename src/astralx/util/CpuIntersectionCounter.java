package astralx.util;

import astralx.cluster.Cluster;
import astralx.preprocess.PreprocessedGeneTrees;

public final class CpuIntersectionCounter implements IntersectionCounter {
    private final PreprocessedGeneTrees prep;

    public CpuIntersectionCounter(PreprocessedGeneTrees prep) {
        this.prep = prep;
    }

    @Override
    public int intersectionSize(Cluster a, Cluster b) {
        int n = prep.totalTaxa;
        int count = 0;
        for (int t = 0; t < n; t++) {
            if (a.containsTaxon(t, prep) && b.containsTaxon(t, prep)) {
                count++;
            }
        }
        return count;
    }

    @Override
    public int clusterSize(Cluster c) {
        return c.size;
    }
}
