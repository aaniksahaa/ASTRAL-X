package astralx.util;

import astralx.cluster.Cluster;
import astralx.preprocess.PreprocessedGeneTrees;

public final class ClusterOps {
    private ClusterOps() {}

    public static int intersectionSize(Cluster a, Cluster b, PreprocessedGeneTrees prep) {
        int n = prep.totalTaxa;
        int count = 0;
        for (int t = 0; t < n; t++) {
            if (a.containsTaxon(t, prep) && b.containsTaxon(t, prep)) {
                count++;
            }
        }
        return count;
    }

    public static int firstTaxon(Cluster c, PreprocessedGeneTrees prep) {
        for (int t = 0; t < prep.totalTaxa; t++) {
            if (c.containsTaxon(t, prep)) {
                return t;
            }
        }
        return -1;
    }
}
