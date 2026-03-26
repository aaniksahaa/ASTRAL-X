package astralx.dp;

import astralx.cluster.Cluster;

public final class CandidateSplit {
    public final Cluster left;
    public final Cluster right;

    public CandidateSplit(Cluster left, Cluster right) {
        this.left = left;
        this.right = right;
    }
}
