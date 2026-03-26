package astralx.partition;

import astralx.cluster.Cluster;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class Partition {
    public final List<Cluster> nonTrivialClusters;
    public final boolean universeIsGlobal;
    public final int universeTreeIndex;

    public Partition(List<Cluster> nonTrivialClusters, boolean universeIsGlobal, int universeTreeIndex) {
        List<Cluster> sorted = new ArrayList<>(nonTrivialClusters);
        sorted.sort(Comparator.comparingInt(c -> c.id));
        this.nonTrivialClusters = sorted;
        this.universeIsGlobal = universeIsGlobal;
        this.universeTreeIndex = universeTreeIndex;
    }
}
