package astralx.partition;

import astralx.cluster.Cluster;
import astralx.hash.ClusterHashVector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

final class PartitionKey {
    private final boolean universeIsGlobal;
    private final int universeTreeIndex;
    private final List<ClusterHashVector> clusterHashes;

    private PartitionKey(boolean universeIsGlobal, int universeTreeIndex, List<ClusterHashVector> clusterHashes) {
        this.universeIsGlobal = universeIsGlobal;
        this.universeTreeIndex = universeTreeIndex;
        this.clusterHashes = clusterHashes;
    }

    static PartitionKey from(Partition partition) {
        List<ClusterHashVector> hs = new ArrayList<>();
        for (Cluster c : partition.nonTrivialClusters) {
            hs.add(c.hash);
        }
        hs.sort(Comparator.comparingInt(Object::hashCode));
        return new PartitionKey(partition.universeIsGlobal, partition.universeTreeIndex, hs);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof PartitionKey)) {
            return false;
        }
        PartitionKey other = (PartitionKey) obj;
        return universeIsGlobal == other.universeIsGlobal
                && universeTreeIndex == other.universeTreeIndex
                && clusterHashes.equals(other.clusterHashes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(universeIsGlobal, universeTreeIndex, clusterHashes);
    }
}
