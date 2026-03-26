package astralx.cluster;

import astralx.hash.ClusterHashVector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class ClusterTable {
    public static final class Entry {
        public final Cluster representative;
        public int frequency;

        public Entry(Cluster representative) {
            this.representative = representative;
            this.frequency = 1;
        }
    }

    private final Map<ClusterHashVector, Entry> byHash = new HashMap<>();
    private final List<Cluster> uniqueClusters = new ArrayList<>();

    public Cluster upsert(Cluster cluster) {
        Entry e = byHash.get(cluster.hash);
        if (e == null) {
            byHash.put(cluster.hash, new Entry(cluster));
            uniqueClusters.add(cluster);
            return cluster;
        }
        e.frequency++;
        return e.representative;
    }

    public Cluster findByHash(ClusterHashVector hash) {
        Entry e = byHash.get(hash);
        return e == null ? null : e.representative;
    }

    public List<Cluster> uniqueClusters() {
        return uniqueClusters;
    }

    public int frequency(Cluster cluster) {
        Entry e = byHash.get(cluster.hash);
        return e == null ? 0 : e.frequency;
    }
}
