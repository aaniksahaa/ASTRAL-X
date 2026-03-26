package astralx.util;

import astralx.cluster.Cluster;

public interface IntersectionCounter {
    int intersectionSize(Cluster a, Cluster b);
    int clusterSize(Cluster c);
}
