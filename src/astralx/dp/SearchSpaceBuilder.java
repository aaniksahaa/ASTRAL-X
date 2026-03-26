package astralx.dp;

import astralx.cluster.Cluster;
import astralx.hash.ClusterHashVector;
import astralx.preprocess.PreprocessedGeneTrees;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class SearchSpaceBuilder {
    public Map<Cluster, List<CandidateSplit>> build(List<Cluster> clusters, Cluster allTaxa, PreprocessedGeneTrees prep) {
        long startNs = System.nanoTime();
        Map<Integer, List<Cluster>> bins = new HashMap<>();
        Map<Integer, Map<ClusterHashVector, List<Cluster>>> bySizeByHash = new HashMap<>();

        List<Cluster> pool = new ArrayList<>(clusters);
        pool.add(allTaxa);

        for (Cluster c : pool) {
            if (c.size <= 0 || c.size > prep.totalTaxa) {
                continue;
            }
            bins.computeIfAbsent(c.size, ignored -> new ArrayList<>()).add(c);
            bySizeByHash.computeIfAbsent(c.size, ignored -> new HashMap<>())
                    .computeIfAbsent(c.hash, ignored -> new ArrayList<>())
                    .add(c);
        }

        Map<Cluster, List<CandidateSplit>> result = new HashMap<>();

        int processed = 0;
        int total = pool.size();
        long lastLogNs = startNs;
        for (Cluster a : pool) {
            if (a.size <= 1) {
                result.put(a, new ArrayList<>());
                processed++;
                continue;
            }
            List<CandidateSplit> splits = new ArrayList<>();
            Set<Long> seen = new HashSet<>();

            for (int sz = 1; sz <= a.size / 2; sz++) {
                List<Cluster> leftBin = bins.get(sz);
                if (leftBin == null) {
                    continue;
                }
                int rightSize = a.size - sz;
                Map<ClusterHashVector, List<Cluster>> rightMap = bySizeByHash.get(rightSize);
                if (rightMap == null) {
                    continue;
                }

                for (Cluster b : leftBin) {
                    ClusterHashVector remainingHash = ClusterHashVector.subtract(a.hash, b.hash);
                    List<Cluster> candidates = rightMap.get(remainingHash);
                    if (candidates == null) {
                        continue;
                    }
                    for (Cluster c : candidates) {
                        if (b.id == c.id) {
                            continue;
                        }
                        if (!isValidDecomposition(a, b, c, prep)) {
                            continue;
                        }
                        Cluster left = b.id < c.id ? b : c;
                        Cluster right = b.id < c.id ? c : b;
                        long key = (((long) left.id) << 32) ^ (long) right.id;
                        if (seen.add(key)) {
                            splits.add(new CandidateSplit(left, right));
                        }
                    }
                }
            }

            result.put(a, splits);
            processed++;
            long now = System.nanoTime();
            if (processed % 100 == 0 || now - lastLogNs >= 2_000_000_000L) {
                System.out.printf("Search-space progress: %d/%d clusters processed%n", processed, total);
                System.out.flush();
                lastLogNs = now;
            }
        }

        double seconds = (System.nanoTime() - startNs) / 1_000_000_000.0;
        System.out.printf("Search-space build done in %.2fs%n", seconds);
        System.out.flush();
        return result;
    }

    private boolean isValidDecomposition(Cluster a, Cluster b, Cluster c, PreprocessedGeneTrees prep) {
        int n = prep.totalTaxa;
        for (int t = 0; t < n; t++) {
            boolean inA = a.containsTaxon(t, prep);
            boolean inB = b.containsTaxon(t, prep);
            boolean inC = c.containsTaxon(t, prep);
            if (inB && inC) {
                return false;
            }
            if ((inB || inC) != inA) {
                return false;
            }
        }
        return true;
    }
}
