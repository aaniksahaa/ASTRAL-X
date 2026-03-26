package astralx.dp;

import astralx.cluster.Cluster;
import astralx.hash.ClusterHashVector;
import astralx.preprocess.PreprocessedGeneTrees;

import java.io.File;
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

        for (Cluster a : pool) {
            result.put(a, new ArrayList<>());
        }

        if (isGpuLookupAvailable()) {
            try {
                buildWithGpuConstruction(pool, bins, prep, result);
            } catch (Exception ex) {
                System.out.println("GPU search-space construction failed: " + ex.getMessage() + "; falling back to CPU lookup.");
                System.out.flush();
                buildWithCpuLookup(pool, bins, bySizeByHash, prep, result);
            }
        } else {
            buildWithCpuLookup(pool, bins, bySizeByHash, prep, result);
        }

        double seconds = (System.nanoTime() - startNs) / 1_000_000_000.0;
        System.out.printf("Search-space build done in %.2fs%n", seconds);
        System.out.flush();
        return result;
    }

    private void buildWithCpuLookup(
            List<Cluster> pool,
            Map<Integer, List<Cluster>> bins,
            Map<Integer, Map<ClusterHashVector, List<Cluster>>> bySizeByHash,
            PreprocessedGeneTrees prep,
            Map<Cluster, List<CandidateSplit>> result) {

        int processed = 0;
        int total = pool.size();
        long lastLogNs = System.nanoTime();

        for (Cluster a : pool) {
            if (a.size <= 1) {
                processed++;
                continue;
            }

            List<CandidateSplit> splits = result.get(a);
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
                        tryAddSplit(a, b, c, prep, splits, seen);
                    }
                }
            }

            processed++;
            long now = System.nanoTime();
            if (processed % 100 == 0 || now - lastLogNs >= 2_000_000_000L) {
                System.out.printf("Search-space progress: %d/%d clusters processed%n", processed, total);
                System.out.flush();
                lastLogNs = now;
            }
        }
    }

    private void buildWithGpuConstruction(
            List<Cluster> pool,
            Map<Integer, List<Cluster>> bins,
            PreprocessedGeneTrees prep,
            Map<Cluster, List<CandidateSplit>> result) throws Exception {

        System.out.println("Search-space: using GPU construction backend");
        System.out.flush();

        int total = pool.size();
        long lastLogNs = System.nanoTime();

        long[] clusterKeys = new long[pool.size()];
        int[] clusterIds = new int[pool.size()];
        int words = (prep.totalTaxa + 63) / 64;
        long[] bitsets = new long[pool.size() * words];
        Map<Integer, Integer> poolIndexByClusterId = new HashMap<>();
        for (int i = 0; i < pool.size(); i++) {
            Cluster c = pool.get(i);
            clusterKeys[i] = signature(c.hash, c.size);
            clusterIds[i] = c.id;
            poolIndexByClusterId.put(c.id, i);
            fillBitset(c, prep, bitsets, i * words, words);
        }

        final class Query {
            final int aIdx;
            final int bIdx;
            Query(int aIdx, int bIdx) { this.aIdx = aIdx; this.bIdx = bIdx; }
        }

        List<Query> queries = new ArrayList<>();
        List<Long> queryKeys = new ArrayList<>();

        for (int aIdx = 0; aIdx < pool.size(); aIdx++) {
            Cluster a = pool.get(aIdx);
            if (a.size <= 1) {
                continue;
            }
            for (int sz = 1; sz <= a.size / 2; sz++) {
                List<Cluster> leftBin = bins.get(sz);
                if (leftBin == null) {
                    continue;
                }
                int rightSize = a.size - sz;
                for (Cluster b : leftBin) {
                    Integer bIdxObj = poolIndexByClusterId.get(b.id);
                    if (bIdxObj == null) {
                        continue;
                    }
                    int bIdx = bIdxObj;
                    ClusterHashVector rem = ClusterHashVector.subtract(a.hash, b.hash);
                    queries.add(new Query(aIdx, bIdx));
                    queryKeys.add(signature(rem, rightSize));
                }
            }
            long now = System.nanoTime();
            if (aIdx % 100 == 0 || now - lastLogNs >= 2_000_000_000L) {
                System.out.printf("Search-space query prep: %d/%d clusters%n", aIdx, total);
                System.out.flush();
                lastLogNs = now;
            }
        }

        long[] qk = new long[queryKeys.size()];
        for (int i = 0; i < queryKeys.size(); i++) qk[i] = queryKeys.get(i);

        int[] qA = new int[queries.size()];
        int[] qB = new int[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            qA[i] = queries.get(i).aIdx;
            qB[i] = queries.get(i).bIdx;
        }

        int outCap = Math.max(1, queries.size() * 4);
        GpuSearchSpaceBuildRunner.Result gpu = new GpuSearchSpaceBuildRunner()
                .run(clusterKeys, clusterIds, bitsets, words, qA, qB, qk, outCap);

        if (gpu.overflow) {
            throw new IllegalStateException("GPU split output overflow; increase output capacity or use chunked GPU build");
        }

        Map<Integer, Set<Long>> seenByA = new HashMap<>();
        for (int i = 0; i < gpu.triples.length; i += 3) {
            int aIdx = gpu.triples[i];
            int bIdx = gpu.triples[i + 1];
            int cIdx = gpu.triples[i + 2];
            Cluster a = pool.get(aIdx);
            Cluster b = pool.get(bIdx);
            Cluster c = pool.get(cIdx);
            List<CandidateSplit> splits = result.get(a);
            Set<Long> seen = seenByA.computeIfAbsent(aIdx, ignored -> new HashSet<>());
            Cluster left = b.id < c.id ? b : c;
            Cluster right = b.id < c.id ? c : b;
            long key = (((long) left.id) << 32) ^ (long) right.id;
            if (seen.add(key)) {
                splits.add(new CandidateSplit(left, right));
            }
        }

        System.out.printf("Search-space GPU construction complete (queries=%d, splits=%d)%n",
                queries.size(), gpu.triples.length / 3);
        System.out.flush();
    }

    private void tryAddSplit(Cluster a, Cluster b, Cluster c, PreprocessedGeneTrees prep,
                             List<CandidateSplit> splits, Set<Long> seen) {
        if (b.id == c.id) {
            return;
        }
        if (!isValidDecomposition(a, b, c, prep)) {
            return;
        }
        Cluster left = b.id < c.id ? b : c;
        Cluster right = b.id < c.id ? c : b;
        long key = (((long) left.id) << 32) ^ (long) right.id;
        if (seen.add(key)) {
            splits.add(new CandidateSplit(left, right));
        }
    }

    private boolean isGpuLookupAvailable() {
        File bin = new File("./src/cuda/astralx_search_space_build");
        return bin.exists() && bin.canExecute() && hasCommand("nvidia-smi");
    }

    private boolean hasCommand(String cmd) {
        try {
            Process p = new ProcessBuilder("bash", "-lc", "command -v " + cmd + " >/dev/null 2>&1").start();
            return p.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    private void fillBitset(Cluster c, PreprocessedGeneTrees prep, long[] bitsets, int base, int words) {
        for (int t = 0; t < prep.totalTaxa; t++) {
            if (c.containsTaxon(t, prep)) {
                int w = t >>> 6;
                int b = t & 63;
                bitsets[base + w] |= (1L << b);
            }
        }
    }

    private long signature(ClusterHashVector hash, int size) {
        long x = 0x9e3779b97f4a7c15L ^ ((long) size * 0x100000001b3L);
        for (int i = 0; i < hash.sumHash.length; i++) {
            long s = hash.sumHash[i];
            long y = hash.xorHash[i];
            x ^= mix64(s + 0x9e3779b97f4a7c15L * (i + 1));
            x ^= mix64(y + 0xbf58476d1ce4e5b9L * (i + 1));
            x = Long.rotateLeft(x, 7);
        }
        return x;
    }

    private long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        z = z ^ (z >>> 31);
        return z;
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
