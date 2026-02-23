package astralx.weight;

import astralx.cluster.Cluster;
import astralx.dp.CandidateSplit;
import astralx.partition.Partition;
import astralx.partition.PartitionTable;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.util.CpuIntersectionCounter;
import astralx.util.IntersectionCounter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class QuartetWeightCalculator {
    private final PreprocessedGeneTrees prep;
    private final IntersectionCounter intersections;
    private final ConcurrentMap<Long, Double> splitScoreCache = new ConcurrentHashMap<>();

    public QuartetWeightCalculator(PreprocessedGeneTrees prep, IntersectionCounter intersections) {
        this.prep = prep;
        this.intersections = intersections;
    }

    public double score(CandidateSplit split, Cluster parent, PartitionTable partitions) {
        long key = packKey(split.left.id, split.right.id, parent.id);
        Double cached = splitScoreCache.get(key);
        if (cached != null) {
            return cached;
        }
        double total = computeScore(split, parent, partitions, intersections);
        splitScoreCache.putIfAbsent(key, total);
        return total;
    }

    private double computeScore(CandidateSplit split, Cluster parent, PartitionTable partitions, IntersectionCounter counter) {
        Cluster x = split.left;
        Cluster y = split.right;
        double total = 0.0;

        for (PartitionTable.Entry e : partitions.entries()) {
            Partition p = e.representative;
            if (p.nonTrivialClusters.size() != 2) {
                continue;
            }
            Cluster a = p.nonTrivialClusters.get(0);
            Cluster b = p.nonTrivialClusters.get(1);
            int treeIndex = p.universeTreeIndex;
            Cluster lgt = geneAllCluster(treeIndex);

            int xA = counter.intersectionSize(x, a);
            int xB = counter.intersectionSize(x, b);
            int xL = counter.intersectionSize(x, lgt);
            int sizeA = a.size;
            int sizeB = b.size;
            int sizeL = lgt.size;
            int sizeC = sizeL - sizeA - sizeB;
            int xC = xL - xA - xB;

            int yA = counter.intersectionSize(y, a);
            int yB = counter.intersectionSize(y, b);
            int yL = counter.intersectionSize(y, lgt);
            int yC = yL - yA - yB;

            int zA = sizeA - xA - yA;
            int zB = sizeB - xB - yB;
            int zC = sizeC - xC - yC;

            double qi = qiTripartition(
                    xA, xB, xC,
                    yA, yB, yC,
                    zA, zB, zC
            );

            total += 0.5 * qi * e.frequency;
        }

        return total;
    }

    public void precomputeAll(Map<Cluster, java.util.List<CandidateSplit>> searchSpace, PartitionTable partitions) {
        long startNs = System.nanoTime();
        int total = 0;
        for (Map.Entry<Cluster, java.util.List<CandidateSplit>> e : searchSpace.entrySet()) {
            if (e.getKey().allTaxa) {
                continue;
            }
            total += e.getValue().size();
        }

        int done = 0;
        long lastLogNs = startNs;
        for (Map.Entry<Cluster, java.util.List<CandidateSplit>> e : searchSpace.entrySet()) {
            Cluster parent = e.getKey();
            if (parent.allTaxa) {
                continue;
            }
            for (CandidateSplit split : e.getValue()) {
                score(split, parent, partitions);
                done++;
                long now = System.nanoTime();
                if (done % 200 == 0 || now - lastLogNs >= 2_000_000_000L) {
                    System.out.printf("Weight precompute progress: %d/%d candidate bipartitions%n", done, total);
                    System.out.flush();
                    lastLogNs = now;
                }
            }
        }

        double seconds = (System.nanoTime() - startNs) / 1_000_000_000.0;
        System.out.printf("Weight precompute done in %.2fs (cached=%d)%n", seconds, splitScoreCache.size());
        System.out.flush();
    }

    public void precomputeAllCpuParallel(Map<Cluster, List<CandidateSplit>> searchSpace, PartitionTable partitions, int threads) {
        long startNs = System.nanoTime();
        List<Task> tasks = new ArrayList<>();
        for (Map.Entry<Cluster, List<CandidateSplit>> e : searchSpace.entrySet()) {
            Cluster parent = e.getKey();
            if (parent.allTaxa) {
                continue;
            }
            for (CandidateSplit split : e.getValue()) {
                tasks.add(new Task(parent, split));
            }
        }

        int total = tasks.size();
        if (total == 0) {
            return;
        }

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch latch = new CountDownLatch(threads);
        java.util.concurrent.atomic.AtomicInteger done = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicLong lastLogNs = new java.util.concurrent.atomic.AtomicLong(startNs);

        int chunk = Math.max(1, (total + threads - 1) / threads);
        for (int tid = 0; tid < threads; tid++) {
            final int from = tid * chunk;
            final int to = Math.min(total, from + chunk);
            pool.submit(() -> {
                try {
                    if (from >= to) {
                        return;
                    }
                    IntersectionCounter counter = new CpuIntersectionCounter(prep);
                    for (int i = from; i < to; i++) {
                        Task t = tasks.get(i);
                        long key = packKey(t.split.left.id, t.split.right.id, t.parent.id);
                        splitScoreCache.computeIfAbsent(key, ignored -> computeScore(t.split, t.parent, partitions, counter));

                        int curr = done.incrementAndGet();
                        long now = System.nanoTime();
                        long prev = lastLogNs.get();
                        if (curr % 500 == 0 || now - prev >= 2_000_000_000L) {
                            if (lastLogNs.compareAndSet(prev, now)) {
                                System.out.printf("Weight precompute progress: %d/%d candidate bipartitions%n", curr, total);
                                System.out.flush();
                            }
                        }
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Weight precompute interrupted", e);
        } finally {
            pool.shutdownNow();
        }

        double seconds = (System.nanoTime() - startNs) / 1_000_000_000.0;
        System.out.printf("Weight precompute (CPU_PARALLEL) done in %.2fs (cached=%d, threads=%d)%n",
                seconds, splitScoreCache.size(), threads);
        System.out.flush();
    }

    public int cachedWeightCount() {
        return splitScoreCache.size();
    }

    public void cachePrecomputedScore(CandidateSplit split, Cluster parent, double score) {
        long key = packKey(split.left.id, split.right.id, parent.id);
        splitScoreCache.put(key, score);
    }

    private Cluster geneAllCluster(int treeIndex) {
        int present = prep.treeInfos.get(treeIndex).presentTaxaCount;
        Cluster pseudo = new Cluster(
                -1000000 - treeIndex,
                treeIndex,
                0,
                present - 1,
                false,
                false,
                false,
                null,
                present
        );
        return pseudo;
    }

    private static double qiTripartition(
            int a1, int a2, int a3,
            int b1, int b2, int b3,
            int c1, int c2, int c3) {

        return term(a1, b2, c3)
                + term(a1, b3, c2)
                + term(a2, b1, c3)
                + term(a2, b3, c1)
                + term(a3, b1, c2)
                + term(a3, b2, c1);
    }

    private static double term(int a, int b, int c) {
        return ((a + b + c - 3) / 2.0) * a * b * c;
    }

    private static long packKey(int a, int b, int c) {
        long x = (((long) a) & 0x1FFFFFL) << 42;
        long y = (((long) b) & 0x1FFFFFL) << 21;
        long z = (((long) c) & 0x1FFFFFL);
        return x ^ y ^ z;
    }

    private static final class Task {
        final Cluster parent;
        final CandidateSplit split;

        Task(Cluster parent, CandidateSplit split) {
            this.parent = parent;
            this.split = split;
        }
    }
}
