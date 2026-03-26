package astralx.dp;

import astralx.cluster.Cluster;
import astralx.model.SpeciesNode;
import astralx.partition.PartitionTable;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.util.ClusterOps;
import astralx.weight.QuartetWeightCalculator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class InferenceDP {
    private final PreprocessedGeneTrees prep;
    private final Map<Cluster, List<CandidateSplit>> searchSpace;
    private final QuartetWeightCalculator weightCalculator;
    private final PartitionTable partitions;

    private final Map<Integer, Double> memoScore = new HashMap<>();
    private final Map<Integer, CandidateSplit> bestSplit = new HashMap<>();
    private final int totalStates;
    private int solvedStates = 0;
    private long lastProgressLogNs = System.nanoTime();

    public InferenceDP(PreprocessedGeneTrees prep,
                       Map<Cluster, List<CandidateSplit>> searchSpace,
                       QuartetWeightCalculator weightCalculator,
                       PartitionTable partitions) {
        this.prep = prep;
        this.searchSpace = searchSpace;
        this.weightCalculator = weightCalculator;
        this.partitions = partitions;
        this.totalStates = searchSpace.size();
    }

    public SpeciesNode infer(Cluster root) {
        solve(root);
        return buildTree(root);
    }

    private double solve(Cluster c) {
        Double cached = memoScore.get(c.id);
        if (cached != null) {
            return cached;
        }

        if (c.size <= 1) {
            memoScore.put(c.id, 0.0);
            return 0.0;
        }

        List<CandidateSplit> candidates = searchSpace.get(c);
        if (candidates == null || candidates.isEmpty()) {
            memoScore.put(c.id, Double.NEGATIVE_INFINITY);
            return Double.NEGATIVE_INFINITY;
        }

        double best = Double.NEGATIVE_INFINITY;
        CandidateSplit bestCand = null;

        for (CandidateSplit split : candidates) {
            double leftScore = solve(split.left);
            double rightScore = solve(split.right);
            if (!Double.isFinite(leftScore) || !Double.isFinite(rightScore)) {
                continue;
            }
            double ownScore = (c.allTaxa ? 0.0 : weightCalculator.score(split, c, partitions));
            double total = leftScore + rightScore + ownScore;
            if (total > best) {
                best = total;
                bestCand = split;
            }
        }

        memoScore.put(c.id, best);
        if (bestCand != null) {
            bestSplit.put(c.id, bestCand);
        }
        solvedStates++;
        maybeLogProgress();
        return best;
    }

    private void maybeLogProgress() {
        long now = System.nanoTime();
        if (solvedStates % 100 == 0 || now - lastProgressLogNs >= 2_000_000_000L) {
            System.out.printf("DP progress: %d/%d states solved%n", solvedStates, totalStates);
            System.out.flush();
            lastProgressLogNs = now;
        }
    }

    private SpeciesNode buildTree(Cluster c) {
        if (c.size == 1) {
            int t = ClusterOps.firstTaxon(c, prep);
            if (t < 0) {
                throw new IllegalStateException("Singleton cluster has no taxon: " + c);
            }
            return SpeciesNode.leaf(t);
        }

        CandidateSplit split = bestSplit.get(c.id);
        if (split == null) {
            int t = ClusterOps.firstTaxon(c, prep);
            if (t >= 0) {
                return SpeciesNode.leaf(t);
            }
            throw new IllegalStateException("No DP split for cluster: " + c);
        }

        SpeciesNode left = buildTree(split.left);
        SpeciesNode right = buildTree(split.right);
        return SpeciesNode.internal(left, right);
    }
}
