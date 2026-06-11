package astralx;

import astralx.cluster.ClusterTable;
import astralx.completion.DistanceMatrix;
import astralx.completion.DistanceMatrixBuilder;
import astralx.completion.SimilarityMatrix;
import astralx.completion.SimilarityMatrixBuilder;
import astralx.completion.TreeCompleter;
import astralx.completion.UPGMAClusterer;
import astralx.gpu.GPUDistanceMatrix;
import astralx.gpu.GPUSimilarityMatrix;
import astralx.dp.DPTable;
import astralx.dp.Inference;
import astralx.greedy.EmissionBridge;
import astralx.greedy.GreedyConsensus;
import astralx.greedy.GreedyConsensusVerifier;
import astralx.gpu.GPUDPBuilder;
import astralx.gpu.GPUWeightCalculator;
import astralx.partition.PartitionTable;
import astralx.weight.WeightTable;
import astralx.hash.PrefixHashArrays;
import astralx.hash.TaxonHasher;
import astralx.taxon.TaxonRegistry;
import astralx.tree.Tree;
import astralx.tree.TreeParser;
import astralx.util.Threading;

import astralx.cluster.Cluster;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main {
    public static final String VERSION = "0.1.0";

    public static void main(String[] args) throws Exception {
        Config cfg = Config.getInstance();
        if (!parseArgs(args, cfg)) { printUsage(); System.exit(1); }

        Logging.setLevel(cfg.getVerbosity());
        Banner.print(cfg);

        Threading.start(cfg.getThreadCount());
        long t0 = System.nanoTime();

        try {
            // ── Phase 1: Parse gene trees ─────────────────────────────────────
            long t1 = PhaseLogger.begin("Phase 1  Parse gene trees", false);
            TaxonRegistry registry = new TaxonRegistry();
            List<Tree> trees = TreeParser.parseGeneTrees(cfg.getInputFile(), registry);
            PhaseLogger.end("Phase 1  Parse gene trees", t1, false);

            if (cfg.isVerifyParse()) {
                Phase1Verifier.dump(trees, registry, cfg.getOutputFile());
                return;
            }

            // ── Phase 1b: Auto-complete incomplete gene trees (optional) ──────
            // Entered only when --autocomplete-incomplete-gene-trees or
            // --verify-distance-matrix is explicitly requested.  The baseline
            // (complete trees, no flag) skips this block entirely — no library
            // load, no stream scan, zero overhead.
            //
            // IMPORTANT: originalTrees is saved BEFORE completion and is used for
            // tripartition extraction (Phase 4) and weight calculation (Phase 6).
            // X (ClusterTable) and DP transitions are built from the completed trees
            // so that all bipartitions span the full taxon set — exactly what ASTRAL-MP
            // does.  Weight calculation must use the ORIGINAL gene trees (as ASTRAL-MP
            // does via inference.trees = originalInompleteGeneTrees) so the QI scores
            // reflect actual gene-tree signal, not the artificially inserted taxa.
            List<Tree> originalTrees = trees; // always points to pre-completion trees
            SimilarityMatrix similarityMatrix = null; // visible to Phase 3.5 (Step A)
            if (cfg.isAutoCompleteIncompleteTrees() || cfg.isVerifyDistanceMatrix()
                    || cfg.isVerifySimilarityMatrix() || cfg.isVerifyUpgma()) {
                boolean gpuDist = (cfg.getComputeMode() == Config.ComputeMode.GPU)
                                  && GPUDistanceMatrix.tryLoad();
                boolean gpuSim  = (cfg.getComputeMode() == Config.ComputeMode.GPU)
                                  && GPUSimilarityMatrix.tryLoad();

                if (cfg.isVerifyDistanceMatrix()) {
                    DistanceMatrix dm = gpuDist
                        ? DistanceMatrixBuilder.buildGPU(trees, registry.size())
                        : DistanceMatrixBuilder.buildCPU(trees, registry.size());
                    dumpDistanceMatrix(dm, registry);
                    return;
                }

                if (cfg.isVerifySimilarityMatrix()) {
                    SimilarityMatrix sm = gpuSim
                        ? SimilarityMatrixBuilder.buildGPU(trees, registry.size())
                        : SimilarityMatrixBuilder.buildCPU(trees, registry.size());
                    dumpSimilarityMatrix(sm, registry);
                    return;
                }

                if (cfg.isVerifyUpgma()) {
                    SimilarityMatrix sm = gpuSim
                        ? SimilarityMatrixBuilder.buildGPU(trees, registry.size())
                        : SimilarityMatrixBuilder.buildCPU(trees, registry.size());
                    int n = registry.size();
                    Tree upgmaTree = UPGMAClusterer.build(sm.sim, n, trees.size());
                    dumpUpgmaBipartitions(upgmaTree, registry);
                    return;
                }

                long incompleteCount = trees.stream().filter(t -> !t.isComplete).count();
                boolean useSim = cfg.getCompletionMethod() == Config.CompletionMethod.SIMILARITY;
                boolean gpuActive = useSim ? gpuSim : gpuDist;
                long t1b = PhaseLogger.begin("Phase 1b Auto-complete gene trees ("
                    + (useSim ? "similarity" : "distance") + ") + UPGMA guide", gpuActive);

                // Always build the similarity matrix: needed for UPGMA guide tree regardless of
                // completion method or whether any trees are actually incomplete.
                SimilarityMatrix smForUpgma = gpuSim
                    ? SimilarityMatrixBuilder.buildGPU(trees, registry.size())
                    : SimilarityMatrixBuilder.buildCPU(trees, registry.size());
                similarityMatrix = smForUpgma; // retained for Phase 3.5 (greedy consensus Step A)

                if (incompleteCount > 0) {
                    // The four-point algorithm always needs the similarity matrix for the
                    // scoring formula (sim[x][a] + sim[b][c] - ...).  smForUpgma.sim is
                    // already available regardless of completionMethod.
                    // dist is used only to build sortedRows (nearest-neighbour order);
                    // for SIMILARITY mode we reuse smForUpgma.dist (= 1 - sim).
                    double[] completionSim  = smForUpgma.sim;
                    double[] completionDist;
                    if (useSim) {
                        completionDist = smForUpgma.dist;   // reuse already-built matrix
                    } else {
                        DistanceMatrix dm = gpuDist
                            ? DistanceMatrixBuilder.buildGPU(trees, registry.size())
                            : DistanceMatrixBuilder.buildCPU(trees, registry.size());
                        completionDist = dm.dist;
                    }
                    // originalTrees already saved above; trees is reassigned to completed list
                    trees = TreeCompleter.completeAll(trees, completionSim, completionDist, registry.size());
                    Logging.info("Phase 1b: using original incomplete trees for weight scoring, completed trees for X");

                    if (cfg.getDumpCompletedTreesFile() != null) {
                        dumpCompletedTrees(trees, registry, cfg.getDumpCompletedTreesFile());
                    }
                } else {
                    Logging.info("Phase 1b: all gene trees already complete");
                }

                // Build UPGMA guide tree from similarity matrix and append to the completed
                // trees list so Phase 2/3 include its bipartitions in the cluster set X.
                // It is NOT added to originalTrees, so tripartition scoring (Phase 4/6)
                // is unaffected.
                int nTaxa = registry.size();
                Tree upgmaGuideTree = UPGMAClusterer.build(smForUpgma.sim, nTaxa, trees.size());
                trees = new ArrayList<>(trees);
                trees.add(upgmaGuideTree);
                Logging.info("Phase 1b: UPGMA guide tree (%d taxa) added to cluster search space", nTaxa);

                PhaseLogger.end("Phase 1b Auto-complete gene trees", t1b, gpuActive);
            }
            // After Phase 1b:
            //   trees         = completed gene trees (or original if no autocomplete / no incomplete)
            //   originalTrees = original gene trees (same reference as trees when no autocomplete)

            // ── Phase 2: Taxon hashing + prefix arrays ────────────────────────
            // pref     — built from completed trees; used for ClusterTable and DPTable
            // prefParts — built from original trees; used for PartitionTable (tripartition scoring)
            // When no autocomplete (originalTrees == trees), prefParts == pref (same object).
            long t2 = PhaseLogger.begin("Phase 2  Taxon hashing", false);
            TaxonHasher hasher = new TaxonHasher(
                registry.size(), cfg.getNumHashSeeds(), cfg.getBaseSeed());
            PrefixHashArrays pref = new PrefixHashArrays(trees, hasher);
            PrefixHashArrays prefParts = (originalTrees == trees)
                ? pref
                : new PrefixHashArrays(originalTrees, hasher);
            PhaseLogger.end("Phase 2  Taxon hashing", t2, false);

            if (cfg.isVerifyHash()) {
                Phase2Verifier.dump(trees, registry, hasher, pref, cfg.getOutputFile());
                return;
            }
            // hasher is retained — Phase 3.5 (greedy consensus) needs per-taxon
            // hashes to build consensus-tree prefix arrays whose signatures
            // match those derived from the gene-tree prefix arrays (cross-source
            // signature parity, design §7.2 / verification §13.5).

            // ── Phase 3: Cluster extraction -> X (from COMPLETED trees) ──────
            long t3 = PhaseLogger.begin("Phase 3  Cluster extraction", false);
            ClusterTable clusterTable = new ClusterTable(trees, pref, registry.size());
            PhaseLogger.end("Phase 3  Cluster extraction", t3, false);

            if (cfg.isVerifyClusters()) {
                Phase3Verifier.dump(trees, registry, pref, clusterTable, cfg.getOutputFile());
                return;
            }

            if (cfg.getDumpClustersFile() != null) {
                dumpClusters(clusterTable, trees, registry, cfg.getDumpClustersFile());
            }

            // ── Phase 3.6: Gene-tree polytomy X-enrichment (mechanism B, opt-in) ──
            // Resolve each INPUT gene-tree polytomy against the UPGMA guide tree into
            // arm-union (multi-range) clusters (ASTRAL-MP addBipartitionsFromSignleIndTreesToX).
            // Distinct from d-partition QI scoring; gated since it enlarges X.
            if (cfg.isResolveInputGeneTreePolytomies()
                    && anyPolytomous(trees, originalTrees.size())) {
                long t36 = PhaseLogger.begin("Phase 3.6 Gene-tree polytomy enrichment", false);
                int nT = registry.size();
                if (similarityMatrix == null) {
                    similarityMatrix = SimilarityMatrixBuilder.buildCPU(trees, nT);
                }
                Tree guide = UPGMAClusterer.build(similarityMatrix.sim, nT, trees.size());
                astralx.greedy.GeneTreePolytomySampler.run(
                    trees, originalTrees.size(), guide, pref, nT, pref.numSeeds(),
                    cfg.getBaseSeed() ^ 0xC0FFEEL, clusterTable);
                PhaseLogger.end("Phase 3.6 Gene-tree polytomy enrichment", t36, false);
            }

            // ── Phase 3.5: Greedy consensus + polytomy resolution (EXPERIMENTAL) ──
            // INCOMPLETE feature — emission of resolved polytomies into X is not
            // yet wired up.  Skipped entirely by default: no compute, no memory.
            // Enabled only via --consensus-experimental (build path) or the
            // explicit --verify-greedy-consensus entry point.
            //
            // Use ONLY the gene trees (no UPGMA guide tree) so the bipartition
            // frequencies match ASTRAL-MP's `addExtraBipartitionByHeuristics`
            // (which runs greedy consensus over the gene trees alone).
            //
            // `originalTrees.size()` is the gene-tree count regardless of
            // completion mode:
            //   - autocomplete OFF:  trees == originalTrees (no UPGMA appended)
            //   - autocomplete ON :  trees = completed gene trees + UPGMA
            //                        originalTrees still references the
            //                        pre-completion / pre-UPGMA list size.
            // Exemplar list the WEIGHT phase uses for cluster position lookups.
            // Defaults to the completed/gene trees; the consensus bridge may append
            // consensus snapshot trees (membership-only exemplars) for synthesized
            // multi-range clusters. The DP (Phase 5) keeps using `trees` directly,
            // so consensus trees are never mined for local transitions.
            List<Tree> weightClusterTrees = trees;
            if (cfg.isVerifyGreedyConsensus() || cfg.isConsensusExperimental()) {
                List<Tree> geneTreesForGreedy = trees.subList(0, originalTrees.size());

                if (cfg.isVerifyGreedyConsensus()) {
                    GreedyConsensusVerifier.dump(geneTreesForGreedy, registry, clusterTable,
                                                 pref, hasher, similarityMatrix, cfg.getOutputFile());
                    return;
                }
                long t35 = PhaseLogger.begin("Phase 3.5 Greedy consensus build + polytomy resolution", false);
                GreedyConsensus.Result gcResult =
                    GreedyConsensus.build(clusterTable, geneTreesForGreedy, pref, hasher,
                                           similarityMatrix, registry.size());
                PhaseLogger.end("Phase 3.5 Greedy consensus build + polytomy resolution", t35, false);

                // ── Bridge emissions into X (Tier-1 lookup / Tier-2 synthesize) ──
                List<Tree> ext = new ArrayList<>(trees);
                int[] bridged = EmissionBridge.bridge(gcResult.emissions, clusterTable,
                                                      ext, registry.size());
                // Only switch to the extended list if exemplar trees were actually
                // appended — preserves the `trees == originalTrees` identity that the
                // weight path's autocomplete detection relies on when nothing changed.
                if (ext.size() > trees.size()) weightClusterTrees = ext;
                Logging.info("Consensus emission → X: %d already in X (tier-1), "
                    + "%d synthesized multi-range (tier-2); +%d exemplar trees",
                    bridged[0], bridged[1], ext.size() - trees.size());
                if (cfg.getSearchMode() != Config.SearchMode.FULL && bridged[1] > 0) {
                    Logging.info("Note: synthesized multi-range clusters gain DP transitions only "
                        + "via Mode 2 (--search-mode full); in local mode they remain inert.");
                }
            }

            // ── Phase 4: Gene-tree tripartition extraction (from ORIGINAL trees) ──
            // Uses originalTrees so tripartitions reflect actual gene-tree signal.
            long t4 = PhaseLogger.begin("Phase 4  Tripartition extraction", false);
            PartitionTable partTable = new PartitionTable(originalTrees, prefParts);
            PhaseLogger.end("Phase 4  Tripartition extraction", t4, false);

            if (cfg.isVerifyPartitions()) {
                Phase4Verifier.dump(originalTrees, registry, prefParts, partTable, cfg.getOutputFile());
                return;
            }

            // ── Phase 5: DP search space (from COMPLETED trees) ───────────────
            long t5 = PhaseLogger.begin("Phase 5  DP local transitions", false);
            DPTable dpTable = new DPTable(trees, pref, clusterTable);
            PhaseLogger.end("Phase 5  DP local transitions", t5, false);

            // ── Phase 5b: Cross-tree transitions (Mode 2, optional) ───────────
            if (cfg.getSearchMode() == Config.SearchMode.FULL) {
                boolean gpuDP = (cfg.getComputeMode() == Config.ComputeMode.GPU)
                                && GPUDPBuilder.tryLoad();
                long t5b = PhaseLogger.begin("Phase 5b Cross-tree transitions", gpuDP);
                dpTable.addCrossTreeTransitions(clusterTable, gpuDP);
                PhaseLogger.end("Phase 5b Cross-tree transitions", t5b, gpuDP);
            }

            if (cfg.isVerifyDPSpace()) {
                Phase5Verifier.dump(trees, registry, pref, clusterTable, dpTable, cfg.getOutputFile());
                return;
            }
            pref = null;     // no longer needed after Phase 5; free before Phase 6
            prefParts = null; // likewise (may be same object as pref — both nulled safely)

            // Hint JVM to collect Phase 3-5 intermediates before Phase 6 allocates its working set.
            // System.gc() is a hint — JVM may ignore it if -XX:+DisableExplicitGC is set.
            long gcHeapBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            System.gc();
            long gcHeapAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            Logging.debug("Pre-Phase-6 GC hint: heap %d MB → %d MB (freed %d MB)",
                gcHeapBefore / 1_000_000, gcHeapAfter / 1_000_000,
                (gcHeapBefore - gcHeapAfter) / 1_000_000);

            // ── Phase 6: Weight calculation ───────────────────────────────────
            boolean gpuWeight = (cfg.getComputeMode() == Config.ComputeMode.GPU)
                                && GPUWeightCalculator.isLoaded();
            long t6 = PhaseLogger.begin("Phase 6  Weight calculation", gpuWeight);
            // weightClusterTrees = completed trees (+ any consensus exemplar trees from
            //                      the emission bridge) for cluster exemplar position lookups
            // originalTrees      = original trees (for gene-tree quartet scoring)
            WeightTable weightTable = new WeightTable(dpTable, partTable, clusterTable, weightClusterTrees, originalTrees);
            PhaseLogger.end("Phase 6  Weight calculation", t6, gpuWeight);

            if (cfg.isVerifyWeights()) {
                Phase6Verifier.dump(trees, registry, clusterTable, dpTable, weightTable, cfg.getOutputFile());
                return;
            }

            // ── Phase 7: Inference DP + tree reconstruction ───────────────────
            long t7 = PhaseLogger.begin("Phase 7  Inference", false);
            Inference inference = new Inference();
            String speciesTree = inference.run(dpTable, weightTable, clusterTable, trees, registry);
            PhaseLogger.end("Phase 7  Inference", t7, false);

            // Write or print the species tree
            if (cfg.getOutputFile() != null) {
                try (java.io.PrintStream out = new java.io.PrintStream(
                        new java.io.FileOutputStream(cfg.getOutputFile()))) {
                    out.println(speciesTree);
                }
                Logging.info("Species tree written to %s", cfg.getOutputFile());
            } else {
                System.out.println(speciesTree);
            }

        } finally {
            Threading.shutdown();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            Logging.info("Total time: %d ms", ms);
        }
    }

    private static boolean parseArgs(String[] args, Config cfg) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i","--input"    -> { if (++i>=args.length) return false; cfg.setInputFile(args[i]); }
                case "-o","--output"   -> { if (++i>=args.length) return false; cfg.setOutputFile(args[i]); }
                case "-t","--threads"  -> { if (++i>=args.length) return false; cfg.setThreadCount(Integer.parseInt(args[i])); }
                case "--cpu"           -> cfg.setComputeMode(Config.ComputeMode.CPU);
                case "--gpu"           -> cfg.setComputeMode(Config.ComputeMode.GPU);
                case "--search-mode"   -> {
                    if (++i >= args.length) return false;
                    cfg.setSearchMode(args[i].equalsIgnoreCase("full")
                        ? Config.SearchMode.FULL : Config.SearchMode.LOCAL);
                }
                case "--weight-intersection-method" -> {
                    if (++i >= args.length) return false;
                    String m = args[i].toLowerCase().replace('_', '-');
                    switch (m) {
                        case "prefix-sum", "prefixsum", "prefix" ->
                            cfg.setWeightIntersectionMethod(Config.WeightIntersectionMethod.PREFIX_SUM);
                        case "smaller-side-traversal", "smaller-side", "smallerside", "legacy" ->
                            cfg.setWeightIntersectionMethod(Config.WeightIntersectionMethod.SMALLER_SIDE_TRAVERSAL);
                        default -> {
                            System.err.println("Unknown --weight-intersection-method: " + args[i]
                                + "  (expected: prefix-sum | smaller-side-traversal)");
                            return false;
                        }
                    }
                }
                case "--large-n-score-type", "--large-score-type" -> {
                    if (++i >= args.length) return false;
                    String t = args[i].toLowerCase().replace('_', '-');
                    switch (t) {
                        case "int128", "i128", "exact" ->
                            cfg.setLargeScoreType(Config.LargeScoreType.INT128);
                        case "double", "fp64", "float" ->
                            cfg.setLargeScoreType(Config.LargeScoreType.DOUBLE);
                        default -> {
                            System.err.println("Unknown --large-n-score-type: " + args[i]
                                + "  (expected: int128 | double)");
                            return false;
                        }
                    }
                }
                case "-v"              -> cfg.setVerbosity(Logging.INFO);
                case "-vv"             -> cfg.setVerbosity(Logging.DEBUG);
                case "-vvv"            -> cfg.setVerbosity(Logging.TRACE);
                case "-q","--quiet"    -> cfg.setVerbosity(Logging.QUIET);
                case "-m","--seeds"    -> { if (++i>=args.length) return false; cfg.setNumHashSeeds(Integer.parseInt(args[i])); }
                case "--rooted"        -> cfg.setTreatAsUnrooted(false);
                case "--unrooted"      -> cfg.setTreatAsUnrooted(true);
                case "--no-gpu-batch"    -> cfg.setGpuBatch(false);
                case "--gpu-batch-size"  -> { if (++i>=args.length) return false; cfg.setGpuBatchSize(Integer.parseInt(args[i])); }
                case "--gpu-batches"     -> { if (++i>=args.length) return false; cfg.setGpuNumBatches(Integer.parseInt(args[i])); }
                case "--gpu-vram-control-factor"   -> { if (++i>=args.length) return false; cfg.setGpuVramControlFactor(Double.parseDouble(args[i])); }
                case "--gpu-vram-occupancy-factor" -> { if (++i>=args.length) return false; cfg.setGpuVramFraction(Double.parseDouble(args[i])); }
                case "--gpu-dp-state-space-construction-output-cap" -> { if (++i>=args.length) return false; cfg.setGpuDpStateSpaceConstructionOutputCap(args[i]); }
                case "--gpu-dp-state-space-progress-time-interval"  -> { if (++i>=args.length) return false; cfg.setGpuDpProgressInterval(Double.parseDouble(args[i])); }
                case "--gpu-dp-state-space-progress-max-steps"      -> { if (++i>=args.length) return false; cfg.setGpuDpProgressMaxSteps(Integer.parseInt(args[i])); }
                case "--verify-parse"  -> cfg.setVerifyParse(true);
                case "--verify-hash"      -> cfg.setVerifyHash(true);
                case "--verify-clusters"    -> cfg.setVerifyClusters(true);
                case "--verify-partitions" -> cfg.setVerifyPartitions(true);
                case "--verify-dp"         -> cfg.setVerifyDPSpace(true);
                case "--verify-weights"    -> cfg.setVerifyWeights(true);
                case "--verify-distance-matrix"    -> cfg.setVerifyDistanceMatrix(true);
                case "--verify-similarity-matrix"  -> cfg.setVerifySimilarityMatrix(true);
                case "--verify-upgma"              -> cfg.setVerifyUpgma(true);
                case "--verify-greedy-consensus"   -> cfg.setVerifyGreedyConsensus(true);
                case "--consensus-experimental"    -> cfg.setConsensusExperimental(true);
                case "--stepb-restriction"         -> {
                    if (++i >= args.length) return false;
                    String v = args[i].toLowerCase();
                    if (v.equals("dlogd") || v.equals("fast"))      cfg.setStepBFastRestriction(true);
                    else if (v.equals("n") || v.equals("full"))     cfg.setStepBFastRestriction(false);
                    else { System.err.println("--stepb-restriction expects dlogd|n"); return false; }
                }
                case "--stepb-quadratic-nn-balls"          -> cfg.setStepBQuadraticNnBalls(true);
                case "--stepb-random-leftover-resolution"  -> cfg.setStepBRandomLeftoverResolution(true);
                case "--stepb-process-large-polytomies"    -> cfg.setStepBProcessLargePolytomies(true);
                case "--resolve-input-gene-tree-polytomies" -> cfg.setResolveInputGeneTreePolytomies(true);
                case "--autocomplete-incomplete-gene-trees" -> cfg.setAutoCompleteIncompleteTrees(true);
                case "--completion-method" -> {
                    if (++i >= args.length) return false;
                    cfg.setCompletionMethod(args[i].equalsIgnoreCase("distance")
                        ? Config.CompletionMethod.DISTANCE : Config.CompletionMethod.SIMILARITY);
                }
                case "--dump-clusters"         -> { if (++i>=args.length) return false; cfg.setDumpClustersFile(args[i]); }
                case "--dump-completed-gene-trees" -> { if (++i>=args.length) return false; cfg.setDumpCompletedTreesFile(args[i]); }
                case "--gpu-dist-tile-size" -> { if (++i>=args.length) return false; cfg.setGpuDistTileSizeB(Integer.parseInt(args[i])); }
                case "-h","--help"     -> { printUsage(); System.exit(0); }
                default -> { System.err.println("Unknown arg: " + args[i]); return false; }
            }
        }
        return cfg.getInputFile() != null;
    }

    /**
     * Print distance matrix to stdout in a machine-parseable format:
     *   DISTANCE_MATRIX
     *   n=<count>
     *   taxa=name0,name1,...
     *   row0=d00,d01,...
     *   row1=d10,d11,...
     *   ...
     */
    private static void dumpDistanceMatrix(astralx.completion.DistanceMatrix dm,
                                            TaxonRegistry registry) {
        int n = dm.n;
        StringBuilder taxa = new StringBuilder("taxa=");
        for (int i = 0; i < n; i++) {
            if (i > 0) taxa.append(',');
            taxa.append(registry.getName(i));
        }
        System.out.println("DISTANCE_MATRIX");
        System.out.println("n=" + n);
        System.out.println(taxa);
        for (int i = 0; i < n; i++) {
            StringBuilder row = new StringBuilder("row").append(i).append('=');
            for (int j = 0; j < n; j++) {
                if (j > 0) row.append(',');
                double d = dm.dist[i * n + j];
                if (d == Double.MAX_VALUE) row.append("inf");
                else row.append(String.format("%.6f", d));
            }
            System.out.println(row);
        }
    }

    /**
     * Print similarity matrix to stdout in the same machine-parseable format as
     * dumpDistanceMatrix — just with SIMILARITY_MATRIX header and sim_rowN keys.
     */
    private static void dumpSimilarityMatrix(astralx.completion.SimilarityMatrix sm,
                                              TaxonRegistry registry) {
        int n = sm.n;
        StringBuilder taxa = new StringBuilder("taxa=");
        for (int i = 0; i < n; i++) {
            if (i > 0) taxa.append(',');
            taxa.append(registry.getName(i));
        }
        System.out.println("SIMILARITY_MATRIX");
        System.out.println("n=" + n);
        System.out.println(taxa);
        for (int i = 0; i < n; i++) {
            StringBuilder row = new StringBuilder("sim_row").append(i).append('=');
            for (int j = 0; j < n; j++) {
                if (j > 0) row.append(',');
                row.append(String.format("%.8f", sm.getSim(i, j)));
            }
            System.out.println(row);
        }
    }

    /**
     * Dump UPGMA bipartitions to stdout.
     *
     * Format:
     *   UPGMA_BIPARTITIONS
     *   n=<count>
     *   taxa=name0,name1,...
     *   bipartition=nameA,nameB,...   (one line per internal non-root node, names sorted)
     *
     * Each bipartition is the sorted set of taxon names in that subtree.
     * Root (all-taxa) is skipped automatically (rangeSize == n).
     */
    /** True iff any of the first {@code numGeneTrees} trees contains a polytomous node. */
    private static boolean anyPolytomous(java.util.List<Tree> trees, int numGeneTrees) {
        for (int g = 0; g < numGeneTrees && g < trees.size(); g++) {
            if (hasPolytomousNode(trees.get(g).root)) return true;
        }
        return false;
    }

    private static boolean hasPolytomousNode(astralx.tree.TreeNode node) {
        if (node == null || node.isLeaf()) return false;
        if (node.isPolytomous()) return true;
        return hasPolytomousNode(node.left) || hasPolytomousNode(node.right);
    }

    private static void dumpUpgmaBipartitions(Tree upgmaTree, TaxonRegistry registry) {
        int n = registry.size();
        StringBuilder taxaLine = new StringBuilder("taxa=");
        for (int i = 0; i < n; i++) {
            if (i > 0) taxaLine.append(',');
            taxaLine.append(registry.getName(i));
        }
        System.out.println("UPGMA_BIPARTITIONS");
        System.out.println("n=" + n);
        System.out.println(taxaLine);

        // Post-order walk; print subtree leaf set for every internal non-root node
        dumpUpgmaNode(upgmaTree.root, upgmaTree, registry, n);
    }

    private static void dumpUpgmaNode(astralx.tree.TreeNode node, Tree tree,
                                       TaxonRegistry registry, int n) {
        if (node.isLeaf()) return;
        dumpUpgmaNode(node.left,  tree, registry, n);
        dumpUpgmaNode(node.right, tree, registry, n);
        if (node.isRoot()) return;  // skip all-taxa bipartition

        // Collect and sort taxon names in [rangeStart, rangeEnd)
        java.util.List<String> names = new java.util.ArrayList<>(node.rangeSize());
        for (int pos = node.rangeStart; pos < node.rangeEnd; pos++) {
            names.add(registry.getName(tree.postorderArray[pos]));
        }
        java.util.Collections.sort(names);
        System.out.println("bipartition=" + String.join(",", names));
    }

    /**
     * Dump completed gene trees to a file, one Newick per line.
     * Ordering matches the original input gene tree order.
     */
    static void dumpCompletedTrees(List<Tree> trees, TaxonRegistry registry,
                                    String outFile) throws IOException {
        try (PrintStream out = new PrintStream(new FileOutputStream(outFile))) {
            for (Tree t : trees) {
                out.println(t.toNewick(registry));
            }
        }
        Logging.info("Completed gene trees written to %s (%d trees)", outFile, trees.size());
    }

    /**
     * Dump all clusters in ClusterTable to a file in canonical sorted format.
     * Each line: {A,B,C} with taxon names sorted alphabetically, lines sorted lexicographically.
     * This format matches the ASTRAL-MP --dump-clusters output for head-to-head comparison.
     */
    static void dumpClusters(ClusterTable clusterTable, List<Tree> trees,
                              TaxonRegistry registry, String outFile) throws IOException {
        List<String> lines = new ArrayList<>(clusterTable.size());
        for (ClusterTable.Entry e : clusterTable.entries()) {
            Cluster ex = e.exemplar;
            Tree tree = trees.get(ex.treeIndex);
            int[] arr = tree.postorderArray;
            List<String> taxa = new ArrayList<>(e.hash.size);
            if (!ex.complement) {
                for (int i = ex.left; i < ex.right; i++)
                    taxa.add(registry.getName(arr[i]));
            } else {
                for (int i = 0; i < tree.leafCount; i++) {
                    if (i >= ex.left && i < ex.right) continue;
                    taxa.add(registry.getName(arr[i]));
                }
            }
            Collections.sort(taxa);
            lines.add("{" + String.join(",", taxa) + "}");
        }
        Collections.sort(lines);
        try (PrintStream out = new PrintStream(new FileOutputStream(outFile))) {
            for (String line : lines) out.println(line);
        }
        Logging.info("Cluster dump written to %s (%d entries)", outFile, lines.size());
    }

    private static void printUsage() {
        System.err.println("ASTRAL-X v" + VERSION);
        System.err.println("Usage: astralx -i <input.tre> [-o <out>] [options]");
        System.err.println("  --verify-parse   dump Phase-1 output and exit");
        System.err.println("  --verify-hash    dump Phase-2 output and exit");
        System.err.println("  -v/-vv/-vvv      verbosity levels");
    }
}
