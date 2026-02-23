package astralx;

import astralx.cli.CliParser;
import astralx.cli.Config;
import astralx.cluster.Cluster;
import astralx.cluster.ClusterExtractor;
import astralx.cluster.ClusterTable;
import astralx.dp.InferenceDP;
import astralx.dp.SearchSpaceBuilder;
import astralx.hash.PrefixHashIndex;
import astralx.hash.SeededTaxonHashes;
import astralx.model.SpeciesNode;
import astralx.parse.GeneTreeLoader;
import astralx.partition.PartitionTable;
import astralx.partition.TripartitionExtractor;
import astralx.preprocess.GeneTreePreprocessor;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.util.CpuIntersectionCounter;
import astralx.util.IntersectionCounter;
import astralx.util.NewickWriter;
import astralx.util.PerTreeWaveletIntersectionCounter;
import astralx.weight.GpuWeightPrecomputer;
import astralx.weight.QuartetWeightCalculator;

import java.nio.file.Files;
import java.nio.file.Path;

public final class Main {
    public static void main(String[] args) {
        try {
            long pipelineStart = System.nanoTime();
            Config cfg = CliParser.parse(args);
            System.out.println("ASTRAL-X run started");
            System.out.println("Input: " + cfg.inputPath);
            System.out.println("Output: " + cfg.outputPath);
            System.out.println("Intersection: " + cfg.intersectionMode);
            System.out.println("Weight mode: " + cfg.weightMode);
            System.out.flush();

            long t0 = System.nanoTime();
            System.out.println("[1/9] Loading gene trees...");
            System.out.flush();
            GeneTreeLoader.LoadedGeneTrees loaded = new GeneTreeLoader().load(cfg.inputPath);
            logStageDone(t0, "Loaded gene trees: " + loaded.trees.size() + ", taxa: " + loaded.taxa.size());

            t0 = System.nanoTime();
            System.out.println("[2/9] Preprocessing trees...");
            System.out.flush();
            PreprocessedGeneTrees prep = new GeneTreePreprocessor().preprocess(loaded.trees, loaded.taxa.size());
            logStageDone(t0, "Preprocessing done");

            t0 = System.nanoTime();
            System.out.println("[3/9] Building taxon and prefix hashes...");
            System.out.flush();
            SeededTaxonHashes taxonHashes = new SeededTaxonHashes(prep.totalTaxa, cfg.hashReplicates, cfg.randomSeed);
            PrefixHashIndex prefixHash = new PrefixHashIndex(prep, taxonHashes);
            logStageDone(t0, "Hash index built");

            t0 = System.nanoTime();
            System.out.println("[4/9] Extracting unique clusters...");
            System.out.flush();
            ClusterExtractor clusterExtractor = new ClusterExtractor();
            ClusterExtractor.Result clusterResult = clusterExtractor.build(prep, prefixHash, cfg.treatInputAsUnrooted);
            ClusterTable clusterTable = clusterResult.table;
            Cluster allTaxa = clusterResult.allTaxaCluster;
            logStageDone(t0, "Clusters ready: " + clusterTable.uniqueClusters().size());

            t0 = System.nanoTime();
            System.out.println("[5/9] Extracting unique tripartitions...");
            System.out.flush();
            TripartitionExtractor tripartitionExtractor = new TripartitionExtractor();
            PartitionTable tripartitions = tripartitionExtractor.extract(
                    prep,
                    prefixHash,
                    clusterExtractor,
                    clusterTable,
                    cfg.treatInputAsUnrooted,
                    false
            );
            logStageDone(t0, "Tripartitions ready: " + tripartitions.entries().size());

            t0 = System.nanoTime();
            System.out.println("[6/9] Building DP search space...");
            System.out.flush();
            SearchSpaceBuilder ssBuilder = new SearchSpaceBuilder();
            var searchSpace = ssBuilder.build(clusterTable.uniqueClusters(), allTaxa, prep);
            logStageDone(t0, "Search space states: " + searchSpace.size());

            IntersectionCounter counter = cfg.intersectionMode == Config.IntersectionMode.CPU
                    ? new CpuIntersectionCounter(prep)
                    : new PerTreeWaveletIntersectionCounter(prep);

            QuartetWeightCalculator weightCalc = new QuartetWeightCalculator(prep, counter);
            t0 = System.nanoTime();
            System.out.println("[7/9] Precomputing candidate bipartition weights...");
            System.out.flush();
            String weightBackend;
            if (cfg.weightMode == Config.WeightMode.GPU) {
                GpuWeightPrecomputer.Result r = new GpuWeightPrecomputer().precomputeOrFallback(
                        weightCalc,
                        searchSpace,
                        tripartitions,
                        prep,
                        Runtime.getRuntime().availableProcessors());
                weightBackend = r.backendLabel;
            } else {
                weightCalc.precomputeAllCpuParallel(searchSpace, tripartitions, Runtime.getRuntime().availableProcessors());
                weightBackend = "CPU_PARALLEL";
            }
            logStageDone(t0, "Weight precomputation done (cached=" + weightCalc.cachedWeightCount() + ", backend=" + weightBackend + ")");

            t0 = System.nanoTime();
            System.out.println("[8/9] Running inference DP...");
            System.out.flush();
            InferenceDP dp = new InferenceDP(prep, searchSpace, weightCalc, tripartitions);
            SpeciesNode inferred = dp.infer(allTaxa);
            logStageDone(t0, "Inference DP done");

            t0 = System.nanoTime();
            System.out.println("[9/9] Writing species tree...");
            System.out.flush();
            String outNewick = NewickWriter.toNewick(inferred, loaded.taxa);
            Files.writeString(Path.of(cfg.outputPath), outNewick + System.lineSeparator());
            logStageDone(t0, "Output written");

            System.out.println("Input gene trees: " + loaded.trees.size());
            System.out.println("Total taxa: " + loaded.taxa.size());
            System.out.println("Unique clusters in X: " + clusterTable.uniqueClusters().size());
            System.out.println("Unique tripartitions: " + tripartitions.entries().size());
            System.out.println("Intersection engine: " + cfg.intersectionMode);
            System.out.println("Weight precompute mode: " + cfg.weightMode);
            System.out.println("Wrote species tree to: " + cfg.outputPath);
            System.out.printf("Total runtime: %.2fs%n", (System.nanoTime() - pipelineStart) / 1_000_000_000.0);
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.err.println(CliParser.usage());
            System.exit(1);
        }
    }

    private static void logStageDone(long startNs, String msg) {
        double seconds = (System.nanoTime() - startNs) / 1_000_000_000.0;
        System.out.printf("%s (%.2fs)%n", msg, seconds);
        System.out.flush();
    }
}
