package astralx;

import astralx.cluster.ClusterTable;
import astralx.dp.DPTable;
import astralx.dp.Inference;
import astralx.gpu.GPUDPBuilder;
import astralx.partition.PartitionTable;
import astralx.weight.WeightTable;
import astralx.hash.PrefixHashArrays;
import astralx.hash.TaxonHasher;
import astralx.taxon.TaxonRegistry;
import astralx.tree.Tree;
import astralx.tree.TreeParser;
import astralx.util.Threading;

import java.util.List;

public class Main {
    public static final String VERSION = "0.1.0";

    public static void main(String[] args) throws Exception {
        Config cfg = Config.getInstance();
        if (!parseArgs(args, cfg)) { printUsage(); System.exit(1); }

        Logging.setLevel(cfg.getVerbosity());
        Logging.info("ASTRAL-X v%s", VERSION);
        Logging.info("Input: %s", cfg.getInputFile());
        Logging.info("Mode: %s  Threads: %d  Seeds: %d",
            cfg.getComputeMode(), cfg.getThreadCount(), cfg.getNumHashSeeds());

        Threading.start(cfg.getThreadCount());
        long t0 = System.nanoTime();

        try {
            // ── Phase 1: Parse gene trees ─────────────────────────────────────
            TaxonRegistry registry = new TaxonRegistry();
            List<Tree> trees = TreeParser.parseGeneTrees(cfg.getInputFile(), registry);

            if (cfg.isVerifyParse()) {
                Phase1Verifier.dump(trees, registry, cfg.getOutputFile());
                return;
            }

            // ── Phase 2: Taxon hashing + prefix arrays ────────────────────────
            TaxonHasher hasher = new TaxonHasher(
                registry.size(), cfg.getNumHashSeeds(), cfg.getBaseSeed());
            PrefixHashArrays pref = new PrefixHashArrays(trees, hasher);

            if (cfg.isVerifyHash()) {
                Phase2Verifier.dump(trees, registry, hasher, pref, cfg.getOutputFile());
                return;
            }

            // ── Phase 3: Cluster extraction -> X ─────────────────────────────
            ClusterTable clusterTable = new ClusterTable(trees, pref, registry.size());

            if (cfg.isVerifyClusters()) {
                Phase3Verifier.dump(trees, registry, pref, clusterTable, cfg.getOutputFile());
                return;
            }

            // ── Phase 4: Gene-tree tripartition extraction ────────────────────
            PartitionTable partTable = new PartitionTable(trees, pref);

            if (cfg.isVerifyPartitions()) {
                Phase4Verifier.dump(trees, registry, pref, partTable, cfg.getOutputFile());
                return;
            }

            // ── Phase 5: DP search space (tree-local transitions) ─────────────
            DPTable dpTable = new DPTable(trees, pref, clusterTable);

            // ── Phase 5b: Cross-tree transitions (Mode 2, optional) ───────────
            if (cfg.getSearchMode() == Config.SearchMode.FULL) {
                boolean gpuDP = (cfg.getComputeMode() == Config.ComputeMode.GPU)
                                && GPUDPBuilder.tryLoad();
                dpTable.addCrossTreeTransitions(clusterTable, gpuDP);
            }

            if (cfg.isVerifyDPSpace()) {
                Phase5Verifier.dump(trees, registry, pref, clusterTable, dpTable, cfg.getOutputFile());
                return;
            }

            // ── Phase 6: Weight calculation ───────────────────────────────────
            WeightTable weightTable = new WeightTable(dpTable, partTable, clusterTable, trees);

            if (cfg.isVerifyWeights()) {
                Phase6Verifier.dump(trees, registry, clusterTable, dpTable, weightTable, cfg.getOutputFile());
                return;
            }

            // ── Phase 7: Inference DP + tree reconstruction ───────────────────
            Inference inference = new Inference();
            String speciesTree = inference.run(dpTable, weightTable, clusterTable, trees, registry);

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
                case "-v"              -> cfg.setVerbosity(Logging.INFO);
                case "-vv"             -> cfg.setVerbosity(Logging.DEBUG);
                case "-vvv"            -> cfg.setVerbosity(Logging.TRACE);
                case "-q","--quiet"    -> cfg.setVerbosity(Logging.QUIET);
                case "-m","--seeds"    -> { if (++i>=args.length) return false; cfg.setNumHashSeeds(Integer.parseInt(args[i])); }
                case "--rooted"        -> cfg.setTreatAsUnrooted(false);
                case "--unrooted"      -> cfg.setTreatAsUnrooted(true);
                case "--verify-parse"  -> cfg.setVerifyParse(true);
                case "--verify-hash"      -> cfg.setVerifyHash(true);
                case "--verify-clusters"    -> cfg.setVerifyClusters(true);
                case "--verify-partitions" -> cfg.setVerifyPartitions(true);
                case "--verify-dp"         -> cfg.setVerifyDPSpace(true);
                case "--verify-weights"    -> cfg.setVerifyWeights(true);
                case "-h","--help"     -> { printUsage(); System.exit(0); }
                default -> { System.err.println("Unknown arg: " + args[i]); return false; }
            }
        }
        return cfg.getInputFile() != null;
    }

    private static void printUsage() {
        System.err.println("ASTRAL-X v" + VERSION);
        System.err.println("Usage: astralx -i <input.tre> [-o <out>] [options]");
        System.err.println("  --verify-parse   dump Phase-1 output and exit");
        System.err.println("  --verify-hash    dump Phase-2 output and exit");
        System.err.println("  -v/-vv/-vvv      verbosity levels");
    }
}
