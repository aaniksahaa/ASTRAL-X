package astralx;

import astralx.util.Threading;

/**
 * ASTRAL-X: Scalable species tree inference via quartet scoring
 * with integer-tuple cluster representation and GPU acceleration.
 */
public class Main {

    public static final String VERSION = "0.1.0";

    public static void main(String[] args) {
        Config config = Config.getInstance();

        // Parse command-line arguments
        if (!parseArgs(args, config)) {
            printUsage();
            System.exit(1);
        }

        Logging.setLevel(config.getVerbosity());
        Logging.info("ASTRAL-X v%s", VERSION);
        Logging.info("Input: %s", config.getInputFile());
        Logging.info("Output: %s", config.getOutputFile() != null ? config.getOutputFile() : "stdout");
        Logging.info("Threads: %d", config.getThreadCount());
        Logging.info("Mode: %s", config.getComputeMode());
        Logging.info("Hash seeds: %d", config.getNumHashSeeds());

        // Initialize thread pool
        Threading.start(config.getThreadCount());

        try {
            long startTime = System.nanoTime();

            // === PIPELINE ===
            // Phase 1: Parse gene trees
            // Phase 2: Compute taxon hashes and prefix arrays
            // Phase 3: Extract clusters -> build X
            // Phase 4: Extract gene tree tripartitions
            // Phase 5: Build DP search space
            // Phase 6: Weight calculation (lazy, during DP)
            // Phase 7: Inference DP + tree reconstruction

            Logging.info("Pipeline not yet implemented -- scaffold complete");

            long elapsed = (System.nanoTime() - startTime) / 1_000_000;
            Logging.info("Total time: %d ms", elapsed);

        } catch (Exception e) {
            Logging.error("Fatal error: %s", e.getMessage());
            e.printStackTrace(System.err);
            System.exit(2);
        } finally {
            Threading.shutdown();
        }
    }

    private static boolean parseArgs(String[] args, Config config) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i", "--input" -> {
                    if (++i >= args.length) return false;
                    config.setInputFile(args[i]);
                }
                case "-o", "--output" -> {
                    if (++i >= args.length) return false;
                    config.setOutputFile(args[i]);
                }
                case "-t", "--threads" -> {
                    if (++i >= args.length) return false;
                    config.setThreadCount(Integer.parseInt(args[i]));
                }
                case "--cpu" -> config.setComputeMode(Config.ComputeMode.CPU);
                case "--gpu" -> config.setComputeMode(Config.ComputeMode.GPU);
                case "-v" -> config.setVerbosity(Logging.INFO);
                case "-vv" -> config.setVerbosity(Logging.DEBUG);
                case "-vvv" -> config.setVerbosity(Logging.TRACE);
                case "-q", "--quiet" -> config.setVerbosity(Logging.QUIET);
                case "-m", "--seeds" -> {
                    if (++i >= args.length) return false;
                    config.setNumHashSeeds(Integer.parseInt(args[i]));
                }
                case "--rooted" -> config.setTreatAsUnrooted(false);
                case "--unrooted" -> config.setTreatAsUnrooted(true);
                case "-h", "--help" -> {
                    printUsage();
                    System.exit(0);
                }
                default -> {
                    System.err.println("Unknown argument: " + args[i]);
                    return false;
                }
            }
        }
        return config.getInputFile() != null;
    }

    private static void printUsage() {
        System.err.println("ASTRAL-X v" + VERSION);
        System.err.println("Usage: astralx -i <input.tre> [-o <output.tre>] [options]");
        System.err.println();
        System.err.println("Options:");
        System.err.println("  -i, --input <file>    Input gene trees (Newick, one per line)");
        System.err.println("  -o, --output <file>   Output species tree (default: stdout)");
        System.err.println("  -t, --threads <N>     Number of threads (default: available cores)");
        System.err.println("  --cpu                 Use CPU computation (default)");
        System.err.println("  --gpu                 Use GPU-accelerated computation");
        System.err.println("  -m, --seeds <N>       Number of hash seeds (default: 2)");
        System.err.println("  --rooted              Treat gene trees as rooted");
        System.err.println("  --unrooted            Treat gene trees as unrooted (default)");
        System.err.println("  -v / -vv / -vvv       Verbosity levels (INFO / DEBUG / TRACE)");
        System.err.println("  -q, --quiet           Suppress all output except errors");
        System.err.println("  -h, --help            Show this help message");
    }
}
