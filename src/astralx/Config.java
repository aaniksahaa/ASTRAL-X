package astralx;

/**
 * Global configuration singleton for ASTRAL-X runtime parameters.
 */
public class Config {

    public enum ComputeMode { CPU, GPU }

    // Singleton
    private static Config instance;

    // Input/Output
    private String inputFile;
    private String outputFile;

    // Computation
    private ComputeMode computeMode = ComputeMode.CPU;
    private int threadCount = Runtime.getRuntime().availableProcessors();

    // Hashing
    private int numHashSeeds = 2;
    private long baseSeed = 0xDEADBEEFCAFEL; // fixed for reproducibility

    // Logging
    private int verbosity = 1; // 0=quiet, 1=INFO, 2=DEBUG, 3=TRACE

    // Tree treatment
    private boolean treatAsUnrooted = true; // treat gene trees as unrooted for cluster/partition extraction

    private Config() {}

    public static Config getInstance() {
        if (instance == null) {
            instance = new Config();
        }
        return instance;
    }

    /** Reset for testing */
    public static void reset() {
        instance = null;
    }

    // --- Getters and Setters ---

    public String getInputFile() { return inputFile; }
    public void setInputFile(String f) { this.inputFile = f; }

    public String getOutputFile() { return outputFile; }
    public void setOutputFile(String f) { this.outputFile = f; }

    public ComputeMode getComputeMode() { return computeMode; }
    public void setComputeMode(ComputeMode m) { this.computeMode = m; }

    public int getThreadCount() { return threadCount; }
    public void setThreadCount(int t) { this.threadCount = Math.max(1, t); }

    public int getNumHashSeeds() { return numHashSeeds; }
    public void setNumHashSeeds(int m) { this.numHashSeeds = m; }

    public long getBaseSeed() { return baseSeed; }
    public void setBaseSeed(long s) { this.baseSeed = s; }

    public int getVerbosity() { return verbosity; }
    public void setVerbosity(int v) { this.verbosity = v; }

    public boolean getTreatAsUnrooted() { return treatAsUnrooted; }
    public void setTreatAsUnrooted(boolean u) { this.treatAsUnrooted = u; }

    @Override
    public String toString() {
        return String.format(
            "Config{input=%s, output=%s, mode=%s, threads=%d, seeds=%d, verbosity=%d, unrooted=%b}",
            inputFile, outputFile, computeMode, threadCount, numHashSeeds, verbosity, treatAsUnrooted
        );
    }
}
