package astralx;

public class Config {
    public enum ComputeMode { CPU, GPU }
    public enum SearchMode  { LOCAL, FULL }

    private static Config instance;

    private String inputFile;
    private String outputFile;
    private ComputeMode computeMode = ComputeMode.CPU;
    private int threadCount = Runtime.getRuntime().availableProcessors();
    private int numHashSeeds = 2;
    private long baseSeed = 0xDEADBEEFCAFEL;
    private int verbosity = 1; // 0=quiet 1=INFO 2=DEBUG 3=TRACE
    private boolean treatAsUnrooted = true;
    private SearchMode searchMode = SearchMode.LOCAL;

    /**
     * GPU split-batching control.
     *   true  (default) — adaptive: batch size computed from free VRAM at runtime.
     *   false           — disabled: all splits sent in one kernel launch (original behaviour).
     */
    private boolean gpuBatch = true;

    /**
     * Manual GPU batch size override (ignored when gpuBatch=false).
     *   0 (default) — auto: derived from free VRAM via cudaMemGetInfo.
     *   > 0         — use exactly this many splits per kernel launch.
     */
    private int gpuBatchSize = 0;

    /**
     * Explicit number of GPU batches (highest priority when > 0).
     * batchSize is computed as ceil(numSplits / gpuNumBatches) at runtime.
     * Overrides gpuBatchSize and the auto-VRAM logic.
     *   0 (default) — not set; fall back to gpuBatchSize or auto.
     */
    private int gpuNumBatches = 0;

    /**
     * Fraction of free VRAM to occupy when computing auto batch size.
     * Default 0.75 means use 75% of free VRAM, reserving 25% as headroom
     * for driver overhead, kernel stack, and page tables.
     * Configured via --gpu-vram-occupancy-factor.  Must be in (0, 1].
     */
    private double gpuVramFraction = 0.75;

    private Config() {}

    public static Config getInstance() {
        if (instance == null) instance = new Config();
        return instance;
    }
    public static void reset() { instance = null; }

    public String getInputFile()      { return inputFile; }
    public void setInputFile(String f){ this.inputFile = f; }
    public String getOutputFile()     { return outputFile; }
    public void setOutputFile(String f){ this.outputFile = f; }
    public ComputeMode getComputeMode()        { return computeMode; }
    public void setComputeMode(ComputeMode m)  { this.computeMode = m; }
    public int getThreadCount()               { return threadCount; }
    public void setThreadCount(int t)         { this.threadCount = Math.max(1, t); }
    public int getNumHashSeeds()              { return numHashSeeds; }
    public void setNumHashSeeds(int m)        { this.numHashSeeds = m; }
    public long getBaseSeed()                 { return baseSeed; }
    public int getVerbosity()                 { return verbosity; }
    public void setVerbosity(int v)           { this.verbosity = v; }
    public boolean getTreatAsUnrooted()       { return treatAsUnrooted; }
    public void setTreatAsUnrooted(boolean u) { this.treatAsUnrooted = u; }
    public SearchMode getSearchMode()          { return searchMode; }
    public void setSearchMode(SearchMode s)   { this.searchMode = s; }
    public boolean isGpuBatch()               { return gpuBatch; }
    public void setGpuBatch(boolean b)        { this.gpuBatch = b; }
    public int getGpuBatchSize()              { return gpuBatchSize; }
    public void setGpuBatchSize(int s)        { this.gpuBatchSize = s; }
    public int getGpuNumBatches()             { return gpuNumBatches; }
    public void setGpuNumBatches(int n)       { this.gpuNumBatches = Math.max(1, n); }
    public double getGpuVramFraction()        { return gpuVramFraction; }
    public void setGpuVramFraction(double f)  { this.gpuVramFraction = Math.max(0.01, Math.min(1.0, f)); }

    // Testing flags
    private boolean verifyParse = false;
    public boolean isVerifyParse()          { return verifyParse; }
    public void setVerifyParse(boolean v)   { this.verifyParse = v; }

    private boolean verifyHash = false;
    public boolean isVerifyHash()           { return verifyHash; }
    public void setVerifyHash(boolean v)    { this.verifyHash = v; }

    private boolean verifyClusters = false;
    public boolean isVerifyClusters()       { return verifyClusters; }
    public void setVerifyClusters(boolean v){ this.verifyClusters = v; }

    private boolean verifyPartitions = false;
    public boolean isVerifyPartitions()        { return verifyPartitions; }
    public void setVerifyPartitions(boolean v) { this.verifyPartitions = v; }

    private boolean verifyDPSpace = false;
    public boolean isVerifyDPSpace()           { return verifyDPSpace; }
    public void setVerifyDPSpace(boolean v)    { this.verifyDPSpace = v; }

    private boolean verifyWeights = false;
    public boolean isVerifyWeights()           { return verifyWeights; }
    public void setVerifyWeights(boolean v)    { this.verifyWeights = v; }
}
