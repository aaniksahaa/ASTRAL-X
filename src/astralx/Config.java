package astralx;

public class Config {
    public enum ComputeMode      { CPU, GPU }
    public enum SearchMode       { LOCAL, FULL }
    /**
     * Which matrix is used to guide taxon insertion when auto-completing
     * incomplete gene trees (Phase 1b).
     *   SIMILARITY — quartet-based similarity matrix (default, matches ASTRAL-MP)
     *   DISTANCE   — topological distance matrix (legacy behaviour)
     */
    public enum CompletionMethod { SIMILARITY, DISTANCE }

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
     * Fraction of free VRAM (after static upload) to use for the batch buffer.
     * This is the DEFAULT auto batching mode — the native code queries free VRAM
     * via cudaMemGetInfo after uploading static data, then allocates:
     *
     *   batchSize = floor(freeVRAM × gpuVramFraction / 48 B)
     *
     * This adapts automatically to whatever GPU and dataset are in use.
     * Default 0.75 (use 75% of remaining free VRAM for splits + scores buffers,
     * leaving 25% headroom for driver, kernel stack, page tables).
     *
     * Configured via --gpu-vram-occupancy-factor.  Must be in (0, 1].
     */
    private double gpuVramFraction = 0.75;

    /**
     * VRAM control factor for GPU weight-calculation split batching.
     * Manual override — resident-relative sizing:
     *
     *   resident   = mem(orderings) + mem(invIndex) + mem(parts)
     *   mem(batch) = F × resident
     *   batchSize  = F × resident / 48 B
     *
     * Hardware-independent (same batchSize on any GPU).  Only active when
     * explicitly set via --gpu-vram-control-factor; otherwise the auto
     * free-VRAM adaptive path (gpuVramFraction) is used.
     *
     * Priority: --no-gpu-batch  >  --gpu-batches  >  --gpu-batch-size
     *         >  --gpu-vram-control-factor  >  auto (--gpu-vram-occupancy-factor)
     */
    private double gpuVramControlFactor    = 1.0;
    private boolean gpuVramControlFactorSet = false;

    /**
     * GPU output buffer size for the cross-tree DP state-space construction phase
     * (Phase 5b), stored in bytes.  Each transition triple occupies 12 bytes
     * (3 × sizeof(int)), so the number of triples the buffer can hold is
     * gpuDpOutputCapBytes / 12.
     *
     * Default: 120 MB = 10 000 000 triples.
     *
     * Sub-batching normally guarantees no overflow, but if a dataset has an
     * extraordinarily large single size-bin the kernel will overflow and print
     * a CRITICAL WARNING.  Raise this value (e.g. "1g") to avoid the overflow
     * at the cost of more VRAM.
     *
     * Configured via --gpu-dp-state-space-construction-output-cap.
     * Accepts memory-unit suffixes: k/K (×10³), m/M (×10⁶), g/G (×10⁹).
     * Examples: "120m"  "1.2g"  "500k"  "1500000000"
     */
    private long gpuDpOutputCapBytes = 128_000_000L; // 128 MB default

    /**
     * Minimum seconds between DP progress bar updates.
     * Configured via --gpu-dp-state-space-progress-time-interval.
     */
    private double gpuDpProgressInterval = 1.0;

    /**
     * Maximum number of progress bar print steps for DP phase.
     * When > 0, switches to step-based mode (% advancement) and disables the
     * time-interval trigger entirely.
     * 0 = not set; use time-interval mode.
     * Default: 1000 (step mode, print every 0.1% advancement).
     * Configured via --gpu-dp-state-space-progress-max-steps.
     */
    private int gpuDpProgressMaxSteps = 1000;

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
    public double getGpuVramFraction()            { return gpuVramFraction; }
    public void setGpuVramFraction(double f)      { this.gpuVramFraction = Math.max(0.01, Math.min(1.0, f)); }
    public double getGpuVramControlFactor()       { return gpuVramControlFactor; }
    public boolean isGpuVramControlFactorSet()    { return gpuVramControlFactorSet; }
    public void setGpuVramControlFactor(double f) { this.gpuVramControlFactor = Math.max(0.001, Math.min(1.0, f)); this.gpuVramControlFactorSet = true; }
    /** Raw byte count of the GPU DP output buffer. */
    public long getGpuDpOutputCapBytes()      { return gpuDpOutputCapBytes; }

    /**
     * Number of transition triples the GPU output buffer can hold
     * (= bytes / 12, clamped to at least 1).
     */
    public int getGpuDpOutputCapTriples()     { return (int) Math.max(1, gpuDpOutputCapBytes / 12); }

    /**
     * Set the GPU DP output-buffer cap from a human-readable memory string.
     * Accepts optional suffixes k/K (×1 000), m/M (×1 000 000), g/G (×1 000 000 000).
     * A bare integer is interpreted as bytes.
     * Examples: "120m", "1.2g", "500k", "1500000000"
     */
    public void setGpuDpStateSpaceConstructionOutputCap(String spec) {
        String s = spec.trim().toLowerCase();
        double value;
        long multiplier;
        if (s.endsWith("g")) {
            value = Double.parseDouble(s.substring(0, s.length() - 1));
            multiplier = 1_000_000_000L;
        } else if (s.endsWith("m")) {
            value = Double.parseDouble(s.substring(0, s.length() - 1));
            multiplier = 1_000_000L;
        } else if (s.endsWith("k")) {
            value = Double.parseDouble(s.substring(0, s.length() - 1));
            multiplier = 1_000L;
        } else {
            value = Double.parseDouble(s);
            multiplier = 1L;
        }
        long bytes = (long)(value * multiplier);
        this.gpuDpOutputCapBytes = Math.max(12L, bytes); // at least 1 triple
    }

    public double getGpuDpProgressInterval()          { return gpuDpProgressInterval; }
    public void setGpuDpProgressInterval(double v)    { this.gpuDpProgressInterval = Math.max(0.0, v); }
    public int getGpuDpProgressMaxSteps()             { return gpuDpProgressMaxSteps; }
    public void setGpuDpProgressMaxSteps(int v)       { this.gpuDpProgressMaxSteps = Math.max(1, v); }

    // Completion flags
    private boolean autoCompleteIncompleteTrees = false;
    public boolean isAutoCompleteIncompleteTrees()          { return autoCompleteIncompleteTrees; }
    public void setAutoCompleteIncompleteTrees(boolean v)   { this.autoCompleteIncompleteTrees = v; }

    /** Which matrix guides taxon insertion in tree completion. Default: SIMILARITY. */
    private CompletionMethod completionMethod = CompletionMethod.SIMILARITY;
    public CompletionMethod getCompletionMethod()             { return completionMethod; }
    public void setCompletionMethod(CompletionMethod m)       { this.completionMethod = m; }

    /**
     * Tile side-length B for the GPU distance-matrix kernel.
     * Controls GPU VRAM for the output tile: B² × 12 bytes.
     * Default 0 = auto: B = min(n, ceil(sqrt(n * k))), capped by available VRAM.
     * Configured via --gpu-dist-tile-size.
     */
    private int gpuDistTileSizeB = 0;
    public int  getGpuDistTileSizeB()          { return gpuDistTileSizeB; }
    public void setGpuDistTileSizeB(int b)     { this.gpuDistTileSizeB = Math.max(0, b); }

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

    private boolean verifyDistanceMatrix = false;
    public boolean isVerifyDistanceMatrix()          { return verifyDistanceMatrix; }
    public void setVerifyDistanceMatrix(boolean v)   { this.verifyDistanceMatrix = v; }

    private boolean verifySimilarityMatrix = false;
    public boolean isVerifySimilarityMatrix()        { return verifySimilarityMatrix; }
    public void setVerifySimilarityMatrix(boolean v) { this.verifySimilarityMatrix = v; }

    private boolean verifyUpgma = false;
    public boolean isVerifyUpgma()          { return verifyUpgma; }
    public void setVerifyUpgma(boolean v)   { this.verifyUpgma = v; }

    // ── Comparison / debug dump flags ─────────────────────────────────────────
    private String dumpClustersFile = null;
    public String getDumpClustersFile()       { return dumpClustersFile; }
    public void setDumpClustersFile(String f) { this.dumpClustersFile = f; }
}
