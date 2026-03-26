package astralx;

public class Config {
    public enum ComputeMode { CPU, GPU }

    private static Config instance;

    private String inputFile;
    private String outputFile;
    private ComputeMode computeMode = ComputeMode.CPU;
    private int threadCount = Runtime.getRuntime().availableProcessors();
    private int numHashSeeds = 2;
    private long baseSeed = 0xDEADBEEFCAFEL;
    private int verbosity = 1; // 0=quiet 1=INFO 2=DEBUG 3=TRACE
    private boolean treatAsUnrooted = true;

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
}
