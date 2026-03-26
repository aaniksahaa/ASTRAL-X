package astralx.cli;

public final class Config {
    public enum IntersectionMode {
        WAVELET,
        CPU
    }
    public enum WeightMode {
        GPU,
        CPU
    }

    public final String inputPath;
    public final String outputPath;
    public final boolean treatInputAsUnrooted;
    public final int hashReplicates;
    public final long randomSeed;
    public final IntersectionMode intersectionMode;
    public final WeightMode weightMode;

    public Config(String inputPath, String outputPath, boolean treatInputAsUnrooted, int hashReplicates, long randomSeed,
                  IntersectionMode intersectionMode, WeightMode weightMode) {
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.treatInputAsUnrooted = treatInputAsUnrooted;
        this.hashReplicates = hashReplicates;
        this.randomSeed = randomSeed;
        this.intersectionMode = intersectionMode;
        this.weightMode = weightMode;
    }
}
