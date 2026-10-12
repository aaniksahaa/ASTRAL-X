package astralx.gpu;

import astralx.Logging;

/**
 * JNI bridge to libastralx_support.so (CUDA branch-support kernel).  Loading is
 * optional: when the library or a usable GPU is missing, callers use the CPU path.
 */
public final class GPUBranchSupport {

    private static boolean loaded = false;
    private static String status = "not loaded";

    static {
        try {
            System.loadLibrary("astralx_support");
            status = queryStatus();
            loaded = "ok".equals(status);
        } catch (Throwable t) {
            status = "native library unavailable: " + t.getClass().getSimpleName()
                + (t.getMessage() != null ? ": " + t.getMessage() : "");
            loaded = false;
        }
    }

    private GPUBranchSupport() {}

    public static boolean isLoaded() { return loaded; }
    public static String getStatus() { return status; }

    private static native String queryStatus();

    /**
     * Per-branch (f1, f2, f3, effn) as a flat double array, or null if the GPU
     * path is unusable (caller falls back to CPU).
     */
    public static native double[] computeSupportGPU(
        long[] clusterBits, long[] geneBits, int[] tokens, int[] offsets,
        int numBranches, int numTrees, int wordsPerSet, int maxFrontier, int vramCapMiB);

    public static void logStatus() {
        Logging.info("Branch support GPU kernel: %s", loaded ? "ready" : status);
    }
}
