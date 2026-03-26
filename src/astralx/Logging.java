package astralx;

/**
 * Simple leveled logging for ASTRAL-X.
 *
 * Levels:
 *   0 = QUIET  (errors only)
 *   1 = INFO   (major pipeline stages, summaries)
 *   2 = DEBUG  (per-stage details)
 *   3 = TRACE  (per-element detail -- guard with level check before string construction)
 */
public class Logging {

    public static final int QUIET = 0;
    public static final int INFO  = 1;
    public static final int DEBUG = 2;
    public static final int TRACE = 3;

    private static int level = INFO;

    public static void setLevel(int l) { level = l; }
    public static int getLevel() { return level; }

    public static void info(String msg) {
        if (level >= INFO) log("INFO", msg);
    }

    public static void info(String fmt, Object... args) {
        if (level >= INFO) log("INFO", String.format(fmt, args));
    }

    public static void debug(String msg) {
        if (level >= DEBUG) log("DEBUG", msg);
    }

    public static void debug(String fmt, Object... args) {
        if (level >= DEBUG) log("DEBUG", String.format(fmt, args));
    }

    public static void trace(String msg) {
        if (level >= TRACE) log("TRACE", msg);
    }

    public static void trace(String fmt, Object... args) {
        if (level >= TRACE) log("TRACE", String.format(fmt, args));
    }

    public static void error(String msg) {
        log("ERROR", msg);
    }

    public static void error(String fmt, Object... args) {
        log("ERROR", String.format(fmt, args));
    }

    public static boolean isDebug() { return level >= DEBUG; }
    public static boolean isTrace() { return level >= TRACE; }

    private static void log(String tag, String msg) {
        long elapsed = (System.nanoTime() - startTime) / 1_000_000;
        System.err.printf("[%5dms] [%-5s] %s%n", elapsed, tag, msg);
    }

    private static final long startTime = System.nanoTime();
}
