package astralx;

import astralx.gpu.GPUWeightCalculator;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;

/** Best-effort fatal report for Java exceptions and memory failures. */
public final class FatalReporter {
    /** Crash reports are collected under this directory (relative to the working directory). */
    public static final String CRASH_LOG_DIRECTORY = "crash-logs";
    /** Optional system property overriding the crash-log directory (absolute or cwd-relative). */
    public static final String CRASH_LOG_DIRECTORY_PROPERTY = "astralx.crashLogDir";

    private FatalReporter() {}

    public static void report(Throwable failure, String[] args) {
        String text = build(failure, args);
        System.err.println(text);

        String stamp = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
            .withZone(ZoneOffset.UTC).format(Instant.now());
        String name = "astralx-crash-" + stamp + "-" + ProcessHandle.current().pid() + ".log";
        String configured = System.getProperty(CRASH_LOG_DIRECTORY_PROPERTY, "").trim();
        Path directory = Path.of(System.getProperty("user.dir", "."))
            .resolve(configured.isEmpty() ? CRASH_LOG_DIRECTORY : configured);
        Path report = directory.resolve(name);
        try {
            Files.createDirectories(directory);
            Files.writeString(report, text + System.lineSeparator(), StandardCharsets.UTF_8);
            System.err.println("Crash report written to: " + report.toAbsolutePath());
        } catch (Throwable writeFailure) {
            System.err.println("Could not write crash report: " + writeFailure.getMessage());
        }
    }

    private static String build(Throwable failure, String[] args) {
        Runtime rt = Runtime.getRuntime();
        StringWriter sw = new StringWriter();
        PrintWriter out = new PrintWriter(sw);
        out.println();
        out.println("================ ASTRAL-X FATAL ERROR ================");
        out.println("Version: " + Main.VERSION);
        out.println("Time (UTC): " + Instant.now());
        out.println("Phase: " + PhaseLogger.currentPhase());
        out.println("Failure: " + failure.getClass().getName());
        out.println("Message: " + String.valueOf(failure.getMessage()));
        out.println("OS/arch: " + System.getProperty("os.name") + " "
            + System.getProperty("os.version") + " / " + System.getProperty("os.arch"));
        out.println("Runtime: " + System.getProperty("java.runtime.version"));
        out.println("Memory: used=" + mib(rt.totalMemory() - rt.freeMemory())
            + " MiB, committed=" + mib(rt.totalMemory()) + " MiB, max=" + mib(rt.maxMemory()) + " MiB");
        GPUWeightCalculator.Probe gpu = GPUWeightCalculator.probe();
        out.println("CUDA: " + (gpu.cudaAvailable()
            ? gpu.deviceName() + " (CC " + gpu.computeMajor() + "." + gpu.computeMinor() + ")"
            : "unavailable: " + gpu.detail()));
        out.println("Command: astralx " + quoteArgs(args));
        if (failure instanceof OutOfMemoryError) {
            out.println();
            out.println("Likely remedy: close other memory-heavy jobs, use a machine with more RAM,");
            out.println("or reduce search-space enrichment. The portable launcher allows the JVM");
            out.println("to use up to 85% of physical/container memory by default.");
        }
        out.println();
        failure.printStackTrace(out);
        out.println("======================================================");
        out.flush();
        return sw.toString();
    }

    private static long mib(long bytes) { return bytes / (1024L * 1024L); }

    private static String quoteArgs(String[] args) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < args.length; i++) {
            if (i > 0) sb.append(' ');
            String a = args[i];
            if (a.matches("[A-Za-z0-9_./:=+,-]+")) sb.append(a);
            else sb.append('\'').append(a.replace("'", "'\\''")).append('\'');
        }
        return sb.toString();
    }
}
