package astralx;

import astralx.gpu.GPUWeightCalculator;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.concurrent.TimeUnit;

/**
 * Startup banner: system info + run configuration.
 * Written to stderr so it appears even when stdout is redirected.
 * ANSI colours are enabled only when stderr is a real terminal
 * (or FORCE_COLOR is set), and suppressed when NO_COLOR is set.
 */
public class Banner {

    // ── ANSI colour codes ─────────────────────────────────────────────────────
    private static final String RST  = "\033[0m";
    private static final String BOLD = "\033[1m";
    private static final String DIM  = "\033[2m";
    private static final String CYAN = "\033[36m";
    private static final String GRN  = "\033[32m";
    private static final String YLW  = "\033[33m";
    private static final String WHT  = "\033[97m";

    private static final boolean USE_COLOR = detectColor();

    /** Exposed so PhaseLogger and other classes can share the same colour decision. */
    public static boolean useColor() { return USE_COLOR; }

    private static boolean detectColor() {
        if (System.getenv("NO_COLOR")    != null) return false;
        if (System.getenv("FORCE_COLOR") != null) return true;
        // System.console() requires stdin+stdout to be ttys, so it returns null when
        // the JVM is spawned by a script (even if the terminal is visible).
        // Fall back to checking whether stderr's file descriptor points to a tty.
        if (System.console() != null) return true;
        try {
            String fd2 = java.nio.file.Files.readSymbolicLink(
                java.nio.file.Paths.get("/proc/self/fd/2")).toString();
            return fd2.startsWith("/dev/pts") || fd2.startsWith("/dev/tty");
        } catch (Exception e) {
            return false;
        }
    }

    private static String c(String code, String text) {
        return USE_COLOR ? code + text + RST : text;
    }

    // ── GPU info via nvidia-smi (500 ms timeout, non-fatal) ───────────────────

    private record GpuInfo(String name, long totalMiB, long freeMiB) {}

    private static GpuInfo queryGpuInfo() {
        try {
            Process p = new ProcessBuilder(
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"
            ).redirectErrorStream(true).start();

            boolean done = p.waitFor(500, TimeUnit.MILLISECONDS);
            if (!done) { p.destroyForcibly(); return null; }

            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(p.getInputStream()))) {
                String line = br.readLine();
                if (line == null) return null;
                String[] parts = line.split(",");
                if (parts.length < 3) return null;
                return new GpuInfo(
                    parts[0].trim(),
                    Long.parseLong(parts[1].trim()),
                    Long.parseLong(parts[2].trim())
                );
            }
        } catch (Exception e) {
            return null;
        }
    }

    private static String fmtMiB(long mib) {
        return mib >= 1024 ? String.format("%.1f GB", mib / 1024.0) : mib + " MB";
    }

    // ── Public entry point ────────────────────────────────────────────────────

    public static void print(Config cfg) {
        PrintStream out = System.err;
        // w = number of ═ characters (= total inner + 2 for the space padding each side)
        final int w = 63;

        // ── Title box ──────────────────────────────────────────────────────
        String title = "ASTRAL-X  v" + Main.VERSION;
        int inner = w - 2;
        int lpad  = (inner - title.length()) / 2;
        int rpad  = inner - title.length() - lpad;
        String paddedTitle = " ".repeat(Math.max(0, lpad))
                           + title
                           + " ".repeat(Math.max(0, rpad));

        out.println();
        out.println("  " + c(BOLD + CYAN, "╔" + "═".repeat(w) + "╗"));
        out.println("  " + c(BOLD + CYAN, "║") + " "
                         + c(BOLD + WHT,  paddedTitle)
                         + " " + c(BOLD + CYAN, "║"));
        out.println("  " + c(BOLD + CYAN, "╚" + "═".repeat(w) + "╝"));
        out.println();

        String sep = "─".repeat(w - 2);

        // ── System section ─────────────────────────────────────────────────
        out.println("  " + c(BOLD, "System") + "  " + c(DIM, sep.substring(0, sep.length() - 4)));

        int available = Runtime.getRuntime().availableProcessors();
        int using     = cfg.getThreadCount();
        out.println("    " + String.format("%-8s %s  →  %s",
            "CPU",
            c(WHT, available + " cores available"),
            c(available == using ? WHT : YLW, using + " threads configured")));

        boolean libLoaded = GPUWeightCalculator.tryLoad();
        GpuInfo gpu = queryGpuInfo();
        if (gpu != null) {
            String libTag = libLoaded
                ? "  " + c(GRN, "✓ library loaded")
                : "  " + c(YLW, "⚠ library not found (CPU fallback)");
            out.println("    " + String.format("%-8s %s  ·  %s total  ·  %s free%s",
                "GPU",
                c(WHT, gpu.name()),
                c(WHT, fmtMiB(gpu.totalMiB())),
                c(gpu.freeMiB() > gpu.totalMiB() / 4 ? GRN : YLW, fmtMiB(gpu.freeMiB())),
                libTag));
        } else {
            out.println("    " + String.format("%-8s %s",
                "GPU", c(YLW, "not detected  (nvidia-smi unavailable)")));
        }
        out.println();

        // ── Run configuration section ───────────────────────────────────────
        out.println("  " + c(BOLD, "Run Configuration") + "  " + c(DIM, sep.substring(0, sep.length() - 15)));
        out.println();

        boolean gpuMode = cfg.getComputeMode() == Config.ComputeMode.GPU;
        String naTag    = c(DIM, "(n/a)");

        // ── I/O ────────────────────────────────────────────────────────────
        String inputPath = cfg.getInputFile();
        String displayInput = "(none)";
        if (inputPath != null) {
            String[] parts = inputPath.replace('\\', '/').split("/");
            displayInput = parts.length > 2
                ? "…/" + parts[parts.length - 2] + "/" + parts[parts.length - 1]
                : inputPath;
        }
        out.println("    " + row("Input file",    c(WHT, displayInput)));
        String outputPath = cfg.getOutputFile();
        out.println("    " + row("Output file",   outputPath != null
                                                  ? c(WHT, outputPath)
                                                  : c(DIM, "(stdout)")));
        out.println();

        // ── Compute ────────────────────────────────────────────────────────
        out.println("    " + row("Compute mode",   gpuMode ? c(GRN, "GPU") : c(WHT, "CPU")));
        out.println("    " + row("CPU threads",    c(available == using ? WHT : YLW, String.valueOf(using))
                                                  + c(DIM, "  (" + available + " available)")));
        out.println("    " + row("Tree treatment", c(WHT, cfg.getTreatAsUnrooted() ? "unrooted" : "rooted")));
        out.println();

        // ── Search ─────────────────────────────────────────────────────────
        out.println("    " + row("Search mode",   c(WHT, cfg.getSearchMode().name().toLowerCase())));
        out.println("    " + row("Hash seeds",    c(WHT, String.valueOf(cfg.getNumHashSeeds()))));

        String verbStr = switch (cfg.getVerbosity()) {
            case Logging.QUIET -> c(DIM, "quiet");
            case Logging.DEBUG -> c(YLW, "debug");
            case Logging.TRACE -> c(YLW, "trace");
            default            -> c(WHT, "info");
        };
        out.println("    " + row("Verbosity",     verbStr));
        out.println();

        // ── GPU parameters ─────────────────────────────────────────────────
        // Weight batching
        String batchStr;
        if (!gpuMode) {
            batchStr = naTag;
        } else if (!cfg.isGpuBatch()) {
            batchStr = c(WHT, "disabled");
        } else if (cfg.getGpuNumBatches() > 0) {
            batchStr = c(WHT, cfg.getGpuNumBatches() + " batches") + c(DIM, "  (--gpu-batches)");
        } else if (cfg.getGpuBatchSize() > 0) {
            batchStr = c(WHT, "batch-size " + cfg.getGpuBatchSize()) + c(DIM, "  (--gpu-batch-size)");
        } else {
            batchStr = c(WHT, "auto") + c(DIM, "  (adaptive from free VRAM)");
        }
        out.println("    " + row("Weight batching",    batchStr));

        // VRAM occupancy fraction
        out.println("    " + row("VRAM occupancy",     gpuMode
                ? c(WHT, String.format("%.0f%%", cfg.getGpuVramFraction() * 100))
                : naTag));

        // DP state-space cap (only meaningful for GPU + FULL search)
        boolean dpRelevant = gpuMode && cfg.getSearchMode() == Config.SearchMode.FULL;
        out.println("    " + row("DP state-space construction memory cap (GPU)", dpRelevant
                ? c(WHT, fmtBytes(cfg.getGpuDpOutputCapBytes()))
                  + c(DIM, "  (" + cfg.getGpuDpOutputCapTriples() + " triples)")
                : naTag));

        out.println();
        out.println("  " + c(DIM, "─".repeat(w)));
        out.println();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Label column width for Run Configuration rows
    private static String row(String label, String value) {
        return String.format("%-46s %s", c(DIM, label), value);
    }

    private static String fmtBytes(long bytes) {
        if (bytes >= 1_000_000_000L) return String.format("%.1f GB", bytes / 1e9);
        if (bytes >= 1_000_000L)     return String.format("%.0f MB", bytes / 1e6);
        if (bytes >= 1_000L)         return String.format("%.0f KB", bytes / 1e3);
        return bytes + " B";
    }
}
