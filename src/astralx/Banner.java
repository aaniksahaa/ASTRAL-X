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

    private static boolean detectColor() {
        if (System.getenv("NO_COLOR")    != null) return false;
        if (System.getenv("FORCE_COLOR") != null) return true;
        return System.console() != null;
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
        // inner content width = w - 2  (one space padding each side inside ║…║)
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

        // separator used for section headers
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

        String inputPath = cfg.getInputFile();
        if (inputPath != null) {
            String[] parts = inputPath.replace('\\', '/').split("/");
            String display = parts.length > 2
                ? "…/" + parts[parts.length - 2] + "/" + parts[parts.length - 1]
                : inputPath;
            out.println("    " + row("Input",   c(WHT, display)));
        }

        boolean gpuMode = cfg.getComputeMode() == Config.ComputeMode.GPU;
        out.println("    " + row("Compute",
            gpuMode ? c(GRN, "GPU") + batchDetail(cfg)
                    : c(WHT, "CPU") + "  ·  threads: " + c(WHT, String.valueOf(using))));

        out.println("    " + row("Search",  c(WHT, cfg.getSearchMode().name().toLowerCase())));

        String verbStr = switch (cfg.getVerbosity()) {
            case Logging.QUIET -> c(DIM, "quiet");
            case Logging.DEBUG -> c(YLW, "debug");
            case Logging.TRACE -> c(YLW, "trace");
            default            -> "info";
        };
        out.println("    " + row("Seeds", c(WHT, String.valueOf(cfg.getNumHashSeeds()))
                + "   " + c(DIM, "verbosity") + "  " + verbStr));

        out.println("  " + c(DIM, "─".repeat(w)));
        out.println();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static String row(String label, String value) {
        return String.format("%-10s %s", c(DIM, label), value);
    }

    private static String batchDetail(Config cfg) {
        if (!cfg.isGpuBatch())
            return "  " + c(DIM, "(batching off)");
        if (cfg.getGpuNumBatches() > 0)
            return "  " + c(DIM, "· batches: " + cfg.getGpuNumBatches());
        if (cfg.getGpuBatchSize() > 0)
            return "  " + c(DIM, "· batch-size: " + cfg.getGpuBatchSize());
        return "  " + c(DIM, String.format("· batching: auto  (occupancy %.0f%%)",
                cfg.getGpuVramFraction() * 100));
    }
}
