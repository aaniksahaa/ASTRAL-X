package astralx;

import astralx.gpu.GPUWeightCalculator;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Step-by-step phase logging with per-phase peak GPU VRAM.
 *
 * For GPU phases a background thread polls queryVRAMMiB() every 50 ms and
 * tracks the minimum free VRAM seen.  On phase end, peak VRAM in use during
 * the phase = totalVRAM − minFreeObserved.
 *
 * Output format:
 *   ▶  Phase N  Name  [GPU/CPU]
 *   ...kernel/Java output interleaved...
 *        ✓  1234 ms  │  peak VRAM: 231 MiB
 */
public class PhaseLogger {

    // ── ANSI colours (shared detection with Banner) ───────────────────────────
    private static final boolean COLOR = Banner.useColor();
    private static final String RST  = "\033[0m";
    private static final String BOLD = "\033[1m";
    private static final String DIM  = "\033[2m";
    private static final String CYAN = "\033[36m";
    private static final String GRN  = "\033[32m";
    private static final String YLW  = "\033[33m";

    private static String c(String code, String text) {
        return COLOR ? code + text + RST : text;
    }

    // ── Background VRAM poller ────────────────────────────────────────────────
    private static final AtomicLong minFreeVRAM = new AtomicLong(Long.MAX_VALUE);
    private static final AtomicLong totalVRAM   = new AtomicLong(0);
    private static volatile boolean polling     = false;
    private static Thread           pollThread  = null;

    private static void startVramPolling() {
        if (!GPUWeightCalculator.isLoaded()) return;
        minFreeVRAM.set(Long.MAX_VALUE);
        totalVRAM.set(0);
        polling = true;
        pollThread = new Thread(() -> {
            while (polling) {
                long[] v = GPUWeightCalculator.queryVRAMMiB();
                if (v != null) {
                    long free  = v[0];
                    long total = v[1];
                    totalVRAM.set(total);
                    // CAS-loop to atomically track minimum
                    long cur;
                    do { cur = minFreeVRAM.get(); }
                    while (free < cur && !minFreeVRAM.compareAndSet(cur, free));
                }
                try { Thread.sleep(50); } catch (InterruptedException e) { break; }
            }
        }, "vram-poller");
        pollThread.setDaemon(true);
        pollThread.start();
    }

    /** Stop polling and return a formatted "peak VRAM: X MiB" string, or null. */
    private static String stopVramPolling() {
        if (pollThread == null) return null;
        polling = false;
        try { pollThread.join(500); } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        pollThread = null;

        long minFree = minFreeVRAM.get();
        long total   = totalVRAM.get();
        if (minFree == Long.MAX_VALUE || total == 0) return null;

        long peakUsed = total - minFree;
        // colour: green if peak < 50% of total, yellow otherwise
        String colCode = (peakUsed * 2 < total) ? GRN : YLW;
        return c(DIM, "peak VRAM: ") + c(colCode, peakUsed + " MiB");
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Print phase-start header and return a start timestamp.
     * @param label human-readable phase label
     * @param gpu   true if this phase executes on the GPU
     * @return System.nanoTime() snapshot for passing to {@link #end}
     */
    public static long begin(String label, boolean gpu) {
        if (gpu) startVramPolling();
        String tag = gpu ? c(GRN, "[GPU]") : c(CYAN, "[CPU]");
        System.err.println();
        System.err.println("  " + c(BOLD, "▶  " + label) + "  " + tag);
        return System.nanoTime();
    }

    /**
     * Print phase-completion line with elapsed time and (for GPU) peak VRAM.
     * @param label same label passed to {@link #begin}
     * @param t0    timestamp returned by {@link #begin}
     * @param gpu   true if this phase ran on the GPU
     */
    public static void end(String label, long t0, boolean gpu) {
        long ms = (System.nanoTime() - t0) / 1_000_000;
        String vramStr = gpu ? stopVramPolling() : null;

        StringBuilder sb = new StringBuilder();
        sb.append("     ").append(c(DIM, "✓")).append("  ").append(c(YLW, ms + " ms"));
        if (vramStr != null) {
            sb.append("  ").append(c(DIM, "│")).append("  ").append(vramStr);
        }
        System.err.println(sb);
    }
}
