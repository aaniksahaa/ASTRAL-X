package astralx.util;

import astralx.Logging;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple inline progress bar for CPU work loops, written to stderr.
 *
 * Always active at INFO level — prints unconditionally regardless of TTY,
 * matching the behaviour of the native GPU progress output which uses
 * fprintf(stderr,...) without any TTY check.
 *
 * Uses \r to overwrite the current line; works correctly both on a real
 * terminal and through a pipe/tee (the terminal side of tee renders \r fine;
 * the file side retains \r characters, which is harmless for stderr logs).
 *
 * Thread-safe: update() may be called from multiple threads concurrently;
 * a CAS-based rate-limiter ensures at most one repaint per INTERVAL_NS.
 *
 * Usage:
 *   ProgressBar bar = new ProgressBar("Parsing trees", total);
 *   for (int i = 0; i < total; i++) {
 *       doWork(i);
 *       bar.update(i + 1);
 *   }
 *   bar.done();
 */
public class ProgressBar {

    private static final boolean COLOR = astralx.Banner.useColor();

    private static final String RST  = "\033[0m";
    private static final String DIM  = "\033[2m";
    private static final String CYAN = "\033[36m";

    private static final int  BAR_WIDTH   = 28;
    private static final long INTERVAL_NS = 150_000_000L; // 150 ms

    private final String     label;
    private final int        total;
    private final AtomicLong lastPaintNs = new AtomicLong(0L);

    // -------------------------------------------------------------------------

    public ProgressBar(String label, int total) {
        this.label = label;
        this.total = total;
        if (active() && total > 0) {
            render(0);
            lastPaintNs.set(System.nanoTime());
        }
    }

    /**
     * Report progress. Safe to call from multiple threads.
     * @param done number of items completed so far (1-based)
     */
    public void update(int done) {
        if (!active() || total <= 0) return;
        long now  = System.nanoTime();
        long last = lastPaintNs.get();
        if (now - last < INTERVAL_NS && done < total) return;
        if (!lastPaintNs.compareAndSet(last, now)) return;
        render(done);
    }

    /** Call once after the loop finishes. Prints the completed bar on its own line. */
    public void done() {
        if (!active() || total <= 0) return;
        render(total);
        System.err.println();
    }

    // -------------------------------------------------------------------------

    private boolean active() {
        return Logging.getLevel() >= Logging.INFO;
    }

    private void render(int done) {
        int pct    = (int)(100L * done / total);
        int filled = (int)(1L   * BAR_WIDTH * done / total);

        StringBuilder sb = new StringBuilder("     ");
        sb.append(COLOR ? DIM + "▸  " + RST : "▸  ");
        sb.append(label).append("  ");
        sb.append(COLOR ? CYAN : "");
        sb.append('[');
        for (int i = 0; i < BAR_WIDTH; i++) sb.append(i < filled ? '█' : '░');
        sb.append(']');
        if (COLOR) sb.append(RST);
        sb.append(String.format("  %d/%d (%d%%)", done, total, pct));
        System.err.print(sb + "\r");
    }
}
