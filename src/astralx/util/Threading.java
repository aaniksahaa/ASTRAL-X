package astralx.util;

import java.util.List;
import java.util.concurrent.*;
import java.util.function.Consumer;

public class Threading {
    private static ExecutorService executor;
    private static int numThreads;

    public static void start(int threads) {
        numThreads = Math.max(1, threads);
        executor = Executors.newFixedThreadPool(numThreads);
    }

    public static void shutdown() {
        if (executor != null) {
            executor.shutdown();
            try { executor.awaitTermination(60, TimeUnit.SECONDS); }
            catch (InterruptedException e) { executor.shutdownNow(); Thread.currentThread().interrupt(); }
        }
    }

    public static int getNumThreads() { return numThreads; }
    public static Future<?> submit(Runnable r) { return executor.submit(r); }
    public static <T> Future<T> submit(Callable<T> c) { return executor.submit(c); }

    /** Divide list across threads, apply action in parallel, wait for all. */
    public static <T> void processParallel(List<T> items, Consumer<T> action) {
        if (items.isEmpty()) return;
        int chunk = Math.max(1, (items.size() + numThreads - 1) / numThreads);
        int actual = Math.min(numThreads, (items.size() + chunk - 1) / chunk);
        CountDownLatch latch = new CountDownLatch(actual);
        for (int t = 0; t < actual; t++) {
            int lo = t * chunk, hi = Math.min(lo + chunk, items.size());
            executor.submit(() -> {
                try { for (int i = lo; i < hi; i++) action.accept(items.get(i)); }
                finally { latch.countDown(); }
            });
        }
        try { latch.await(); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new RuntimeException(e); }
    }

    /** Divide range [0,count) across threads in parallel, wait for all. */
    public static void processRangeParallel(int count, Consumer<Integer> action) {
        if (count == 0) return;
        int chunk = Math.max(1, (count + numThreads - 1) / numThreads);
        int actual = Math.min(numThreads, (count + chunk - 1) / chunk);
        CountDownLatch latch = new CountDownLatch(actual);
        for (int t = 0; t < actual; t++) {
            int lo = t * chunk, hi = Math.min(lo + chunk, count);
            executor.submit(() -> {
                try { for (int i = lo; i < hi; i++) action.accept(i); }
                finally { latch.countDown(); }
            });
        }
        try { latch.await(); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new RuntimeException(e); }
    }
}
