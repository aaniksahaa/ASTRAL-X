package astralx.util;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Thread pool management for ASTRAL-X parallel operations.
 */
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
            try {
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    public static int getNumThreads() { return numThreads; }

    public static Future<?> submit(Runnable task) {
        return executor.submit(task);
    }

    public static <T> Future<T> submit(Callable<T> task) {
        return executor.submit(task);
    }

    /**
     * Process a list in parallel using chunked work distribution.
     */
    public static <T> void processParallel(List<T> items, Consumer<T> action) {
        if (items.isEmpty()) return;

        int chunkSize = Math.max(1, (items.size() + numThreads - 1) / numThreads);
        int actualThreads = Math.min(numThreads, (items.size() + chunkSize - 1) / chunkSize);
        CountDownLatch latch = new CountDownLatch(actualThreads);

        for (int t = 0; t < actualThreads; t++) {
            int start = t * chunkSize;
            int end = Math.min(start + chunkSize, items.size());
            executor.submit(() -> {
                try {
                    for (int i = start; i < end; i++) {
                        action.accept(items.get(i));
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel processing interrupted", e);
        }
    }

    /**
     * Process a list in parallel, collecting results.
     */
    public static <T, R> List<R> processParallelWithResults(List<T> items, Function<T, R> action) {
        if (items.isEmpty()) return new ArrayList<>();

        int chunkSize = Math.max(1, (items.size() + numThreads - 1) / numThreads);
        int actualThreads = Math.min(numThreads, (items.size() + chunkSize - 1) / chunkSize);

        @SuppressWarnings("unchecked")
        List<R>[] results = new List[actualThreads];
        CountDownLatch latch = new CountDownLatch(actualThreads);

        for (int t = 0; t < actualThreads; t++) {
            int threadIdx = t;
            int start = t * chunkSize;
            int end = Math.min(start + chunkSize, items.size());
            executor.submit(() -> {
                try {
                    List<R> localResults = new ArrayList<>();
                    for (int i = start; i < end; i++) {
                        localResults.add(action.apply(items.get(i)));
                    }
                    results[threadIdx] = localResults;
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel processing interrupted", e);
        }

        List<R> combined = new ArrayList<>();
        for (List<R> r : results) {
            if (r != null) combined.addAll(r);
        }
        return combined;
    }

    /**
     * Process a range [0, count) in parallel with chunked work distribution.
     */
    public static void processRangeParallel(int count, Consumer<Integer> action) {
        if (count == 0) return;

        int chunkSize = Math.max(1, (count + numThreads - 1) / numThreads);
        int actualThreads = Math.min(numThreads, (count + chunkSize - 1) / chunkSize);
        CountDownLatch latch = new CountDownLatch(actualThreads);

        for (int t = 0; t < actualThreads; t++) {
            int start = t * chunkSize;
            int end = Math.min(start + chunkSize, count);
            executor.submit(() -> {
                try {
                    for (int i = start; i < end; i++) {
                        action.accept(i);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel processing interrupted", e);
        }
    }
}
