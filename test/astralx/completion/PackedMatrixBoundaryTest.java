package astralx.completion;

import astralx.Config;

/** Arithmetic-only checks at and beyond the Java dense-array boundary. */
public final class PackedMatrixBoundaryTest {
    public static void main(String[] args) {
        if (SimilarityMatrix.requiresPacked(46_340)) {
            throw new AssertionError("46,340 should retain the established dense path");
        }
        if (!SimilarityMatrix.requiresPacked(46_341)
                || !SimilarityMatrix.requiresPacked(50_000)) {
            throw new AssertionError("large-N packed dispatch boundary is wrong");
        }

        int n = 50_000;
        long cells = SimilarityMatrix.triangleCellCount(n);
        if (cells != 1_250_025_000L) throw new AssertionError("triangle size=" + cells);
        if (SimilarityMatrix.packedIndex(n, 0, 0) != 0L
                || SimilarityMatrix.packedIndex(n, 0, n - 1) != n - 1L
                || SimilarityMatrix.packedIndex(n, n - 1, n - 1) != cells - 1L) {
            throw new AssertionError("packed boundary indices are wrong");
        }
        int[][] pairs = {{1, 49_999}, {23_456, 45_678}, {49_998, 49_999}};
        for (int[] p : pairs) {
            long ab = SimilarityMatrix.packedIndex(n, p[0], p[1]);
            long ba = SimilarityMatrix.packedIndex(n, p[1], p[0]);
            if (ab != ba || ab < 0 || ab >= cells) {
                throw new AssertionError("invalid symmetric index for " + p[0] + "," + p[1]);
            }
        }

        SegmentedDoubleArray segmented = new SegmentedDoubleArray(40, 4);
        if (segmented.segments().length != 3
                || segmented.segments()[0].length != 16
                || segmented.segments()[1].length != 16
                || segmented.segments()[2].length != 8) {
            throw new AssertionError("test segment layout is wrong");
        }
        long[] positions = {0, 15, 16, 31, 32, 39};
        for (long p : positions) segmented.set(p, p + 0.25);
        segmented.add(16, 2.0);
        for (long p : positions) {
            double expected = p + 0.25 + (p == 16 ? 2.0 : 0.0);
            if (segmented.get(p) != expected) {
                throw new AssertionError("segment-boundary access failed at " + p);
            }
        }

        System.setProperty("astralx.similarity.forcePacked", "true");
        SimilarityMatrix tinyPacked;
        try {
            tinyPacked = new SimilarityMatrix(3);
        } finally {
            System.clearProperty("astralx.similarity.forcePacked");
        }
        Config cfg = Config.getInstance();
        if (SimilarityMatrixBuilder.effectiveTreeCapMiB(tinyPacked, cfg) != 8192) {
            throw new AssertionError("large-N automatic GPU batching ceiling was not raised");
        }
        cfg.setGpuSimilarityVramCapMiB(256);
        if (SimilarityMatrixBuilder.effectiveTreeCapMiB(tinyPacked, cfg) != 256) {
            throw new AssertionError("explicit GPU batching ceiling was not respected");
        }

        // The reported 100k × 1000-tree case has E=4096 and LOG=12.  Its
        // compact sparse table cannot be one Java array, but every planned
        // streamed batch must fit both the array limit and the host-byte cap.
        int streamed = SimilarityMatrixBuilder.compactBatchTreeCount(
            100_000, 1_000, 4_096, 12, Integer.MAX_VALUE - 8L, 4L << 30);
        if (streamed != 19_072) {
            throw new AssertionError("100k compact batch size changed: " + streamed);
        }
        long sparseCells = (long)streamed * 12 * 4_096;
        long flatBytes = (long)streamed
            * (30L * 4_096 + 2L * 12 * 4_096 + 4L * 1_000 + 8L);
        if (sparseCells > Integer.MAX_VALUE - 8L || flatBytes > (4L << 30)) {
            throw new AssertionError("streamed compact batch exceeds a safety bound");
        }
        long nextFlatBytes = (long)(streamed + 1)
            * (30L * 4_096 + 2L * 12 * 4_096 + 4L * 1_000 + 8L);
        if (nextFlatBytes <= (4L << 30)) {
            throw new AssertionError("streamed compact batch is smaller than necessary");
        }

        // Small fitting inputs retain one batch, and artificial tiny limits
        // exercise the exact per-array planner boundary without large arrays.
        if (SimilarityMatrixBuilder.compactBatchTreeCount(
                7, 10, 16, 4, 1_000, 1_000_000) != 7) {
            throw new AssertionError("fitting compact input was unnecessarily split");
        }
        if (SimilarityMatrixBuilder.compactBatchTreeCount(
                100, 10, 16, 4, 1_000, 1_000_000) != 15) {
            throw new AssertionError("sparse-array boundary was not enforced exactly");
        }
        System.out.println("Packed matrix 46,340/46,341/50,000 boundaries: PASS");
    }
}
