package astralx.dp;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

final class GpuSearchLookup {
    private static final int INPUT_MAGIC = 0x41534C31; // ASL1
    private static final int OUTPUT_MAGIC = 0x41534C32; // ASL2

    static final class Result {
        final int[] sortedIds;
        final int[] lo;
        final int[] hi;

        Result(int[] sortedIds, int[] lo, int[] hi) {
            this.sortedIds = sortedIds;
            this.lo = lo;
            this.hi = hi;
        }
    }

    Result lookup(long[] clusterKeys, long[] queryKeys) throws IOException, InterruptedException {
        Path dir = Files.createTempDirectory("astralx-search-gpu-");
        Path in = dir.resolve("in.bin");
        Path out = dir.resolve("out.bin");
        Path log = dir.resolve("log.txt");

        try {
            writeInput(in, clusterKeys, queryKeys);
            Process p = new ProcessBuilder("./src/cuda/astralx_search_lookup", in.toString(), out.toString())
                    .redirectErrorStream(true)
                    .redirectOutput(log.toFile())
                    .start();
            int rc = p.waitFor();
            if (rc != 0) {
                throw new IOException("GPU search lookup runner failed with exit=" + rc + " (see " + log + ")");
            }
            return readOutput(out, clusterKeys.length, queryKeys.length);
        } finally {
            try {
                Files.walk(dir)
                        .sorted((a, b) -> b.getNameCount() - a.getNameCount())
                        .forEach(path -> {
                            try {
                                Files.deleteIfExists(path);
                            } catch (IOException ignored) {
                            }
                        });
            } catch (IOException ignored) {
            }
        }
    }

    private void writeInput(Path path, long[] clusterKeys, long[] queryKeys) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(path.toFile()))) {
            writeIntBE(out, INPUT_MAGIC);
            writeIntBE(out, clusterKeys.length);
            writeIntBE(out, queryKeys.length);
            for (int i = 0; i < clusterKeys.length; i++) {
                writeLongBE(out, clusterKeys[i]);
                writeIntBE(out, i);
            }
            for (long q : queryKeys) {
                writeLongBE(out, q);
            }
        }
    }

    private Result readOutput(Path path, int nClusters, int nQueries) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(path.toFile()))) {
            int magic = readIntBE(in);
            int n = readIntBE(in);
            int q = readIntBE(in);
            if (magic != OUTPUT_MAGIC || n != nClusters || q != nQueries) {
                throw new IOException("Invalid GPU lookup output header");
            }
            int[] sortedIds = new int[n];
            for (int i = 0; i < n; i++) sortedIds[i] = readIntBE(in);
            int[] lo = new int[q];
            int[] hi = new int[q];
            for (int i = 0; i < q; i++) {
                lo[i] = readIntBE(in);
                hi[i] = readIntBE(in);
            }
            return new Result(sortedIds, lo, hi);
        }
    }

    private static void writeIntBE(DataOutputStream out, int v) throws IOException {
        out.writeByte((v >>> 24) & 0xFF);
        out.writeByte((v >>> 16) & 0xFF);
        out.writeByte((v >>> 8) & 0xFF);
        out.writeByte(v & 0xFF);
    }

    private static int readIntBE(DataInputStream in) throws IOException {
        int b1 = in.readUnsignedByte();
        int b2 = in.readUnsignedByte();
        int b3 = in.readUnsignedByte();
        int b4 = in.readUnsignedByte();
        return (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
    }

    private static void writeLongBE(DataOutputStream out, long v) throws IOException {
        out.writeByte((int) ((v >>> 56) & 0xFF));
        out.writeByte((int) ((v >>> 48) & 0xFF));
        out.writeByte((int) ((v >>> 40) & 0xFF));
        out.writeByte((int) ((v >>> 32) & 0xFF));
        out.writeByte((int) ((v >>> 24) & 0xFF));
        out.writeByte((int) ((v >>> 16) & 0xFF));
        out.writeByte((int) ((v >>> 8) & 0xFF));
        out.writeByte((int) (v & 0xFF));
    }
}
