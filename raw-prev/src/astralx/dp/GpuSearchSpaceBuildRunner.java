package astralx.dp;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

final class GpuSearchSpaceBuildRunner {
    private static final int IN_MAGIC = 0x41534231;  // ASB1
    private static final int OUT_MAGIC = 0x41534232; // ASB2

    static final class Result {
        final int[] triples;
        final boolean overflow;

        Result(int[] triples, boolean overflow) {
            this.triples = triples;
            this.overflow = overflow;
        }
    }

    Result run(long[] clusterSignatures,
               int[] clusterIds,
               long[] clusterBitsets,
               int words,
               int[] qA,
               int[] qB,
               long[] qKey,
               int outCap) throws IOException, InterruptedException {

        Path dir = Files.createTempDirectory("astralx-search-build-");
        Path in = dir.resolve("in.bin");
        Path out = dir.resolve("out.bin");
        Path log = dir.resolve("log.txt");

        try {
            writeInput(in, clusterSignatures, clusterIds, clusterBitsets, words, qA, qB, qKey, outCap);
            Process p = new ProcessBuilder("./src/cuda/astralx_search_space_build", in.toString(), out.toString())
                    .redirectErrorStream(true)
                    .redirectOutput(log.toFile())
                    .start();
            int rc = p.waitFor();
            if (rc != 0) {
                throw new IOException("GPU search-space runner failed with exit=" + rc + " (see " + log + ")");
            }
            return readOutput(out);
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

    private void writeInput(Path p,
                            long[] sig,
                            int[] ids,
                            long[] bits,
                            int words,
                            int[] qA,
                            int[] qB,
                            long[] qKey,
                            int outCap) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(p.toFile()))) {
            writeIntBE(out, IN_MAGIC);
            writeIntBE(out, sig.length);
            writeIntBE(out, 0);
            writeIntBE(out, words);
            writeIntBE(out, qKey.length);
            writeIntBE(out, outCap);

            for (int i = 0; i < sig.length; i++) {
                writeLongBE(out, sig[i]);
                writeIntBE(out, ids[i]);
            }
            for (long b : bits) writeLongBE(out, b);
            for (int i = 0; i < qKey.length; i++) {
                writeIntBE(out, qA[i]);
                writeIntBE(out, qB[i]);
                writeLongBE(out, qKey[i]);
            }
        }
    }

    private Result readOutput(Path p) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(p.toFile()))) {
            int magic = readIntBE(in);
            if (magic != OUT_MAGIC) {
                throw new IOException("Invalid GPU output magic");
            }
            int used = readIntBE(in);
            int overflow = readIntBE(in);
            int[] triples = new int[used * 3];
            for (int i = 0; i < triples.length; i++) triples[i] = readIntBE(in);
            return new Result(triples, overflow != 0);
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
