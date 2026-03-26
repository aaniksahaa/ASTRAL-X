package astralx.weight;

import astralx.cluster.Cluster;
import astralx.dp.CandidateSplit;
import astralx.partition.Partition;
import astralx.partition.PartitionTable;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.preprocess.TreePreprocessInfo;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * GPU-first precompute coordinator.
 * Uses an external CUDA runner binary for real GPU kernel execution.
 */
public final class GpuWeightPrecomputer {
    private static final String RUNNER_PATH = "./src/cuda/astralx_weight_precompute";
    private static final int INPUT_MAGIC = 0x41585731; // "AWX1"

    public static final class Result {
        public final boolean usedGpu;
        public final String backendLabel;

        public Result(boolean usedGpu, String backendLabel) {
            this.usedGpu = usedGpu;
            this.backendLabel = backendLabel;
        }
    }

    private static final class Task {
        final Cluster parent;
        final CandidateSplit split;

        Task(Cluster parent, CandidateSplit split) {
            this.parent = parent;
            this.split = split;
        }
    }

    private static final class PartitionRow {
        final Cluster a;
        final Cluster b;
        final Cluster lgt;
        final int freq;

        PartitionRow(Cluster a, Cluster b, Cluster lgt, int freq) {
            this.a = a;
            this.b = b;
            this.lgt = lgt;
            this.freq = freq;
        }
    }

    public Result precomputeOrFallback(
            QuartetWeightCalculator calculator,
            Map<Cluster, List<CandidateSplit>> searchSpace,
            PartitionTable partitions,
            PreprocessedGeneTrees prep,
            int cpuThreads) {

        if (!isGpuAvailable()) {
            System.out.println("GPU backend not detected; using CPU_PARALLEL weight precompute fallback.");
            System.out.flush();
            calculator.precomputeAllCpuParallel(searchSpace, partitions, Math.max(1, cpuThreads));
            return new Result(false, "CPU_PARALLEL_FALLBACK");
        }

        List<Task> tasks = flattenTasks(searchSpace);
        List<PartitionRow> rows = flattenPartitions(partitions, prep);

        if (tasks.isEmpty() || rows.isEmpty()) {
            calculator.precomputeAllCpuParallel(searchSpace, partitions, Math.max(1, cpuThreads));
            return new Result(false, "CPU_PARALLEL_EMPTY_INPUT_FALLBACK");
        }

        Path tmpDir = null;
        try {
            tmpDir = Files.createTempDirectory("astralx-gpu-weight-");
            Path inBin = tmpDir.resolve("input.bin");
            Path outTxt = tmpDir.resolve("weights.txt");
            Path logTxt = tmpDir.resolve("runner.log");

            long t0 = System.nanoTime();
            writeRunnerInput(inBin, tasks, rows, prep);
            System.out.printf("GPU precompute input: %d candidates, %d partitions, %d trees, %d taxa%n",
                    tasks.size(), rows.size(), prep.treeInfos.size(), prep.totalTaxa);
            System.out.flush();

            Process p = new ProcessBuilder(RUNNER_PATH, inBin.toString(), outTxt.toString())
                    .redirectErrorStream(true)
                    .redirectOutput(logTxt.toFile())
                    .start();

            int rc = p.waitFor();
            if (rc != 0) {
                System.out.println("GPU runner failed (exit=" + rc + "); falling back to CPU_PARALLEL.");
                dumpRunnerLog(logTxt);
                calculator.precomputeAllCpuParallel(searchSpace, partitions, Math.max(1, cpuThreads));
                return new Result(false, "CPU_PARALLEL_GPU_ERROR_FALLBACK");
            }

            List<Double> weights = readWeights(outTxt, tasks.size());
            for (int i = 0; i < tasks.size(); i++) {
                Task task = tasks.get(i);
                calculator.cachePrecomputedScore(task.split, task.parent, weights.get(i));
            }

            double seconds = (System.nanoTime() - t0) / 1_000_000_000.0;
            System.out.printf("GPU weight precompute done in %.2fs (cached=%d)%n", seconds, calculator.cachedWeightCount());
            System.out.flush();
            return new Result(true, "GPU_CUDA_WAVELET_RUNNER");
        } catch (Exception e) {
            System.out.println("GPU path exception: " + e.getMessage() + "; falling back to CPU_PARALLEL.");
            System.out.flush();
            calculator.precomputeAllCpuParallel(searchSpace, partitions, Math.max(1, cpuThreads));
            return new Result(false, "CPU_PARALLEL_EXCEPTION_FALLBACK");
        } finally {
            if (tmpDir != null) {
                try {
                    Files.walk(tmpDir)
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
    }

    private List<Task> flattenTasks(Map<Cluster, List<CandidateSplit>> searchSpace) {
        List<Task> tasks = new ArrayList<>();
        for (Map.Entry<Cluster, List<CandidateSplit>> e : searchSpace.entrySet()) {
            Cluster parent = e.getKey();
            if (parent.allTaxa) {
                continue;
            }
            for (CandidateSplit split : e.getValue()) {
                tasks.add(new Task(parent, split));
            }
        }
        return tasks;
    }

    private List<PartitionRow> flattenPartitions(PartitionTable partitions, PreprocessedGeneTrees prep) {
        List<PartitionRow> rows = new ArrayList<>();
        for (PartitionTable.Entry e : partitions.entries()) {
            Partition p = e.representative;
            if (p.nonTrivialClusters.size() != 2) {
                continue;
            }
            Cluster a = p.nonTrivialClusters.get(0);
            Cluster b = p.nonTrivialClusters.get(1);
            int treeIndex = p.universeTreeIndex;
            int present = prep.treeInfos.get(treeIndex).presentTaxaCount;
            Cluster lgt = new Cluster(
                    -1000000 - treeIndex,
                    treeIndex,
                    0,
                    present - 1,
                    false,
                    false,
                    false,
                    null,
                    present
            );
            rows.add(new PartitionRow(a, b, lgt, e.frequency));
        }
        return rows;
    }

    private void writeRunnerInput(Path input,
                                  List<Task> tasks,
                                  List<PartitionRow> partitions,
                                  PreprocessedGeneTrees prep) throws IOException {

        int c = tasks.size();
        int p = partitions.size();
        int n = prep.totalTaxa;
        int k = prep.treeInfos.size();

        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(input.toFile()))) {
            writeIntBE(out, INPUT_MAGIC);
            writeIntBE(out, c);
            writeIntBE(out, p);
            writeIntBE(out, n);
            writeIntBE(out, k);

            // tree metadata
            for (int ti = 0; ti < k; ti++) {
                TreePreprocessInfo info = prep.treeInfos.get(ti);
                writeIntBE(out, info.presentTaxaCount);
            }
            for (int ti = 0; ti < k; ti++) {
                TreePreprocessInfo info = prep.treeInfos.get(ti);
                for (int x = 0; x < n; x++) {
                    writeIntBE(out, info.taxaByPostorderLeaf[x]);
                }
            }
            for (int ti = 0; ti < k; ti++) {
                TreePreprocessInfo info = prep.treeInfos.get(ti);
                for (int x = 0; x < n; x++) {
                    writeIntBE(out, info.positionByTaxon[x]);
                }
            }

            // candidate descriptors
            for (Task t : tasks) {
                writeCluster(out, t.split.left);
            }
            for (Task t : tasks) {
                writeCluster(out, t.split.right);
            }

            // partition descriptors
            for (PartitionRow row : partitions) {
                writeCluster(out, row.a);
            }
            for (PartitionRow row : partitions) {
                writeCluster(out, row.b);
            }
            for (PartitionRow row : partitions) {
                writeCluster(out, row.lgt);
            }

            for (PartitionRow row : partitions) {
                writeIntBE(out, row.freq);
            }
        }
    }

    private void writeCluster(DataOutputStream out, Cluster c) throws IOException {
        int flags = 0;
        if (c.localComplement) flags |= 1;
        if (c.globalComplement) flags |= 2;
        if (c.allTaxa) flags |= 4;

        writeIntBE(out, c.sourceTreeIndex);
        writeIntBE(out, c.left);
        writeIntBE(out, c.right);
        writeIntBE(out, flags);
        writeIntBE(out, c.size);
    }

    private List<Double> readWeights(Path outTxt, int expected) throws IOException {
        List<Double> w = new ArrayList<>(expected);
        try (BufferedReader br = new BufferedReader(new FileReader(outTxt.toFile()))) {
            String line;
            while ((line = br.readLine()) != null) {
                String t = line.trim();
                if (t.isEmpty()) {
                    continue;
                }
                w.add(Double.parseDouble(t));
            }
        }
        if (w.size() != expected) {
            throw new IOException("Expected " + expected + " weights, got " + w.size());
        }
        return w;
    }

    private void dumpRunnerLog(Path logPath) {
        if (logPath == null || !Files.exists(logPath)) {
            return;
        }
        try (BufferedReader br = new BufferedReader(new FileReader(logPath.toFile()))) {
            String line;
            int shown = 0;
            while ((line = br.readLine()) != null && shown < 60) {
                System.out.println("[gpu-runner] " + line);
                shown++;
            }
        } catch (IOException ignored) {
        }
    }

    private void writeIntBE(DataOutputStream out, int v) throws IOException {
        out.writeByte((v >>> 24) & 0xFF);
        out.writeByte((v >>> 16) & 0xFF);
        out.writeByte((v >>> 8) & 0xFF);
        out.writeByte(v & 0xFF);
    }

    private boolean isGpuAvailable() {
        return hasCommand("nvidia-smi") && new File(RUNNER_PATH).exists() && new File(RUNNER_PATH).canExecute();
    }

    private boolean hasCommand(String cmd) {
        try {
            Process p = new ProcessBuilder("bash", "-lc", "command -v " + cmd + " >/dev/null 2>&1").start();
            return p.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }
}
