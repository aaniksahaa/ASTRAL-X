# ASTRAL-MP — Editable Source Development Guide

This directory (`astral-my/`) is a **developer workspace** for ASTRAL-MP (the multi-threaded,
GPU-capable version of ASTRAL).  The goal is to run ASTRAL from its own Java source files
so that you can freely add logging, inspect intermediate data structures (distance matrices,
bipartition weights, cluster sets, etc.), and understand the internal algorithm — all without
changing any functional behaviour of the tool.

---

## Directory layout

```
astral-my/
├── dev.sh                        ← YOUR MAIN ENTRY POINT (compile + run from project root)
├── test.tre                      ← small 6-taxon gene tree file for quick testing
├── run-astral-with-monitor.sh    ← performance wrapper (time / GPU / memory monitoring)
└── ASTRAL/
    ├── compile_astral.sh         ← compiles Java source → .class files
    ├── run_astral.sh             ← runs ASTRAL from compiled .class files
    ├── lib/
    │   ├── main.jar              ← PhyloNet library (tree I/O, data structures) — NOT edited
    │   ├── colt.jar              ← numerical library
    │   ├── JSAP-2.1.jar          ← CLI argument parsing
    │   └── jocl-2.0.0.jar        ← Java/OpenCL bindings for GPU
    └── main/                     ← EDITABLE Java source tree
        └── phylonet/
            ├── coalescent/       ← core ASTRAL-MP logic  ← EDIT HERE
            │   ├── CommandLine.java          main() entry point, argument parsing
            │   ├── AbstractInference.java    top-level inference orchestration
            │   ├── WQInferenceConsumer.java  dynamic programming / tree scoring
            │   ├── WQDataCollection.java     builds the cluster search space X
            │   ├── WQWeightCalculator.java   quartet weight calculation (CPU path)
            │   ├── SimilarityMatrix.java     pairwise taxon similarity
            │   ├── DistanceMatrix.java       pairwise taxon distance, tree completion
            │   ├── BipartitionWeightCalculator.java  bipartition weight scores
            │   ├── Polytree.java             polytree weight calculation (fast path)
            │   ├── AstralCudaLib.java        CUDA / GPU weight calculation
            │   ├── GlobalMaps.java           global singletons (taxon IDs, etc.)
            │   ├── Logging.java              all output goes through here
            │   └── ...
            ├── tree/             ← tree model (mostly from PhyloNet, rarely edited)
            └── util/             ← BitSet, etc.
```

> **Key insight:** `main.jar` is a *dependency* (the PhyloNet tree library), not the
> compiled ASTRAL code.  The ASTRAL code lives in `main/phylonet/coalescent/*.java` and is
> compiled to `.class` files that take priority on the classpath.  Editing those `.java`
> files and recompiling is all you need.

---

## Quickstart

### 1. Compile and run in one shot (most common)

```bash
# from astral-my/
bash dev.sh -i test.tre -o out.tre -C
```

`-C` = CPU-only mode (no GPU).  Relative paths like `test.tre` are resolved automatically.

### 2. Compile only (after editing source)

```bash
bash dev.sh --compile-only
```

### 3. Run without recompiling

```bash
bash dev.sh --run-only -i test.tre -o out.tre -C
```

### 4. Run with threads / GPU

```bash
bash dev.sh -i test.tre -o out.tre -T 8        # 8 CPU threads
bash dev.sh -i test.tre -o out.tre -G 0,1      # GPUs 0 and 1
bash dev.sh -i test.tre -o out.tre -T 4 -G 0   # 4 threads + GPU 0
```

### 5. Run with monitoring (time, memory, GPU usage)

```bash
bash run-astral-with-monitor.sh test.tre out.tre --astral-root ASTRAL --no-gpu-monitor
```

---

## The edit → compile → run cycle

This is the core loop for all instrumentation work:

```
1. Open a .java file under ASTRAL/main/phylonet/coalescent/
2. Add your System.err.println / Logging.log / file-write calls
3. bash dev.sh --compile-only          (takes ~3 seconds)
4. bash dev.sh --run-only -i test.tre -o out.tre -C
5. Inspect output / logs
6. Repeat
```

Compilation only touches the files that changed (incremental javac).  A full clean rebuild
takes about 3 seconds on this machine.

---

## How to add logging / print intermediate data

All ASTRAL console output goes through `Logging.log(String)`.  You can use it anywhere:

```java
// in any .java file under main/phylonet/coalescent/
Logging.log("[DEBUG] my message: " + someValue);
```

Or write to `System.err` directly if you prefer not to go through the logger:

```java
System.err.println("[DEBUG] matrix size: " + n);
```

### Example — printing the distance matrix

Open [ASTRAL/main/phylonet/coalescent/DistanceMatrix.java](ASTRAL/main/phylonet/coalescent/DistanceMatrix.java)
and find the method where the matrix is populated (e.g. `populateByBranchDistance`).
Add before the return:

```java
System.err.println("[DEBUG] Distance matrix (" + n + "x" + n + "):");
for (int r = 0; r < n; r++) {
    StringBuilder sb = new StringBuilder();
    for (int c = 0; c < n; c++) sb.append(String.format("%8.4f ", matrix[r][c]));
    System.err.println(sb);
}
```

Then recompile and run.

### Example — printing bipartition weights

In [ASTRAL/main/phylonet/coalescent/WQWeightCalculator.java](ASTRAL/main/phylonet/coalescent/WQWeightCalculator.java),
find where weights are computed and accumulated, and log them similarly.

---

## Useful ASTRAL options (reference)

| Flag | Description |
|------|-------------|
| `-i FILE` | Input gene trees (Newick, one per line) — **required** |
| `-o FILE` | Output species tree file |
| `-C` | CPU-only mode (skip all GPU code paths) |
| `-T N` | Number of CPU threads (default: all cores) |
| `-G LIST` | Comma-separated GPU indices to use |
| `-t N` | Branch annotation level: 0=none, 1=quartet support, 2=full, 3=posterior (default), 4=3 posteriors |
| `-q FILE` | Score a given species tree and exit (no inference) |
| `-e FILE` | Extra gene trees to add to search space X |
| `-a FILE` | Taxon name map (gene label → species label) |
| `-p 0/1/2` | Extra bipartitions: 0=none, 1=greedy (default), 2=quadratic |
| `-x` | Exact solution (only feasible for ≤18 taxa) |
| `--xms SIZE` | JVM initial heap (default 4g, env: `ASTRAL_XMS`) |
| `--xmx SIZE` | JVM max heap (default 128g, env: `ASTRAL_XMX`) |

---

## Source file map — what to read for each topic

| Topic | File(s) |
|-------|---------|
| Entry point, argument parsing | `CommandLine.java` |
| Top-level inference flow | `AbstractInference.java` |
| Building the cluster search space (X) | `WQDataCollection.java` |
| Dynamic programming (species tree DP) | `WQInferenceConsumer.java`, `WQInferenceProducer.java` |
| Quartet weight calculation (CPU) | `WQWeightCalculator.java` |
| Quartet weight calculation (native/AVX) | `Polytree.java`, `AstralCudaLib.java` |
| Distance matrix (used to complete missing taxa) | `DistanceMatrix.java`, `SimilarityMatrix.java` |
| Bipartition scoring | `BipartitionWeightCalculator.java` |
| Tree completion / missing taxa | `DistanceMatrix.java` → `PhyDstar.java` |
| Taxon identity and global state | `GlobalMaps.java`, `TaxonIdentifier.java` |
| Threading / task queues | `Threading.java`, `WQComputeMinCostTaskProducer.java` |
| All logging / console output | `Logging.java` |
| GPU (CUDA via OpenCL JNI) | `AstralCudaLib.java`, `CudaGPUManager.java` |
| GPU kernel source (OpenCL C) | `calculateWeight.cl`, `calculateWeightNVidia.cl`, `calculateWeightAMD.cl` |

---

## Notes on the native library (libAstral.so)

ASTRAL-MP ships a precompiled native library (`lib/libAstral.so`) that uses AVX2 SIMD
instructions for fast weight calculation.  If the CPU does not support AVX2 the JVM will
crash with `SIGILL`.  The library is **optional** — if it cannot be loaded, ASTRAL
automatically falls back to a pure-Java implementation that is ~4× slower but produces
identical results.

- The `lib/no_avx2/` directory contains a non-AVX2 build for older CPUs.
- Do not pass `-Djava.library.path` unless you know your CPU supports AVX2.  The current
  setup deliberately omits it so the safe Java fallback is always used.

---

## The test tree (`test.tre`)

`test.tre` contains 24 small gene trees over 6 taxa (A–F), with some trees missing certain
taxa.  It is ideal for fast iteration:

- Runs in ~1 second (Java fallback mode).
- Has missing taxa → triggers the tree-completion code path.
- Has conflicting signal → tests how ASTRAL resolves quartet disagreement.
- Expected output species tree: `(A,(B,((E,F),(C,D))));`

---

## Compilation details (for reference)

`compile_astral.sh` runs (from `ASTRAL/`):

```bash
cd main
javac -g \
  -classpath ../lib/main.jar:../lib/colt.jar:../lib/JSAP-2.1.jar:../lib/jocl-2.0.0.jar \
  phylonet/util/BitSet*.java \
  phylonet/coalescent/*.java \
  phylonet/tree/model/sti/*.java \
  phylonet/tree/io/NewickWriter.java
```

The deprecation warnings from `PhyDstar.java` and `SpeciesMapper.java` are pre-existing in
the upstream codebase and are harmless — do not fix them as that would be a functional
change to someone else's code.

`run_astral.sh` then runs (from `ASTRAL/main/`):

```bash
java -Xms4g -Xmx128g \
  -classpath ".:../lib/main.jar:../lib/colt.jar:../lib/JSAP-2.1.jar:../lib/jocl-2.0.0.jar" \
  phylonet.coalescent.CommandLine \
  "$@"
```

The `.` at the start of the classpath means the compiled `.class` files in `main/` are
found **before** anything in the JARs, so your edits always take effect.
