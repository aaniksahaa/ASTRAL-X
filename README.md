# ASTRAL-X

## A Fundamental Computational Redesign for Scalable Coalescent-Based Species Tree Inference

ASTRAL-X is a complete algorithmic redesign of the ASTRAL framework for highly
scalable, statistically consistent species tree inference from collections of
gene trees. Compact data representations, memory-efficient search-space
construction, and GPU-accelerated computation preserve ASTRAL's quartet-based
optimization and statistical guarantees while making analyses with hundreds of
thousands of taxa practical.

ASTRAL-X reconstructed a 300,000-taxon species tree within only
12 hours using 100 GB of memory, and inferred the evolutionary history of 9,524
angiosperm species in just 16 minutes.

An NVIDIA CUDA GPU is strongly recommended, particularly for large datasets,
and is the primary high-performance execution path. CPU execution remains
available as a reliable fallback for compatibility, smaller analyses, and
development. Portable releases include their own Java runtime and automatically
use CUDA when it is available.

## One-time setup

Keep the complete extracted application directory together: the launcher needs
the accompanying `bin/` and `lib/` directories.

Download `astralx-1.0.0-linux-x86_64.tar.gz` from the
GitHub Releases page. The
release archive is self-contained, so cloning this repository is not required.
Replace `/path/to/downloaded/` below with the archive's actual location, then
run this single setup block. It installs ASTRAL-X under `~/.local/opt/` and adds
the launcher to your user `PATH` without requiring `sudo`:

```bash
mkdir -p "$HOME/.local/opt/astralx/1.0.0"
tar -xzf /path/to/downloaded/astralx-1.0.0-linux-x86_64.tar.gz \
  -C "$HOME/.local/opt/astralx/1.0.0"
ASTRALX_DIR="$(realpath "$HOME/.local/opt/astralx/1.0.0/astralx-1.0.0-linux-x86_64")"
mkdir -p "$HOME/.local/bin"
ln -sfn "$ASTRALX_DIR/astralx" "$HOME/.local/bin/astralx"
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  printf '%s\n' 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
export PATH="$HOME/.local/bin:$PATH"
```

### Check the installation

Verify that the launcher is available:

```bash
astralx --version
```

## Quick start

After the one-time setup, ASTRAL-X can be run from any directory:

```bash
astralx -i /path/to/gene_trees.tre -o /path/to/output_species_tree.tre
```

Input and output may be relative to the current directory or given as absolute
paths. For a directly runnable example, use the included 37-taxon dataset from
either the extracted application directory or this repository root:

```bash
astralx -i example/all_gt_37.tre -o example/out_astralx_37.tre
```

The defaults are `--auto`, `--search-space S1`, and
`--intersection-method I2`: ASTRAL-X tries CUDA first, safely falls back to CPU,
uses the smallest search space, and uses prefix-sum intersections.

For a broader search on incomplete gene trees using cross-tree recombined
transitions as well as tree-local ones:

```bash
astralx -i /path/to/gene_trees.tre -o /path/to/output_species_tree.tre \
  --search-space S2 --intersection-method I2
```

Use `astralx --help` for the complete option list and `astralx --diagnose` to
check the packaged runtime, native libraries, driver, and GPU selection without
loading a dataset.

## Search-space presets

`--search-space` controls how broadly ASTRAL-X explores candidate species-tree
topologies. Choose `S1`, `S2`, or `S3`; bare numbers such as
`--search-space 2` are also accepted. “Complete” below means that missing taxa
are inserted into incomplete gene trees while constructing the search space.
Quartet weights are still calculated from the original gene trees.

| Preset | Search space | What it enables |
|---|---|---|
| **S1** | Incomplete, local | Uses topology candidates found directly within each original gene tree. Fastest and the default. |
| **S2** | Complete, full | Completes incomplete gene trees, constructs a distance-based guide tree, and combines compatible candidates across trees. Recommended broader search. |
| **S3** | Exhaustive | Includes everything in S2, then adds consensus-derived candidates, denser nearest-neighbour groups, remaining consensus-polytomy resolutions, large-polytomy handling, and resolutions derived from polytomous input gene trees. |

Moving from S1 to S3 progressively broadens the candidate topology set. A
larger preset can increase runtime and memory substantially and is not
guaranteed to change the inferred tree.

Most analyses only need one search-space preset. Individual search controls are
also available for specialized workflows; `astralx --help` lists them. When
options are combined, they are applied from left to right. A later individual
option can refine a preset, while a later preset selects its complete predefined
configuration.

## Intersection methods

Select an intersection method with `--intersection-method` (short form `--im`).
All four methods compute the same quartet weights; they differ only in their
memory use and performance strategy.

| Method | Name | Typical use |
|---|---|---|
| **I1** | Smaller-side traversal | Low auxiliary memory; directly walks the smaller side of each intersection. |
| **I2** | Prefix sum | General-purpose default with constant-time range intersections. |
| **I3** | Simple tree walk | Lean traversal that is often useful with very large full-search candidate sets. |
| **I4** | Bitset | Popcount-based path, usually strongest for smaller taxon sets and many gene trees. |

For example, `--im I3` and
`--weight-intersection-method simple-tree-walk` are equivalent; both forms are
accepted.

## Portable releases

The portable builder is the single local entry point for compilation, packaging,
smoke testing, checksumming, and manifest generation:

```bash
./build_portable.sh --version 1.0.0
```

The newly built application and release files are written under `dist/1.0.0/`.
On Linux x86-64, you may relink the `astralx` command to the newly generated application:

```bash
ASTRALX_DIR="$(realpath dist/1.0.0/astralx-1.0.0-linux-x86_64)"
mkdir -p "$HOME/.local/bin"
ln -sfn "$ASTRALX_DIR/astralx" "$HOME/.local/bin/astralx"
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  printf '%s\n' 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
export PATH="$HOME/.local/bin:$PATH"
```

On Linux and Windows, CUDA is included automatically when `nvcc` is available;
the resulting application always retains CPU fallback. Use `--without-cuda` to
deliberately create a CPU-only build. The resulting layout is:

```text
dist/
└── 1.0.0/
    ├── astralx-1.0.0-<os>-<arch>/
    ├── astralx-1.0.0-<os>-<arch>.<archive>
    ├── astralx-1.0.0-<os>-<arch>.<archive>.sha256
    └── astralx-1.0.0-<os>-<arch>.manifest.json
```

The entire local `dist/` directory is ignored by Git. Portable archives,
checksums, and manifests are distributed through GitHub Releases rather than
committed to the source repository.

Different version directories are retained. Rebuilding an existing version and
platform is refused by default; pass `--force` only when you intentionally want
to replace that exact artifact. If `--version` is omitted, the source version in
`src/astralx/Version.java` is used and embedded into the packaged JAR.

A portable archive is portable across machines with the same operating-system
and CPU family; no single native launcher can span Linux, Windows, macOS, x86-64,
and ARM64. CUDA builds contain CPU fallback and code for the GPU generations
supported by the build machine's CUDA toolkit, plus forward-compatible PTX.
CUDA acceleration requires an NVIDIA GPU and compatible installed driver; AMD,
Intel, and other GPUs currently use the CPU path. See
[`DOCS/portable-artifacts.md`](DOCS/portable-artifacts.md) for the release matrix
and compatibility details. Linux manifests record the actual minimum glibc
version used by the bundled ELF files, making the release baseline auditable.

---

# Developer guide

This section is for contributors working from a source checkout. Portable-release
users do not need these development dependencies.

## Development requirements

The development scripts target Linux and expect:

- JDK 21 or newer (`java`, `javac`, and `jar`)
- Python 3.9 or newer with the `venv` module
- Bash and common GNU tools, including `realpath`, `awk`, `sed`, `grep`, and
  `find`
- DendroPy 5, installed automatically into the repository's `.venv`

CUDA development is optional. Building the native GPU libraries requires an
NVIDIA CUDA toolkit with `nvcc`; running them also requires a compatible NVIDIA
GPU and driver. CPU development and all CPU tests work without CUDA. GNU
`time`, `curl`, `nvidia-smi`, and the bundled Linux x86-64 SimPhy executable are
optional integrations used for resource monitoring, notifications, GPU
monitoring, and simulation respectively.

On Ubuntu or Debian, the base packages can be installed with:

```bash
sudo apt install openjdk-21-jdk python3 python3-venv time curl
```

## Prepare a checkout

From the repository root, run:

```bash
./setup_dev.sh
```

The setup script checks the machine, creates `.venv`, installs
`requirements-dev.txt`, builds ASTRAL-X, builds the CUDA libraries when `nvcc`
is available, and runs the CPU test suite. It never installs system packages or
uses `sudo`. Useful variants are:

```bash
./setup_dev.sh --cpu-only       # explicitly skip the CUDA build
./setup_dev.sh --no-tests       # prepare and build without running tests
./setup_dev.sh --check          # verify an existing setup without changing it
```

The repository scripts automatically use `.venv/bin/python`, so activating the
virtual environment is optional. Run `./setup_dev.sh --help` for every setup
option.

## Build and run from source

Build Java only, or build the optional CUDA libraries separately:

```bash
./build.sh
./build_native.sh
```

`run.sh` builds stale sources automatically and uses CUDA with safe CPU fallback
by default:

```bash
./run.sh -i gene_trees.tre -o species_tree.tre --search-space S1
```

Use `--cpu` for a CPU-only run or `--no-build` when the current build should be
reused. JVM memory can be adjusted with `--xms`, `--xmx`, or the
`ASTRALX_XMS` and `ASTRALX_XMX` environment variables.

## Monitored runs

The monitoring wrapper records elapsed time, peak CPU memory, optional GPU
memory, exit status, and RF distance when a true tree is provided:

```bash
./run-astralx-with-monitor.sh \
  -i gene_trees.tre \
  -o results/species_tree.tre \
  --search-space S2 \
  --intersection-method I2 \
  --no-notify
```

Output parent directories are created automatically. Add `-t true_tree.tre` to
calculate RF distance, `--no-gpu-monitor` to disable GPU sampling, and
`--no-build` to reuse the current build. Set `NTFY_CHANNEL_NAME` to choose a
notification channel, or keep `--no-notify` for local development.

## Simulated experiments

The bundled SimPhy workflow can create a small dataset and run ASTRAL-X on it:

```bash
./sim.sh -t 10 -g 10 -rs 1 -r R1
./test-astralx-simulated.sh \
  -t 10 -g 10 -r R1 \
  --opts "--cpu --search-space S1" \
  --no-notify
```

For a controlled parameter sweep, pass all experiment sizes explicitly:

```bash
./run-bulk-simulated.sh \
  --taxa-list "10,20" \
  --genes-list "10,50" \
  --num-replicates 2 \
  --opts-list "--cpu --search-space S1;--cpu --search-space S2;--cpu --search-space S3" \
  --no-gpu-monitor \
  --no-notify
```

The bulk script intentionally defaults to one small 10-taxon, 10-gene run. Use
`--fresh` only when existing simulation outputs should be regenerated. A custom
simulation location can be supplied to the individual scripts with
`--simphy-data-dir`.

## Standard-dataset experiments

`run-bulk-standard.sh` works with the repository's configured benchmark layout.
Provide the dataset location and select ASTRAL-X explicitly:

```bash
./run-bulk-standard.sh \
  --base-dir /path/to/research \
  --dataset-dir /path/to/datasets/standard \
  --method astralx \
  --folder "37-taxon" \
  --opts "--search-space S2 -vv" \
  --no-notify
```

The configured folders expect their original benchmark subdirectory and file
names; inspect `./run-bulk-standard.sh --help` before launching a sweep.
ASTRAL-X-only runs need no baseline installation. The ASTER, ASTRAL, TreeQMC,
wQFMtree, SuperTriplets, and TMC binaries are required only when their
corresponding methods are selected.

Combine generated statistics with:

```bash
./collect-stats-simulated.sh --help
./collect-stats-standard.sh --help
```

## Tests

Run the CLI contract tests and the complete CPU regression suite with:

```bash
bash test/run_cli_tests.sh
bash test/run_tests.sh --cpu
```

All primary developer launchers resolve their resources from the repository
location, so they can be invoked by absolute path from another working
directory. Set `ASTRALX_PYTHON` only when an interpreter other than the managed
`.venv` should be used.
