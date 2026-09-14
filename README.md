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
ASTRALX_INSTALL_ROOT="$HOME/.local/opt/astralx/1.0.0"
mkdir -p "$ASTRALX_INSTALL_ROOT"
tar -xzf /path/to/downloaded/astralx-1.0.0-linux-x86_64.tar.gz \
  -C "$ASTRALX_INSTALL_ROOT"
ASTRALX_DIR="$(realpath "$ASTRALX_INSTALL_ROOT/astralx-1.0.0-linux-x86_64")"
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

Every successful analysis ends with a built-in summary of the quartet score,
running time, maximum CPU RAM, and maximum GPU VRAM. The reported quartet score
always uses the native input gene-tree topology, including unresolved
multifurcations, and respects `--taxa-file` when supplied. No monitoring wrapper
is required for these core statistics.

For a broader search on incomplete gene trees using cross-tree recombined
transitions as well as tree-local ones:

```bash
astralx -i /path/to/gene_trees.tre -o /path/to/output_species_tree.tre \
  --search-space S2 --intersection-method I2
```

Use `astralx --help` for the complete option list and `astralx --diagnose` to
check the packaged runtime, native libraries, driver, and GPU selection without
loading a dataset.

## Taxa extraction and taxon-restricted analysis

`extract-taxa.sh` uses the same Newick leaf-token scanner as ASTRAL-X. It writes
only taxon names, sorted deterministically with one name per line. For a file
containing multiple Newick trees (one tree per non-empty line), the default is
the union of their taxa:

```bash
./extract-taxa.sh \
  --input /path/to/trees.nwk \
  --output /path/to/taxa.txt
```

Use `--intersection` to retain only taxa present in every tree. The equivalent
core CLI is `astralx -i trees.nwk --extract-taxa --taxa-set union|intersection`;
when `-o` is omitted, names are written to standard output.

Fixed-tree scoring normally remains strict: the species tree must have exactly
the gene-tree union taxon set. To score an induced common subset instead, supply
a one-name-per-line allow-list:

```bash
./run.sh \
  --input /path/to/gene_trees.nwk \
  --score-species-tree /path/to/species_tree.nwk \
  --taxa-file /path/to/taxa.txt \
  --output /path/to/quartet_score.txt \
  --intersection-method I3 \
  --auto
```

The effective scoring universe is the listed taxa present in both the gene-tree
union and the supplied species tree. Taxa outside the list are pruned, unary
nodes created by pruning are suppressed, and gene trees retaining fewer than
four selected taxa are discarded because they contribute zero quartets. The run
reports duplicate list entries, taxa absent from the gene-tree union or species
tree, ignored outside taxa, and the mean/minimum/maximum number of listed taxa
missing per gene tree. Missing taxa are never inserted or assigned an arbitrary
placement.

The same allow-list can restrict species-tree inference:

```bash
./run.sh \
  --input /path/to/gene_trees.nwk \
  --taxa-file /path/to/taxa.txt \
  --output /path/to/species_tree.nwk \
  --intersection-method I3 \
  --auto
```

For inference, the effective universe is the listed taxa that occur in at least
one gene tree. Listed taxa absent from every gene tree are ignored, as are all
unlisted leaves. Each gene tree is induced onto that universe while it is parsed:
excluded leaves and resulting unary nodes are removed before compact tree arrays,
hashes, clusters, partitions, matrices, or GPU inputs are built. Induced trees
with two or three leaves remain available to the candidate search, while trees
with fewer than two selected leaves are discarded. The inferred species tree and
its reported quartet score therefore contain and score only effective taxa.

Without `--taxa-file`, the established strict score-only and inference paths are
unchanged.

## Running ASTRAL-X on Biological Datasets

Run the commands below from the directory where you want the `data/` folder.

### Angiosperms

Download, extract, clean up, and verify the angiosperm dataset with:

```bash
mkdir -p data/angio
wget -c \
  -O data/angio.tar \
  "https://zenodo.org/records/21722787/files/angio.tar?download=1"
tar -xf data/angio.tar \
  -C data/angio \
  --strip-components=1 &&
  rm data/angio.tar
test -s data/angio/all_gt_angio.tre &&
  ls -lh data/angio/all_gt_angio.tre
```

Because this dataset has relatively few gene trees compared with its number of
taxa and includes incomplete gene trees, we use ASTRAL-X's exhaustive,
consensus-enriched search space (S3) with simple-tree-walk intersections (I3):

```bash
astralx -i data/angio/all_gt_angio.tre \
  -o data/angio/out-astralx-angio-S3-I3.tre \
  --search-space S3 \
  --intersection-method I3 \
  -vv
```

### Avian-48

Download, extract, clean up, and verify the 48-taxon avian dataset with:

```bash
mkdir -p data/avian-48
wget -c \
  -O data/avian-48.tar \
  "https://zenodo.org/records/21722787/files/avian-48.tar?download=1"
tar -xf data/avian-48.tar \
  -C data/avian-48 \
  --strip-components=1 &&
  rm data/avian-48.tar
test -s data/avian-48/avian-48-gt.tre &&
  ls -lh data/avian-48/avian-48-gt.tre
```

Run ASTRAL-X with the S2 search space and bitset intersections (I4):

```bash
astralx -i data/avian-48/avian-48-gt.tre \
  -o data/avian-48/out-astralx-avian-48-S2-I4.tre \
  --search-space S2 \
  --intersection-method I4 \
  -vv
```

### Avian-363

Download, extract, clean up, and verify the 363-taxon avian dataset with:

```bash
mkdir -p data/avian-363
wget -c \
  -O data/avian-363.tar \
  "https://zenodo.org/records/21722787/files/avian-363.tar?download=1"
tar -xf data/avian-363.tar \
  -C data/avian-363 \
  --strip-components=1 &&
  rm data/avian-363.tar
test -s data/avian-363/63430.gene.trees &&
  ls -lh data/avian-363/63430.gene.trees
```

Run ASTRAL-X with the tree-local search space (S1) and simple-tree-walk intersections (I3):

```bash
astralx \
  -i data/avian-363/63430.gene.trees \
  -o data/avian-363/out-astralx-avian-363-complete-local-S1-I3.tre \
  --search-space S1 \
  --autocomplete-incomplete-gene-trees \
  --intersection-method I3 \
  -vv
```

For the broader search space with cross-tree transitions (S2) with the same I3 intersection method:

```bash
astralx -i data/avian-363/63430.gene.trees \
  -o data/avian-363/out-astralx-avian-363-S2-I3.tre \
  --search-space S2 \
  --intersection-method I3 \
  -vv
```

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

By default, inference detects input polytomies and refines them to binary trees
with the deterministic first-pair algorithm used by
`clean.py --deterministic`. This preprocessing is parallel when multiple CPU
threads are configured and is reproducible regardless of worker scheduling.
Use `--keep-polytomy-during-inference` to retain unresolved input nodes in the
inference search and weight calculation instead. An unrooted three-way Newick
root is normalized during default refinement but does not represent a biological
polytomy; its unrooted topology is unchanged.

This inference choice never changes the scoring definition. Score-only mode and
the final score reported after inference always preserve native input
polytomies. When default inference had to refine a genuine multifurcation,
ASTRAL-X re-reads the original gene trees and scores the inferred topology on
their unresolved quartets. If the input has no genuine polytomy, or
`--keep-polytomy-during-inference` was used, the equivalent inference DP score is
reused without an additional scoring pass. A taxa allow-list is applied to both
inference and final scoring.

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
`--fresh` only when existing simulation outputs should be regenerated.

Individual replicates that must never be inferred are listed in
`EXCLUDED_SIMULATED_CONFIGS` near the top of `run-bulk-simulated.sh`, as exact
`TAXA,GENE_TREES,SB,SPMIN,SPMAX,REPLICATE` tuples. `sim.sh` still prepares the
surrounding dataset batch; only the listed per-replicate inference is skipped,
and the run plan reports it inline (`… / R3-R4 / … (excluded: R1-R2)`) so the
omission is visible before anything starts.

Every script that touches the simulated data tree — `sim.sh`,
`sim_incomplete.sh`, `run-bulk-simulated.sh`, `test-astralx-simulated.sh`,
`collect-stats-simulated.sh`, `download-bulk-simulated.sh`,
`upload-bulk-simulated.sh`, and the mirror tools — resolves its location the
same way, so they never disagree about where the datasets live:

1. an explicit `--simphy-data-dir` (or `--local-dir` / `--data-dir`) argument,
2. otherwise `$PHYLOGENY_DATA_DIR/simphy/data`,
3. otherwise this checkout's own `simphy/data`.

Experiment output directories and CSV rows derive their `setting` name from the
meaningful algorithm options. For example,
`--search-space S1 --intersection-method I2 -vv` becomes
`search-space_S1__intersection-method_I2`; verbosity flags are omitted, while
additional options are appended using the same `option_value` format. Legacy
intersection-method names are normalized to their compact `I1`-`I4` names.

Each simulated run also maintains a compact reproducibility mirror containing
only result artifacts and the SimPhy `.command`/`.params` files. For the normal
`simphy/data` tree it is written to `outputs/simphy`, organized as
`astralx_outputs/<dataset>/<replicate>/<setting>`. Simulated gene trees, true
trees, databases, ZIPs, and `stat-sim.csv` are explicitly refused.

```bash
./sync-simulated-outputs.sh --dry-run       # preview a back-fill
./sync-simulated-outputs.sh                 # mirror older results
./upload-bulk-simulated-outputs.sh --dry-run
./upload-bulk-simulated-outputs.sh --sync   # refresh, confirm, then publish
```

Use `--simphy-outputs-dir` to override the mirror root or
`--no-outputs-mirror` for an individual/bulk run. See
[`DOCS/simulated-outputs-reproducibility.md`](DOCS/simulated-outputs-reproducibility.md)
for the layout, safeguards, and reproduction procedure.

## Standard-dataset experiments

`run-bulk-standard.sh` works with the repository's configured benchmark layout.
Provide the dataset location and select ASTRAL-X explicitly:

```bash
./run-bulk-standard.sh \
  --base-dir /path/to/research \
  --dataset-dir /path/to/datasets/standard \
  --method astralx \
  --folder "37-taxon" \
  --opts "--search-space S2 --intersection-method I2 -vv" \
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

To remove one method's bulk-standard statistics before collecting again, first
preview and then confirm the cleanup:

```bash
./clear-bulk-standard.sh --method astralx --dry-run
./clear-bulk-standard.sh --method astralx --yes
./collect-stats-standard.sh
```

The cleaner preserves output trees and logs by default. Add `--all-results` to
remove the selected method's complete output directories. Historical
`stelar_outputs` can be selected with `--method stelar`; use `--method all` to
clean statistics for every supported method.

The A10K runner accepts one tree type or a quoted semicolon-separated list. For
example, this runs every selected replicate and setting once with true gene
trees and once with estimated gene trees:

```bash
./run-a10k.sh \
  --data-dir /path/to/10k-astral-dataset \
  --tree-type "true;estimated" \
  --opts "--search-space S1 --intersection-method I1"
```

To clear every result produced by the A10K runner, including all settings and
both tree types across every replicate, preview the exact targets first:

```bash
./clear-a10k.sh --data-dir /path/to/10k-astral-dataset --dry-run
./clear-a10k.sh --data-dir /path/to/10k-astral-dataset --yes
```

This removes only `10k-simphy/R*/astralx_outputs` and the A10K merged scores
CSV; gene trees, rooted gene trees, and species trees are preserved.

Each A10K run also maintains a compact reproducibility mirror, exactly like the
simulated runs: the results stay in the data tree unchanged and are copied
(inferred tree, CSVs, command record, run log; never gene trees or species
trees) to an `outputs` directory next to the dataset, so
`$PHYLOGENY_DATA_DIR/10k-astral-dataset` mirrors into
`$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset/astralx_outputs/10k-simphy/<replicate>/<tree_type>/<setting>`,
beside the SimPhy mirror in `$PHYLOGENY_DATA_DIR/outputs/simphy`.
The merged scores CSV from `collect-scores-a10k.sh` and the rooting command for
estimated gene trees are mirrored as well.

```bash
./sync-a10k-outputs.sh --data-dir data/10k-astral-dataset --dry-run   # preview a back-fill
./sync-a10k-outputs.sh --data-dir data/10k-astral-dataset             # mirror older results
./upload-a10k-outputs.sh --data-dir data/10k-astral-dataset --dry-run
./upload-a10k-outputs.sh --data-dir data/10k-astral-dataset --sync    # refresh, confirm, then publish
```

Use `--outputs-dir` to override the mirror root or `--no-outputs-mirror` to
skip it for one run; `clear-a10k.sh` keeps the mirror unless `--include-mirror`
is given. See
[`DOCS/a10k-outputs-reproducibility.md`](DOCS/a10k-outputs-reproducibility.md)
for the layout, safeguards, and reproduction procedure.

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
