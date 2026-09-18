# ASTRAL-X

## Scaling Coalescent-Based Species Tree Inference to 300,000 Taxa

ASTRAL-X is a complete algorithmic redesign of ASTRAL for scalable,
statistically consistent species tree inference from gene trees. Compact data
representations, memory-efficient search-space construction, and GPU
acceleration preserve ASTRAL's quartet-based optimization while making analyses
with hundreds of thousands of taxa practical: it reconstructed a 300,000-taxon
species tree in 12 hours using 100 GB of memory, and a 9,524-species angiosperm
tree in 16 minutes. An NVIDIA CUDA GPU is strongly recommended and is used
automatically when available; CPU execution remains a reliable fallback for
smaller analyses, compatibility, and development.

## Quick start

Download the archive for your platform (for example
`astralx-1.0.0-linux-x86_64.tar.gz`) from the GitHub Releases page. The archive
is self-contained: it bundles its own Java runtime and, on Linux and Windows,
the CUDA libraries, so no Java installation, CUDA toolkit, or repository clone
is needed. Extract it and run the bundled 37-taxon example:

```bash
tar -xzf astralx-1.0.0-linux-x86_64.tar.gz
cd astralx-1.0.0-linux-x86_64
./astralx --version
./astralx -i example/all_gt_37.tre -o example/predicted_st_37.tre
```

The example finishes in about a second. Every successful run ends with a
summary of the quartet score, running time, peak CPU RAM, and peak GPU VRAM,
and the inferred species tree is written to `example/predicted_st_37.tre`. The
reference tree `example/true_37.tre` is provided for comparison only; ASTRAL-X
never reads it during inference. On Windows use `astralx.exe`.

Keep the extracted directory together: the launcher needs its `bin/` and
`lib/` siblings. `./astralx --diagnose` reports the packaged runtime, native
libraries, driver, and GPU selection without loading a dataset, and
`./astralx --help` lists every option.

The same command works from a source checkout with JDK 21 or newer: the
repository's `./astralx` launcher compiles stale sources automatically before
running. See the [Developer guide](#developer-guide).

## Install the launcher on your PATH

The one-time block below installs the release under `~/.local/opt/` and links
the launcher into `~/.local/bin`, without `sudo`. Replace `/path/to/downloaded/`
with the archive's location:

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

Verify with `astralx --version`. Every remaining command in this README uses
`astralx` from the `PATH`; if you skipped this step, substitute the full path
to the extracted launcher.

## Basic usage

```bash
astralx -i /path/to/gene_trees.tre -o /path/to/species_tree.tre
```

The input file holds one Newick gene tree per line. Gene trees may be
incomplete (missing taxa) and may contain polytomies. Input and output paths
may be relative to the current directory or absolute. When `-o` is omitted the
species tree is written to standard output; `--log-file FILE` additionally
saves the run messages.

The defaults are `--auto`, `--search-space S1`, and `--intersection-method I2`:
ASTRAL-X tries CUDA first, falls back to CPU when CUDA is unavailable, uses the
smallest search space, and computes quartet weights with prefix-sum
intersections. Use `--cpu` to force CPU execution, `--gpu-strict` to fail
instead of falling back, and `-t N` to set the number of CPU worker threads.

The quartet score printed at the end always uses the native input gene-tree
topology, including unresolved multifurcations, and respects `--taxa-file` when
supplied. No monitoring wrapper is required for these core statistics.

For a broader search on incomplete gene trees that combines candidate
topologies across trees rather than using only tree-local ones:

```bash
astralx -i /path/to/gene_trees.tre -o /path/to/species_tree.tre \
  --search-space S2 --intersection-method I2
```

To score a given species tree instead of inferring one:

```bash
astralx -i /path/to/gene_trees.tre \
  --score-species-tree /path/to/species_tree.tre \
  -o /path/to/quartet_score.txt
```

## Search-space presets

`--search-space` controls how broadly ASTRAL-X explores candidate species-tree
topologies. Choose `S1`, `S2`, or `S3`; bare numbers such as
`--search-space 2` are also accepted. "Complete" below means that missing taxa
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
`scripts/clean.py --deterministic`. This preprocessing is parallel when multiple
CPU threads are configured and is reproducible regardless of worker scheduling.
Use `--keep-polytomy-during-inference` to retain unresolved input nodes in the
inference search and weight calculation instead. An unrooted three-way Newick
root is normalized during default refinement but does not represent a
biological polytomy; its unrooted topology is unchanged.

This inference choice never changes the scoring definition. Score-only mode and
the final score reported after inference always preserve native input
polytomies. When default inference had to refine a genuine multifurcation,
ASTRAL-X re-reads the original gene trees and scores the inferred topology on
their unresolved quartets. If the input has no genuine polytomy, or
`--keep-polytomy-during-inference` was used, the equivalent inference DP score
is reused without an additional scoring pass. A taxa allow-list is applied to
both inference and final scoring.

Most analyses only need one search-space preset. Individual search controls are
also available for specialized workflows; `astralx --help` lists them. When
options are combined, they are applied from left to right. A later individual
option can refine a preset, while a later preset selects its complete
predefined configuration.

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

For example, `--im I3` and `--weight-intersection-method simple-tree-walk` are
equivalent; both forms are accepted.

## Taxa extraction and taxon-restricted analysis

`--extract-taxa` writes the taxon names found in the input, sorted
deterministically with one name per line, and exits. For a file containing
multiple Newick trees, the default is the union of their taxa; add
`--taxa-set intersection` to retain only taxa present in every tree:

```bash
astralx -i /path/to/trees.nwk --extract-taxa -o /path/to/taxa.txt
astralx -i /path/to/trees.nwk --extract-taxa --taxa-set intersection -o /path/to/common_taxa.txt
```

From a source checkout, `./scripts/extract-taxa.sh --input trees.nwk --output taxa.txt`
wraps the same scanner and accepts `--union` or `--intersection`.

Fixed-tree scoring normally remains strict: the species tree must have exactly
the gene-tree union taxon set. To score an induced common subset instead, supply
a one-name-per-line allow-list:

```bash
astralx \
  --input /path/to/gene_trees.nwk \
  --score-species-tree /path/to/species_tree.nwk \
  --taxa-file /path/to/taxa.txt \
  --output /path/to/quartet_score.txt \
  --intersection-method I3
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
astralx \
  --input /path/to/gene_trees.nwk \
  --taxa-file /path/to/taxa.txt \
  --output /path/to/species_tree.nwk \
  --intersection-method I3
```

For inference, the effective universe is the listed taxa that occur in at least
one gene tree. Listed taxa absent from every gene tree are ignored, as are all
unlisted leaves. Each gene tree is induced onto that universe while it is
parsed: excluded leaves and resulting unary nodes are removed before compact
tree arrays, hashes, clusters, partitions, matrices, or GPU inputs are built.
Induced trees with two or three leaves remain available to the candidate
search, while trees with fewer than two selected leaves are discarded. The
inferred species tree and its reported quartet score therefore contain and
score only effective taxa.

Without `--taxa-file`, the established strict score-only and inference paths
are unchanged.

## Data directory

The dataset commands in this README, the research command sheet `cmd.txt`, and
the experiment scripts all locate data through one environment variable,
`PHYLOGENY_DATA_DIR`. Set it once, using `$HOME` rather than `~` because a
quoted `~` is not expanded by the shell:

```bash
grep -qF 'PHYLOGENY_DATA_DIR' "$HOME/.bashrc" || \
  printf '%s\n' 'export PHYLOGENY_DATA_DIR="$HOME/phylogeny/data"' >> "$HOME/.bashrc"
export PHYLOGENY_DATA_DIR="$HOME/phylogeny/data"
mkdir -p "$PHYLOGENY_DATA_DIR"
```

The conventional layout beneath it is:

| Path | Contents |
|---|---|
| `$PHYLOGENY_DATA_DIR/angio`, `avian-48`, `avian-363` | Biological datasets from the section below. |
| `$PHYLOGENY_DATA_DIR/simphy/data` | Simulated SimPhy datasets; the default for every simulation script. |
| `$PHYLOGENY_DATA_DIR/10k-astral-dataset` | The A10K benchmark. |
| `$PHYLOGENY_DATA_DIR/outputs/` | Compact reproducibility mirrors of experiment results. |

`astralx` itself accepts any path; the variable is only a shared convention so
that runs, mirrors, and collection tools never disagree about where data
lives. Scripts accept explicit overrides such as `--simphy-data-dir` and
`--data-dir`, and the simulation scripts fall back to the checkout's own
`simphy/data` when the variable is unset.

## Running ASTRAL-X on biological datasets

The three datasets below are hosted on Zenodo. Each block downloads one
archive into `$PHYLOGENY_DATA_DIR`, extracts it, removes the archive, and
lists the gene-tree file.

| Dataset | Taxa | Gene trees | Download | Recommended run |
|---|---:|---:|---:|---|
| Angiosperms | 9,524 | 353 | 25 MB | `--search-space S3 --intersection-method I3` |
| Avian-48 | 48 | 14,446 | 5 MB | `--search-space S2 --intersection-method I4` |
| Avian-363 | 363 | 63,430 | 1.0 GB | `--search-space S1` or `S2` with `--intersection-method I3` |

### Angiosperms

```bash
mkdir -p "$PHYLOGENY_DATA_DIR/angio"
wget -c -O "$PHYLOGENY_DATA_DIR/angio.tar" \
  "https://zenodo.org/records/21722787/files/angio.tar?download=1"
tar -xf "$PHYLOGENY_DATA_DIR/angio.tar" \
  -C "$PHYLOGENY_DATA_DIR/angio" --strip-components=1 &&
  rm "$PHYLOGENY_DATA_DIR/angio.tar"
ls -lh "$PHYLOGENY_DATA_DIR/angio/all_gt_angio.tre"
```

This dataset has relatively few gene trees for its number of taxa and includes
incomplete gene trees, so we use the exhaustive, consensus-enriched search
space (S3) with simple-tree-walk intersections (I3):

```bash
astralx -i "$PHYLOGENY_DATA_DIR/angio/all_gt_angio.tre" \
  -o "$PHYLOGENY_DATA_DIR/angio/out-astralx-angio-S3-I3.tre" \
  --search-space S3 \
  --intersection-method I3 \
  -vv
```

### Avian-48

```bash
mkdir -p "$PHYLOGENY_DATA_DIR/avian-48"
wget -c -O "$PHYLOGENY_DATA_DIR/avian-48.tar" \
  "https://zenodo.org/records/21722787/files/avian-48.tar?download=1"
tar -xf "$PHYLOGENY_DATA_DIR/avian-48.tar" \
  -C "$PHYLOGENY_DATA_DIR/avian-48" --strip-components=1 &&
  rm "$PHYLOGENY_DATA_DIR/avian-48.tar"
ls -lh "$PHYLOGENY_DATA_DIR/avian-48/avian-48-gt.tre"
```

With few taxa and many gene trees, the S2 search space with bitset
intersections (I4) is the strongest combination:

```bash
astralx -i "$PHYLOGENY_DATA_DIR/avian-48/avian-48-gt.tre" \
  -o "$PHYLOGENY_DATA_DIR/avian-48/out-astralx-avian-48-S2-I4.tre" \
  --search-space S2 \
  --intersection-method I4 \
  -vv
```

### Avian-363

```bash
mkdir -p "$PHYLOGENY_DATA_DIR/avian-363"
wget -c -O "$PHYLOGENY_DATA_DIR/avian-363.tar" \
  "https://zenodo.org/records/21722787/files/avian-363.tar?download=1"
tar -xf "$PHYLOGENY_DATA_DIR/avian-363.tar" \
  -C "$PHYLOGENY_DATA_DIR/avian-363" --strip-components=1 &&
  rm "$PHYLOGENY_DATA_DIR/avian-363.tar"
ls -lh "$PHYLOGENY_DATA_DIR/avian-363/63430.gene.trees"
```

Tree-local search space (S1) on completed gene trees with simple-tree-walk
intersections (I3):

```bash
astralx -i "$PHYLOGENY_DATA_DIR/avian-363/63430.gene.trees" \
  -o "$PHYLOGENY_DATA_DIR/avian-363/out-astralx-avian-363-complete-local-S1-I3.tre" \
  --search-space S1 \
  --autocomplete-incomplete-gene-trees \
  --intersection-method I3 \
  -vv
```

The broader S2 search space with the same intersection method:

```bash
astralx -i "$PHYLOGENY_DATA_DIR/avian-363/63430.gene.trees" \
  -o "$PHYLOGENY_DATA_DIR/avian-363/out-astralx-avian-363-S2-I3.tre" \
  --search-space S2 \
  --intersection-method I3 \
  -vv
```

---

# Developer guide

This section is for contributors working from a source checkout.
Portable-release users do not need these development dependencies.

## Repository layout

The repository root deliberately holds only the `./astralx` launcher, this
`README.md`, the research command sheet `cmd.txt`, and the top-level
directories:

| Path | Contents |
|---|---|
| `astralx` | Developer launcher; forwards every argument to `scripts/run.sh`. |
| `cmd.txt` | Research command sheet: the exact experiment invocations, written against `$PHYLOGENY_DATA_DIR`. |
| `src/` | Java sources (`src/astralx`) and CUDA kernels (`src/native`). |
| `scripts/` | Every build, run, simulation, benchmark, and upload script, plus the Python helpers (`rf.py`, `clean.py`, `root_by_outgroups.py`, `analyze-dataset.py`) and `requirements-dev.txt`. |
| `example/` | Bundled inputs: `all_gt_37.tre`, `all_gt_48.tre`, `all_gt_200.tre` with their reference trees `true_37.tre`, `true_48.tre`, `true_200.tre`, and the sample output `out_astralx_37.tre`. |
| `test/` | CLI contract tests, regression suite, and script-level tests. |
| `DOCS/`, `DESIGN/` | Design notes and reproducibility documentation. |
| `packaging/` | The portable Unix launcher installed into release images. |
| `simphy/` | Bundled SimPhy executable and simulator driver. |
| `build/`, `native/`, `dist/`, `.venv/`, `crash-logs/`, `data/` | Generated or local: compiled classes, CUDA libraries, portable releases, the Python environment, crash reports, and local data. All but `native/` are git-ignored. |

Every script under `scripts/` locates the repository root from its own
location, so `./scripts/<name>.sh` works from the root and by absolute path
from anywhere else. When ASTRAL-X aborts with a Java exception or memory
failure it writes a report to `crash-logs/` inside the working directory (the
developer launcher pins this to the repository's `crash-logs/`, including
HotSpot `hs_err`-style files); that directory is git-ignored.

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
`time`, `curl`, `nvidia-smi`, and the bundled Linux x86-64 SimPhy executable
are optional integrations used for resource monitoring, notifications, GPU
monitoring, and simulation respectively.

On Ubuntu or Debian, the base packages can be installed with:

```bash
sudo apt install openjdk-21-jdk python3 python3-venv time curl
```

## Prepare a checkout

From the repository root, run:

```bash
./scripts/setup_dev.sh
```

The setup script checks the machine, creates `.venv`, installs
`scripts/requirements-dev.txt`, builds ASTRAL-X, builds the CUDA libraries when
`nvcc` is available, and runs the CPU test suite. It never installs system
packages or uses `sudo`. Useful variants are:

```bash
./scripts/setup_dev.sh --cpu-only       # explicitly skip the CUDA build
./scripts/setup_dev.sh --no-tests       # prepare and build without running tests
./scripts/setup_dev.sh --check          # verify an existing setup without changing it
```

The repository scripts automatically use `.venv/bin/python`, so activating the
virtual environment is optional. Run `./scripts/setup_dev.sh --help` for every
setup option.

## Build and run from source

Build Java only, or build the optional CUDA libraries separately:

```bash
./scripts/build.sh
./scripts/build_native.sh
```

The `./astralx` launcher at the repository root forwards to `scripts/run.sh`.
It builds stale sources automatically and uses CUDA with safe CPU fallback by
default:

```bash
./astralx -i example/all_gt_37.tre -o example/predicted_st_37.tre --search-space S1
```

Use `--cpu` for a CPU-only run or `--no-build` when the current build should be
reused. JVM memory can be adjusted with `--xms`, `--xmx`, or the
`ASTRALX_XMS` and `ASTRALX_XMX` environment variables. The developer launcher
prints a short build banner before the program output, so pass `-o` rather
than relying on standard output when capturing results such as
`--extract-taxa` lists.

## Monitored runs

The monitoring wrapper records elapsed time, peak CPU memory, optional GPU
memory, exit status, and the RF distance to a reference tree when one is
provided:

```bash
./scripts/run-astralx-with-monitor.sh \
  -i gene_trees.tre \
  -o results/species_tree.tre \
  --search-space S2 \
  --intersection-method I2 \
  --no-notify
```

Output parent directories are created automatically. Add
`--reference-species-tree true_tree.tre` to calculate the RF distance,
`--no-gpu-monitor` to disable GPU sampling, and `--no-build` to reuse the
current build. Set `NTFY_CHANNEL_NAME` to choose a notification channel, or
keep `--no-notify` for local development. `cmd.txt` collects the monitored
invocations used for the published experiments.

## Simulated experiments

The bundled SimPhy workflow can create a small dataset and run ASTRAL-X on it.
Datasets are written under `$PHYLOGENY_DATA_DIR/simphy/data` (see
[Data directory](#data-directory)):

```bash
./scripts/sim.sh -t 10 -g 10 -rs 1 -r R1
./scripts/test-astralx-simulated.sh \
  -t 10 -g 10 -r R1 \
  --opts "--cpu --search-space S1" \
  --no-notify
```

For a controlled parameter sweep, pass all experiment sizes explicitly:

```bash
./scripts/run-bulk-simulated.sh \
  --taxa-list "10,20" \
  --genes-list "10,50" \
  --num-replicates 2 \
  --opts-list "--cpu --search-space S1;--cpu --search-space S2;--cpu --search-space S3" \
  --no-gpu-monitor \
  --no-notify
```

The bulk script intentionally defaults to one small 10-taxon, 10-gene run. Use
`--dry-run` to print the run plan without starting anything and `--fresh` only
when existing simulation outputs should be regenerated.

To avoid recomputing replicates whose results you already have (for example
from a sweep on another machine), list them in
`ALREADY_COMPLETED_SIMULATED_CONFIGS` near the top of
`scripts/run-bulk-simulated.sh` as exact
`TAXA,GENE_TREES,SB,SPMIN,SPMAX,REPLICATE` tuples. The list ships empty, so
every replicate runs by default. `sim.sh` still prepares the surrounding
dataset batch; only the listed per-replicate inference is skipped, and the run
plan reports it inline (`… / R3-R4 / … (already completed earlier, skipped:
R1-R2)`) so the skip is visible before anything starts.

Every script that touches the simulated data tree (`sim.sh`,
`sim_incomplete.sh`, `run-bulk-simulated.sh`, `test-astralx-simulated.sh`,
`collect-stats-simulated.sh`, `download-bulk-simulated.sh`,
`upload-bulk-simulated.sh`, and the mirror tools) resolves its location the
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
only result artifacts and the SimPhy `.command`/`.params` files. For the
default data tree it is written to `$PHYLOGENY_DATA_DIR/outputs/simphy`,
organized as `astralx_outputs/<dataset>/<replicate>/<setting>`. Simulated gene
trees, true trees, databases, ZIPs, and `stat-sim.csv` are explicitly refused.

```bash
./scripts/sync-simulated-outputs.sh --dry-run       # preview a back-fill
./scripts/sync-simulated-outputs.sh                 # mirror older results
./scripts/upload-bulk-simulated-outputs.sh --dry-run
./scripts/upload-bulk-simulated-outputs.sh --sync   # refresh, confirm, then publish
```

Use `--simphy-outputs-dir` to override the mirror root or
`--no-outputs-mirror` for an individual/bulk run. See
[`DOCS/simulated-outputs-reproducibility.md`](DOCS/simulated-outputs-reproducibility.md)
for the layout, safeguards, and reproduction procedure.

## Standard-dataset experiments

`scripts/run-bulk-standard.sh` works with the repository's configured benchmark
layout. Provide the dataset location and select ASTRAL-X explicitly:

```bash
./scripts/run-bulk-standard.sh \
  --base-dir "$HOME/phylogeny" \
  --dataset-dir "$HOME/phylogeny/datasets/standard" \
  --method astralx \
  --folder "37-taxon" \
  --opts "--search-space S2 --intersection-method I2 -vv" \
  --no-notify
```

The configured folders expect their original benchmark subdirectory and file
names; inspect `./scripts/run-bulk-standard.sh --help` before launching a
sweep. ASTRAL-X-only runs need no baseline installation. The ASTER, ASTRAL,
TreeQMC, wQFMtree, SuperTriplets, and TMC binaries are required only when their
corresponding methods are selected.

Combine generated statistics with:

```bash
./scripts/collect-stats-simulated.sh --help
./scripts/collect-stats-standard.sh --help
```

To remove one method's bulk-standard statistics before collecting again, first
preview and then confirm the cleanup:

```bash
./scripts/clear-bulk-standard.sh --method astralx --dry-run
./scripts/clear-bulk-standard.sh --method astralx --yes
./scripts/collect-stats-standard.sh
```

The cleaner preserves output trees and logs by default. Add `--all-results` to
remove the selected method's complete output directories. Historical
`stelar_outputs` can be selected with `--method stelar`; use `--method all` to
clean statistics for every supported method.

## A10K experiments

The A10K runner takes the dataset root and one tree type or a quoted
semicolon-separated list. This runs every selected replicate and setting once
with true gene trees and once with estimated gene trees:

```bash
./scripts/run-a10k.sh \
  --data-dir "$PHYLOGENY_DATA_DIR/10k-astral-dataset" \
  --tree-type "true;estimated" \
  --opts "--search-space S1 --intersection-method I1"
```

To clear every result produced by the A10K runner, including all settings and
both tree types across every replicate, preview the exact targets first:

```bash
./scripts/clear-a10k.sh --data-dir "$PHYLOGENY_DATA_DIR/10k-astral-dataset" --dry-run
./scripts/clear-a10k.sh --data-dir "$PHYLOGENY_DATA_DIR/10k-astral-dataset" --yes
```

This removes only `10k-simphy/R*/astralx_outputs` and the A10K merged scores
CSV; gene trees, rooted gene trees, and species trees are preserved.

Each A10K run also maintains a compact reproducibility mirror, exactly like the
simulated runs: the results stay in the data tree unchanged and are copied
(inferred tree, CSVs, command record, run log; never gene trees or species
trees) to an `outputs` directory next to the dataset, so
`$PHYLOGENY_DATA_DIR/10k-astral-dataset` mirrors into
`$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset/astralx_outputs/10k-simphy/<replicate>/<tree_type>/<setting>`,
beside the SimPhy mirror in `$PHYLOGENY_DATA_DIR/outputs/simphy`. The merged
scores CSV from `collect-scores-a10k.sh` and the rooting command for estimated
gene trees are mirrored as well.

```bash
./scripts/sync-a10k-outputs.sh --data-dir "$PHYLOGENY_DATA_DIR/10k-astral-dataset" --dry-run   # preview a back-fill
./scripts/sync-a10k-outputs.sh --data-dir "$PHYLOGENY_DATA_DIR/10k-astral-dataset"             # mirror older results
./scripts/upload-a10k-outputs.sh --data-dir "$PHYLOGENY_DATA_DIR/10k-astral-dataset" --dry-run
./scripts/upload-a10k-outputs.sh --data-dir "$PHYLOGENY_DATA_DIR/10k-astral-dataset" --sync    # refresh, confirm, then publish
```

Use `--outputs-dir` to override the mirror root or `--no-outputs-mirror` to
skip it for one run; `scripts/clear-a10k.sh` keeps the mirror unless
`--include-mirror` is given. See
[`DOCS/a10k-outputs-reproducibility.md`](DOCS/a10k-outputs-reproducibility.md)
for the layout, safeguards, and reproduction procedure.

## Portable releases

The portable builder is the single local entry point for compilation,
packaging, smoke testing, checksumming, and manifest generation:

```bash
./scripts/build_portable.sh --version 1.0.0
```

The newly built application and release files are written under `dist/1.0.0/`.
On Linux x86-64, you may relink the `astralx` command to the newly generated
application:

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
to replace that exact artifact. If `--version` is omitted, the source version
in `src/astralx/Version.java` is used and embedded into the packaged JAR.

A portable archive is portable across machines with the same operating-system
and CPU family; no single native launcher can span Linux, Windows, macOS,
x86-64, and ARM64. CUDA builds contain CPU fallback and code for the GPU
generations supported by the build machine's CUDA toolkit, plus
forward-compatible PTX. CUDA acceleration requires an NVIDIA GPU and compatible
installed driver; AMD, Intel, and other GPUs currently use the CPU path. See
[`DOCS/portable-artifacts.md`](DOCS/portable-artifacts.md) for the release
matrix and compatibility details. Linux manifests record the actual minimum
glibc version used by the bundled ELF files, making the release baseline
auditable.

## Tests

Run the CLI contract tests and the complete CPU regression suite with:

```bash
bash test/run_cli_tests.sh
bash test/run_tests.sh --cpu
```

Script-level tests such as `test/test_phylogeny_data_dir.sh`,
`test/test_bulk_simulated_skip_list.sh`, and the outputs-mirror tests can be run
individually with `bash`. All developer scripts resolve their resources from
the repository location, so they can be invoked by absolute path from another
working directory. Set `ASTRALX_PYTHON` only when an interpreter other than the
managed `.venv` should be used.
