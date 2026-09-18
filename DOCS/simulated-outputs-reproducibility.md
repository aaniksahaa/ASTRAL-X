# Reproducibility fingerprint for simulated runs

Status: implemented (2026-09-12). This document is the source of truth for the
outputs mirror, command records, and the bulk uploader. Any change to that
implementation must be reflected here.

## 1. Problem

SimPhy datasets and every inference result live in one tree:

```
$PHYLOGENY_DATA_DIR/simphy/data/<dataset>/<R>/{all_gt.tre, s_tree.trees, <method>_outputs/<setting>/...}
```

`all_gt.tre` dominates the size; the results are tiny. Sharing results for
several methods along with everything needed to regenerate the data and rerun
the inference required either shipping the gene trees or hand-collecting files.

## 2. Requirements

1. The data tree keeps its exact layout. Reading and writing there is unchanged
   and fully backward compatible.
2. Every run additionally writes its results into a separate outputs tree,
   organised method-first:
   `<outputs>/<method>_outputs/<dataset>/<R>/<setting>/`.
3. Each `<dataset>` directory in the outputs tree carries the SimPhy command
   file that generated the data (the `.command` file; `.params` alongside).
   Without it the mirror is not considered reproducible.
4. The mirror contains everything the run wrote into its results directory
   (inferred tree, CSVs, run markers, log) and never the simulated inputs:
   `all_gt.tre`, `s_tree.trees`, SimPhy `.db` files.
5. Each results directory contains the exact command that produced it: every
   flag and parameter, absolute paths, code revision, and the outer commands.
6. A bulk uploader publishes the outputs tree to the Hugging Face dataset repo
   with the same layout under `ph/d/simulated/outputs/`, uploading only outputs
   and command files.
7. `run-bulk-simulated.sh` and the uploader print the exact cases they are
   about to process, one line per `<dataset> / <replicates> / <setting>`, and
   ask for confirmation.

## 3. Layout

Data tree (unchanged):

```
$PHYLOGENY_DATA_DIR/simphy/data/
  t_<taxa>_g_<genes>_sb_<sb>_spmin_<min>_spmax_<max>/          # <dataset>
    <dataset>.command  <dataset>.params  <dataset>.db          # written by SimPhy
    R1/ all_gt.tre  s_tree.trees  stat-sim.csv
        astralx_outputs/<setting>/ out-astralx.tre  out-astralx.command
                                   out-astralx_stats.csv  stat-astralx.csv
                                   .astralx.lock  .astralx.success  .astralx_run.log
  <dataset>_incomplete/                                         # from sim_incomplete.sh
    <dataset>_incomplete.command                                # derivation record
    R1/ all_gt.tre  s_tree.trees  astralx_outputs/...
```

Outputs mirror (new):

```
$PHYLOGENY_DATA_DIR/outputs/simphy/
  astralx_outputs/
    <dataset>/
      <dataset>.command                 # copied from the data tree
      <dataset>.params
      R1/<setting>/  <exact copy of the results leaf above>
      R2/...
    <dataset>_incomplete/
      <dataset>.command                 # base SimPhy command
      <dataset>_incomplete.command      # pruning parameters (fraction, seed, min-keep)
      R1/<setting>/...
  <other-method>_outputs/<dataset>/...  # identical shape for any method
```

Remote (Hugging Face, repo `imAniksahA/blab`, type dataset):

```
ph/d/simulated/outputs/<method>_outputs/<dataset>/...   # same as the local mirror
```

`<setting>` is the option-derived folder name from `experiment-setting-name.sh`
(for example `search-space_S1__intersection-method_I1`; `default` when no
result-affecting option is given).

### Data root

Every script that reads or writes the data tree resolves it through
`astralx_resolve_simphy_data_dir` (`scripts/phylogeny-data-dir.sh`), in this
order:

1. an explicit argument (`--simphy-data-dir`, `--local-dir`, `--data-dir`),
2. `$PHYLOGENY_DATA_DIR/simphy/data`,
3. a repository-local fallback (`<checkout>/simphy/data`, or
   `<simphy-dir>/data` where the script takes `--simphy-dir`).

This applies to `sim.sh`, `simphy/run_simulator.sh`, `sim_incomplete.sh`,
`run-bulk-simulated.sh`, `test-astralx-simulated.sh`,
`collect-stats-simulated.sh`, `download-bulk-simulated.sh`,
`upload-bulk-simulated.sh`, `sync-simulated-outputs.sh`,
`upload-bulk-simulated-outputs.sh`, and
`test/simmat_comparison/run_upgma_tests.sh`. A script that resolved the tree differently
would generate, collect, mirror, or publish from a directory the others never
look at, which is why the rule lives in one function.

### Default mirror location rule

The mirror root is derived from the resolved data directory:

| data directory              | mirror root                |
|-----------------------------|----------------------------|
| `<root>/simphy/data`        | `<root>/outputs/simphy`    |
| any other `<parent>/data`   | `<parent>/outputs`         |
| any other `<dir>`           | `<dir>_outputs`            |

`--simphy-outputs-dir PATH` overrides the rule. A mirror root equal to, inside,
or containing the data directory is refused.

## 4. Components

| File | Role |
|---|---|
| `scripts/phylogeny-data-dir.sh` | Shared helper: the data-root resolution order above (`astralx_prepare_simphy_data_dir`, `astralx_resolve_simphy_data_dir`). |
| `scripts/simphy-outputs-dir.sh` | Shared helper: default-root rule, guards, atomic leaf mirroring (`astralx_mirror_directory_atomic`, also used by the A10K mirror), command-file copying, forbidden-file detection, dataset-name validation. |
| `test-astralx-simulated.sh` | Mirrors its results leaf after every run and on the "already completed" skip path (free back-fill). Appends the simulated-run context to the command record. Flags `--simphy-outputs-dir`, `--no-outputs-mirror`. |
| `run-astralx-with-monitor.sh` | Writes `<output>.command` (e.g. `out-astralx.command`) beside the output tree before the run and appends exit code and running time after. Applies to real-dataset runs too. |
| `run-bulk-simulated.sh` | Prints the run plan, confirms, forwards the mirror flags. Flags `--dry-run`, `--yes`. |
| `sim_incomplete.sh` | Writes `<dataset>_incomplete.command` recording fraction, seed, min-keep and the base command path. Resolves the data root and forwards it to `sim.sh`. |
| `sim.sh`, `simphy/run_simulator.sh` | Generate into the resolved data root instead of a checkout-relative `data/`. |
| `collect-stats-simulated.sh` | Reads the resolved data root (`--simphy-data-dir`); the combined CSV carries `optimal-quartet-score`. |
| `download-bulk-simulated.sh` | Downloads into the resolved data root. |
| `sync-simulated-outputs.sh` | Back-fills or refreshes the whole mirror from the data tree. Flags `--dry-run`, `--methods`, `--quiet`. |
| `upload-bulk-simulated-outputs.sh` | Uploads `<method>_outputs/<dataset>` directories as folders. Flags `--sync`, `--methods`, `--min-taxa`, `--min-gene-trees`, `--exclude-incomplete`, `--allow-missing-command`, `--dry-run`, `--yes`, `--repo-id`, `--remote-dir`, `--python`. |
| `scripts/hf-python.sh` | Finds a Python interpreter that imports `huggingface_hub` (an active `.venv` shadows the conda base one). Used by both uploaders. |
| `upload-bulk-simulated.sh` | Unchanged purpose (dataset ZIPs to `ph/d/simulated/astralx-datasets/raw`); now uses the interpreter finder and the resolved data root. |

## 5. Behaviour and invariants

- **Mirror leaf equals results leaf.** The leaf is rebuilt in a temporary
  directory and swapped in atomically, so stale files from earlier runs never
  survive in the mirror. Hidden markers (`.astralx.lock`, `.astralx.success`,
  `.astralx_run.log`) are included because the stats collector treats them as
  validity markers.
- **Forbidden names** never enter the mirror and block upload if found:
  `all_gt.tre`, `s_tree.trees`, `l_trees.trees`, `g_trees*.trees`, `*.db`,
  `*.db-journal`, `*.zip`, `stat-sim.csv`. Only directories of the exact shape
  `<dataset>/<R>/<method>_outputs/<setting>` are mirrored; raw SimPhy replicate
  directories (`1/`, `2/`) are ignored.
- **Command files.** `<dataset>.command` and `.params` are copied per dataset
  (idempotent, byte-compared). For `_incomplete` datasets the base dataset's
  files are copied too. A missing `.command` produces a warning at mirror time
  and blocks upload unless `--allow-missing-command` is given.
- **Mirror failures never change a run's exit code.** They print a `WARNING`
  line; `sync-simulated-outputs.sh` repairs the mirror afterwards.
- **Command record** (`out-astralx.command`) contains, as `#` comment lines:
  date, host, ASTRAL-X root, git commit (marked when the tree has uncommitted
  changes), input, output, reference tree, the wrapper invocation; then one
  executable line `cd <root> && ./astralx --input ... --output ... <flags>`;
  then `exit_code` and `running_time`. For simulated runs, a trailing block
  adds dataset, replicate, setting, true tree, RF rate, the exact
  `test-astralx-simulated.sh` invocation, and the wrapper command. All values
  are shell-quoted (`printf %q`).
- **Plans.** Both bulk scripts print
  `<dataset> / <replicate range> / <setting>` lines. Replicates are collapsed
  into ranges (`R1-R4, R6`); replicates listed in
  `ALREADY_COMPLETED_SIMULATED_CONFIGS` (results already produced by an earlier
  sweep, so not recomputed) are noted inline. The run plan is followed by a
  `[y/N]` prompt; `--yes` skips it and a non-terminal stdin proceeds with a
  note so `nohup` launches keep working. The uploader revalidates every
  directory immediately before uploading and refuses if anything forbidden
  appeared.
- **Uploader preflight.** The outputs uploader requires the folder-aware
  `hf_upload.py` (its `--help` lists `--include`/`--exclude`) and a Python that
  imports `huggingface_hub`; both are checked before the plan is printed.
  Older file-only copies of the helper are rejected with instructions instead
  of failing once per dataset after confirmation.
- **Upload granularity.** One `hf_upload.py` folder upload per
  `<method>_outputs/<dataset>` to `<remote-dir>/<method>_outputs/<dataset>`.
  Re-running is an incremental sync because the Hub skips unchanged files.

## 6. Usage

```bash
# normal experiments: mirror happens automatically
./scripts/run-bulk-simulated.sh --taxa-list 1000 --genes-list 1000 -n 5 \
  --opts-list "--search-space S1 --intersection-method I1;" --no-notify
./scripts/run-bulk-simulated.sh ... --dry-run          # plan only
./scripts/run-bulk-simulated.sh ... --yes              # no prompt

# back-fill results produced before the mirror existed
./scripts/sync-simulated-outputs.sh --dry-run
./scripts/sync-simulated-outputs.sh

# publish
./scripts/upload-bulk-simulated-outputs.sh --dry-run
./scripts/upload-bulk-simulated-outputs.sh --sync
./scripts/upload-bulk-simulated-outputs.sh --methods astralx --min-taxa 1000

# fetch a published mirror (generic folder downloader)
~/utils/hf-data-transfer/hf_download.sh --repo-id imAniksahA/blab \
  --path-in-repo ph/d/simulated/outputs/astralx_outputs \
  --local-path $PHYLOGENY_DATA_DIR/outputs/simphy/astralx_outputs
```

Reproducing a published result: run the SimPhy line in `<dataset>.command`
(and the `_incomplete.command` line if applicable) to regenerate the data, then
run the `cd ... && ./astralx ...` line from `out-astralx.command` at the recorded
git commit.

## 7. Tests

- `test/test_simulated_outputs_mirror.sh` covers the default-root rule and
  guards, back-fill (command copying, forbidden files refused, incomplete
  fallback), real runs including the skip path and `--no-outputs-mirror`, the
  command record contents, and uploader planning and refusals.
- `test/test_simulated_success_detection.sh` asserts the mirror leaf equals the
  results leaf and that the collected CSV carries the quartet score.
- `test/test_bulk_simulated_skip_list.sh` checks that
  `ALREADY_COMPLETED_SIMULATED_CONFIGS` ships empty, that a listed tuple does not
  leak into a neighbouring taxa/gene-tree/sb/spmax setting, and drives the skip
  branch used by the replicate loops with a test-only tuple.
- `test/test_phylogeny_data_dir.sh` covers the data-root resolution order
  (environment default, explicit override, repository fallback, error cases)
  and checks that `sim.sh`, `sim_incomplete.sh`, the stats collector, and the
  download/upload tools all land on the same tree.
- The mirror tests are registered in `test/run_cli_tests.sh`.

## 8. Adding another method

Write results to `<data>/<dataset>/<R>/<method>_outputs/<setting>/` and call
`astralx_mirror_simulated_results DATA_DIR OUTPUTS_ROOT RESULTS_DIR` after the
run (source `scripts/simphy-outputs-dir.sh`). Write the method's exact command
into the results directory as `<output-basename>.command`. Nothing else needs
to change: sync and upload discover `<method>_outputs` directories generically.

## 9. Related

The A10K (10k-astral-dataset) runs have the same kind of mirror under
`$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset`, next to this one; see
[`a10k-outputs-reproducibility.md`](a10k-outputs-reproducibility.md).

## 10. Change log

- 2026-09-12: ported the reference implementation to ASTRAL-X. Mirror root is
  `outputs/simphy`; the older `simphy/outputs` location is not read.
- 2026-09-12: added the per-replicate skip list to `run-bulk-simulated.sh` so
  replicates whose results already existed from earlier sweeps were not
  recomputed during the large runs.
- 2026-09-18: renamed the list to `ALREADY_COMPLETED_SIMULATED_CONFIGS` to
  state its purpose (compute-time saving only) and emptied it for the public
  release, so every replicate runs by default. Covered by
  `test/test_bulk_simulated_skip_list.sh`.
- 2026-09-12: every SimPhy-data-touching script now resolves the data root
  through `astralx_resolve_simphy_data_dir`, so `sim.sh`, `sim_incomplete.sh`,
  `collect-stats-simulated.sh`, `download-bulk-simulated.sh` and
  `upload-bulk-simulated.sh` no longer read a checkout-local `simphy/data`
  while the runs write under `$PHYLOGENY_DATA_DIR`. The collected CSV regained
  its `optimal-quartet-score` column.
- 2026-09-12: uploaders auto-detect a Python with `huggingface_hub`
  (`scripts/hf-python.sh`); the outputs uploader rejects file-only
  `hf_upload.py` builds up front.
- 2026-09-13: extracted `astralx_mirror_directory_atomic` from
  `astralx_mirror_simulated_results` so the A10K mirror shares the atomic leaf
  swap; behaviour of the SimPhy mirror is unchanged.
- 2026-09-13: `test/test_simulated_outputs_mirror.sh` passes `--all-methods` to
  the uploader where it expects the second synthetic method, matching the
  astralx-only default introduced on 2026-09-12.
