# Reproducibility fingerprint for A10K runs

Status: implemented (2026-09-13). This document is the source of truth for the
A10K (10k-astral-dataset) outputs mirror, command records, and uploader. It is
the A10K counterpart of
[`simulated-outputs-reproducibility.md`](simulated-outputs-reproducibility.md);
any change to the implementation must be reflected here.

## 1. Problem

The A10K dataset and every inference result live in one tree:

```
<data-dir>/10k-simphy/R<n>/{truegenetrees, s_tree.trees,
                            estimatedgenetrees/estimatedgenetrees{,.rooted}.tre,
                            astralx_outputs/<tree_type>/<setting>/...}
```

`<data-dir>` is given explicitly to every A10K tool and is normally
`data/10k-astral-dataset` (see `cmd.txt`). The gene trees dominate the size;
the results are tiny. Sharing the results and everything needed to rerun them
required shipping the gene trees or hand-collecting files.

## 2. Requirements

1. The data tree keeps its exact layout. Reading and writing there is unchanged
   and fully backward compatible.
2. Every run additionally writes its results into a separate outputs tree,
   method-first:
   `<outputs>/<method>_outputs/10k-simphy/<R>/<tree_type>/<setting>/`.
3. The mirror contains everything the run wrote into its results directory
   (inferred tree, CSVs, command record, run log) and never the inputs:
   `truegenetrees`, `estimatedgenetrees*.tre`, `s_tree.trees`, archives.
4. Each results directory contains the exact command that produced it: every
   flag and parameter, absolute paths, code revision, the rooting command for
   estimated gene trees, and the outer commands.
5. The merged scores CSV and any dataset-level provenance files are mirrored
   too, so the mirror alone documents where the data came from and what the
   aggregate results were.
6. A bulk uploader publishes the outputs tree to the Hugging Face dataset repo
   with the same layout under `ph/d/a10k/outputs/`.

## 3. Layout

Data tree (unchanged, plus two small additive records marked `+`):

```
<data-dir>/                                   # e.g. data/10k-astral-dataset
  <dataset>.command|.source|.params, README*  # optional provenance (not produced here)
  a10k_astralx_scores_merged.csv              # collect-scores-a10k.sh
  10k-simphy/
    R1/ truegenetrees  s_tree.trees
        estimatedgenetrees/ estimatedgenetrees.tre  estimatedgenetrees.rooted.tre
                          + estimatedgenetrees.rooted.command
        astralx_outputs/<tree_type>/<setting>/
            out-astralx.tre  out-astralx.command  out-astralx_stats.csv
            stat-astralx.csv  + .astralx_run.log
```

Outputs mirror (new):

```
<outputs>/                                    # e.g. $PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset
  <dataset>.command|.source|.params, README*  # copied when present
  a10k_astralx_scores_merged.csv              # copied by the collector / sync
  astralx_outputs/
    10k-simphy/
      R1/
        estimatedgenetrees.rooted.command     # how the rooted input was derived
        estimated/<setting>/  <exact copy of the results leaf above>
        true/<setting>/       ...
      R2/...
  <other-method>_outputs/10k-simphy/...       # identical shape for any method
```

Remote (Hugging Face, repo `imAniksahA/blab`, type dataset):

```
ph/d/a10k/outputs/<method>_outputs/...        # same as the local mirror
ph/d/a10k/outputs/<root-level files>
```

`<setting>` is the option-derived folder name from `experiment-setting-name.sh`
(for example `search-space_S1__intersection-method_I1`). `<tree_type>` is
`true` or `estimated`.

### Default mirror location rule

The mirror root is derived from the resolved data directory:

| data directory                 | mirror root                        |
|--------------------------------|------------------------------------|
| `<parent>/<dataset>`           | `<parent>/outputs/<dataset>`       |

So the usual `$PHYLOGENY_DATA_DIR/10k-astral-dataset` mirrors into
`$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset`, next to the SimPhy mirror in
`$PHYLOGENY_DATA_DIR/outputs/simphy`. The rule does not depend on the parent
directory's name. `--outputs-dir PATH` overrides the rule. A mirror
root equal to, inside, or containing the data directory is refused.

## 4. Components

| File | Role |
|---|---|
| `scripts/a10k-outputs-dir.sh` | Shared helper: default-root rule, guards, results-path parsing, forbidden-file detection, leaf mirroring, rooting-record / provenance / merged-CSV copying. Sources `scripts/simphy-outputs-dir.sh` for the shared primitives. |
| `scripts/simphy-outputs-dir.sh` | Provides `astralx_mirror_directory_atomic` (temp dir + rename), used by both mirrors. |
| `run-a10k.sh` | Mirrors each results leaf after every run (also failed runs) and on the "already completed" skip path (free back-fill). Captures the wrapper output to `.astralx_run.log`. Appends the A10K context to the command record. Writes `estimatedgenetrees.rooted.command` when it roots estimated gene trees. Flags `--outputs-dir`, `--no-outputs-mirror`. |
| `run-astralx-with-monitor.sh` | Unchanged: writes `out-astralx.command` (exact `./astralx` line, git commit, wrapper invocation) before the run, appends exit code and running time after. |
| `collect-scores-a10k.sh` | Copies the merged CSV into the mirror root. Flags `--outputs-dir`, `--no-outputs-mirror`. |
| `sync-a10k-outputs.sh` | Back-fills or refreshes the whole mirror from the data tree. Flags `--data-dir` (required), `--outputs-dir`, `--methods`, `--dry-run`, `--quiet`. |
| `upload-a10k-outputs.sh` | Uploads each `<method>_outputs` directory as a folder plus the root-level files. Flags `--data-dir` / `--outputs-dir`, `--no-sync` / `--sync` (with `--data-dir` the mirror is refreshed first by default), `--methods`, `--all-methods`, `--dry-run`, `--yes`, `--repo-id`, `--remote-dir`, `--python`. |
| `clear-a10k.sh` | Preserves the mirror by default and says where it is; `--include-mirror` also removes the mirrored `astralx_outputs` and merged CSV (provenance files stay). |

## 5. Behaviour and invariants

- **Mirror leaf equals results leaf.** The leaf is rebuilt in a temporary
  directory and swapped in atomically, so stale files from earlier runs never
  survive in the mirror. `.astralx_run.log` is included.
- **Forbidden names** never enter the mirror and block upload if found:
  `estimatedgenetrees*.tre`, `truegenetrees`, `truegenetrees*.tre(es)`,
  `s_tree.trees`, `*.trees`, `all_gt.tre`, `*.db`, `*.db-journal`, `*.zip`,
  `*.tar`, `*.tar.gz`, `*.tgz`. Only directories of the exact shape
  `10k-simphy/<R>/<method>_outputs/<tree_type>/<setting>` are mirrored.
- **Command record** (`out-astralx.command`) contains the wrapper's block
  (date, host, ASTRAL-X root, git commit, input, output, wrapper invocation,
  the executable `cd <root> && ./astralx ...` line, exit code, running time)
  followed by an A10K block: data directory, replicate, tree type, setting,
  gene-tree file, true tree, RF rate, run exit code, the rooting command (for
  estimated gene trees), the exact `run-a10k.sh` invocation, and the wrapper
  command. All values are shell-quoted (`printf %q`).
- **Rooting record** (`estimatedgenetrees.rooted.command`) is written beside
  the rooted gene trees the first time a replicate is rooted and mirrored at
  `<outputs>/<method>_outputs/10k-simphy/<R>/`. Replicates rooted before this
  record existed have no record; their run command records still carry the
  `rooting cmd` line.
- **Mirror failures never change a run's exit code.** They print a `WARNING`
  line; `sync-a10k-outputs.sh` repairs the mirror afterwards.
- **Failed runs are mirrored too**, so the command record (with `run_exit`)
  and the run log of a failure are part of the fingerprint. Their leaf has no
  `stat-astralx.csv`, so the runner reruns them next time as before.
- **Uploader.** Requires the folder-aware `hf_upload.py` and a Python that
  imports `huggingface_hub`; prints the cases
  (`<replicates> / <tree_type>/<setting>`) and destinations, asks once, and
  revalidates for forbidden files immediately before uploading. A non-terminal
  stdin proceeds with a note so `nohup` launches keep working.

## 6. Usage

```bash
# normal experiments: mirror happens automatically
./scripts/run-a10k.sh --data-dir data/10k-astral-dataset --tree-type "true;estimated" \
  --opts "--search-space S1 --intersection-method I1"
./scripts/run-a10k.sh --data-dir data/10k-astral-dataset ... --outputs-dir /elsewhere/10k-outputs
./scripts/run-a10k.sh --data-dir data/10k-astral-dataset ... --no-outputs-mirror

# merged scores (also copied into the mirror)
./scripts/collect-scores-a10k.sh --data-dir data/10k-astral-dataset --start-rep 1 --end-rep 20

# back-fill results produced before the mirror existed
./scripts/sync-a10k-outputs.sh --data-dir data/10k-astral-dataset --dry-run
./scripts/sync-a10k-outputs.sh --data-dir data/10k-astral-dataset

# publish
./scripts/upload-a10k-outputs.sh --data-dir data/10k-astral-dataset --dry-run
./scripts/upload-a10k-outputs.sh --data-dir data/10k-astral-dataset          # refreshes the mirror first
./scripts/upload-a10k-outputs.sh --data-dir data/10k-astral-dataset --no-sync

# clear results; the mirror is kept unless asked
./scripts/clear-a10k.sh --data-dir data/10k-astral-dataset --dry-run
./scripts/clear-a10k.sh --data-dir data/10k-astral-dataset --yes --include-mirror

# fetch a published mirror (generic folder downloader)
~/utils/hf-data-transfer/hf_download.sh --repo-id imAniksahA/blab \
  --path-in-repo ph/d/a10k/outputs/astralx_outputs \
  --local-path outputs/10k-astral-dataset/astralx_outputs
```

Reproducing a published result: obtain the A10K dataset (see the mirrored
provenance file when present), run the `rooting cmd` line from the command
record for estimated gene trees, then run the `cd ... && ./astralx ...` line
from `out-astralx.command` at the recorded git commit.

## 7. Tests

- `test/test_a10k_outputs_mirror.sh` covers the default-root rule and guards,
  back-fill (results, rooting record, provenance, merged CSV; forbidden files
  refused; stale files replaced; method filter), real `run-a10k.sh` runs with
  true and estimated gene trees including the skip path, `--no-outputs-mirror`
  and `--outputs-dir`, the command and rooting records, the collector's CSV
  mirroring, uploader planning and refusals, and the cleaner's mirror handling.
- Registered in `test/run_cli_tests.sh`.

## 8. Change log

- 2026-09-16: `upload-a10k-outputs.sh` refreshes the mirror by default when
  `--data-dir` is given (`--no-sync` opts out; `--sync` forces it).
- 2026-09-15: default mirror root changed to `<parent>/outputs/<dataset>`
  (`$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset`). The first version mirrored
  into the `outputs` sibling of a parent named `data`, which put the A10K mirror
  one level above the SimPhy mirror (`/…/phylogeny/outputs/` instead of
  `/…/phylogeny/data/outputs/`). Re-running `run-a10k.sh` or
  `sync-a10k-outputs.sh` re-mirrors finished runs into the new location; the old
  directory can be deleted by hand.
- 2026-09-13: initial implementation. `astralx_mirror_directory_atomic` was
  extracted from the SimPhy mirror so both mirrors share the atomic leaf swap.
