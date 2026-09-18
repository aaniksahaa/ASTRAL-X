#!/usr/bin/env bash
# Verifies the reproducibility mirror of A10K (10k-astral-dataset) run outputs:
#   * default outputs-directory derivation and containment guards,
#   * back-fill through sync-a10k-outputs.sh (results, rooting record,
#     provenance files, merged CSV copied; gene trees / species trees never
#     copied; stale mirror files replaced),
#   * automatic mirroring by run-a10k.sh on real runs (true and estimated gene
#     trees), on the "already completed" skip path, and its --no-outputs-mirror
#     and --outputs-dir switches,
#   * collect-scores-a10k.sh mirroring the merged CSV,
#   * upload-a10k-outputs.sh planning, remote paths, and refusals,
#   * clear-a10k.sh preserving the mirror unless --include-mirror is given.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT}/scripts/a10k-outputs-dir.sh"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/astralx-a10k-mirror-test.XXXXXX")"
trap 'status=$?; rm -rf -- "$TMP"; exit "$status"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# ---------------------------------------------------------------- helpers ---
BASE="${TMP}/research root/data"
DATA="${BASE}/10k-astral-dataset"
OUTPUTS="${BASE}/outputs/10k-astral-dataset"
mkdir -p "${DATA}/10k-simphy"

[[ "$(astralx_default_a10k_outputs_dir "$DATA")" == "$OUTPUTS" ]] || \
  fail "<parent>/<dataset> should mirror into <parent>/outputs/<dataset>"
[[ "$(astralx_default_a10k_outputs_dir "${TMP}/elsewhere/10k-astral-dataset/")" == "${TMP}/elsewhere/outputs/10k-astral-dataset" ]] || \
  fail "the default rule must not depend on the parent being named 'data'"
RESOLVED="$(astralx_prepare_a10k_outputs_dir "" "$DATA")"
[[ "$RESOLVED" == "$OUTPUTS" && -d "$OUTPUTS" ]] || fail "default outputs dir was not created: $RESOLVED"
if astralx_prepare_a10k_outputs_dir "${DATA}/outputs" "$DATA" >/dev/null 2>"${TMP}/inside.err"; then
  fail "an outputs dir inside the data dir was accepted"
fi
grep -q "inside it" "${TMP}/inside.err" || fail "containment error was unclear"
if astralx_prepare_a10k_outputs_dir "${BASE}" "$DATA" >/dev/null 2>/dev/null; then
  fail "an outputs dir containing the data dir was accepted"
fi
if astralx_prepare_a10k_outputs_dir_standalone "" >/dev/null 2>/dev/null; then
  fail "standalone resolver accepted an empty outputs dir"
fi

# ------------------------------------------------------- synthetic dataset ---
SETTING="search-space_S1__cpu_true"
make_replicate_inputs() {
  local rep_dir="$1"
  mkdir -p "${rep_dir}/estimatedgenetrees"
  printf '((1,2),((3,4),0));\n((1,3),((2,4),0));\n' > "${rep_dir}/truegenetrees"
  printf '((1,2),((3,4),0));\n' > "${rep_dir}/s_tree.trees"
  printf '((1,2),(3,4),0);\n((1,3),(2,4),0);\n' > "${rep_dir}/estimatedgenetrees/estimatedgenetrees.tre"
}
make_results() {
  local results="$1" tag="$2"
  mkdir -p "$results"
  printf '((1,2),((3,4),0));\n' > "${results}/out-astralx.tre"
  printf 'alg,setting\nastralx,%s\n' "$tag" > "${results}/stat-astralx.csv"
  printf 'algorithm\nastral-x\n' > "${results}/out-astralx_stats.csv"
  printf '# cmd %s\n' "$tag" > "${results}/out-astralx.command"
  printf 'log %s\n' "$tag" > "${results}/.astralx_run.log"
}

for r in R1 R2 R3; do
  make_replicate_inputs "${DATA}/10k-simphy/${r}"
  make_results "${DATA}/10k-simphy/${r}/astralx_outputs/estimated/search-mode_full" "${r}-est-full"
done
make_results "${DATA}/10k-simphy/R1/astralx_outputs/true/search-mode_full" "R1-true-full"
make_results "${DATA}/10k-simphy/R1/astralx_outputs/estimated/search-mode_local" "R1-est-local"
make_results "${DATA}/10k-simphy/R2/aster_outputs/estimated/default" "R2-aster"
printf 'rooted\n' > "${DATA}/10k-simphy/R1/estimatedgenetrees/estimatedgenetrees.rooted.tre"
printf '# rooting\nprocess_unrooted.sh -og 0\n' > "${DATA}/10k-simphy/R1/estimatedgenetrees/estimatedgenetrees.rooted.command"
printf 'downloaded from the ASTRAL 10k dataset\n' > "${DATA}/10k-astral-dataset.source"
printf '# 10k dataset\n' > "${DATA}/README.md"
printf 'alg,setting\nastralx,x\n' > "${DATA}/a10k_astralx_scores_merged.csv"

# ------------------------------------------------------------ sync dry run ---
"${ROOT}/scripts/sync-a10k-outputs.sh" --data-dir "$DATA" --dry-run >"${TMP}/sync-dry.out" 2>&1
grep -q "would mirror=6 filtered-out=0 problems=0 merged-csv=would copy" "${TMP}/sync-dry.out" || \
  fail "dry run summary unexpected: $(cat "${TMP}/sync-dry.out")"
[[ -z "$(find "$OUTPUTS" -mindepth 1 -print -quit)" ]] || fail "dry run wrote into the outputs dir"

# ---------------------------------------------------------------- sync run ---
"${ROOT}/scripts/sync-a10k-outputs.sh" --data-dir "$DATA" >"${TMP}/sync.out" 2>&1
grep -q "mirrored=6 filtered-out=0 failed=0 merged-csv=copied" "${TMP}/sync.out" || \
  fail "sync summary unexpected: $(cat "${TMP}/sync.out")"

M="${OUTPUTS}/astralx_outputs/10k-simphy"
for leaf in R1/estimated/search-mode_full R1/estimated/search-mode_local R1/true/search-mode_full \
            R2/estimated/search-mode_full R3/estimated/search-mode_full; do
  [[ -s "${M}/${leaf}/out-astralx.tre" ]] || fail "missing mirrored tree: $leaf"
  [[ -s "${M}/${leaf}/stat-astralx.csv" ]] || fail "missing mirrored stat csv: $leaf"
  [[ -s "${M}/${leaf}/out-astralx.command" ]] || fail "missing mirrored command record: $leaf"
  [[ -f "${M}/${leaf}/.astralx_run.log" ]] || fail "missing mirrored run log: $leaf"
done
[[ -s "${OUTPUTS}/aster_outputs/10k-simphy/R2/estimated/default/out-astralx.tre" ]] || fail "second method was not mirrored"
[[ -s "${M}/R1/estimatedgenetrees.rooted.command" ]] || fail "rooting record was not mirrored"
[[ ! -e "${M}/R2/estimatedgenetrees.rooted.command" ]] || fail "rooting record appeared for a replicate without one"
[[ -s "${OUTPUTS}/10k-astral-dataset.source" ]] || fail "dataset provenance file was not mirrored"
[[ -s "${OUTPUTS}/README.md" ]] || fail "dataset README was not mirrored"
cmp -s "${DATA}/a10k_astralx_scores_merged.csv" "${OUTPUTS}/a10k_astralx_scores_merged.csv" || fail "merged CSV was not mirrored"
[[ -z "$(astralx_a10k_find_forbidden_in_mirror "$OUTPUTS")" ]] || \
  fail "forbidden files leaked into the mirror: $(astralx_a10k_find_forbidden_in_mirror "$OUTPUTS")"
[[ ! -e "${M}/R1/truegenetrees" && ! -e "${M}/R1/s_tree.trees" && ! -e "${M}/R1/estimatedgenetrees" ]] || \
  fail "input data leaked into the mirror"
# The data tree keeps its exact layout.
[[ -f "${DATA}/10k-simphy/R1/astralx_outputs/estimated/search-mode_full/out-astralx.tre" && -f "${DATA}/10k-simphy/R1/truegenetrees" ]] || \
  fail "sync altered the data tree"
[[ ! -e "${DATA}/outputs" ]] || fail "sync created an outputs dir inside the data dir"

# Re-sync replaces a stale mirror leaf exactly (stale extra file disappears).
printf 'stale\n' > "${M}/R1/estimated/search-mode_full/stale.txt"
printf '((1,3),((2,4),0));\n' > "${DATA}/10k-simphy/R1/astralx_outputs/estimated/search-mode_full/out-astralx.tre"
"${ROOT}/scripts/sync-a10k-outputs.sh" --data-dir "$DATA" --methods astralx --quiet >"${TMP}/resync.out" 2>&1
grep -q "mirrored=5 filtered-out=1 failed=0" "${TMP}/resync.out" || fail "method filter summary unexpected: $(cat "${TMP}/resync.out")"
[[ ! -e "${M}/R1/estimated/search-mode_full/stale.txt" ]] || fail "stale mirror file survived a re-sync"
cmp -s "${DATA}/10k-simphy/R1/astralx_outputs/estimated/search-mode_full/out-astralx.tre" "${M}/R1/estimated/search-mode_full/out-astralx.tre" || \
  fail "re-sync did not refresh the tree"
[[ -z "$(find "${M}/R1/estimated" -maxdepth 1 -name '.*mirror*' -print -quit)" ]] || fail "temporary mirror dir left behind"

# Results containing input data are refused.
mkdir -p "${DATA}/10k-simphy/R2/astralx_outputs/estimated/bad"
printf 'x\n' > "${DATA}/10k-simphy/R2/astralx_outputs/estimated/bad/s_tree.trees"
if "${ROOT}/scripts/sync-a10k-outputs.sh" --data-dir "$DATA" --quiet >"${TMP}/bad.out" 2>&1; then
  fail "sync succeeded although a results dir contained s_tree.trees"
fi
grep -q "A10K input data" "${TMP}/bad.out" || fail "refusal reason unclear: $(cat "${TMP}/bad.out")"
[[ ! -e "${M}/R2/estimated/bad" ]] || fail "forbidden results dir was mirrored anyway"
rm -rf "${DATA}/10k-simphy/R2/astralx_outputs/estimated/bad"

# ------------------------------------------------ real ASTRAL-X run mirror ---
RUN_BASE="${TMP}/run/data"
RUN_DATA="${RUN_BASE}/10k-astral-dataset"
RUN_OUTPUTS="${RUN_BASE}/outputs/10k-astral-dataset"
make_replicate_inputs "${RUN_DATA}/10k-simphy/R1"
make_replicate_inputs "${RUN_DATA}/10k-simphy/R2"
printf 'downloaded from the ASTRAL 10k dataset\n' > "${RUN_DATA}/10k-astral-dataset.source"
COMMON=(--data-dir "$RUN_DATA" --opts '--search-space S1 --cpu -q'
  --no-time-monitor --no-gpu-monitor --no-notify)

"${ROOT}/scripts/run-a10k.sh" "${COMMON[@]}" --tree-type "true;estimated" --replicates R1 >"${TMP}/run1.out" 2>&1 || \
  fail "run-a10k.sh failed: $(tail -n 30 "${TMP}/run1.out")"
grep -q "outputs mirror: ${RUN_OUTPUTS}" "${TMP}/run1.out" || fail "run did not report the default outputs mirror: $(grep -i mirror "${TMP}/run1.out")"
for tree_type in true estimated; do
  RUN_LEAF="${RUN_OUTPUTS}/astralx_outputs/10k-simphy/R1/${tree_type}/${SETTING}"
  SRC_LEAF="${RUN_DATA}/10k-simphy/R1/astralx_outputs/${tree_type}/${SETTING}"
  grep -q "Mirrored outputs to: ${RUN_LEAF}" "${TMP}/run1.out" || fail "run did not mirror its ${tree_type} outputs"
  [[ -s "${RUN_LEAF}/out-astralx.tre" && -s "${RUN_LEAF}/stat-astralx.csv" && -s "${RUN_LEAF}/out-astralx_stats.csv" ]] || \
    fail "mirrored ${tree_type} run is missing tree or CSVs"
  [[ -s "${RUN_LEAF}/.astralx_run.log" ]] || fail "mirrored ${tree_type} run lacks the run log"
  grep -q "ASTRAL-X" "${RUN_LEAF}/.astralx_run.log" || fail "run log does not contain the wrapper output"
  # The exact ASTRAL-X command is recorded beside the tree and mirrored with it.
  [[ -s "${RUN_LEAF}/out-astralx.command" ]] || fail "run command record was not mirrored (${tree_type})"
  grep -q "&& ./astralx --input .* --output .*out-astralx.tre --search-space S1 --cpu -q\$" "${RUN_LEAF}/out-astralx.command" || \
    fail "command record lacks the exact ./astralx invocation: $(cat "${RUN_LEAF}/out-astralx.command")"
  grep -q "^# git_commit: " "${RUN_LEAF}/out-astralx.command" || fail "command record lacks the git commit"
  grep -q "^# exit_code:    0$" "${RUN_LEAF}/out-astralx.command" || fail "command record lacks the exit code"
  grep -q "^# --- A10K run context (run-a10k.sh) ---$" "${RUN_LEAF}/out-astralx.command" || fail "command record lacks the A10K context"
  grep -q "^# invoked as: .*run-a10k.sh" "${RUN_LEAF}/out-astralx.command" || fail "command record lacks the outer invocation"
  grep -q "^# tree_type:    ${tree_type}$" "${RUN_LEAF}/out-astralx.command" || fail "command record lacks the tree type"
  grep -q "^# setting:      ${SETTING}$" "${RUN_LEAF}/out-astralx.command" || fail "command record lacks the setting"
  grep -q "^# rf_rate:      [0-9.]*$" "${RUN_LEAF}/out-astralx.command" || fail "command record lacks the RF rate"
  diff -r "$SRC_LEAF" "$RUN_LEAF" >/dev/null || fail "mirror leaf differs from the results dir (${tree_type})"
done
grep -q "^# rooting cmd: .*process_unrooted.sh .* -og 0" "${RUN_OUTPUTS}/astralx_outputs/10k-simphy/R1/estimated/${SETTING}/out-astralx.command" || \
  fail "estimated command record lacks the rooting command"
[[ -s "${RUN_OUTPUTS}/astralx_outputs/10k-simphy/R1/estimatedgenetrees.rooted.command" ]] || fail "rooting record was not written and mirrored"
grep -q "process_unrooted.sh .* -og 0$" "${RUN_OUTPUTS}/astralx_outputs/10k-simphy/R1/estimatedgenetrees.rooted.command" || \
  fail "rooting record lacks the rooting command"
[[ -f "${RUN_DATA}/10k-simphy/R1/estimatedgenetrees/estimatedgenetrees.rooted.tre" ]] || fail "rooted gene trees were not produced in the data tree"
[[ -s "${RUN_OUTPUTS}/10k-astral-dataset.source" ]] || fail "run did not mirror the dataset provenance file"
[[ -z "$(astralx_a10k_find_forbidden_in_mirror "$RUN_OUTPUTS")" ]] || fail "run mirror contains input data"
[[ ! -e "${RUN_OUTPUTS}/astralx_outputs/10k-simphy/R1/estimatedgenetrees" ]] || fail "run mirror contains the estimated gene tree dir"
[[ -f "${RUN_DATA}/10k-simphy/R1/astralx_outputs/true/${SETTING}/out-astralx.tre" && -f "${RUN_DATA}/10k-simphy/R1/truegenetrees" ]] || \
  fail "data tree layout changed"

# The skip path (already completed) rebuilds a deleted mirror.
rm -rf "$RUN_OUTPUTS"
"${ROOT}/scripts/run-a10k.sh" "${COMMON[@]}" --tree-type true --replicates R1 >"${TMP}/run2.out" 2>&1 || fail "second run failed"
grep -q "SKIPPING: .*stat-astralx.csv exists" "${TMP}/run2.out" || fail "second run did not skip"
[[ -s "${RUN_OUTPUTS}/astralx_outputs/10k-simphy/R1/true/${SETTING}/out-astralx.tre" ]] || fail "skip path did not rebuild the mirror"

# --no-outputs-mirror leaves the mirror untouched; explicit dir is honored.
rm -rf "$RUN_OUTPUTS"
"${ROOT}/scripts/run-a10k.sh" "${COMMON[@]}" --tree-type true --replicates R1 --no-outputs-mirror >"${TMP}/run3.out" 2>&1 || fail "third run failed"
[[ ! -e "$RUN_OUTPUTS" ]] || fail "--no-outputs-mirror still wrote a mirror"
grep -q "SKIPPING" "${TMP}/run3.out" || fail "third run did not take the skip path"
"${ROOT}/scripts/run-a10k.sh" "${COMMON[@]}" --tree-type true --replicates R1 --outputs-dir "${TMP}/explicit outputs" >"${TMP}/run4.out" 2>&1 || fail "fourth run failed"
[[ -s "${TMP}/explicit outputs/astralx_outputs/10k-simphy/R1/true/${SETTING}/out-astralx.tre" ]] || \
  fail "explicit --outputs-dir was not used"
if "${ROOT}/scripts/run-a10k.sh" "${COMMON[@]}" --tree-type true --replicates R1 --outputs-dir "${RUN_DATA}/outputs" >"${TMP}/run5.out" 2>&1; then
  fail "an outputs dir inside the data dir was accepted by the run script"
fi

# ------------------------------------------------- merged scores collector ---
"${ROOT}/scripts/collect-scores-a10k.sh" --data-dir "$RUN_DATA" --start-rep 1 --end-rep 2 >"${TMP}/collect.out" 2>&1 || \
  fail "collector failed: $(cat "${TMP}/collect.out")"
[[ -s "${RUN_DATA}/a10k_astralx_scores_merged.csv" ]] || fail "collector did not write the merged CSV"
[[ "$(wc -l < "${RUN_DATA}/a10k_astralx_scores_merged.csv")" == 3 ]] || fail "merged CSV should hold the header and two rows"
grep -q "Mirrored merged stats to: ${RUN_OUTPUTS}/a10k_astralx_scores_merged.csv" "${TMP}/collect.out" || \
  fail "collector did not report the mirrored CSV: $(cat "${TMP}/collect.out")"
cmp -s "${RUN_DATA}/a10k_astralx_scores_merged.csv" "${RUN_OUTPUTS}/a10k_astralx_scores_merged.csv" || fail "mirrored merged CSV differs"
"${ROOT}/scripts/collect-scores-a10k.sh" --data-dir "$RUN_DATA" --start-rep 1 --end-rep 2 --no-outputs-mirror >"${TMP}/collect2.out" 2>&1
! grep -q "Mirrored merged stats" "${TMP}/collect2.out" || fail "--no-outputs-mirror still mirrored the merged CSV"

# ------------------------------------------------------- uploader dry run ---
UP="${ROOT}/scripts/upload-a10k-outputs.sh"
"$UP" --data-dir "$DATA" --dry-run --uploader /bin/true --python /bin/true >"${TMP}/up.out" 2>&1 || \
  fail "uploader dry run failed: $(cat "${TMP}/up.out")"
grep -q "Plan: upload 1 method director" "${TMP}/up.out" || fail "uploader did not plan the astralx upload: $(cat "${TMP}/up.out")"
grep -q "(5 replicate/tree-type/setting result set(s)) and 3 root-level file(s)" "${TMP}/up.out" || \
  fail "uploader counted the wrong result sets or root files: $(cat "${TMP}/up.out")"
grep -q "^  R1-R3 / estimated/search-mode_full$" "${TMP}/up.out" || fail "uploader did not collapse replicates: $(cat "${TMP}/up.out")"
grep -Fq -- "--path-in-repo ph/d/a10k/outputs/astralx_outputs " "${TMP}/up.out" || fail "remote path for astralx outputs is wrong"
grep -Fq -- "--path-in-repo ph/d/a10k/outputs/a10k_astralx_scores_merged.csv " "${TMP}/up.out" || fail "merged CSV was not planned"
grep -Fq -- "--local-path $(printf '%q' "${OUTPUTS}/astralx_outputs") " "${TMP}/up.out" || fail "local path is wrong"
! grep -q "aster_outputs" "${TMP}/up.out" || fail "default method filter did not exclude aster"
grep -q "nothing was uploaded" "${TMP}/up.out" || fail "dry run did not state that nothing was uploaded"
"$UP" --outputs-dir "$OUTPUTS" --dry-run --uploader /bin/true --python /bin/true --all-methods >"${TMP}/up-all.out" 2>&1
grep -q "Plan: upload 2 method director" "${TMP}/up-all.out" || fail "--all-methods did not widen the plan: $(cat "${TMP}/up-all.out")"
if "$UP" --dry-run --uploader /bin/true --python /bin/true >"${TMP}/up-none.out" 2>&1; then
  fail "uploader ran without any location"
fi
if "$UP" --outputs-dir "$OUTPUTS" --sync --dry-run --uploader /bin/true --python /bin/true >"${TMP}/up-sync-nodata.out" 2>&1; then
  fail "--sync without --data-dir was accepted"
fi

# --sync refreshes the mirror before planning (a new result appears).
make_results "${DATA}/10k-simphy/R3/astralx_outputs/true/search-mode_local" "R3-true-local"
"$UP" --data-dir "$DATA" --sync --yes --uploader /bin/true --python /bin/true >"${TMP}/up-s.out" 2>&1 || \
  fail "--sync upload with a stub uploader failed: $(cat "${TMP}/up-s.out")"
grep -q "Refreshing the outputs mirror" "${TMP}/up-s.out" || fail "--sync did not run the mirror sync"
[[ -s "${M}/R3/true/search-mode_local/out-astralx.tre" ]] || fail "--sync did not back-fill the new result"

# With --data-dir the refresh is the default; --no-sync and --outputs-dir skip it.
"$UP" --data-dir "$DATA" --yes --dry-run --uploader /bin/true --python /bin/true >"${TMP}/up-auto.out" 2>&1 || \
  fail "default upload with --data-dir failed: $(cat "${TMP}/up-auto.out")"
grep -q "Refreshing the outputs mirror" "${TMP}/up-auto.out" || fail "--data-dir did not refresh the mirror by default"
"$UP" --data-dir "$DATA" --no-sync --yes --dry-run --uploader /bin/true --python /bin/true >"${TMP}/up-nosync.out" 2>&1 || \
  fail "--no-sync upload failed: $(cat "${TMP}/up-nosync.out")"
grep -q "Refreshing the outputs mirror" "${TMP}/up-nosync.out" && fail "--no-sync still refreshed the mirror"
"$UP" --outputs-dir "$OUTPUTS" --yes --dry-run --uploader /bin/true --python /bin/true >"${TMP}/up-od.out" 2>&1 || \
  fail "--outputs-dir upload failed: $(cat "${TMP}/up-od.out")"
grep -q "Refreshing the outputs mirror" "${TMP}/up-od.out" && fail "--outputs-dir alone refreshed the mirror"
grep -q "uploaded=4 failed=0" "${TMP}/up-s.out" || fail "stub upload summary unexpected: $(cat "${TMP}/up-s.out")"

# Input data inside the mirror blocks the upload.
printf 'leak\n' > "${M}/R1/truegenetrees"
if "$UP" --data-dir "$DATA" --dry-run --uploader /bin/true --python /bin/true >"${TMP}/up-leak.out" 2>&1; then
  fail "uploader accepted a mirror containing truegenetrees"
fi
grep -q "BLOCKED astralx_outputs: contains truegenetrees" "${TMP}/up-leak.out" || fail "leak was not reported: $(cat "${TMP}/up-leak.out")"
grep -q "nothing was uploaded" "${TMP}/up-leak.out" || fail "leak refusal did not state nothing was uploaded"
rm -f "${M}/R1/truegenetrees"

# Non-interactive launches proceed (for nohup) but print a visible note.
"$UP" --data-dir "$DATA" --uploader /bin/true --python /bin/true </dev/null >"${TMP}/up-noyes.out" 2>&1 || \
  fail "non-interactive uploader failed: $(cat "${TMP}/up-noyes.out")"
grep -q "Non-interactive session: proceeding without confirmation" "${TMP}/up-noyes.out" || \
  fail "non-interactive uploader did not explain that it would proceed"

# ------------------------------------------------------------- cleaner ---
# Earlier steps removed the run mirror; rebuild it so the cleaner has targets.
"${ROOT}/scripts/sync-a10k-outputs.sh" --data-dir "$RUN_DATA" --quiet >"${TMP}/resync-run.out" 2>&1 || \
  fail "re-sync of the run mirror failed: $(cat "${TMP}/resync-run.out")"
[[ -d "${RUN_OUTPUTS}/astralx_outputs" && -f "${RUN_OUTPUTS}/10k-astral-dataset.source" ]] || fail "re-sync did not rebuild the run mirror"
"${ROOT}/scripts/clear-a10k.sh" --data-dir "$RUN_DATA" --dry-run >"${TMP}/clear-dry.out" 2>&1 || fail "cleaner dry run failed"
grep -q "Mirror: .*preserved" "${TMP}/clear-dry.out" || fail "cleaner did not state that the mirror is preserved"
! grep -q "^  ${RUN_OUTPUTS}" "${TMP}/clear-dry.out" || fail "cleaner listed mirror targets without --include-mirror"
"${ROOT}/scripts/clear-a10k.sh" --data-dir "$RUN_DATA" --dry-run --include-mirror >"${TMP}/clear-dry2.out" 2>&1 || fail "cleaner dry run with mirror failed"
grep -Fq "  ${RUN_OUTPUTS}/astralx_outputs" "${TMP}/clear-dry2.out" || fail "--include-mirror did not list the mirrored results: $(cat "${TMP}/clear-dry2.out")"
grep -Fq "  ${RUN_OUTPUTS}/a10k_astralx_scores_merged.csv" "${TMP}/clear-dry2.out" || fail "--include-mirror did not list the mirrored CSV"
[[ -d "${RUN_OUTPUTS}/astralx_outputs" ]] || fail "dry run removed the mirror"
"${ROOT}/scripts/clear-a10k.sh" --data-dir "$RUN_DATA" --yes >"${TMP}/clear.out" 2>&1 || fail "cleaner failed"
[[ ! -e "${RUN_DATA}/10k-simphy/R1/astralx_outputs" ]] || fail "cleaner did not remove the results"
[[ -d "${RUN_OUTPUTS}/astralx_outputs" && -f "${RUN_OUTPUTS}/10k-astral-dataset.source" ]] || fail "cleaner removed the mirror without --include-mirror"
"${ROOT}/scripts/clear-a10k.sh" --data-dir "$RUN_DATA" --yes --include-mirror >"${TMP}/clear2.out" 2>&1 || fail "cleaner with mirror failed"
[[ ! -e "${RUN_OUTPUTS}/astralx_outputs" && ! -e "${RUN_OUTPUTS}/a10k_astralx_scores_merged.csv" ]] || fail "--include-mirror did not remove the mirror"
[[ -f "${RUN_OUTPUTS}/10k-astral-dataset.source" ]] || fail "--include-mirror removed the provenance file"
[[ -f "${RUN_DATA}/10k-simphy/R1/truegenetrees" && -f "${RUN_DATA}/10k-simphy/R1/estimatedgenetrees/estimatedgenetrees.rooted.tre" ]] || \
  fail "cleaner removed input data"

echo "A10K outputs mirror: PASS"
