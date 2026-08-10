#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/astralx-success-test.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

RUN_DIR="$TMP/t_4_g_1_sb_0.000001_spmin_100000_spmax_200000/R1"
mkdir -p "$RUN_DIR"
printf '((a,b),(c,d));\n' > "$RUN_DIR/all_gt.tre"
printf '((a,b),(c,d));\n' > "$RUN_DIR/s_tree.trees"

COMMON=(--simphy-data-dir "$TMP" -t 4 -g 1 -r R1
  --sb 0.000001 --spmin 100000 --spmax 200000
  --opts '--search-space S1 --cpu -q'
  --no-time-monitor --no-gpu-monitor --no-notify)

"$ROOT/test-astralx-simulated.sh" "${COMMON[@]}" >/dev/null
RESULTS_DIR=$(find "$RUN_DIR/astralx_outputs" -mindepth 1 -maxdepth 1 -type d)
OUTPUT="$RESULTS_DIR/out-astralx.tre"
SIDE="$RESULTS_DIR/out-astralx_stats.csv"
SUCCESS="$RESULTS_DIR/.astralx.success"
[[ -s "$OUTPUT" && -s "$SIDE" && -s "$SUCCESS" ]]

# A failed sidecar plus a stale tree must never be accepted as completed.
rm -f "$SUCCESS"
sed -i '2s/,0$/,1/' "$SIDE"
rerun_log=$("$ROOT/test-astralx-simulated.sh" "${COMMON[@]}" 2>&1)
[[ "$rerun_log" == *"Previous statistics exist but no successful output was recorded; rerunning."* ]]
[[ -s "$OUTPUT" && -s "$SUCCESS" ]]
[[ "$(awk -F, 'NR==2 {print $9}' "$SIDE")" == "0" ]]

skip_log=$("$ROOT/test-astralx-simulated.sh" "${COMMON[@]}" 2>&1)
[[ "$skip_log" == *"SKIPPING: successful output already exists"* ]]

echo "Simulated-run success detection: PASS"
