#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/astralx-success-test.XXXXXX")"
trap 'status=$?; rm -rf -- "$TMP"; exit "$status"' EXIT

DATA="$TMP/data"
RUN_DIR="$DATA/t_4_g_1_sb_0.000001_spmin_100000_spmax_200000/R1"
mkdir -p "$RUN_DIR"
printf '((a,b),(c,d));\n' > "$RUN_DIR/all_gt.tre"
printf '((a,b),(c,d));\n' > "$RUN_DIR/s_tree.trees"

COMMON=(--simphy-data-dir "$DATA" -t 4 -g 1 -r R1
  --sb 0.000001 --spmin 100000 --spmax 200000
  --opts '--search-space S1 --cpu -q'
  --no-time-monitor --no-gpu-monitor --no-notify)

"$ROOT/scripts/test-astralx-simulated.sh" "${COMMON[@]}" >/dev/null
RESULTS_DIR=$(find "$RUN_DIR/astralx_outputs" -mindepth 1 -maxdepth 1 -type d)
OUTPUT="$RESULTS_DIR/out-astralx.tre"
SIDE="$RESULTS_DIR/out-astralx_stats.csv"
SUCCESS="$RESULTS_DIR/.astralx.success"
[[ -s "$OUTPUT" && -s "$SIDE" && -s "$SUCCESS" ]]

# The mirror sits beside the custom data tree and equals the result leaf.
MIRROR_LEAF="$TMP/outputs/astralx_outputs/t_4_g_1_sb_0.000001_spmin_100000_spmax_200000/R1/$(basename "$RESULTS_DIR")"
[[ -s "$MIRROR_LEAF/out-astralx.tre" && -s "$MIRROR_LEAF/stat-astralx.csv" ]]
diff -r "$RESULTS_DIR" "$MIRROR_LEAF" >/dev/null
[[ ! -e "$TMP/outputs/astralx_outputs/t_4_g_1_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre" ]]

# A failed sidecar plus a stale tree must never be accepted as completed.
rm -f "$SUCCESS"
sed -i '2s/,0$/,1/' "$SIDE"
rerun_log=$("$ROOT/scripts/test-astralx-simulated.sh" "${COMMON[@]}" 2>&1)
[[ "$rerun_log" == *"Previous statistics exist but no successful output was recorded; rerunning."* ]]
[[ -s "$OUTPUT" && -s "$SUCCESS" ]]
[[ "$(awk -F, 'NR==2 {print $9}' "$SIDE")" == "0" ]]

skip_log=$("$ROOT/scripts/test-astralx-simulated.sh" "${COMMON[@]}" 2>&1)
[[ "$skip_log" == *"SKIPPING: successful output already exists"* ]]

# The stats collector reads the same data tree the run wrote to.
COMBINED="${TMP}/combined.csv"
"${ROOT}/scripts/collect-stats-simulated.sh" --simphy-data-dir "$DATA" --out "$COMBINED" >/dev/null
grep -q 'optimal-quartet-score' "$COMBINED"
! grep -qi 'triplet' "$COMBINED"
[[ "$(awk -F, 'NR==1 {print $10}' "$COMBINED")" == "optimal-quartet-score" ]]
[[ "$(awk -F, 'NR==2 {print $10}' "$COMBINED")" =~ ^[0-9]+([.][0-9]+)?$ ]]

echo "Simulated-run success detection: PASS"
