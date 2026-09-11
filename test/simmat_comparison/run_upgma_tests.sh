#!/usr/bin/env bash
# run_upgma_tests.sh — Run UPGMA guide tree comparison tests (ASTRAL-X vs ASTRAL-MP)
#
# Tests all available simulated-incomplete datasets and the hand-crafted
# inputs that are large enough for ASTRAL-MP's -C completion to succeed.
#
# Usage:
#   bash run_upgma_tests.sh [--verbose]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${ROOT}/scripts/phylogeny-data-dir.sh"
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose|-v) VERBOSE="--verbose"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== UPGMA Guide Tree Comparison Tests ==="
echo ""

PASS=0; FAIL=0; SKIP=0

run_one() {
    local label="$1"
    local input="$2"
    echo "--- $label ---"
    set +e
    python3 "$SCRIPT_DIR/compare_upgma.py" "$input" $VERBOSE 2>&1
    local rc=$?
    set -e
    echo ""
    if   [[ $rc -eq 0 ]]; then PASS=$((PASS+1))
    elif [[ $rc -eq 2 ]]; then SKIP=$((SKIP+1))
    else                       FAIL=$((FAIL+1))
    fi
}

# Simulated incomplete datasets — skip very large ones (t>=1000) that OOM ASTRAL-MP locally
SIMPHY_DATA="$(astralx_resolve_simphy_data_dir "" "$ROOT/simphy/data")"
for d in "$SIMPHY_DATA"/*_incomplete; do
    [[ -d "$d" ]] || continue
    # Extract t= value from directory name and skip if >= 1000
    taxa=$(basename "$d" | grep -oP '(?<=t_)\d+' | head -1)
    if [[ -n "$taxa" && "$taxa" -ge 1000 ]]; then
        echo "--- $(basename $d) --- SKIP (t=$taxa >= 1000, would OOM ASTRAL-MP locally)"
        echo ""
        SKIP=$((SKIP+1))
        continue
    fi
    for rep in "$d"/R*/; do
        gt="$rep/all_gt.tre"
        [[ -f "$gt" ]] || continue
        label="$(basename $d)/$(basename $rep)"
        run_one "$label" "$gt"
    done
done

echo "=== Summary ==="
echo "  PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
echo ""
if [[ $FAIL -gt 0 ]]; then echo "Overall: FAIL"; exit 1
else                        echo "Overall: PASS"; exit 0
fi
