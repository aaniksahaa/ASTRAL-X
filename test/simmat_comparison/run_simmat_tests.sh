#!/usr/bin/env bash
# run_simmat_tests.sh — Run similarity matrix comparison tests (ASTRAL-X vs ASTRAL-MP)
#
# Usage:
#   bash run_simmat_tests.sh [--tol TOL] [--verbose] [--no-gen] [--mode cpu|gpu]
#
# Options:
#   --tol TOL      Float tolerance (default: 1e-5)
#   --verbose      Pass --verbose to compare_simmat.py
#   --no-gen       Skip input generation (assume inputs already exist)
#   --mode MODE    ASTRAL-X compute mode: cpu or gpu (default: cpu)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/input"

TOL="1e-5"
VERBOSE=""
NO_GEN=false
MODE="cpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tol)     TOL="$2"; shift 2 ;;
        --verbose) VERBOSE="--verbose"; shift ;;
        --no-gen)  NO_GEN=true; shift ;;
        --mode)    MODE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Step 1: Generate inputs if needed ─────────────────────────────────────────
if [[ "$NO_GEN" == false ]]; then
    if [[ ! -f "${INPUT_DIR}/tc_sm1.tre" || ! -f "${INPUT_DIR}/tc_sm2.tre" ]]; then
        echo "=== Generating test inputs ==="
        python3 "${SCRIPT_DIR}/gen_test_inputs.py"
        echo ""
    else
        echo "(Test inputs already exist; skipping generation. Use --no-gen to suppress this check.)"
        echo ""
    fi
fi

# ── Step 2: Run comparisons ───────────────────────────────────────────────────
echo "=== Similarity Matrix Comparison Tests ==="
echo "Tolerance: ${TOL}  Mode: ${MODE}"
echo ""

PASS=0
FAIL=0
SKIP=0
declare -A RESULTS

for tc_file in "${INPUT_DIR}"/tc_sm*.tre; do
    tc_name="$(basename "$tc_file" .tre)"
    echo "--- ${tc_name} ---"

    if [[ ! -f "$tc_file" ]]; then
        echo "  SKIP (file not found: $tc_file)"
        RESULTS["$tc_name"]="SKIP"
        SKIP=$((SKIP + 1))
        echo ""
        continue
    fi

    set +e
    python3 "${SCRIPT_DIR}/compare_simmat.py" "$tc_file" --tol "$TOL" --mode "$MODE" $VERBOSE
    exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
        RESULTS["$tc_name"]="PASS"
        PASS=$((PASS + 1))
    elif [[ $exit_code -eq 2 ]]; then
        RESULTS["$tc_name"]="ERROR"
        FAIL=$((FAIL + 1))
    else
        RESULTS["$tc_name"]="FAIL"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

# ── Step 3: Summary ───────────────────────────────────────────────────────────
echo "=== Summary ==="
printf "  %-12s  %s\n" "Test Case" "Result"
printf "  %-12s  %s\n" "----------" "------"
for tc_name in "${!RESULTS[@]}"; do
    printf "  %-12s  %s\n" "$tc_name" "${RESULTS[$tc_name]}"
done | sort

echo ""
echo "  PASS: ${PASS}  FAIL: ${FAIL}  SKIP: ${SKIP}"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo "Overall: FAIL"
    exit 1
else
    echo "Overall: PASS"
    exit 0
fi
