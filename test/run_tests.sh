#!/usr/bin/env bash
# run_tests.sh — Run all ASTRAL-X Python verifier test cases.
#
# Usage:  bash test/run_tests.sh [TC_FILTER]
#   TC_FILTER (optional): pattern to match test names, e.g. "tc1" or "tc1[23]"
#
# Each TC_* is run against verify_weights.py.
# If a tc*_true.tre file exists the inferred tree is compared (RF distance).
#
# Exit code: 0 if all tests pass, 1 if any fail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/input"
VERIFIER="${SCRIPT_DIR}/verify_weights.py"
FILTER="${1:-tc[0-9]*}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

pass=0
fail=0
skip=0

run_tc () {
    local input="$1"
    local name
    name="$(basename "$input" .tre)"
    local tc_id="${name%%_*}"   # e.g. "tc1"

    # skip true-tree files
    [[ "$name" == *_true ]] && return 0

    # True tree for RF comparison (optional)
    local true_file="${INPUT_DIR}/${tc_id}_true.tre"
    local compare_arg=""
    if [[ -f "$true_file" ]]; then
        local true_newick
        true_newick="$(cat "$true_file")"
        compare_arg="--compare ${true_newick}"
    fi

    printf "  %-45s" "$name"

    local out
    # shellcheck disable=SC2086
    if out="$(python3 "$VERIFIER" "$input" $compare_arg 2>&1)"; then
        local score line
        score=$(echo "$out" | grep -oP 'quartet score = \K[0-9]+' || true)
        if [[ -n "$score" ]]; then
            printf "${GREEN}PASS${NC}  score=%-10s" "$score"
        else
            printf "${GREEN}PASS${NC}  "
        fi
        if [[ -n "$compare_arg" ]]; then
            local rf
            rf=$(echo "$out" | grep -oP 'RF = \K[0-9]+' || true)
            [[ -n "$rf" ]] && printf "RF=%-4s" "$rf"
        fi
        echo
        ((pass++)) || true
    else
        printf "${RED}FAIL${NC}\n"
        echo "$out" | tail -5 | sed 's/^/    /'
        ((fail++)) || true
    fi
}

echo -e "\n${BOLD}=== ASTRAL-X Test Suite ===${NC}"
echo

# Collect matching inputs (exclude true-tree files and non-TC files)
mapfile -t inputs < <(
    find "$INPUT_DIR" -maxdepth 1 -name "${FILTER}_*.tre" ! -name "*_true.tre" | sort
)

if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "No test inputs matched filter '${FILTER}' in ${INPUT_DIR}/"
    exit 1
fi

for f in "${inputs[@]}"; do
    run_tc "$f"
done

echo
echo -e "  Results: ${GREEN}${pass} passed${NC}  ${RED}${fail} failed${NC}"
echo

[[ $fail -eq 0 ]]
