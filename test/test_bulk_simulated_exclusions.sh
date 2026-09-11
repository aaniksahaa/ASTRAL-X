#!/usr/bin/env bash
# Verifies the per-replicate exclusion list of run-bulk-simulated.sh: the exact
# configured tuples, that no exclusion leaks into a neighbouring configuration,
# and that the predicate drives the skip branch used by the replicate loops.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# Source mode loads only the uppercase exclusion configuration and predicate;
# run-bulk-simulated.sh must not start any simulation or inference work.
source "${ROOT}/run-bulk-simulated.sh"

# Pinned copy of the configured list, so an accidental edit is caught here.
EXPECTED_EXCLUSIONS=(
  "30000,1000,0.000001,100000,150000,R5"
  "30000,1000,0.000001,100000,250000,R4"
  "30000,1000,0.000001,100000,300000,R3"
  "40000,1000,0.000001,100000,150000,R2"
  "40000,1000,0.000001,100000,150000,R5"
  "40000,1000,0.000001,100000,200000,R5"
  "40000,1000,0.000001,100000,300000,R5"
  "1000,25000,0.000001,100000,200000,R3"
  "1000,25000,0.000001,100000,250000,R2"
  "75000,1000,0.000001,100000,200000,R5"
  "100000,1000,0.000001,100000,200000,R3"
  "100000,1000,0.000001,100000,200000,R4"
  "125000,1000,0.000001,100000,200000,R2"
  "125000,1000,0.000001,100000,200000,R3"
  "125000,1000,0.000001,100000,200000,R4"
  "150000,1000,0.000001,100000,200000,R4"
  "175000,1000,0.000001,100000,200000,R2"
  "175000,1000,0.000001,100000,200000,R4"
  "200000,1000,0.000001,100000,200000,R1"
  "200000,1000,0.000001,100000,200000,R2"
  "225000,1000,0.000001,100000,200000,R1"
  "225000,1000,0.000001,100000,200000,R3"
  "1000,25000,0.000001,100000,200000,R3"
  "1000,75000,0.000001,100000,200000,R3"
  "1000,100000,0.000001,100000,200000,R4"
  "1000,125000,0.000001,100000,200000,R4"
  "1000,150000,0.000001,100000,200000,R2"
  "1000,200000,0.000001,100000,200000,R4"
  "1000,225000,0.000001,100000,200000,R2"
  "1000,275000,0.000001,100000,200000,R3"
  "1000,300000,0.000001,100000,200000,R2"
)

[[ ${#EXCLUDED_SIMULATED_CONFIGS[@]} -eq ${#EXPECTED_EXCLUSIONS[@]} ]] || \
  fail "expected ${#EXPECTED_EXCLUSIONS[@]} configured exclusions, found ${#EXCLUDED_SIMULATED_CONFIGS[@]}"
[[ "${EXCLUDED_SIMULATED_CONFIGS[*]}" == "${EXPECTED_EXCLUSIONS[*]}" ]] || \
  fail "the configured exclusion list does not match the pinned one"

for EXPECTED_CONFIG in "${EXPECTED_EXCLUSIONS[@]}"; do
  [[ "$EXPECTED_CONFIG" =~ ^[0-9]+,[0-9]+,[0-9.]+,[0-9]+,[0-9]+,R[0-9]+$ ]] || \
    fail "malformed exclusion tuple: $EXPECTED_CONFIG"
  IFS=',' read -r TAXA GENES SB SPMIN SPMAX REPLICATE <<< "$EXPECTED_CONFIG"
  IS_SIMULATED_CONFIG_EXCLUDED \
    "$TAXA" "$GENES" "$SB" "$SPMIN" "$SPMAX" "$REPLICATE" || \
    fail "configured tuple was not excluded: $EXPECTED_CONFIG"
done

if IS_SIMULATED_CONFIG_EXCLUDED 75000 1000 0.000001 100000 200000 R4; then
  fail "an unlisted replicate was excluded"
fi
if IS_SIMULATED_CONFIG_EXCLUDED 75000 2000 0.000001 100000 200000 R5; then
  fail "an exclusion leaked into another gene-tree count"
fi
if IS_SIMULATED_CONFIG_EXCLUDED 75000 1000 0.000002 100000 200000 R5; then
  fail "an exclusion leaked into another SB setting"
fi
if IS_SIMULATED_CONFIG_EXCLUDED 75000 1000 0.000001 100000 300000 R5; then
  fail "an exclusion leaked into another spmax setting"
fi

# Add a test-only dummy tuple and drive the same branch used by the production
# replicate loops. Exactly the dummy R2 run must be skipped; its neighbours run.
DUMMY_CONFIG="42,7,0.125,11,22,R2"
EXCLUDED_SIMULATED_CONFIGS+=("$DUMMY_CONFIG")
EXECUTED_REPLICATES=()
SKIPPED_REPLICATES=()
for REPLICATE in R1 R2 R3; do
  if IS_SIMULATED_CONFIG_EXCLUDED 42 7 0.125 11 22 "$REPLICATE"; then
    SKIPPED_REPLICATES+=("$REPLICATE")
    continue
  fi
  EXECUTED_REPLICATES+=("$REPLICATE")
done

[[ "${SKIPPED_REPLICATES[*]}" == "R2" ]] || \
  fail "dummy exclusion did not skip exactly R2: ${SKIPPED_REPLICATES[*]}"
[[ "${EXECUTED_REPLICATES[*]}" == "R1 R3" ]] || \
  fail "dummy exclusion suppressed a neighbouring run: ${EXECUTED_REPLICATES[*]}"

echo "Bulk-simulated replicate exclusions: PASS"
