#!/usr/bin/env bash
# Verifies the already-completed skip list of run-bulk-simulated.sh: the list is
# empty in the published configuration (every replicate runs), a listed tuple
# never leaks into a neighbouring configuration, and the predicate drives the
# skip branch used by the replicate loops.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# Source mode loads only the uppercase skip-list configuration and predicate;
# run-bulk-simulated.sh must not start any simulation or inference work.
source "${ROOT}/scripts/run-bulk-simulated.sh"

# The published list is empty: nothing is skipped unless a user adds tuples for
# replicates whose results they already have.
[[ ${#ALREADY_COMPLETED_SIMULATED_CONFIGS[@]} -eq 0 ]] || \
  fail "expected an empty already-completed list, found ${#ALREADY_COMPLETED_SIMULATED_CONFIGS[@]} entries"

if IS_SIMULATED_CONFIG_ALREADY_COMPLETED 75000 1000 0.000001 100000 200000 R4; then
  fail "a replicate was skipped although the list is empty"
fi

# Add a test-only tuple and drive the same branch used by the production
# replicate loops. Exactly the listed R2 run must be skipped; its neighbours run,
# and nothing leaks into other gene-tree/SB/spmax settings.
TEST_CONFIG="42,7,0.125,11,22,R2"
[[ "$TEST_CONFIG" =~ ^[0-9]+,[0-9]+,[0-9.]+,[0-9]+,[0-9]+,R[0-9]+$ ]] || \
  fail "malformed tuple: $TEST_CONFIG"
ALREADY_COMPLETED_SIMULATED_CONFIGS+=("$TEST_CONFIG")
EXECUTED_REPLICATES=()
SKIPPED_REPLICATES=()
for REPLICATE in R1 R2 R3; do
  if IS_SIMULATED_CONFIG_ALREADY_COMPLETED 42 7 0.125 11 22 "$REPLICATE"; then
    SKIPPED_REPLICATES+=("$REPLICATE")
    continue
  fi
  EXECUTED_REPLICATES+=("$REPLICATE")
done

[[ "${SKIPPED_REPLICATES[*]}" == "R2" ]] || \
  fail "listed tuple did not skip exactly R2: ${SKIPPED_REPLICATES[*]}"
[[ "${EXECUTED_REPLICATES[*]}" == "R1 R3" ]] || \
  fail "listed tuple suppressed a neighbouring run: ${EXECUTED_REPLICATES[*]}"

if IS_SIMULATED_CONFIG_ALREADY_COMPLETED 42 8 0.125 11 22 R2; then
  fail "the tuple leaked into another gene-tree count"
fi
if IS_SIMULATED_CONFIG_ALREADY_COMPLETED 42 7 0.25 11 22 R2; then
  fail "the tuple leaked into another SB setting"
fi
if IS_SIMULATED_CONFIG_ALREADY_COMPLETED 42 7 0.125 11 33 R2; then
  fail "the tuple leaked into another spmax setting"
fi

echo "Bulk-simulated already-completed skip list: PASS"
