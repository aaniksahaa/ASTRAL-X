#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_CLASSES="$(mktemp -d "${TMPDIR:-/tmp}/astralx-cli-tests.XXXXXX")"
trap 'rm -rf "$TEST_CLASSES"' EXIT

"${ROOT}/build.sh" >/dev/null
javac -cp "${ROOT}/build" -d "$TEST_CLASSES" "${ROOT}/test/CliPresetsTest.java"
java -cp "${ROOT}/build:${TEST_CLASSES}" astralx.CliPresetsTest

VERSION_TEXT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --version)"
[[ "$VERSION_TEXT" == "Welcome to ASTRAL-X version 1.0.0!" ]]

VERSION_COLOR="$(env -u NO_COLOR FORCE_COLOR=1 java -cp "${ROOT}/build" astralx.Main --version)"
EXPECTED_COLOR="$(printf '\033[36mWelcome\033[0m to \033[32mASTRAL-X\033[0m version 1.0.0!')"
[[ "$VERSION_COLOR" == "$EXPECTED_COLOR" ]]

run_score() {
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc1_complete_only.tre" "$@" 2>&1 |
    sed -n 's/.*optimal quartet score = \([0-9][0-9]*\).*/\1/p'
}

[[ "$(run_score --search-space S1 --intersection-method I2)" == \
   "$(run_score --search-mode local --weight-intersection-method prefix-sum)" ]]
[[ "$(run_score --search-space S2 --im I3)" == \
   "$(run_score --autocomplete-incomplete-gene-trees --search-mode full \
      --weight-intersection-method simple-tree-walk)" ]]
[[ "$(run_score --search-space S3 --intersection-method I2)" == \
   "$(run_score --autocomplete-incomplete-gene-trees --search-mode full \
      --consensus-experimental --stepb-quadratic-nn-balls \
      --stepb-random-leftover-resolution --stepb-process-large-polytomies \
      --resolve-input-gene-tree-polytomies --weight-intersection-method prefix-sum)" ]]

if java -cp "${ROOT}/build" astralx.Main --cpu --diagnose --search-space S4 \
    >/dev/null 2>&1; then
  echo "invalid search preset was unexpectedly accepted" >&2
  exit 1
fi

echo "CLI end-to-end aliases: PASS"
