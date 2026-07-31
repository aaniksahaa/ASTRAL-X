#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_CLASSES="$(mktemp -d "${TMPDIR:-/tmp}/astralx-cli-tests.XXXXXX")"
trap 'rm -rf "$TEST_CLASSES"' EXIT

"${ROOT}/build.sh" >/dev/null
javac -cp "${ROOT}/build" -d "$TEST_CLASSES" "${ROOT}/test/CliPresetsTest.java"
java -cp "${ROOT}/build:${TEST_CLASSES}" astralx.CliPresetsTest

run_score() {
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc1_complete_only.tre" "$@" 2>&1 |
    sed -n 's/.*optimal quartet score = \([0-9][0-9]*\).*/\1/p'
}

[[ "$(run_score --search-space S1 --intersection-method I2)" == \
   "$(run_score --search-mode local --weight-intersection-method prefix-sum)" ]]
[[ "$(run_score --search-space S3 --im I3)" == \
   "$(run_score --search-mode full --weight-intersection-method simple-tree-walk)" ]]

if java -cp "${ROOT}/build" astralx.Main --cpu --diagnose --search-space S9 \
    >/dev/null 2>&1; then
  echo "invalid search preset was unexpectedly accepted" >&2
  exit 1
fi

echo "CLI end-to-end aliases: PASS"
