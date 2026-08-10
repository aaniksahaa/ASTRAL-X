#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_CLASSES="$(mktemp -d "${TMPDIR:-/tmp}/astralx-cli-tests.XXXXXX")"
trap 'rm -rf "$TEST_CLASSES"' EXIT

source "${ROOT}/experiment-setting-name.sh"
[[ "$(build_setting_name_from_opts '--search-space S1 --intersection-method I2 -vv')" == \
   "search-space_S1__intersection-method_I2" ]]
[[ "$(build_setting_name_from_opts '--search-space complete-full --weight-intersection-method prefix-sum --cpu')" == \
   "search-space_S2__intersection-method_I2__cpu_true" ]]
[[ "$(build_setting_name_from_opts '--search-mode local --im I1 --threads 8 -q')" == \
   "search-mode_local__intersection-method_I1__threads_8" ]]
[[ "$(build_setting_name_from_opts '-vv --quiet')" == "default" ]]

"${ROOT}/build.sh" >/dev/null
javac -cp "${ROOT}/build" -d "$TEST_CLASSES" \
  "${ROOT}/test/CliPresetsTest.java" \
  "${ROOT}/test/PackedSimilarityParityTest.java" \
  "${ROOT}/test/PackedPreflightTest.java" \
  "${ROOT}/test/astralx/completion/PackedMatrixBoundaryTest.java" \
  "${ROOT}/test/SimilarityArgminTest.java" \
  "${ROOT}/test/ThreadingFailureTest.java" \
  "${ROOT}/test/WideSimilarityBoundaryTest.java"
java -cp "${ROOT}/build:${TEST_CLASSES}" astralx.CliPresetsTest
java -cp "${ROOT}/build:${TEST_CLASSES}" astralx.completion.PackedMatrixBoundaryTest
java -Xmx1g -cp "${ROOT}/build:${TEST_CLASSES}" PackedPreflightTest
java -cp "${ROOT}/build:${TEST_CLASSES}" PackedSimilarityParityTest \
  "${ROOT}/test/input/tc5_heavy_incomplete.tre"
java -cp "${ROOT}/build:${TEST_CLASSES}" ThreadingFailureTest
java -Xmx4g -cp "${ROOT}/build:${TEST_CLASSES}" WideSimilarityBoundaryTest
java -cp "${ROOT}/build:${TEST_CLASSES}" SimilarityArgminTest \
  "${ROOT}/test/input/tc10_unrooted_8taxa.tre" \
  "${ROOT}/test/input/tc14_polytomy_6taxa.tre"
bash "${ROOT}/test/test_simulated_success_detection.sh"

VERSION_TEXT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --version)"
[[ "$VERSION_TEXT" == *"ASTRAL-X  v1.0.0"* ]]
[[ "$VERSION_TEXT" == *"Welcome to ASTRAL-X version 1.0.0!"* ]]
[[ "$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main -v)" == "$VERSION_TEXT" ]]

VERSION_COLOR="$(env -u NO_COLOR FORCE_COLOR=1 java -cp "${ROOT}/build" astralx.Main --version)"
WHITE_GREETING="$(printf '\033[97mWelcome to ASTRAL-X version 1.0.0!\033[0m')"
[[ "$VERSION_COLOR" == *"$WHITE_GREETING"* ]]

HELP_TEXT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --help 2>&1)"
[[ "$HELP_TEXT" == *"ASTRAL-X  v1.0.0"* ]]
[[ "$HELP_TEXT" == *"Usage:"* ]]
[[ "$HELP_TEXT" == *"--log-file FILE"* ]]
[[ "$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main -h 2>&1)" == "$HELP_TEXT" ]]

run_score() {
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc1_complete_only.tre" "$@" 2>&1 |
    sed -n 's/.*Quartet score[[:space:]]*\([0-9][0-9]*\).*/\1/p'
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

# Force the large-N matrix implementation through complete S2/S3 inference on a
# small incomplete dataset.  Final Newick must remain byte-for-byte identical.
for preset in S2 S3; do
  dense_tree="${TEST_CLASSES}/${preset}-dense.tre"
  packed_tree="${TEST_CLASSES}/${preset}-packed.tre"
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc5_heavy_incomplete.tre" -o "$dense_tree" \
    --search-space "$preset" --intersection-method I1 >/dev/null 2>&1
  java -Dastralx.similarity.forcePacked=true -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc5_heavy_incomplete.tre" -o "$packed_tree" \
    --search-space "$preset" --intersection-method I1 >/dev/null 2>&1
  cmp "$dense_tree" "$packed_tree"
done

if java -cp "${ROOT}/build" astralx.Main --cpu --diagnose --search-space S4 \
    >/dev/null 2>&1; then
  echo "invalid search preset was unexpectedly accepted" >&2
  exit 1
fi

SUMMARY_TREE="${TEST_CLASSES}/summary-tree.tre"
SUMMARY_OUTPUT="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/tc1_complete_only.tre" -o "$SUMMARY_TREE" 2>&1)"
[[ -s "$SUMMARY_TREE" ]]
[[ "$SUMMARY_OUTPUT" == *"Run Summary"* ]]
[[ "$SUMMARY_OUTPUT" == *"Quartet score"* ]]
[[ "$SUMMARY_OUTPUT" == *"Running time"* ]]
[[ "$SUMMARY_OUTPUT" == *"Max CPU RAM"* ]]
[[ "$SUMMARY_OUTPUT" == *"Max GPU VRAM"* ]]
[[ "$SUMMARY_OUTPUT" == *"N/A (CPU execution)"* ]]

LOG_TREE="${TEST_CLASSES}/logged-tree.tre"
LOG_FILE="${TEST_CLASSES}/nested/astralx.log"
LOG_TERMINAL="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --cpu \
  -i "${ROOT}/test/input/tc1_complete_only.tre" -o "$LOG_TREE" \
  --log-file "$LOG_FILE" 2>&1)"
[[ -s "$LOG_TREE" && -s "$LOG_FILE" ]]
[[ "$LOG_TERMINAL" == *"Run Summary"* ]]
[[ "$LOG_TERMINAL" == *$'\r'* ]]
grep -q "Run Summary" "$LOG_FILE"
grep -q "Quartet score" "$LOG_FILE"
if [[ "$(LC_ALL=C tr -cd '\r' < "$LOG_FILE" | wc -c)" -ne 0 ]]; then
  echo "carriage-return progress repaint leaked into --log-file" >&2
  exit 1
fi
if grep -q "Parsing trees.*it/s" "$LOG_FILE"; then
  echo "progress bar leaked into --log-file" >&2
  exit 1
fi

if java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc1_complete_only.tre" \
    --log-file "${ROOT}/test/input/tc1_complete_only.tre" >/dev/null 2>&1; then
  echo "input/log collision was unexpectedly accepted" >&2
  exit 1
fi

SCORE_ONLY_OUTPUT="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/tc1_complete_only.tre" \
  --score-species-tree "${ROOT}/test/input/tc1_true.tre" 2>&1)"
[[ "$SCORE_ONLY_OUTPUT" == *"QUARTET_SCORE: 10"* ]]
[[ "$SCORE_ONLY_OUTPUT" == *"Run Summary"* ]]
[[ "$SCORE_ONLY_OUTPUT" == *"Quartet score"*"10"* ]]

echo "CLI end-to-end aliases: PASS"
