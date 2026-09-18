#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_CLASSES="$(mktemp -d "${TMPDIR:-/tmp}/astralx-cli-tests.XXXXXX")"
trap 'rm -rf "$TEST_CLASSES"' EXIT

source "${ROOT}/scripts/experiment-setting-name.sh"
[[ "$(build_setting_name_from_opts '--search-space S1 --intersection-method I2 -vv')" == \
   "search-space_S1__intersection-method_I2" ]]
[[ "$(build_setting_name_from_opts '--search-space complete-full --weight-intersection-method prefix-sum --cpu')" == \
   "search-space_S2__intersection-method_I2__cpu_true" ]]
[[ "$(build_setting_name_from_opts '--search-mode local --im I1 --threads 8 -q')" == \
   "search-mode_local__intersection-method_I1__threads_8" ]]
[[ "$(build_setting_name_from_opts '-vv --quiet')" == "default" ]]

"${ROOT}/scripts/build.sh" >/dev/null
javac -cp "${ROOT}/build" -d "$TEST_CLASSES" \
  "${ROOT}/test/CliPresetsTest.java" \
  "${ROOT}/test/PackedSimilarityParityTest.java" \
  "${ROOT}/test/PackedPreflightTest.java" \
  "${ROOT}/test/PolytomyPreprocessingTest.java" \
  "${ROOT}/test/astralx/completion/PackedMatrixBoundaryTest.java" \
  "${ROOT}/test/astralx/completion/TreeCompleterPolytomyTest.java" \
  "${ROOT}/test/SimilarityArgminTest.java" \
  "${ROOT}/test/ThreadingFailureTest.java" \
  "${ROOT}/test/WideSimilarityBoundaryTest.java"
java -cp "${ROOT}/build:${TEST_CLASSES}" astralx.CliPresetsTest
java -cp "${ROOT}/build:${TEST_CLASSES}" astralx.completion.PackedMatrixBoundaryTest
java -cp "${ROOT}/build:${TEST_CLASSES}" PolytomyPreprocessingTest \
  "${ROOT}/test/input/completion/incomplete_polytomy_7taxa.tre"
java -cp "${ROOT}/build:${TEST_CLASSES}" astralx.completion.TreeCompleterPolytomyTest \
  "${ROOT}/test/input/completion/incomplete_polytomy_7taxa.tre"

polytomy_completion_score() {
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc16_polytomy_incomplete.tre" \
    --search-space "$1" --intersection-method I3 "${@:2}" 2>&1 |
    sed -n 's/.*Quartet score[[:space:]]*\([0-9][0-9]*\).*/\1/p'
}
[[ "$(polytomy_completion_score S2)" == "784" ]]
[[ "$(polytomy_completion_score S3)" == "784" ]]
[[ "$(polytomy_completion_score S2 --keep-polytomy-during-inference)" == "794" ]]
[[ "$(polytomy_completion_score S3 --keep-polytomy-during-inference)" == "798" ]]

POLY_DEFAULT_OUTPUT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/tc16_polytomy_incomplete.tre" \
  --search-space S2 --intersection-method I3 2>&1)"
POLY_NATIVE_OUTPUT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/tc16_polytomy_incomplete.tre" \
  --search-space S2 --intersection-method I3 \
  --keep-polytomy-during-inference 2>&1)"
[[ "$POLY_DEFAULT_OUTPUT" == *"Phase 8  Final quartet scoring against unresolved input"* ]]
[[ "$POLY_NATIVE_OUTPUT" != *"Phase 8  Final quartet scoring against unresolved input"* ]]

# Incomplete-tree residual clusters need not have their own ClusterTable exemplar.
# Reconstruction must inherit their exact parent membership instead of emitting
# the historical "?" placeholder, and Phase 8 must accept the resulting tree.
RECONSTRUCTION_OUTPUT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/reconstruction_incomplete_polytomy.tre" \
  --taxa-file "${ROOT}/test/input/reconstruction_incomplete_taxa.txt" \
  --search-space S1 --intersection-method I3 2>&1)"
RECONSTRUCTION_TREE="$(grep ';' <<<"$RECONSTRUCTION_OUTPUT" | tail -n1)"
[[ -n "$RECONSTRUCTION_TREE" && "$RECONSTRUCTION_TREE" != *'?'* ]]
for taxon in A B C D E F G H I J; do
  [[ "$RECONSTRUCTION_TREE" == *"$taxon"* ]]
done
[[ "$RECONSTRUCTION_OUTPUT" == *"Phase 8  Final quartet scoring against unresolved input"* ]]

java -Xmx1g -cp "${ROOT}/build:${TEST_CLASSES}" PackedPreflightTest
java -cp "${ROOT}/build:${TEST_CLASSES}" PackedSimilarityParityTest \
  "${ROOT}/test/input/tc5_heavy_incomplete.tre"
java -cp "${ROOT}/build:${TEST_CLASSES}" ThreadingFailureTest
java -Xmx4g -cp "${ROOT}/build:${TEST_CLASSES}" WideSimilarityBoundaryTest
java -cp "${ROOT}/build:${TEST_CLASSES}" SimilarityArgminTest \
  "${ROOT}/test/input/tc10_unrooted_8taxa.tre" \
  "${ROOT}/test/input/tc14_polytomy_6taxa.tre"
bash "${ROOT}/test/test_simulated_success_detection.sh"
bash "${ROOT}/test/test_simulated_outputs_mirror.sh"
bash "${ROOT}/test/test_a10k_outputs_mirror.sh"
bash "${ROOT}/test/test_phylogeny_data_dir.sh"
bash "${ROOT}/test/test_bulk_simulated_exclusions.sh"

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
[[ "$HELP_TEXT" == *"--keep-polytomy-during-inference"* ]]
[[ "$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main -h 2>&1)" == "$HELP_TEXT" ]]
if java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/tc1_complete_only.tre" --keep-polytomy \
    >/dev/null 2>&1; then
  echo "removed --keep-polytomy option was unexpectedly accepted" >&2
  exit 1
fi

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
[[ "$SUMMARY_OUTPUT" != *"Phase 8  Final quartet scoring against unresolved input"* ]]

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

# Parser-backed taxa extraction: deterministic union by default and explicit
# intersection, with exactly one taxon name per output line.
TAXA_UNION="${TEST_CLASSES}/taxa-union.txt"
TAXA_INTERSECTION="${TEST_CLASSES}/taxa-intersection.txt"
"${ROOT}/scripts/extract-taxa.sh" --no-build \
  -i "${ROOT}/test/input/taxa_extract_multi.tre" -o "$TAXA_UNION" >/dev/null
"${ROOT}/scripts/extract-taxa.sh" --no-build \
  -i "${ROOT}/test/input/taxa_extract_multi.tre" -o "$TAXA_INTERSECTION" \
  --intersection >/dev/null
printf 'A\nB\nC\nD\nE\n' > "${TEST_CLASSES}/expected-union.txt"
printf 'B\nC\nD\n' > "${TEST_CLASSES}/expected-intersection.txt"
cmp "${TEST_CLASSES}/expected-union.txt" "$TAXA_UNION"
cmp "${TEST_CLASSES}/expected-intersection.txt" "$TAXA_INTERSECTION"
TAXA_STDOUT="$(java -cp "${ROOT}/build" astralx.Main -q \
  -i "${ROOT}/test/input/taxa_extract_multi.tre" --extract-taxa)"
[[ "$TAXA_STDOUT" == $'A\nB\nC\nD\nE' ]]

# Taxon-filtered scoring must equal scoring manually induced input trees for
# every intersection implementation. The fixture also covers a duplicate list
# line, a listed taxon absent from both inputs, outside taxa on both sides, and
# a post-filter gene tree with <4 leaves (zero quartet contribution).
score_value() {
  sed -n 's/^QUARTET_SCORE: //p' | tail -n1
}
for method in I1 I2 I3 I4; do
  manual_score="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_genes_manual.tre" \
    --score-species-tree "${ROOT}/test/input/taxa_filter_species_manual.tre" \
    --im "$method" 2>&1 | score_value)"
  filtered_score="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_genes.tre" \
    --score-species-tree "${ROOT}/test/input/taxa_filter_species.tre" \
    --taxa-file "${ROOT}/test/input/taxa_filter_list.txt" \
    --im "$method" 2>&1 | score_value)"
  [[ "$manual_score" == "4" && "$filtered_score" == "$manual_score" ]]
done

# Native-polytomy restriction parity, covering both an unrooted multifurcating
# root and an internal polytomy whose degree decreases after pruning.
for method in I1 I2 I3 I4; do
  manual_score="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_polytomy_genes_manual.tre" \
    --score-species-tree "${ROOT}/test/input/taxa_filter_polytomy_species_manual.tre" \
    --im "$method" 2>&1 | score_value)"
  filtered_score="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_polytomy_genes.tre" \
    --score-species-tree "${ROOT}/test/input/taxa_filter_polytomy_species.tre" \
    --taxa-file "${ROOT}/test/input/taxa_filter_polytomy_list.txt" \
    --im "$method" 2>&1 | score_value)"
  [[ -n "$manual_score" && "$filtered_score" == "$manual_score" ]]
done

FILTER_REPORT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --cpu \
  -i "${ROOT}/test/input/taxa_filter_genes.tre" \
  --score-species-tree "${ROOT}/test/input/taxa_filter_species.tre" \
  --taxa-file "${ROOT}/test/input/taxa_filter_list.txt" --im I2 2>&1)"
[[ "$FILTER_REPORT" == *"Taxon filter report:"* ]]
[[ "$FILTER_REPORT" == *"5 unique name(s) (1 duplicate line(s) ignored)"* ]]
[[ "$FILTER_REPORT" == *"mean=1.25 (25.000%), min=1, max=2"* ]]
[[ "$FILTER_REPORT" == *"Species tree: 1 listed taxa missing (20.000%)"* ]]
[[ "$FILTER_REPORT" == *"Ignored outside taxa: gene-tree union=1, species tree=1"* ]]
[[ "$FILTER_REPORT" == *"Effective common scoring universe: 4 taxa"* ]]

# Taxon-restricted inference must be identical to inference from manually
# induced gene trees. This covers outside taxa, an absent listed taxon, a
# duplicate allow-list line, unary suppression, retention of two/three-leaf trees,
# dropping of zero/one-leaf trees, and restriction before binary refinement.
for method in I1 I2 I3 I4; do
  manual_tree="${TEST_CLASSES}/taxa-inference-manual-${method}.tre"
  filtered_tree="${TEST_CLASSES}/taxa-inference-filtered-${method}.tre"
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_inference_manual.tre" \
    -o "$manual_tree" --im "$method" >/dev/null 2>&1
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_inference_genes.tre" \
    --taxa-file "${ROOT}/test/input/taxa_filter_list.txt" \
    -o "$filtered_tree" --im "$method" >/dev/null 2>&1
  cmp "$manual_tree" "$filtered_tree"
  if grep -Eq '(^|[(,])(X|Y|Z)([),;]|$)' "$filtered_tree"; then
    echo "taxon-restricted inference emitted an excluded/absent taxon" >&2
    exit 1
  fi
done

# Completion/consensus presets must consume only the induced universe too.
for preset in S2 S3; do
  manual_tree="${TEST_CLASSES}/taxa-inference-manual-${preset}.tre"
  filtered_tree="${TEST_CLASSES}/taxa-inference-filtered-${preset}.tre"
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_inference_manual.tre" \
    -o "$manual_tree" --search-space "$preset" --im I3 >/dev/null 2>&1
  java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_inference_genes.tre" \
    --taxa-file "${ROOT}/test/input/taxa_filter_list.txt" \
    -o "$filtered_tree" --search-space "$preset" --im I3 >/dev/null 2>&1
  cmp "$manual_tree" "$filtered_tree"
done

# Native-polytomy inference follows the same induced-tree semantics.
manual_poly_tree="${TEST_CLASSES}/taxa-inference-polytomy-manual.tre"
filtered_poly_tree="${TEST_CLASSES}/taxa-inference-polytomy-filtered.tre"
java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/taxa_filter_polytomy_genes_manual.tre" \
  -o "$manual_poly_tree" --keep-polytomy-during-inference --search-mode full --im I2 \
  >/dev/null 2>&1
java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/taxa_filter_polytomy_genes.tre" \
  --taxa-file "${ROOT}/test/input/taxa_filter_polytomy_list.txt" \
  -o "$filtered_poly_tree" --keep-polytomy-during-inference --search-mode full --im I2 \
  >/dev/null 2>&1
cmp "$manual_poly_tree" "$filtered_poly_tree"

# With default refinement, both runs must re-score the inferred topology against
# unresolved gene trees, and the taxa-file run must use only its induced universe.
manual_poly_default_tree="${TEST_CLASSES}/taxa-inference-polytomy-manual-default.tre"
filtered_poly_default_tree="${TEST_CLASSES}/taxa-inference-polytomy-filtered-default.tre"
manual_poly_default_output="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/taxa_filter_polytomy_genes_manual.tre" \
  -o "$manual_poly_default_tree" --search-mode full --im I2 2>&1)"
filtered_poly_default_output="$(java -cp "${ROOT}/build" astralx.Main --cpu -q \
  -i "${ROOT}/test/input/taxa_filter_polytomy_genes.tre" \
  --taxa-file "${ROOT}/test/input/taxa_filter_polytomy_list.txt" \
  -o "$filtered_poly_default_tree" --search-mode full --im I2 2>&1)"
cmp "$manual_poly_default_tree" "$filtered_poly_default_tree"
manual_poly_final_score="$(sed -n \
  's/.*Quartet score[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
  <<<"$manual_poly_default_output" | tail -n1)"
filtered_poly_final_score="$(sed -n \
  's/.*Quartet score[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
  <<<"$filtered_poly_default_output" | tail -n1)"
[[ "$manual_poly_final_score" == "16" ]]
[[ "$filtered_poly_final_score" == "$manual_poly_final_score" ]]
[[ "$filtered_poly_default_output" == \
   *"Phase 8  Final quartet scoring against unresolved input"* ]]

INFERENCE_FILTER_REPORT="$(NO_COLOR=1 java -cp "${ROOT}/build" astralx.Main --cpu \
  -i "${ROOT}/test/input/taxa_filter_inference_genes.tre" \
  --taxa-file "${ROOT}/test/input/taxa_filter_list.txt" --im I2 2>&1)"
[[ "$INFERENCE_FILTER_REPORT" == *"Effective inference universe: 4 taxa"* ]]
[[ "$INFERENCE_FILTER_REPORT" == *"Listed taxa absent from every gene tree: 1 (20.000%)"* ]]
[[ "$INFERENCE_FILTER_REPORT" == *"Ignored unlisted leaf occurrences: 9"* ]]
[[ "$INFERENCE_FILTER_REPORT" == *"retained 6/8 induced gene tree(s)"* ]]
[[ "$INFERENCE_FILTER_REPORT" == *"Final quartet score ="* ]]

if java -cp "${ROOT}/build" astralx.Main --cpu -q \
    -i "${ROOT}/test/input/taxa_filter_inference_genes.tre" \
    --taxa-file "${ROOT}/test/input/taxa_filter_list.txt" \
    -o "${ROOT}/test/input/taxa_filter_list.txt" >/dev/null 2>&1; then
  echo "taxa-file/output collision was unexpectedly accepted for inference" >&2
  exit 1
fi

echo "CLI end-to-end aliases: PASS"
