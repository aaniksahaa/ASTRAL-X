#!/usr/bin/env bash
# Verifies that every script touching the SimPhy data tree agrees on one root:
#   explicit path  >  $PHYLOGENY_DATA_DIR/simphy/data  >  repository-local
# Without this the runs, the outputs mirror, and the collect/download/upload
# tools silently read and write different trees.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT}/scripts/phylogeny-data-dir.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/astralx-data-dir-test.XXXXXX")"
trap 'status=$?; rm -rf -- "$WORK"; exit "$status"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# ------------------------------------------------------------- resolution ---
DEFAULT_BASE="${WORK}/phylogeny data"
RESOLVED="$(PHYLOGENY_DATA_DIR="${DEFAULT_BASE}/" astralx_prepare_simphy_data_dir "")"
[[ "$RESOLVED" == "${DEFAULT_BASE}/simphy/data" ]] || fail "unexpected default path: $RESOLVED"
[[ -d "${DEFAULT_BASE}/simphy/data" ]] || fail "default directory was not created"

OVERRIDE="${WORK}/explicit data"
RESOLVED="$(PHYLOGENY_DATA_DIR="$DEFAULT_BASE" astralx_resolve_simphy_data_dir "$OVERRIDE" "${WORK}/unused")"
[[ "$RESOLVED" == "$OVERRIDE" ]] || fail "explicit path did not win over the environment: $RESOLVED"
[[ -d "$OVERRIDE" ]] || fail "override directory was not created"

RESOLVED="$(env -u PHYLOGENY_DATA_DIR bash -c \
  'source "$1/scripts/phylogeny-data-dir.sh"; astralx_resolve_simphy_data_dir "" "$2"' \
  _ "$ROOT" "${WORK}/fallback/simphy/data")"
[[ "$RESOLVED" == "${WORK}/fallback/simphy/data" ]] || fail "fallback was not used without the environment: $RESOLVED"

if env -u PHYLOGENY_DATA_DIR bash -c \
  'source "$1/scripts/phylogeny-data-dir.sh"; astralx_resolve_simphy_data_dir "" ""' \
  _ "$ROOT" >"${WORK}/missing.out" 2>&1; then
  fail "resolution succeeded with neither an environment default, an override, nor a fallback"
fi
grep -q "PHYLOGENY_DATA_DIR is not set" "${WORK}/missing.out" || fail "missing-environment error was unclear"

CONFLICT="${WORK}/not-a-directory"
touch "$CONFLICT"
if astralx_prepare_simphy_data_dir "$CONFLICT" >"${WORK}/conflict.out" 2>&1; then
  fail "a file was accepted as the SimPhy data directory"
fi
grep -q "not a directory" "${WORK}/conflict.out" || fail "file-conflict error was unclear"

# ------------------------------------------------------------------ sim.sh ---
DS="t_1_g_1_sb_0.000001_spmin_500000_spmax_1500000"
DEFAULT_DATASET="${DEFAULT_BASE}/simphy/data/${DS}/R1"
mkdir -p "$DEFAULT_DATASET"
: > "${DEFAULT_DATASET}/stat-sim.csv"
PHYLOGENY_DATA_DIR="$DEFAULT_BASE" "${ROOT}/sim.sh" -t 1 -g 1 >"${WORK}/default-sim.out"
grep -Fq "SKIPPING: ${DEFAULT_DATASET}/stat-sim.csv" "${WORK}/default-sim.out" || \
  fail "sim.sh did not use the environment-derived checkpoint path: $(cat "${WORK}/default-sim.out")"

OVERRIDE_DATASET="${OVERRIDE}/t_2_g_3_sb_0.000001_spmin_500000_spmax_1500000/R1"
mkdir -p "$OVERRIDE_DATASET"
: > "${OVERRIDE_DATASET}/stat-sim.csv"
PHYLOGENY_DATA_DIR="$DEFAULT_BASE" "${ROOT}/sim.sh" -t 2 -g 3 --simphy-data-dir "$OVERRIDE" \
  >"${WORK}/override-sim.out"
grep -Fq "SKIPPING: ${OVERRIDE_DATASET}/stat-sim.csv" "${WORK}/override-sim.out" || \
  fail "sim.sh did not honor the explicit override"

# Without the environment variable sim.sh falls back to <simphy-dir>/data.
CHECKOUT_SIMPHY="${WORK}/checkout/simphy"
FALLBACK_DATASET="${CHECKOUT_SIMPHY}/data/t_4_g_5_sb_0.000001_spmin_500000_spmax_1500000/R1"
mkdir -p "$FALLBACK_DATASET"
: > "${FALLBACK_DATASET}/stat-sim.csv"
env -u PHYLOGENY_DATA_DIR "${ROOT}/sim.sh" -t 4 -g 5 --simphy-dir "$CHECKOUT_SIMPHY" \
  >"${WORK}/fallback-sim.out"
grep -Fq "SKIPPING: ${FALLBACK_DATASET}/stat-sim.csv" "${WORK}/fallback-sim.out" || \
  fail "sim.sh did not fall back to <simphy-dir>/data"

# --------------------------------------------------------- sim_incomplete ---
# The incomplete variant must land in the same environment-derived tree.
printf '((a,b),(c,d));\n((a,c),(b,d));\n' > "${DEFAULT_DATASET}/all_gt.tre"
printf '((a,b),(c,d));\n' > "${DEFAULT_DATASET}/s_tree.trees"
PHYLOGENY_DATA_DIR="$DEFAULT_BASE" "${ROOT}/sim_incomplete.sh" -t 1 -g 1 -rs 1 --min-keep 3 \
  >"${WORK}/incomplete.out" 2>&1 || fail "sim_incomplete.sh failed: $(cat "${WORK}/incomplete.out")"
INC_DIR="${DEFAULT_BASE}/simphy/data/${DS}_incomplete"
[[ -s "${INC_DIR}/R1/all_gt.tre" ]] || fail "incomplete trees were not written to the shared tree"
[[ -f "${INC_DIR}/${DS}_incomplete.command" ]] || fail "incomplete derivation record is missing"

# ------------------------------------------------- collect / download / up ---
STATS_BASE="${WORK}/stats root"
PHYLOGENY_DATA_DIR="$STATS_BASE" "${ROOT}/collect-stats-simulated.sh" \
  --out "${WORK}/unused.csv" >"${WORK}/stats.out"
[[ -d "${STATS_BASE}/simphy/data" ]] || fail "stats collector did not use the default directory"
grep -q "No stat files" "${WORK}/stats.out" || fail "empty stats directory was not handled"

DOWNLOAD_BASE="${WORK}/download root"
PHYLOGENY_DATA_DIR="$DOWNLOAD_BASE" "${ROOT}/download-bulk-simulated.sh" \
  --dry-run --download-script /bin/true --taxa-list 1 --gene-trees-list 1 \
  --sb-list 0.1 --spmin-list 1 --spmax-list 1 >"${WORK}/download.out" 2>&1 || \
  fail "download tool failed: $(cat "${WORK}/download.out")"
[[ -d "${DOWNLOAD_BASE}/simphy/data" ]] || fail "download tool did not use the default directory"

UPLOAD_BASE="${WORK}/upload root"
PHYLOGENY_DATA_DIR="$UPLOAD_BASE" "${ROOT}/upload-bulk-simulated.sh" \
  --dry-run --uploader /bin/true --python /bin/true >"${WORK}/upload.out" 2>&1 || true
[[ -d "${UPLOAD_BASE}/simphy/data" ]] || fail "upload tool did not use the default directory"

OUTPUTS_BASE="${WORK}/outputs root"
PHYLOGENY_DATA_DIR="$OUTPUTS_BASE" "${ROOT}/upload-bulk-simulated-outputs.sh" \
  --dry-run --uploader /bin/true --python /bin/true >"${WORK}/upload-outputs.out" 2>&1
[[ -d "${OUTPUTS_BASE}/outputs/simphy" ]] || fail "outputs uploader did not use the default mirror root"
PHYLOGENY_DATA_DIR="$OUTPUTS_BASE" "${ROOT}/sync-simulated-outputs.sh" --dry-run >"${WORK}/sync-outputs.out"
[[ -d "${OUTPUTS_BASE}/simphy/data" && -d "${OUTPUTS_BASE}/outputs/simphy" ]] || \
  fail "outputs sync did not use the default directories"

# Every remaining consumer of the data tree goes through the same resolver
# rather than hardcoding a checkout-relative simphy/data.
for consumer in test/simmat_comparison/run_upgma_tests.sh simphy/run_simulator.sh; do
  grep -q 'astralx_resolve_simphy_data_dir' "${ROOT}/${consumer}" || \
    fail "${consumer} does not resolve the shared data root"
done

echo "PHYLOGENY_DATA_DIR resolution: PASS"
