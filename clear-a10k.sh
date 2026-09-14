#!/usr/bin/env bash
# Remove all ASTRAL-X results produced by run-a10k.sh while preserving the A10K
# source data, rooted gene trees, and simulation files.

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_ROOT}/scripts/a10k-outputs-dir.sh"

DATA_DIR=""
OUTPUTS_DIR=""
INCLUDE_MIRROR=false
DRY_RUN=false
ASSUME_YES=false

show_usage() {
  cat <<'EOF'
Usage: ./clear-a10k.sh --data-dir DIR [options]

Remove all ASTRAL-X A10K results beneath:
  DIR/10k-simphy/R*/astralx_outputs

The merged DIR/a10k_astralx_scores_merged.csv file is also removed when present.
Input gene trees, rooted gene trees, species trees, and all other dataset files
are preserved. The reproducibility mirror (outputs/<dataset>) is preserved too
unless --include-mirror is given.

Required:
  --data-dir DIR   A10K dataset root containing 10k-simphy/

Options:
  --include-mirror Also remove the mirrored astralx_outputs and merged CSV in
                   the outputs mirror
  --outputs-dir D  Mirror root (default: <parent>/outputs/<dataset>, e.g.
                   \$PHYLOGENY_DATA_DIR/10k-astral-dataset ->
                   \$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset)
  --dry-run        List exact targets without deleting anything
  --yes, -y        Delete without interactive confirmation
  --help, -h       Show this help

Examples:
  ./clear-a10k.sh --data-dir /path/to/10k-astral-dataset --dry-run
  ./clear-a10k.sh --data-dir /path/to/10k-astral-dataset --yes

Do not run this cleaner while run-a10k.sh is active on the same dataset.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
      DATA_DIR="$2"
      shift 2
      ;;
    --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
    --outputs-dir|--a10k-outputs-dir)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
      OUTPUTS_DIR="$2"
      shift 2
      ;;
    --outputs-dir=*|--a10k-outputs-dir=*) OUTPUTS_DIR="${1#*=}"; shift ;;
    --include-mirror) INCLUDE_MIRROR=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --yes|-y) ASSUME_YES=true; shift ;;
    --help|-h) show_usage; exit 0 ;;
    *) echo "Error: unknown option: $1" >&2; show_usage >&2; exit 2 ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Error: --data-dir is required." >&2
  show_usage >&2
  exit 2
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: data directory does not exist: $DATA_DIR" >&2
  exit 2
fi

DATA_DIR="$(realpath "$DATA_DIR")"
case "$DATA_DIR" in
  /|"$HOME")
    echo "Error: refusing to clean unsafe data directory: $DATA_DIR" >&2
    exit 2
    ;;
esac

SIMPHY_DIR="${DATA_DIR}/10k-simphy"
if [[ ! -d "$SIMPHY_DIR" ]]; then
  echo "Error: expected A10K dataset directory at $SIMPHY_DIR" >&2
  exit 2
fi

declare -a RESULT_DIRS=()
declare -a TARGETS=()
while IFS= read -r -d '' replicate_dir; do
  result_dir="${replicate_dir}/astralx_outputs"
  [[ -d "$result_dir" ]] && RESULT_DIRS+=("$result_dir")
done < <(find "$SIMPHY_DIR" -mindepth 1 -maxdepth 1 -type d -name 'R*' -print0 | sort -z -V)

TARGETS=("${RESULT_DIRS[@]}")
MERGED_CSV="${DATA_DIR}/${ASTRALX_A10K_MERGED_CSV_NAME}"
[[ -f "$MERGED_CSV" ]] && TARGETS+=("$MERGED_CSV")

# The reproducibility mirror is only touched on request.
MIRROR_RESULTS_DIR=""
MIRROR_MERGED_CSV=""
if [[ -z "$OUTPUTS_DIR" ]]; then
  OUTPUTS_DIR="$(astralx_default_a10k_outputs_dir "$DATA_DIR")"
fi
OUTPUTS_DIR="$(realpath -m -- "$OUTPUTS_DIR")"
if [[ "$INCLUDE_MIRROR" == true ]]; then
  if astralx__path_is_within "$OUTPUTS_DIR" "$DATA_DIR" || astralx__path_is_within "$DATA_DIR" "$OUTPUTS_DIR"; then
    echo "Error: refusing a mirror directory that overlaps the data directory: $OUTPUTS_DIR" >&2
    exit 2
  fi
  MIRROR_RESULTS_DIR="${OUTPUTS_DIR}/astralx_outputs"
  MIRROR_MERGED_CSV="${OUTPUTS_DIR}/${ASTRALX_A10K_MERGED_CSV_NAME}"
  [[ -d "$MIRROR_RESULTS_DIR" ]] && TARGETS+=("$MIRROR_RESULTS_DIR")
  [[ -f "$MIRROR_MERGED_CSV" ]] && TARGETS+=("$MIRROR_MERGED_CSV")
fi

echo "A10K data: $DATA_DIR"
echo "Results:   $SIMPHY_DIR/R*/astralx_outputs"
if [[ "$INCLUDE_MIRROR" == true ]]; then
  echo "Mirror:    $OUTPUTS_DIR (astralx_outputs and merged CSV will be removed)"
else
  echo "Mirror:    $OUTPUTS_DIR (preserved; pass --include-mirror to remove it too)"
fi
echo

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No A10K ASTRAL-X results found. Nothing to remove."
  exit 0
fi

echo "Targets (${#TARGETS[@]}):"
for target in "${TARGETS[@]}"; do
  case "$target" in
    "$SIMPHY_DIR"/R*/astralx_outputs|"$MERGED_CSV") ;;
    "$MIRROR_RESULTS_DIR"|"$MIRROR_MERGED_CSV") [[ -n "$target" ]] || { echo "Error: empty mirror target" >&2; exit 3; } ;;
    *) echo "Error: unsafe target outside the A10K result layout: $target" >&2; exit 3 ;;
  esac
  printf '  %s\n' "$target"
done
echo

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run only; nothing was removed."
  exit 0
fi

if [[ "$ASSUME_YES" != true ]]; then
  if [[ ! -t 0 ]]; then
    echo "Refusing non-interactive deletion without --yes. Use --dry-run to preview." >&2
    exit 4
  fi
  read -r -p "Remove all listed A10K result targets? [y/N] " reply
  case "${reply,,}" in
    y|yes) ;;
    *) echo "Cancelled; nothing was removed."; exit 0 ;;
  esac
fi

for result_dir in "${RESULT_DIRS[@]}"; do
  [[ "$(basename "$result_dir")" == "astralx_outputs" ]] || {
    echo "Error: refusing unexpected result directory: $result_dir" >&2
    exit 3
  }
  [[ "$(basename "$(dirname "$result_dir")")" == R* ]] || {
    echo "Error: result directory is not beneath an R* replicate: $result_dir" >&2
    exit 3
  }
  rm -rf -- "$result_dir"
done

if [[ -f "$MERGED_CSV" ]]; then
  rm -f -- "$MERGED_CSV"
fi

if [[ "$INCLUDE_MIRROR" == true ]]; then
  if [[ -n "$MIRROR_RESULTS_DIR" && -d "$MIRROR_RESULTS_DIR" ]]; then
    [[ "$(basename "$MIRROR_RESULTS_DIR")" == "astralx_outputs" ]] || {
      echo "Error: refusing unexpected mirror directory: $MIRROR_RESULTS_DIR" >&2
      exit 3
    }
    rm -rf -- "$MIRROR_RESULTS_DIR"
  fi
  [[ -n "$MIRROR_MERGED_CSV" && -f "$MIRROR_MERGED_CSV" ]] && rm -f -- "$MIRROR_MERGED_CSV"
fi

echo "Removed ${#RESULT_DIRS[@]} replicate result director$(
  [[ ${#RESULT_DIRS[@]} -eq 1 ]] && printf 'y' || printf 'ies'
) and the merged CSV if it existed."
if [[ "$INCLUDE_MIRROR" == true ]]; then
  echo "Removed the mirrored astralx_outputs and merged CSV under $OUTPUTS_DIR."
else
  echo "The reproducibility mirror under $OUTPUTS_DIR was preserved."
fi
echo "All A10K input and simulation files were preserved."
