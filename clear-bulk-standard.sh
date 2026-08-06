#!/usr/bin/env bash
# Remove one method's generated standard-dataset statistics without touching
# source data or results belonging to other methods.

set -euo pipefail

BASE_DIR="${HOME}/phylogeny"
DATASET_DIR=""
METHOD=""
DRY_RUN=false
ASSUME_YES=false
ALL_RESULTS=false

show_usage() {
  cat <<'EOF'
Usage: ./clear-bulk-standard.sh --method METHOD [options]

Remove generated files for one method from the bulk-standard dataset tree.
By default, only method-specific statistics CSVs and lock markers are removed;
output trees and logs are preserved. This is enough to exclude the method from
the next collect-stats-standard.sh run and allow run-bulk-standard.sh to rerun it.

Required:
  --method, -m METHOD   astralx | aster | astral | treeqmc | wqfmtree |
                        supertriplets | stp-nni | tmc

Paths:
  --base-dir, -b DIR    Base directory (default: $HOME/phylogeny)
  --dataset-dir, -d DIR Standard dataset directory
                        (default: BASE_DIR/datasets/standard)

Modes:
  --dry-run             List matching targets without deleting anything
  --yes, -y             Delete without an interactive confirmation
  --all-results         Remove the selected method's complete *_outputs
                        directories, including trees, CSVs, logs, and locks
  --help, -h            Show this help

Examples:
  ./clear-bulk-standard.sh --method astralx --dry-run
  ./clear-bulk-standard.sh --method astralx
  ./clear-bulk-standard.sh --method astralx --yes
EOF
}

normalize_method() {
  case "${1,,}" in
    astralx|astral-x|stelar|stelar-x) printf 'astralx' ;;
    aster)                            printf 'aster' ;;
    astral)                           printf 'astral' ;;
    treeqmc|tree-qmc)                 printf 'treeqmc' ;;
    wqfm|wqfmtree|wqfm-tree)          printf 'wqfmtree' ;;
    supertriplets|super-triplets)     printf 'supertriplets' ;;
    stp-nni|stpnni)                   printf 'stp-nni' ;;
    tmc)                              printf 'tmc' ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method|-m)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
      METHOD="$2"
      shift 2
      ;;
    --method=*) METHOD="${1#*=}"; shift ;;
    --base-dir|-b)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
      BASE_DIR="$2"
      shift 2
      ;;
    --base-dir=*) BASE_DIR="${1#*=}"; shift ;;
    --dataset-dir|-d)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
      DATASET_DIR="$2"
      shift 2
      ;;
    --dataset-dir=*) DATASET_DIR="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --yes|-y) ASSUME_YES=true; shift ;;
    --all-results) ALL_RESULTS=true; shift ;;
    --help|-h) show_usage; exit 0 ;;
    *) echo "Error: unknown option: $1" >&2; show_usage >&2; exit 2 ;;
  esac
done

if [[ -z "$METHOD" ]]; then
  echo "Error: --method is required." >&2
  show_usage >&2
  exit 2
fi

if ! METHOD="$(normalize_method "$METHOD")"; then
  echo "Error: unsupported method '$METHOD'." >&2
  exit 2
fi

if [[ -z "$DATASET_DIR" ]]; then
  DATASET_DIR="${BASE_DIR%/}/datasets/standard"
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Error: dataset directory does not exist: $DATASET_DIR" >&2
  exit 2
fi

DATASET_DIR="$(realpath "$DATASET_DIR")"
case "$DATASET_DIR" in
  /|"$HOME")
    echo "Error: refusing to clean unsafe dataset directory: $DATASET_DIR" >&2
    exit 2
    ;;
esac

OUTPUT_DIR_NAME="${METHOD}_outputs"
declare -a OUTPUT_DIRS=()
declare -a TARGETS=()
mapfile -d '' OUTPUT_DIRS < <(
  find "$DATASET_DIR" -type d -name "$OUTPUT_DIR_NAME" -print0 2>/dev/null
)

if [[ "$ALL_RESULTS" == true ]]; then
  TARGETS=("${OUTPUT_DIRS[@]}")
else
  for output_dir in "${OUTPUT_DIRS[@]}"; do
    mapfile -d '' -O "${#TARGETS[@]}" TARGETS < <(
      find "$output_dir" -type f \
        \( -name "stat-${METHOD}.csv" \
           -o -name "*-${METHOD}_stats.csv" \
           -o -name ".${METHOD}.lock" \) \
        -print0 2>/dev/null
    )
  done
fi

echo "Dataset: $DATASET_DIR"
echo "Method:  $METHOD"
if [[ "$ALL_RESULTS" == true ]]; then
  echo "Mode:    complete method output directories"
else
  echo "Mode:    statistics CSVs and lock markers (trees/logs preserved)"
fi
echo

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No matching $METHOD results found. Nothing to remove."
  exit 0
fi

echo "Targets (${#TARGETS[@]}):"
for target in "${TARGETS[@]}"; do
  case "$target" in
    "$DATASET_DIR"/*) ;;
    *) echo "Error: unsafe target escaped dataset directory: $target" >&2; exit 3 ;;
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
  read -r -p "Remove these ${#TARGETS[@]} target(s)? [y/N] " reply
  case "${reply,,}" in
    y|yes) ;;
    *) echo "Cancelled; nothing was removed."; exit 0 ;;
  esac
fi

if [[ "$ALL_RESULTS" == true ]]; then
  for target in "${TARGETS[@]}"; do
    [[ "$(basename "$target")" == "$OUTPUT_DIR_NAME" ]] || {
      echo "Error: refusing unexpected directory target: $target" >&2
      exit 3
    }
    rm -rf -- "$target"
  done
else
  for target in "${TARGETS[@]}"; do
    rm -f -- "$target"
  done
fi

echo "Removed ${#TARGETS[@]} $METHOD target(s)."
echo "Run collect-stats-standard.sh again to rebuild the merged CSV without $METHOD rows."
