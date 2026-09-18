#!/usr/bin/env bash
# Back-fill or refresh the reproducibility mirror of A10K run outputs.
#
# Every results directory found in the A10K dataset tree,
#   <data>/10k-simphy/<replicate>/<method>_outputs/<tree_type>/<setting>
# is copied to
#   <outputs>/<method>_outputs/10k-simphy/<replicate>/<tree_type>/<setting>
# together with the replicate's rooting record, the dataset-level provenance
# files, and the merged scores CSV. Gene trees and species trees are never
# copied.
#
# New runs mirror themselves automatically (run-a10k.sh); this tool exists for
# results produced before the mirror existed and for verifying that the mirror
# is complete before uploading it.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/scripts/a10k-outputs-dir.sh"

DATA_DIR=""
OUTPUTS_DIR=""
METHODS_RAW=""
DRY_RUN=false
QUIET=false

print_help() {
  cat <<EOF
sync-a10k-outputs.sh

Copies every 10k-simphy/<replicate>/<method>_outputs/<tree_type>/<setting>
results directory from the A10K dataset tree into the outputs mirror, alongside
each replicate's rooting record, the dataset provenance files, and the merged
scores CSV. Existing mirror leaves are replaced so they equal the current
results exactly. Gene trees and species trees are never copied.

Required:
  --data-dir PATH      A10K dataset root containing 10k-simphy/

Options:
  --outputs-dir PATH   Mirror root to write (default: <parent>/outputs/<dataset>,
                       e.g. \$PHYLOGENY_DATA_DIR/10k-astral-dataset ->
                       \$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset)
  --method, -m METHOD  Only mirror this method (e.g. "astralx"); repeatable
  --methods LIST       Only mirror these methods, comma/space separated
                       (e.g. "astralx" or "astralx_outputs"; default: all)
  --dry-run            Show what would be mirrored without writing
  --quiet, -q          Print only the summary and problems
  --help, -h           Show this message

Examples:
  ./scripts/sync-a10k-outputs.sh --data-dir data/10k-astral-dataset --dry-run
  ./scripts/sync-a10k-outputs.sh --data-dir data/10k-astral-dataset
  ./scripts/sync-a10k-outputs.sh --data-dir data/10k-astral-dataset --method astralx
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir|--outputs-dir|--a10k-outputs-dir|--methods|--method|-m)
      if [[ $# -lt 2 ]]; then
        echo "Error: option '$1' requires a value." >&2
        exit 2
      fi
      ;;
  esac
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
    --outputs-dir|--a10k-outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --outputs-dir=*|--a10k-outputs-dir=*) OUTPUTS_DIR="${1#*=}"; shift ;;
    --methods) METHODS_RAW="$2"; shift 2 ;;
    --methods=*) METHODS_RAW="${1#*=}"; shift ;;
    --method|-m) METHODS_RAW="${METHODS_RAW:+$METHODS_RAW,}$2"; shift 2 ;;
    --method=*) METHODS_RAW="${METHODS_RAW:+$METHODS_RAW,}${1#*=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --quiet|-q) QUIET=true; shift ;;
    --help|-h) print_help; exit 0 ;;
    *)
      echo "Error: unknown option '$1'." >&2
      print_help >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Error: --data-dir is required." >&2
  print_help >&2
  exit 2
fi
if [[ "$DATA_DIR" == "~/"* ]]; then
  DATA_DIR="${HOME}/${DATA_DIR:2}"
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: data directory does not exist: $DATA_DIR" >&2
  exit 2
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd -P)" || exit 2
if [[ ! -d "${DATA_DIR}/${ASTRALX_A10K_SIMPHY_SUBDIR}" ]]; then
  echo "Error: expected ${ASTRALX_A10K_SIMPHY_SUBDIR} at ${DATA_DIR}/${ASTRALX_A10K_SIMPHY_SUBDIR}" >&2
  exit 2
fi
OUTPUTS_DIR="$(astralx_prepare_a10k_outputs_dir "$OUTPUTS_DIR" "$DATA_DIR")" || exit 2

# Normalize the method filter to "<method>_outputs" directory names.
declare -A METHOD_FILTER=()
if [[ -n "$METHODS_RAW" ]]; then
  read -r -a method_items <<< "${METHODS_RAW//,/ }"
  for method in "${method_items[@]}"; do
    [[ -z "$method" ]] && continue
    [[ "$method" == *_outputs ]] || method="${method}_outputs"
    METHOD_FILTER["$method"]=1
  done
  if [[ ${#METHOD_FILTER[@]} -eq 0 ]]; then
    echo "Error: --methods did not name any method." >&2
    exit 2
  fi
fi

echo "ASTRAL-X A10K outputs mirror sync"
echo "Data directory:    $DATA_DIR"
echo "Outputs directory: $OUTPUTS_DIR"
if [[ ${#METHOD_FILTER[@]} -gt 0 ]]; then
  echo "Methods:           ${!METHOD_FILTER[*]}"
fi
[[ "$DRY_RUN" == true ]] && echo "Dry run:           yes"
echo

mirrored=0
skipped_filter=0
failed=0

while IFS= read -r -d '' results_dir; do
  mapfile -t parts < <(astralx_a10k_results_components "$DATA_DIR" "$results_dir" 2>/dev/null)
  if [[ ${#parts[@]} -ne 5 ]]; then
    echo "  Skip (unexpected layout): $results_dir" >&2
    continue
  fi
  subdir="${parts[0]}"; replicate="${parts[1]}"; method_dir="${parts[2]}"
  tree_type="${parts[3]}"; setting="${parts[4]}"

  if [[ ${#METHOD_FILTER[@]} -gt 0 && -z "${METHOD_FILTER[$method_dir]:-}" ]]; then
    ((skipped_filter++)) || true
    continue
  fi

  label="${replicate}/${method_dir}/${tree_type}/${setting}"
  target="${OUTPUTS_DIR}/${method_dir}/${subdir}/${replicate}/${tree_type}/${setting}"
  if [[ "$DRY_RUN" == true ]]; then
    [[ "$QUIET" == true ]] || echo "  Would mirror: ${label} -> ${target}"
    forbidden="$(astralx_a10k_find_forbidden_in_mirror "$results_dir" | head -n1)"
    if [[ -n "$forbidden" ]]; then
      echo "  Error: results directory contains A10K input data and would be refused: $forbidden" >&2
      ((failed++)) || true
    else
      ((mirrored++)) || true
    fi
    continue
  fi

  if mirrored_path="$(astralx_mirror_a10k_results "$DATA_DIR" "$OUTPUTS_DIR" "$results_dir")"; then
    ((mirrored++)) || true
    [[ "$QUIET" == true ]] || echo "  Mirrored: ${label} -> ${mirrored_path}"
  else
    ((failed++)) || true
    echo "  Error: could not mirror $results_dir" >&2
  fi
done < <(astralx_list_a10k_results_dirs "$DATA_DIR")

merged_note="absent"
if [[ -f "${DATA_DIR}/${ASTRALX_A10K_MERGED_CSV_NAME}" ]]; then
  if [[ "$DRY_RUN" == true ]]; then
    merged_note="would copy"
  elif astralx_mirror_a10k_merged_csv "$DATA_DIR" "$OUTPUTS_DIR"; then
    merged_note="copied"
  else
    merged_note="FAILED"
    ((failed++)) || true
  fi
fi
if [[ "$DRY_RUN" != true ]]; then
  astralx_mirror_a10k_dataset_records "$DATA_DIR" "$OUTPUTS_DIR"
fi

echo
if [[ "$DRY_RUN" == true ]]; then
  echo "Summary (dry run): would mirror=$mirrored filtered-out=$skipped_filter problems=$failed merged-csv=$merged_note"
else
  echo "Summary: mirrored=$mirrored filtered-out=$skipped_filter failed=$failed merged-csv=$merged_note"
fi

if (( failed > 0 )); then
  exit 1
fi
