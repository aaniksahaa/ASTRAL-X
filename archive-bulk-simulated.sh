#!/usr/bin/env bash
# Discover complete SimPhy datasets and package them as validated ZIP archives.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${SCRIPT_DIR}/simphy/data"
MIN_TAXA=1
MIN_GENE_TREES=1
FORCE=false
DRY_RUN=false
ASSUME_YES=false

print_help() {
  cat <<EOF
archive-bulk-simulated.sh

Discovers complete SimPhy dataset directories directly under simphy/data and
creates one validated ZIP per dataset. Archives are built in a temporary
directory and moved into place only after they pass an integrity check.

Options:
  --data-dir PATH          SimPhy data directory (default: ${DATA_DIR})
  --min-taxa N             Minimum taxon count (default: ${MIN_TAXA})
  --min-gene-trees N       Minimum gene-tree count (default: ${MIN_GENE_TREES})
  --force                  Rebuild even if an archive looks current and valid
  --dry-run                Show what would be archived without writing files
  --yes, -y                Do not ask for confirmation
  --help, -h               Show this message

Only directories with the complete canonical name below are considered:
  t_<taxa>_g_<genes>_sb_<rate>_spmin_<min>_spmax_<max>

Directories ending in "_incomplete" are intentionally excluded.

Examples:
  ./archive-bulk-simulated.sh --dry-run
  ./archive-bulk-simulated.sh --min-taxa 1000 --min-gene-trees 1000
  ./archive-bulk-simulated.sh --force --yes
EOF
}

expand_home() {
  local path="$1"
  if [[ "$path" == "~/"* ]]; then
    printf '%s/%s\n' "$HOME" "${path:2}"
  else
    printf '%s\n' "$path"
  fi
}

require_positive_integer() {
  local option="$1" value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: $option requires a positive integer; got '$value'." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir|--min-taxa|--min-gene-trees)
      if [[ $# -lt 2 ]]; then
        echo "Error: option '$1' requires a value." >&2
        exit 2
      fi
      ;;
  esac

  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
    --min-taxa) MIN_TAXA="$2"; shift 2 ;;
    --min-taxa=*) MIN_TAXA="${1#*=}"; shift ;;
    --min-gene-trees) MIN_GENE_TREES="$2"; shift 2 ;;
    --min-gene-trees=*) MIN_GENE_TREES="${1#*=}"; shift ;;
    --force) FORCE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --yes|-y) ASSUME_YES=true; shift ;;
    --help|-h) print_help; exit 0 ;;
    *)
      echo "Error: unknown option '$1'." >&2
      print_help >&2
      exit 2
      ;;
  esac
done

require_positive_integer "--min-taxa" "$MIN_TAXA"
require_positive_integer "--min-gene-trees" "$MIN_GENE_TREES"

DATA_DIR="$(expand_home "$DATA_DIR")"
DATA_DIR="$(realpath -m "$DATA_DIR")"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: data directory does not exist: $DATA_DIR" >&2
  exit 2
fi
if ! command -v zip >/dev/null 2>&1; then
  echo "Error: zip is required but was not found." >&2
  exit 2
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip is required but was not found." >&2
  exit 2
fi

dataset_name_is_valid() {
  local name="$1"
  [[ "$name" =~ ^t_([1-9][0-9]*)_g_([1-9][0-9]*)_sb_([0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?)_spmin_([1-9][0-9]*)_spmax_([1-9][0-9]*)$ ]]
}

archive_is_valid() {
  local archive="$1" dataset_name="$2"
  local entry normalized found=false

  if ! unzip -tq "$archive" >/dev/null 2>&1; then
    return 1
  fi

  while IFS= read -r entry; do
    normalized="${entry#./}"
    [[ -z "$normalized" ]] && continue
    if [[ "$normalized" == /* || "$normalized" =~ (^|/)[.][.](/|$) ]]; then
      return 1
    fi
    case "$normalized" in
      "$dataset_name"|"$dataset_name/"*) found=true ;;
      *) return 1 ;;
    esac
  done < <(unzip -Z1 "$archive")

  [[ "$found" == true ]]
}

human_size() {
  du -sh -- "$1" 2>/dev/null | awk '{print $1}'
}

declare -a DATASETS=()
declare -a ACTIONS=()
declare -a SIZES=()

while IFS= read -r -d '' dataset_path; do
  name="${dataset_path##*/}"
  if ! dataset_name_is_valid "$name"; then
    continue
  fi

  taxa="${BASH_REMATCH[1]}"
  gene_trees="${BASH_REMATCH[2]}"
  if (( taxa < MIN_TAXA || gene_trees < MIN_GENE_TREES )); then
    continue
  fi
  if [[ -z "$(find "$dataset_path" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Warning: skipping empty dataset directory: $dataset_path" >&2
    continue
  fi

  archive_path="${DATA_DIR}/${name}.zip"
  action="create"
  if [[ -e "$archive_path" && ! -f "$archive_path" ]]; then
    echo "Warning: skipping because archive path is not a regular file: $archive_path" >&2
    continue
  elif [[ -f "$archive_path" ]]; then
    if [[ "$FORCE" == true ]]; then
      action="rebuild (--force)"
    elif ! archive_is_valid "$archive_path" "$name"; then
      action="rebuild (invalid ZIP)"
    elif [[ -n "$(find "$dataset_path" -type f -newer "$archive_path" -print -quit 2>/dev/null)" ]]; then
      action="rebuild (source is newer)"
    else
      action="skip (current)"
    fi
  fi

  DATASETS+=("$name")
  ACTIONS+=("$action")
  SIZES+=("$(human_size "$dataset_path")")
done < <(find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type d -name 't_*' -print0 | sort -zV)

echo "ASTRAL-X simulated dataset archiver"
echo "Data directory: $DATA_DIR"
echo "Selection:      taxa >= $MIN_TAXA, gene trees >= $MIN_GENE_TREES"
[[ "$DRY_RUN" == true ]] && echo "Dry run:        yes"
echo

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No complete dataset directories matched the selection."
  exit 0
fi

printf '%-7s %-7s %-11s %s\n' "TAXA" "GENES" "SIZE" "ACTION / DATASET"
printf '%-7s %-7s %-11s %s\n' "-------" "-------" "-----------" "----------------"

pending=0
for i in "${!DATASETS[@]}"; do
  name="${DATASETS[$i]}"
  dataset_name_is_valid "$name"
  taxa="${BASH_REMATCH[1]}"
  gene_trees="${BASH_REMATCH[2]}"
  printf '%-7s %-7s %-11s %s: %s\n' "$taxa" "$gene_trees" "${SIZES[$i]}" "${ACTIONS[$i]}" "$name"
  if [[ "${ACTIONS[$i]}" != "skip (current)" ]]; then
    ((pending++)) || true
  fi
done

echo
echo "Output directory: $DATA_DIR"

if (( pending == 0 )); then
  echo "All selected archives are already current and valid."
  exit 0
fi

if [[ "$DRY_RUN" == true ]]; then
  echo "Would create or rebuild $pending archive(s)."
  exit 0
fi

if [[ "$ASSUME_YES" == false ]]; then
  echo
  read -r -p "Create or rebuild $pending archive(s)? [y/N]: " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled; no archives were changed."
    exit 0
  fi
fi

CURRENT_TEMP_DIR=""
cleanup() {
  if [[ -n "$CURRENT_TEMP_DIR" && -d "$CURRENT_TEMP_DIR" ]]; then
    rm -rf -- "$CURRENT_TEMP_DIR"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

created=0
skipped=0
failed=0

for i in "${!DATASETS[@]}"; do
  name="${DATASETS[$i]}"
  action="${ACTIONS[$i]}"
  archive_path="${DATA_DIR}/${name}.zip"

  if [[ "$action" == "skip (current)" ]]; then
    ((skipped++)) || true
    continue
  fi

  echo
  echo "[$((created + failed + 1))/$pending] $action: $name"
  CURRENT_TEMP_DIR="$(mktemp -d "${DATA_DIR}/.${name}.archive.XXXXXX")" || {
    echo "  Error: could not create a temporary directory." >&2
    ((failed++)) || true
    continue
  }
  temp_archive="${CURRENT_TEMP_DIR}/${name}.zip"

  echo "  Compressing..."
  if ! (cd "$DATA_DIR" && zip -rq "$temp_archive" "$name"); then
    echo "  Error: zip failed; existing archive was left untouched." >&2
    rm -rf -- "$CURRENT_TEMP_DIR"
    CURRENT_TEMP_DIR=""
    ((failed++)) || true
    continue
  fi

  echo "  Testing archive integrity..."
  if ! archive_is_valid "$temp_archive" "$name"; then
    echo "  Error: archive test failed; existing archive was left untouched." >&2
    rm -rf -- "$CURRENT_TEMP_DIR"
    CURRENT_TEMP_DIR=""
    ((failed++)) || true
    continue
  fi

  if ! mv -f -- "$temp_archive" "$archive_path"; then
    echo "  Error: could not install archive; existing archive was left untouched." >&2
    rm -rf -- "$CURRENT_TEMP_DIR"
    CURRENT_TEMP_DIR=""
    ((failed++)) || true
    continue
  fi
  rmdir "$CURRENT_TEMP_DIR"
  CURRENT_TEMP_DIR=""
  ((created++)) || true
  echo "  Ready: $archive_path ($(human_size "$archive_path"))"
done

echo
echo "Summary: created=$created skipped=$skipped failed=$failed"
if (( failed > 0 )); then
  exit 1
fi
