#!/usr/bin/env bash
# Discover validated SimPhy ZIP archives and upload them to Hugging Face.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${SCRIPT_DIR}/simphy/data"
REPO_ID="imAniksahA/blab"
REPO_TYPE="dataset"
REMOTE_DIR="ph/d/simulated/astralx-datasets/raw"
UPLOADER="${HOME}/utils/hf-data-transfer/hf_upload.py"
PYTHON_BIN="python3"
MIN_TAXA=1000
MIN_GENE_TREES=1000
DRY_RUN=false
ASSUME_YES=false

print_help() {
  cat <<EOF
upload-bulk-simulated.sh

Discovers canonical SimPhy ZIP archives directly under simphy/data, validates
them, shows the exact upload plan, and asks once before uploading sequentially.

Options:
  --data-dir PATH          Directory containing dataset ZIPs
                            (default: ${DATA_DIR})
  --min-taxa N             Minimum taxon count (default: ${MIN_TAXA})
  --min-gene-trees N       Minimum gene-tree count (default: ${MIN_GENE_TREES})
  --all                    Select all positive taxa/gene-tree counts
  --repo-id ID             Hugging Face repository (default: ${REPO_ID})
  --repo-type TYPE         dataset, model, or space (default: ${REPO_TYPE})
  --remote-dir PATH        Destination directory inside the repository
                            (default: ${REMOTE_DIR})
  --uploader PATH          Path to hf_upload.py (default: ${UPLOADER})
  --python COMMAND         Python interpreter (default: ${PYTHON_BIN})
  --dry-run                Validate and print commands without uploading
  --yes, -y                Do not ask for confirmation
  --help, -h               Show this message

ZIPs for directories ending in "_incomplete" are intentionally excluded.
If a matching source directory has files newer than its ZIP, the upload is
stopped so the stale archive can be rebuilt first.

Examples:
  ./upload-bulk-simulated.sh --dry-run
  ./upload-bulk-simulated.sh
  ./upload-bulk-simulated.sh --all --remote-dir ph/d/simulated/astralx-datasets/raw
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
    --data-dir|--min-taxa|--min-gene-trees|--repo-id|--repo-type|\
    --remote-dir|--uploader|--python)
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
    --all) MIN_TAXA=1; MIN_GENE_TREES=1; shift ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --repo-id=*) REPO_ID="${1#*=}"; shift ;;
    --repo-type) REPO_TYPE="$2"; shift 2 ;;
    --repo-type=*) REPO_TYPE="${1#*=}"; shift ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --remote-dir=*) REMOTE_DIR="${1#*=}"; shift ;;
    --uploader) UPLOADER="$2"; shift 2 ;;
    --uploader=*) UPLOADER="${1#*=}"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --python=*) PYTHON_BIN="${1#*=}"; shift ;;
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

DATA_DIR="$(realpath -m "$(expand_home "$DATA_DIR")")"
UPLOADER="$(realpath -m "$(expand_home "$UPLOADER")")"
REMOTE_DIR="${REMOTE_DIR%/}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: data directory does not exist: $DATA_DIR" >&2
  exit 2
fi
if [[ ! "$REPO_ID" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
  echo "Error: invalid --repo-id '$REPO_ID'; expected owner/repository." >&2
  exit 2
fi
case "$REPO_TYPE" in
  dataset|model|space) ;;
  *) echo "Error: --repo-type must be dataset, model, or space." >&2; exit 2 ;;
esac
if [[ -z "$REMOTE_DIR" || "$REMOTE_DIR" == /* || "$REMOTE_DIR" =~ (^|/)[.][.](/|$) ]]; then
  echo "Error: unsafe --remote-dir '$REMOTE_DIR'." >&2
  exit 2
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip is required but was not found." >&2
  exit 2
fi
if [[ ! -f "$UPLOADER" ]]; then
  echo "Error: uploader was not found: $UPLOADER" >&2
  exit 2
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 2
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python command was not found: $PYTHON_BIN" >&2
  exit 2
fi

dataset_name_is_valid() {
  local name="$1"
  [[ "$name" =~ ^t_([1-9][0-9]*)_g_([1-9][0-9]*)_sb_([0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?)_spmin_([1-9][0-9]*)_spmax_([1-9][0-9]*)$ ]]
}

archive_has_expected_root() {
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
  du -h -- "$1" 2>/dev/null | awk '{print $1}'
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

declare -a ARCHIVES=()
declare -a SIZES=()
invalid=0
stale=0

echo "ASTRAL-X simulated dataset uploader"
echo "Data directory: $DATA_DIR"
echo "Repository:     $REPO_ID ($REPO_TYPE)"
echo "Remote path:    $REMOTE_DIR/"
echo "Selection:      taxa >= $MIN_TAXA, gene trees >= $MIN_GENE_TREES"
[[ "$DRY_RUN" == true ]] && echo "Dry run:        yes"
echo

printf '%-7s %-7s %-9s %-10s %s\n' "TAXA" "GENES" "ZIP SIZE" "STATUS" "ARCHIVE"
printf '%-7s %-7s %-9s %-10s %s\n' "-------" "-------" "---------" "----------" "-------"

while IFS= read -r -d '' archive_path; do
  archive_name="${archive_path##*/}"
  dataset_name="${archive_name%.zip}"
  if ! dataset_name_is_valid "$dataset_name"; then
    continue
  fi

  taxa="${BASH_REMATCH[1]}"
  gene_trees="${BASH_REMATCH[2]}"
  if (( taxa < MIN_TAXA || gene_trees < MIN_GENE_TREES )); then
    continue
  fi

  status="ready"
  if ! archive_has_expected_root "$archive_path" "$dataset_name"; then
    status="INVALID"
    ((invalid++)) || true
  elif [[ -d "${DATA_DIR}/${dataset_name}" ]] &&
       [[ -n "$(find "${DATA_DIR}/${dataset_name}" -type f -newer "$archive_path" -print -quit 2>/dev/null)" ]]; then
    status="STALE"
    ((stale++)) || true
  fi

  printf '%-7s %-7s %-9s %-10s %s\n' "$taxa" "$gene_trees" "$(human_size "$archive_path")" "$status" "$archive_name"
  ARCHIVES+=("$archive_path")
  SIZES+=("$(human_size "$archive_path")")
done < <(find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type f -name 't_*_g_*.zip' -print0 | sort -zV)

echo
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
  echo "No canonical ZIP archives matched the selection."
  echo "Create them first with: ${SCRIPT_DIR}/archive-bulk-simulated.sh --min-taxa $MIN_TAXA --min-gene-trees $MIN_GENE_TREES"
  exit 0
fi

if (( invalid > 0 || stale > 0 )); then
  echo "Error: refusing to upload: invalid=$invalid stale=$stale." >&2
  echo "Rebuild the affected archives with: ${SCRIPT_DIR}/archive-bulk-simulated.sh --min-taxa $MIN_TAXA --min-gene-trees $MIN_GENE_TREES" >&2
  exit 1
fi

echo "Upload plan (${#ARCHIVES[@]} archive(s)):"
for i in "${!ARCHIVES[@]}"; do
  archive_name="${ARCHIVES[$i]##*/}"
  echo "  ${archive_name} (${SIZES[$i]})"
  echo "    -> ${REPO_ID}/${REMOTE_DIR}/${archive_name}"
done

if [[ "$DRY_RUN" == true ]]; then
  echo
  echo "Commands:"
  for archive_path in "${ARCHIVES[@]}"; do
    archive_name="${archive_path##*/}"
    print_command "$PYTHON_BIN" "$UPLOADER" \
      --repo-id "$REPO_ID" \
      --repo-type "$REPO_TYPE" \
      --local-path "$archive_path" \
      --path-in-repo "${REMOTE_DIR}/${archive_name}"
  done
  echo "Dry run complete; nothing was uploaded."
  exit 0
fi

if [[ "$ASSUME_YES" == false ]]; then
  echo
  read -r -p "Upload ${#ARCHIVES[@]} archive(s) to $REPO_ID? [y/N]: " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled; nothing was uploaded."
    exit 0
  fi
fi

succeeded=0
failed=0
for i in "${!ARCHIVES[@]}"; do
  archive_path="${ARCHIVES[$i]}"
  archive_name="${archive_path##*/}"
  echo
  echo "[$((i + 1))/${#ARCHIVES[@]}] Uploading: $archive_name"
  if "$PYTHON_BIN" "$UPLOADER" \
      --repo-id "$REPO_ID" \
      --repo-type "$REPO_TYPE" \
      --local-path "$archive_path" \
      --path-in-repo "${REMOTE_DIR}/${archive_name}"; then
    ((succeeded++)) || true
    echo "  Done: ${REMOTE_DIR}/${archive_name}"
  else
    ((failed++)) || true
    echo "  Error: upload failed; continuing with remaining archives." >&2
  fi
done

echo
echo "Summary: selected=${#ARCHIVES[@]} uploaded=$succeeded failed=$failed"
if (( failed > 0 )); then
  exit 1
fi
