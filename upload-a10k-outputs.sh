#!/usr/bin/env bash
# Upload the reproducibility mirror of A10K run outputs to Hugging Face.
#
# Local layout (maintained by run-a10k.sh / sync-a10k-outputs.sh):
#   <outputs>/<method>_outputs/10k-simphy/<replicate>/<tree_type>/<setting>/<trees, CSVs, records>
#   <outputs>/<method>_outputs/10k-simphy/<replicate>/estimatedgenetrees.rooted.command
#   <outputs>/<dataset>.command|.source|.params, README*     (optional provenance)
#   <outputs>/a10k_astralx_scores_merged.csv                 (collector output)
#
# Remote layout (same shape, one folder upload per <method>_outputs plus the
# root-level files):
#   <remote-dir>/<method>_outputs/...
#   <remote-dir>/<root-level files>
#
# Gene trees and species trees are never part of the mirror; every directory is
# checked for them before upload.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/hf-python.sh"
source "${SCRIPT_DIR}/scripts/a10k-outputs-dir.sh"

OUTPUTS_DIR=""
DATA_DIR=""
SYNC_FIRST=false
REPO_ID="imAniksahA/blab"
REPO_TYPE="dataset"
REMOTE_DIR="ph/d/a10k/outputs"
UPLOADER="${HOME}/utils/hf-data-transfer/hf_upload.py"
PYTHON_BIN=""
DEFAULT_METHODS="astralx"   # this repo's own runs; --all-methods lifts the filter
METHODS_RAW=""
ALL_METHODS=false
DRY_RUN=false
ASSUME_YES=false

print_help() {
  cat <<EOF
upload-a10k-outputs.sh

Uploads the A10K outputs mirror to the Hugging Face repository: one folder
upload per <method>_outputs directory to <remote-dir>/<method>_outputs, plus
the root-level provenance files and merged scores CSV. The complete plan is
shown before one confirmation. Only inferred trees, CSVs, command records, and
run logs are uploaded; a directory containing gene trees or species trees is
refused.

Location (one of):
  --outputs-dir PATH       Outputs mirror to upload
  --data-dir PATH          A10K dataset root; the mirror location is derived
                           from it (data/<name> -> outputs/<name>) and --sync
                           refreshes the mirror from it first

Options:
  --sync                   Run ./sync-a10k-outputs.sh first (needs --data-dir)
  --method, -m METHOD      Only upload this method (e.g. "aster"); repeatable
  --methods LIST           Only upload these methods, comma/space separated
  --all-methods            Upload every <method>_outputs directory found
                           (by default only ${DEFAULT_METHODS}_outputs is uploaded)
  --repo-id ID             Hugging Face repository (default: ${REPO_ID})
  --repo-type TYPE         dataset, model, or space (default: ${REPO_TYPE})
  --remote-dir PATH        Destination directory inside the repository
                            (default: ${REMOTE_DIR})
  --uploader PATH          Path to hf_upload.py (default: ${UPLOADER})
  --python COMMAND         Python interpreter with huggingface_hub (default: the
                           first of python3, python, conda base python that has it)
  --dry-run                Validate and print commands without uploading
  --yes, -y                Do not ask for confirmation
  --help, -h               Show this message

Re-running is cheap: files already present on the Hub are skipped by the
uploader, so this doubles as an incremental sync of the outputs mirror.

Examples:
  ./upload-a10k-outputs.sh --data-dir data/10k-astral-dataset --dry-run
  ./upload-a10k-outputs.sh --data-dir data/10k-astral-dataset --sync
  ./upload-a10k-outputs.sh --outputs-dir outputs/10k-astral-dataset --yes
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outputs-dir|--a10k-outputs-dir|--data-dir|--methods|--method|-m|\
    --repo-id|--repo-type|--remote-dir|--uploader|--python)
      if [[ $# -lt 2 ]]; then
        echo "Error: option '$1' requires a value." >&2
        exit 2
      fi
      ;;
  esac

  case "$1" in
    --outputs-dir|--a10k-outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --outputs-dir=*|--a10k-outputs-dir=*) OUTPUTS_DIR="${1#*=}"; shift ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
    --sync) SYNC_FIRST=true; shift ;;
    --methods) METHODS_RAW="$2"; shift 2 ;;
    --methods=*) METHODS_RAW="${1#*=}"; shift ;;
    --method|-m) METHODS_RAW="${METHODS_RAW:+$METHODS_RAW,}$2"; shift 2 ;;
    --method=*) METHODS_RAW="${METHODS_RAW:+$METHODS_RAW,}${1#*=}"; shift ;;
    --all-methods) ALL_METHODS=true; shift ;;
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

if [[ "$ALL_METHODS" == true ]]; then
  if [[ -n "$METHODS_RAW" ]]; then
    echo "Error: --all-methods cannot be combined with --method/--methods." >&2
    exit 2
  fi
elif [[ -z "$METHODS_RAW" ]]; then
  METHODS_RAW="$DEFAULT_METHODS"
fi

OUTPUTS_DIR="$(expand_home "$OUTPUTS_DIR")"
DATA_DIR="$(expand_home "$DATA_DIR")"
if [[ "$SYNC_FIRST" == true && -z "$DATA_DIR" ]]; then
  echo "Error: --sync needs --data-dir so the mirror can be refreshed from the dataset tree." >&2
  exit 2
fi
if [[ -n "$DATA_DIR" ]]; then
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: data directory does not exist: $DATA_DIR" >&2
    exit 2
  fi
  DATA_DIR="$(cd "$DATA_DIR" && pwd -P)" || exit 2
  OUTPUTS_DIR="$(astralx_prepare_a10k_outputs_dir "$OUTPUTS_DIR" "$DATA_DIR")" || exit 2
else
  OUTPUTS_DIR="$(astralx_prepare_a10k_outputs_dir_standalone "$OUTPUTS_DIR")" || exit 2
fi
UPLOADER="$(realpath -m "$(expand_home "$UPLOADER")")"
REMOTE_DIR="${REMOTE_DIR%/}"

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
if [[ ! -f "$UPLOADER" ]]; then
  echo "Error: uploader was not found: $UPLOADER" >&2
  exit 2
fi
PYTHON_BIN="$(astralx_find_hf_python "$PYTHON_BIN")" || exit 2

# Folder uploads need the folder-aware hf_upload.py (its --help lists
# --include/--exclude). Older copies only accept single files.
if [[ "$UPLOADER" == *.py ]] &&
   ! "$PYTHON_BIN" "$UPLOADER" --help 2>/dev/null | grep -q -- '--include'; then
  echo "Error: $UPLOADER does not support folder uploads (no --include/--exclude in its --help)." >&2
  echo "  This is an outdated hf_upload.py. Replace it with the current folder-aware version, then rerun." >&2
  exit 2
fi

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

if [[ "$SYNC_FIRST" == true ]]; then
  echo "Refreshing the outputs mirror from the dataset tree first..."
  SYNC_CMD=("${SCRIPT_DIR}/sync-a10k-outputs.sh" --data-dir "$DATA_DIR" --outputs-dir "$OUTPUTS_DIR" --quiet)
  [[ -n "$METHODS_RAW" ]] && SYNC_CMD+=(--methods "$METHODS_RAW")
  [[ "$DRY_RUN" == true ]] && SYNC_CMD+=(--dry-run)
  if ! "${SYNC_CMD[@]}"; then
    echo "Error: mirror sync reported problems; nothing was uploaded." >&2
    exit 1
  fi
  echo
fi

human_size() {
  du -sh -- "$1" 2>/dev/null | awk '{print $1}'
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

# Collapse sorted replicate names into "R1-R4, R6" style ranges; names that
# are not R<n> are listed verbatim.
format_replicate_ranges() {
  local out="" start="" prev="" name n
  local -a others=()
  for name in "$@"; do
    if [[ ! "$name" =~ ^R([0-9]+)$ ]]; then
      others+=("$name"); continue
    fi
    n="${BASH_REMATCH[1]}"
    if [[ -z "$start" ]]; then start=$n; prev=$n; continue; fi
    if (( n == prev + 1 )); then prev=$n; continue; fi
    out+="${out:+, }R${start}"; (( start != prev )) && out+="-R${prev}"
    start=$n; prev=$n
  done
  if [[ -n "$start" ]]; then
    out+="${out:+, }R${start}"; (( start != prev )) && out+="-R${prev}"
  fi
  for name in "${others[@]}"; do out+="${out:+, }${name}"; done
  printf '%s' "$out"
}

echo "ASTRAL-X A10K outputs uploader"
echo "Outputs directory: $OUTPUTS_DIR"
echo "Repository:        $REPO_ID ($REPO_TYPE)"
echo "Remote path:       $REMOTE_DIR/"
echo "Python:            $PYTHON_BIN"
if [[ ${#METHOD_FILTER[@]} -gt 0 ]]; then
  echo "Methods:           ${!METHOD_FILTER[*]}"
else
  echo "Methods:           all"
fi
[[ "$DRY_RUN" == true ]] && echo "Dry run:           yes"
echo

declare -a UPLOAD_DIRS=()   # "<method>_outputs" entries
declare -a ROOT_FILES=()    # root-level provenance / merged CSV file names
blocked=0
total_leaves=0

while IFS= read -r -d '' method_path; do
  method_dir="${method_path##*/}"
  if [[ ${#METHOD_FILTER[@]} -gt 0 && -z "${METHOD_FILTER[$method_dir]:-}" ]]; then
    continue
  fi
  files="$(find "$method_path" -type f | wc -l | tr -d ' ')"
  forbidden="$(astralx_a10k_find_forbidden_in_mirror "$method_path" | head -n1)"
  if [[ -n "$forbidden" ]]; then
    echo "  BLOCKED ${method_dir}: contains ${forbidden##*/}"
    ((blocked++)) || true
    continue
  fi
  if (( files == 0 )); then
    echo "  BLOCKED ${method_dir}: no result files"
    ((blocked++)) || true
    continue
  fi
  UPLOAD_DIRS+=("$method_dir")

  # <method>_outputs/10k-simphy/<replicate>/<tree_type>/<setting>
  declare -A case_replicates=()
  while IFS= read -r -d '' leaf; do
    rel="${leaf#"$method_path"/}"          # 10k-simphy/R1/estimated/setting
    rel="${rel#*/}"                         # R1/estimated/setting
    replicate="${rel%%/*}"
    case_replicates["${rel#*/}"]+="${replicate} "
    ((total_leaves++)) || true
  done < <(find "$method_path" -mindepth 4 -maxdepth 4 -type d -not -name '.*' -print0 | sort -zV)
  echo "[${method_dir%_outputs}] ${files} file(s), $(human_size "$method_path"), cases (<replicates> / <tree_type>/<setting>):"
  while IFS= read -r case_key; do
    [[ -z "$case_key" ]] && continue
    read -r -a repl_names <<< "${case_replicates[$case_key]}"
    echo "  $(format_replicate_ranges "${repl_names[@]}") / ${case_key}"
  done < <(printf '%s\n' "${!case_replicates[@]}" | sort)
  unset case_replicates
done < <(find "$OUTPUTS_DIR" -mindepth 1 -maxdepth 1 -type d -name '*_outputs' -print0 | sort -z)

while IFS= read -r -d '' root_file; do
  name="${root_file##*/}"
  ROOT_FILES+=("$name")
done < <(find "$OUTPUTS_DIR" -mindepth 1 -maxdepth 1 -type f -not -name '.*' -print0 | sort -z)

echo
if (( blocked > 0 )); then
  echo "Error: $blocked <method>_outputs director(ies) cannot be uploaded safely; nothing was uploaded." >&2
  exit 1
fi
if [[ ${#UPLOAD_DIRS[@]} -eq 0 && ${#ROOT_FILES[@]} -eq 0 ]]; then
  echo "Nothing to upload under $OUTPUTS_DIR."
  if [[ "$ALL_METHODS" == false ]]; then
    echo "Only ${METHODS_RAW} is selected; pass --all-methods (or --method NAME) to widen the search."
  fi
  echo "Run ./sync-a10k-outputs.sh --data-dir ... (or this tool with --sync) to populate the mirror."
  exit 0
fi

echo "Plan: upload ${#UPLOAD_DIRS[@]} method director(ies) as folders (${total_leaves} replicate/tree-type/setting result set(s)) and ${#ROOT_FILES[@]} root-level file(s)."
echo "Destinations:"
for method_dir in "${UPLOAD_DIRS[@]}"; do
  echo "  ${method_dir}/ -> ${REPO_ID}/${REMOTE_DIR}/${method_dir}/"
done
for name in "${ROOT_FILES[@]}"; do
  echo "  ${name} -> ${REPO_ID}/${REMOTE_DIR}/${name}"
done

build_folder_upload_command() {
  local method_dir="$1"
  UPLOAD_CMD=("$PYTHON_BIN" "$UPLOADER"
    --repo-id "$REPO_ID"
    --repo-type "$REPO_TYPE"
    --local-path "${OUTPUTS_DIR}/${method_dir}"
    --path-in-repo "${REMOTE_DIR}/${method_dir}"
    --commit-message "A10K outputs: ${method_dir}")
}
build_file_upload_command() {
  local name="$1"
  UPLOAD_CMD=("$PYTHON_BIN" "$UPLOADER"
    --repo-id "$REPO_ID"
    --repo-type "$REPO_TYPE"
    --local-path "${OUTPUTS_DIR}/${name}"
    --path-in-repo "${REMOTE_DIR}/${name}"
    --commit-message "A10K outputs: ${name}")
}

if [[ "$DRY_RUN" == true ]]; then
  echo
  echo "Upload commands:"
  for method_dir in "${UPLOAD_DIRS[@]}"; do
    build_folder_upload_command "$method_dir"
    print_command "${UPLOAD_CMD[@]}"
  done
  for name in "${ROOT_FILES[@]}"; do
    build_file_upload_command "$name"
    print_command "${UPLOAD_CMD[@]}"
  done
  echo "Dry run complete; nothing was uploaded."
  exit 0
fi

if [[ "$ASSUME_YES" == false ]]; then
  echo
  if [[ -t 0 ]]; then
    read -r -p "Proceed with the upload(s) listed above? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
      echo "Cancelled; nothing was uploaded."
      exit 0
    fi
  else
    echo "Non-interactive session: proceeding without confirmation (pass --yes to silence this note)."
  fi
fi

echo
echo "Revalidating every directory immediately before upload..."
for method_dir in "${UPLOAD_DIRS[@]}"; do
  forbidden="$(astralx_a10k_find_forbidden_in_mirror "${OUTPUTS_DIR}/${method_dir}" | head -n1)"
  if [[ -n "$forbidden" ]]; then
    echo "Error: A10K input data appeared in the mirror: $forbidden; nothing was uploaded." >&2
    exit 1
  fi
  echo "  Ready: ${method_dir} ($(human_size "${OUTPUTS_DIR}/${method_dir}"))"
done

uploaded=0
failed=0
for method_dir in "${UPLOAD_DIRS[@]}"; do
  build_folder_upload_command "$method_dir"
  echo
  echo "Uploading ${method_dir}/ ..."
  print_command "${UPLOAD_CMD[@]}"
  if "${UPLOAD_CMD[@]}"; then
    ((uploaded++)) || true
  else
    ((failed++)) || true
    echo "Error: upload failed for ${method_dir}" >&2
  fi
done
for name in "${ROOT_FILES[@]}"; do
  build_file_upload_command "$name"
  echo
  echo "Uploading ${name} ..."
  print_command "${UPLOAD_CMD[@]}"
  if "${UPLOAD_CMD[@]}"; then
    ((uploaded++)) || true
  else
    ((failed++)) || true
    echo "Error: upload failed for ${name}" >&2
  fi
done

echo
echo "Summary: uploaded=$uploaded failed=$failed"
(( failed == 0 ))
