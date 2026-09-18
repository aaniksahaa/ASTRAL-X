#!/usr/bin/env bash
# run-a10k.sh
# ASTRAL-X runner for the 10k-simphy dataset layout.

set -euo pipefail

NTFY_CHANNEL_NAME="${NTFY_CHANNEL_NAME:-anik-phylo}"

# Exact invocation of this script, appended to every run's command record.
SCRIPT_ARGV=("$0" "$@")

TREE_TYPES_RAW="estimated"
DATA_DIR=""
OUTPUTS_DIR=""
OUTPUTS_MIRROR=true
REPLICATES_SPEC=""
START_REP=""
END_REP=""
FRESH=false
ASTRALX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASTRALX_OPTS="--search-space S2 -vv"
ASTRALX_OPTS_LIST_RAW=""
TIME_MONITOR=true
GPU_MONITOR=true
NO_NOTIFY=false

source "${ASTRALX_ROOT}/scripts/experiment-setting-name.sh"
source "${ASTRALX_ROOT}/scripts/a10k-outputs-dir.sh"

csv_get_field() {
  local file="$1"
  shift
  local header data
  header="$(head -n1 "$file" 2>/dev/null || true)"
  data="$(sed -n '2p' "$file" 2>/dev/null || true)"
  if [[ -z "$header" || -z "$data" ]]; then
    echo ""
    return 0
  fi
  IFS=',' read -r -a headers <<< "$header"
  IFS=',' read -r -a values <<< "$data"
  for key in "$@"; do
    for i in "${!headers[@]}"; do
      if [[ "${headers[$i]}" == "$key" ]]; then
        echo "${values[$i]:-}"
        return 0
      fi
    done
  done
  echo ""
}

# Example single setting:
# ASTRALX_OPTS="--search-space S2 --intersection-method I2"
#
# Example sweep over search-space presets:
# ASTRALX_OPTS_LIST_RAW="--search-space S1;--search-space S2;--search-space S3"
#
# This becomes search-space_S2__intersection-method_I2. Verbosity flags such as
# -v/-vv are ignored when constructing the setting name.

print_help() {
  cat <<EOF
run-a10k.sh

Required:
  --data-dir           Path to A10K dataset root containing 10k-simphy/

Optional:
  --tree-type          estimated | true, or a semicolon-separated list
                       such as "true;estimated" (default: estimated)
  --replicates         Replicates to run, e.g. "1-20" or "R1,R2"
  --start-rep, -sr     Start replicate number
  --end-rep, -er       End replicate number
  --astralx-root       Path to ASTRAL-X root
  --opts, --alg-opts   Extra options for the selected algorithm
  --opts-list, --alg-opts-list
                       Semicolon-separated list of option strings to loop over
  --fresh              Force rerun even if stat-astralx.csv exists
  --outputs-dir        Reproducibility mirror root for the small run outputs
                       (default: <parent>/outputs/<dataset>, e.g.
                       \$PHYLOGENY_DATA_DIR/10k-astral-dataset mirrors into
                       \$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset)
  --no-outputs-mirror  Do not copy results into the outputs mirror
  --no-time-monitor    Disable time monitoring
  --no-gpu-monitor     Disable GPU monitoring
  --no-notify, -nn     Disable ntfy notifications

Examples:
  ./scripts/run-a10k.sh --data-dir /path/to/10k-astral-dataset --tree-type estimated --opts "--search-space S1 --intersection-method I2 -vv"
  ./scripts/run-a10k.sh --data-dir /path/to/10k-astral-dataset --tree-type "true;estimated" --opts "--search-space S1 --intersection-method I2 -vv"
  ./scripts/run-a10k.sh --data-dir /path/to/10k-astral-dataset --tree-type estimated --opts "--search-space S2 --intersection-method I2 -vv"
  ./scripts/run-a10k.sh --data-dir /path/to/10k-astral-dataset --tree-type estimated --opts-list "--search-space S1 -vv;--search-space S2 -vv;--search-space S3 -vv"
  The first example setting is search-space_S1__intersection-method_I2.
  Verbosity is ignored; other meaningful options are appended to the name.

Results are written to
  <data-dir>/10k-simphy/<replicate>/astralx_outputs/<tree_type>/<setting>/
exactly as before and are also mirrored (tree, CSVs, command record, run log;
never gene trees or species trees) to
  <outputs-dir>/astralx_outputs/10k-simphy/<replicate>/<tree_type>/<setting>/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tree-type) TREE_TYPES_RAW="$2"; shift 2 ;;
    --tree-type=*) TREE_TYPES_RAW="${1#*=}"; shift ;;
    --replicates) REPLICATES_SPEC="$2"; shift 2 ;;
    --start-rep|-sr) START_REP="$2"; shift 2 ;;
    --end-rep|-er) END_REP="$2"; shift 2 ;;
    --astralx-root|--stelar-root) ASTRALX_ROOT="$2"; shift 2 ;;
    --opts|--alg-opts|--astralx-opts|--stelar-opts) ASTRALX_OPTS="$2"; shift 2 ;;
    --opts-list|--alg-opts-list|--astralx-opts-list|--stelar-opts-list) ASTRALX_OPTS_LIST_RAW="$2"; shift 2 ;;
    --fresh) FRESH=true; shift ;;
    --outputs-dir|--a10k-outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --outputs-dir=*|--a10k-outputs-dir=*) OUTPUTS_DIR="${1#*=}"; shift ;;
    --no-outputs-mirror) OUTPUTS_MIRROR=false; shift ;;
    --no-time-monitor) TIME_MONITOR=false; shift ;;
    --no-gpu-monitor) GPU_MONITOR=false; shift ;;
    --no-notify|-nn) NO_NOTIFY=true; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Error: --data-dir is required."
  exit 2
fi

ASTRALX_ROOT="$(realpath "$ASTRALX_ROOT")"
PYTHON_BIN="${ASTRALX_PYTHON:-${ASTRALX_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python3"
DATA_DIR="$(realpath "$DATA_DIR")"
SIMPHY_DIR="${DATA_DIR%/}/10k-simphy"
if [[ ! -d "$SIMPHY_DIR" ]]; then
  echo "Error: expected 10k-simphy at $SIMPHY_DIR"
  exit 3
fi
if [[ "$OUTPUTS_MIRROR" == true ]]; then
  OUTPUTS_DIR="$(astralx_prepare_a10k_outputs_dir "$OUTPUTS_DIR" "$DATA_DIR")" || exit 2
else
  OUTPUTS_DIR="(disabled)"
fi
echo "[DEBUG] data dir: ${DATA_DIR} | outputs mirror: ${OUTPUTS_DIR}"

# Copy one results directory into the reproducibility mirror. A mirror problem
# is reported loudly but never changes the run's own exit status.
mirror_results_dir() {
  local results_dir="$1" mirrored
  [[ "$OUTPUTS_MIRROR" == true ]] || return 0
  [[ -d "$results_dir" ]] || return 0
  if mirrored="$(astralx_mirror_a10k_results "$DATA_DIR" "$OUTPUTS_DIR" "$results_dir")"; then
    echo "Mirrored outputs to: $mirrored"
  else
    echo "WARNING: outputs mirror was not updated for $results_dir" >&2
  fi
}

TREE_TYPES=()
IFS=';' read -r -a raw_tree_types <<< "$TREE_TYPES_RAW"
for tree_type in "${raw_tree_types[@]}"; do
  tree_type="${tree_type//[[:space:]]/}"
  tree_type="${tree_type,,}"
  [[ -n "$tree_type" ]] || continue
  case "$tree_type" in
    true|estimated) ;;
    *)
      echo "Error: invalid --tree-type value '$tree_type' (expected true, estimated, or a semicolon-separated list)."
      exit 2
      ;;
  esac

  duplicate=false
  for existing_tree_type in "${TREE_TYPES[@]}"; do
    if [[ "$existing_tree_type" == "$tree_type" ]]; then
      duplicate=true
      break
    fi
  done
  [[ "$duplicate" == false ]] && TREE_TYPES+=("$tree_type")
done
if [[ ${#TREE_TYPES[@]} -eq 0 ]]; then
  echo "Error: --tree-type must contain at least one of: true, estimated."
  exit 2
fi

ASTRALX_OPTS_LIST=()
if [[ -n "$ASTRALX_OPTS_LIST_RAW" ]]; then
  IFS=';' read -r -a raw_opts_list <<< "$ASTRALX_OPTS_LIST_RAW"
  for opts in "${raw_opts_list[@]}"; do
    opts="$(echo "$opts" | sed 's/^ *//;s/ *$//')"
    [[ -n "$opts" ]] && ASTRALX_OPTS_LIST+=("$opts")
  done
fi
if [[ ${#ASTRALX_OPTS_LIST[@]} -eq 0 ]]; then
  ASTRALX_OPTS_LIST+=("${ASTRALX_OPTS}")
fi
echo "[DEBUG] opts list (${#ASTRALX_OPTS_LIST[@]} items): ${ASTRALX_OPTS_LIST[*]}"
echo "[DEBUG] tree types (${#TREE_TYPES[@]} items): ${TREE_TYPES[*]}"
echo "[DEBUG] replicates spec: '${REPLICATES_SPEC}' | fresh: ${FRESH}"

REPL_LIST=()
if [[ -n "$START_REP" || -n "$END_REP" ]]; then
  for i in $(seq "$START_REP" "$END_REP"); do REPL_LIST+=("R${i}"); done
elif [[ -n "$REPLICATES_SPEC" ]]; then
  if [[ "$REPLICATES_SPEC" =~ ^[0-9]+-[0-9]+$ ]]; then
    start="${REPLICATES_SPEC%-*}"
    end="${REPLICATES_SPEC#*-}"
    for i in $(seq "$start" "$end"); do REPL_LIST+=("R${i}"); done
  else
    IFS=',' read -r -a parts <<< "$REPLICATES_SPEC"
    for p in "${parts[@]}"; do
      p="${p// /}"
      [[ "$p" =~ ^R ]] || p="R${p}"
      REPL_LIST+=("$p")
    done
  fi
else
  while IFS= read -r -d '' d; do
    REPL_LIST+=("$(basename "$d")")
  done < <(find "$SIMPHY_DIR" -maxdepth 1 -type d -name 'R*' -print0 | sort -z -V)
fi

echo "[DEBUG] replicate list (${#REPL_LIST[@]} items): ${REPL_LIST[*]}"

for TREE_TYPE in "${TREE_TYPES[@]}"; do
  echo "==> Processing A10K tree type: ${TREE_TYPE}"
  for REPL in "${REPL_LIST[@]}"; do
  REPL_DIR="${SIMPHY_DIR%/}/${REPL}"
  if [[ ! -d "$REPL_DIR" ]]; then
    echo "[DEBUG] SKIP ${REPL}: directory not found: ${REPL_DIR}"
    continue
  fi

  if [[ "$TREE_TYPE" == "estimated" ]]; then
    GT_DIR="${REPL_DIR}/estimatedgenetrees"
    GT_FILE="${GT_DIR}/estimatedgenetrees.tre"
    ROOTED_GT="${GT_DIR}/estimatedgenetrees.rooted.tre"
    if [[ ! -f "$ROOTED_GT" ]]; then
      if [[ ! -x "${ASTRALX_ROOT%/}/scripts/process_unrooted.sh" ]]; then
        echo "Error: process_unrooted.sh not found or not executable at ${ASTRALX_ROOT%/}/scripts/process_unrooted.sh"
        exit 7
      fi
      echo "Rooting estimated gene trees for ${REPL} with outgroup 0..."
      ROOT_CMD=("${ASTRALX_ROOT%/}/scripts/process_unrooted.sh" -i "$GT_FILE" -o "$ROOTED_GT" -og "0")
      "${ROOT_CMD[@]}"
      # Record how the rooted input was derived; mirrored beside the results.
      {
        echo "# A10K estimated gene trees rooting (written by run-a10k.sh)"
        echo "# date:    $(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "# input:   $GT_FILE"
        echo "# output:  $ROOTED_GT"
        printf '%q' "${ROOT_CMD[0]}"
        printf ' %q' "${ROOT_CMD[@]:1}"
        printf '\n'
      } > "${GT_DIR}/${ASTRALX_A10K_ROOTING_RECORD_NAME}" 2>/dev/null ||
        echo "Warning: could not write rooting record to ${GT_DIR}/${ASTRALX_A10K_ROOTING_RECORD_NAME}" >&2
    fi
    GT_FILE="$ROOTED_GT"
  else
    GT_FILE="${REPL_DIR}/truegenetrees"
  fi
  TRUE_TREE="${REPL_DIR}/s_tree.trees"
  if [[ ! -f "$GT_FILE" || ! -f "$TRUE_TREE" ]]; then
    echo "[DEBUG] SKIP ${REPL}: missing files (gt=${GT_FILE} exists=$([ -f "$GT_FILE" ] && echo yes || echo no), true_tree=${TRUE_TREE} exists=$([ -f "$TRUE_TREE" ] && echo yes || echo no))"
    continue
  fi

  for ASTRALX_OPTS_ITEM in "${ASTRALX_OPTS_LIST[@]}"; do
    SETTING_NAME="$(build_setting_name_from_opts "$ASTRALX_OPTS_ITEM")"
    OUT_DIR="${REPL_DIR}/astralx_outputs/${TREE_TYPE}/${SETTING_NAME}"
    OUT_FILE="${OUT_DIR}/out-astralx.tre"
    STAT_FILE="${OUT_DIR}/stat-astralx.csv"
    RUN_LOG="${OUT_DIR}/.astralx_run.log"
    COMMAND_FILE="${OUT_FILE%.*}.command"

    if [[ "$FRESH" == false && -f "$STAT_FILE" ]]; then
      echo "SKIPPING: ${STAT_FILE} exists."
      # Keep the reproducibility mirror complete even for runs finished earlier.
      mirror_results_dir "$OUT_DIR"
      continue
    elif [[ "$FRESH" == true && -f "$STAT_FILE" ]]; then
      echo "[DEBUG] --fresh set, overwriting existing: ${STAT_FILE}"
    fi

    mkdir -p "$OUT_DIR"
    rm -f "$RUN_LOG"
    CMD=("${ASTRALX_ROOT}/scripts/run-astralx-with-monitor.sh" -i "$GT_FILE" -o "$OUT_FILE" --astralx-root "$ASTRALX_ROOT")
    if [[ "$TIME_MONITOR" == false ]]; then CMD+=(--no-time-monitor); fi
    if [[ "$GPU_MONITOR" == false ]]; then CMD+=(--no-gpu-monitor); fi
    if [[ "$NO_NOTIFY" == true ]]; then CMD+=(--no-notify); fi
    if [[ -n "$ASTRALX_OPTS_ITEM" ]]; then
      CMD+=(--opts "$ASTRALX_OPTS_ITEM")
    fi

    echo "==> Running astralx on ${REPL} (${TREE_TYPE}, ${SETTING_NAME})"
    echo "Command: ${CMD[*]}"
    set +e
    "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
    RUN_EXIT=${PIPESTATUS[0]}
    set -e

    # Complete the command record (out-astralx.command, written by the wrapper
    # with the exact ./astralx invocation) with the A10K context of this run.
    append_command_context() {
      local rf_rate="$1"
      {
        echo "# --- A10K run context (run-a10k.sh) ---"
        echo "# data_dir:     $DATA_DIR"
        echo "# replicate:    $REPL"
        echo "# tree_type:    $TREE_TYPE"
        echo "# setting:      $SETTING_NAME"
        echo "# gene trees:   $GT_FILE"
        echo "# true tree:    $TRUE_TREE"
        echo "# rf_rate:      $rf_rate"
        echo "# run_exit:     $RUN_EXIT"
        if [[ "$TREE_TYPE" == "estimated" ]]; then
          echo "# rooting cmd:  $(printf '%q' "${ASTRALX_ROOT%/}/scripts/process_unrooted.sh"; printf ' %q' -i "${GT_DIR}/estimatedgenetrees.tre" -o "$ROOTED_GT" -og "0")"
        fi
        printf '# invoked as:  '
        printf ' %q' "${SCRIPT_ARGV[@]}"
        printf '\n'
        printf '# wrapper cmd: '
        printf ' %q' "${CMD[@]}"
        printf '\n'
      } >> "$COMMAND_FILE" 2>/dev/null || echo "Warning: could not append to command record $COMMAND_FILE" >&2
    }

    SIDE_STATS="${OUT_FILE%.tre}_stats.csv"
    if [[ "$RUN_EXIT" -ne 0 || ! -f "$SIDE_STATS" ]]; then
      echo "Run failed for ${REPL} (${TREE_TYPE}, ${SETTING_NAME}); skipping RF/stat summary."
      append_command_context "NA"
      mirror_results_dir "$OUT_DIR"
      continue
    fi

    RUNNING_TIME="$(csv_get_field "$SIDE_STATS" "running_time_s" "running-time-s")"
    MAX_CPU_MB="$(csv_get_field "$SIDE_STATS" "max_cpu_mb" "max-cpu-mb")"
    MAX_GPU_MB="$(csv_get_field "$SIDE_STATS" "max_gpu_mb" "max-gpu-mb")"
    OPTIMAL_QUARTET_SCORE="$(csv_get_field "$SIDE_STATS" "optimal_quartet_score" "optimal-quartet-score")"
    EXIT_CODE="$(csv_get_field "$SIDE_STATS" "exit_code" "exit-code")"
    if [[ -z "$EXIT_CODE" ]]; then
      EXIT_CODE="$RUN_EXIT"
    fi

    RF_RATE="NA"
    if [[ -f "$OUT_FILE" && -f "$TRUE_TREE" ]]; then
      rf_output=$("$PYTHON_BIN" "${ASTRALX_ROOT}/scripts/rf.py" "$OUT_FILE" "$TRUE_TREE" 2>&1) || true
      rf_line=$(echo "$rf_output" | grep -i "Robinson-Foulds distance" | tail -n1 || true)
      if [[ -n "$rf_line" ]]; then
        RF_RATE=$(echo "$rf_line" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -n1 || echo "NA")
      fi
    fi

    echo "alg,setting,replicate,tree_type,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb" > "$STAT_FILE"
    echo "astralx,${SETTING_NAME},${REPL},${TREE_TYPE},${RF_RATE},${OPTIMAL_QUARTET_SCORE},${RUNNING_TIME},${MAX_CPU_MB},${MAX_GPU_MB}" >> "$STAT_FILE"
    append_command_context "$RF_RATE"
    mirror_results_dir "$OUT_DIR"
    echo
    echo "=== A10K ASTRAL-X Summary ==="
    echo "Replicate:      ${REPL}"
    echo "Tree type:      ${TREE_TYPE}"
    echo "Setting:        ${SETTING_NAME}"
    echo "RF rate:        ${RF_RATE}"
    echo "Quartet score:  ${OPTIMAL_QUARTET_SCORE}"
    echo "Running time:   ${RUNNING_TIME}s"
    echo "Max CPU RAM:    ${MAX_CPU_MB} MB"
    echo "Max GPU VRAM:   ${MAX_GPU_MB} MB"
    echo "Output tree:    ${OUT_FILE}"
    echo "Stats file:     ${STAT_FILE}"
    echo "Outputs mirror: ${OUTPUTS_DIR}"
    echo "Saved $STAT_FILE"

    if [[ "$NO_NOTIFY" == false ]] && command -v curl >/dev/null 2>&1; then
      curl -s -d "✅ ASTRAL-X A10K completed

Replicate: ${REPL}
Tree type: ${TREE_TYPE}
Setting: ${SETTING_NAME}

RF: ${RF_RATE}
Quartet score: ${OPTIMAL_QUARTET_SCORE}
Time: ${RUNNING_TIME}s
CPU: ${MAX_CPU_MB} MB
GPU: ${MAX_GPU_MB} MB
Exit: ${EXIT_CODE}

Tree: $(basename "$OUT_FILE")
Stats: $(basename "$STAT_FILE")" "https://ntfy.sh/${NTFY_CHANNEL_NAME}" >/dev/null 2>&1 || true
    fi
    done
  done
done
