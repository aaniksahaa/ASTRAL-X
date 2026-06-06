#!/usr/bin/env bash
# run-a10k.sh
# ASTRAL-X runner for the 10k-simphy dataset layout.

set -euo pipefail

NTFY_CHANNEL_NAME="${NTFY_CHANNEL_NAME:-anik-phylo}"

TREE_TYPE="estimated"
DATA_DIR=""
REPLICATES_SPEC=""
START_REP=""
END_REP=""
FRESH=false
ASTRALX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTRALX_OPTS="--search-mode full -vv"
ASTRALX_OPTS_LIST_RAW=""
TIME_MONITOR=true
GPU_MONITOR=true
NO_NOTIFY=false

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
# ASTRALX_OPTS="--search-mode full"
#
# Example sweep over both result-affecting search modes:
# ASTRALX_OPTS_LIST_RAW="--search-mode local;--search-mode full"
#
# Verbosity flags such as -v/-vv are ignored when constructing the setting name.

sanitize_setting_part() {
  local value="$1"
  value="${value// /-}"
  value="${value//\//-}"
  value="${value//:/-}"
  value="${value//=/-}"
  value="${value//,/.-}"
  printf '%s' "$value"
}

build_setting_name_from_opts() {
  local raw="$1"
  local -a tokens=()
  local -a parts=()
  local i key value

  if [[ -z "${raw// }" ]]; then
    printf 'default'
    return
  fi

  read -r -a tokens <<< "$raw"
  i=0
  while (( i < ${#tokens[@]} )); do
    key="${tokens[$i]}"
    case "$key" in
      -v|-vv|-vvv|-q|--quiet)
        ((i+=1))
        continue
        ;;
      --*)
        key="${key#--}"
        if (( i + 1 < ${#tokens[@]} )) && [[ ! "${tokens[$((i + 1))]}" =~ ^- ]]; then
          value="${tokens[$((i + 1))]}"
          parts+=("$(sanitize_setting_part "$key")_$(sanitize_setting_part "$value")")
          ((i+=2))
        else
          parts+=("$(sanitize_setting_part "$key")_true")
          ((i+=1))
        fi
        ;;
      -t)
        if (( i + 1 < ${#tokens[@]} )); then
          parts+=("threads_$(sanitize_setting_part "${tokens[$((i + 1))]}")")
          ((i+=2))
        else
          ((i+=1))
        fi
        ;;
      -m)
        if (( i + 1 < ${#tokens[@]} )); then
          parts+=("seeds_$(sanitize_setting_part "${tokens[$((i + 1))]}")")
          ((i+=2))
        else
          ((i+=1))
        fi
        ;;
      *)
        ((i+=1))
        ;;
    esac
  done

  if [[ ${#parts[@]} -eq 0 ]]; then
    printf 'default'
  else
    local result=""
    for part in "${parts[@]}"; do
      if [[ -z "$result" ]]; then result="$part"; else result="${result}___${part}"; fi
    done
    printf '%s' "$result"
  fi
}

print_help() {
  cat <<EOF
run-a10k.sh

Required:
  --data-dir           Path to A10K dataset root containing 10k-simphy/

Optional:
  --tree-type          estimated | true (default: estimated)
  --replicates         Replicates to run, e.g. "1-20" or "R1,R2"
  --start-rep, -sr     Start replicate number
  --end-rep, -er       End replicate number
  --astralx-root       Path to ASTRAL-X root
  --opts, --alg-opts   Extra options for the selected algorithm
  --opts-list, --alg-opts-list
                       Semicolon-separated list of option strings to loop over
  --fresh              Force rerun even if stat-astralx.csv exists
  --no-time-monitor    Disable time monitoring
  --no-gpu-monitor     Disable GPU monitoring
  --no-notify, -nn     Disable ntfy notifications

Examples:
  ./run-a10k.sh --data-dir /path/to/10k-astral-dataset --tree-type estimated --opts "--search-mode local -vv"
  ./run-a10k.sh --data-dir /path/to/10k-astral-dataset --tree-type estimated --opts "--search-mode full -vv"
  ./run-a10k.sh --data-dir /path/to/10k-astral-dataset --tree-type estimated --opts-list "--search-mode local -vv;--search-mode full -vv"
  Verbosity is ignored when constructing the setting name.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tree-type) TREE_TYPE="$2"; shift 2 ;;
    --replicates) REPLICATES_SPEC="$2"; shift 2 ;;
    --start-rep|-sr) START_REP="$2"; shift 2 ;;
    --end-rep|-er) END_REP="$2"; shift 2 ;;
    --astralx-root|--stelar-root) ASTRALX_ROOT="$2"; shift 2 ;;
    --opts|--alg-opts|--astralx-opts|--stelar-opts) ASTRALX_OPTS="$2"; shift 2 ;;
    --opts-list|--alg-opts-list|--astralx-opts-list|--stelar-opts-list) ASTRALX_OPTS_LIST_RAW="$2"; shift 2 ;;
    --fresh) FRESH=true; shift ;;
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

DATA_DIR="$(realpath "$DATA_DIR")"
SIMPHY_DIR="${DATA_DIR%/}/10k-simphy"
if [[ ! -d "$SIMPHY_DIR" ]]; then
  echo "Error: expected 10k-simphy at $SIMPHY_DIR"
  exit 3
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
      if [[ ! -x "${ASTRALX_ROOT%/}/process_unrooted.sh" ]]; then
        echo "Error: process_unrooted.sh not found or not executable at ${ASTRALX_ROOT%/}/process_unrooted.sh"
        exit 7
      fi
      echo "Rooting estimated gene trees for ${REPL} with outgroup 0..."
      "${ASTRALX_ROOT%/}/process_unrooted.sh" -i "$GT_FILE" -o "$ROOTED_GT" -og "0"
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

    if [[ "$FRESH" == false && -f "$STAT_FILE" ]]; then
      echo "SKIPPING: ${STAT_FILE} exists."
      continue
    elif [[ "$FRESH" == true && -f "$STAT_FILE" ]]; then
      echo "[DEBUG] --fresh set, overwriting existing: ${STAT_FILE}"
    fi

    mkdir -p "$OUT_DIR"
    CMD=("${ASTRALX_ROOT}/run-astralx-with-monitor.sh" -i "$GT_FILE" -o "$OUT_FILE" --astralx-root "$ASTRALX_ROOT")
    if [[ "$TIME_MONITOR" == false ]]; then CMD+=(--no-time-monitor); fi
    if [[ "$GPU_MONITOR" == false ]]; then CMD+=(--no-gpu-monitor); fi
    if [[ "$NO_NOTIFY" == true ]]; then CMD+=(--no-notify); fi
    if [[ -n "$ASTRALX_OPTS_ITEM" ]]; then
      CMD+=(--opts "$ASTRALX_OPTS_ITEM")
    fi

    echo "==> Running astralx on ${REPL} (${TREE_TYPE}, ${SETTING_NAME})"
    echo "Command: ${CMD[*]}"
    set +e
    "${CMD[@]}"
    RUN_EXIT=$?
    set -e

    SIDE_STATS="${OUT_FILE%.tre}_stats.csv"
    if [[ "$RUN_EXIT" -ne 0 || ! -f "$SIDE_STATS" ]]; then
      echo "Run failed for ${REPL} (${TREE_TYPE}, ${SETTING_NAME}); skipping RF/stat summary."
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
      rf_output=$(python3 "${ASTRALX_ROOT}/rf.py" "$OUT_FILE" "$TRUE_TREE" 2>&1) || true
      rf_line=$(echo "$rf_output" | grep -i "Robinson-Foulds distance" | tail -n1 || true)
      if [[ -n "$rf_line" ]]; then
        RF_RATE=$(echo "$rf_line" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -n1 || echo "NA")
      fi
    fi

    echo "alg,setting,replicate,tree_type,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb" > "$STAT_FILE"
    echo "astralx,${SETTING_NAME},${REPL},${TREE_TYPE},${RF_RATE},${OPTIMAL_QUARTET_SCORE},${RUNNING_TIME},${MAX_CPU_MB},${MAX_GPU_MB}" >> "$STAT_FILE"
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
