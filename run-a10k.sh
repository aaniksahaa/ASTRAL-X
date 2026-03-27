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
TIME_MONITOR=true
GPU_MONITOR=true
NO_NOTIFY=false

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
  --astralx-opts       Extra ASTRAL-X options
  --fresh              Force rerun even if stat-astralx.csv exists
  --no-time-monitor    Disable time monitoring
  --no-gpu-monitor     Disable GPU monitoring
  --no-notify, -nn     Disable ntfy notifications
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
    --astralx-opts|--stelar-opts) ASTRALX_OPTS="$2"; shift 2 ;;
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

for REPL in "${REPL_LIST[@]}"; do
  REPL_DIR="${SIMPHY_DIR%/}/${REPL}"
  [[ -d "$REPL_DIR" ]] || continue

  if [[ "$TREE_TYPE" == "estimated" ]]; then
    GT_FILE="${REPL_DIR}/estimatedgenetrees/estimatedgenetrees.tre"
  else
    GT_FILE="${REPL_DIR}/truegenetrees"
  fi
  TRUE_TREE="${REPL_DIR}/s_tree.trees"
  OUT_DIR="${REPL_DIR}/astralx_outputs/${TREE_TYPE}"
  OUT_FILE="${OUT_DIR}/out-astralx.tre"
  STAT_FILE="${OUT_DIR}/stat-astralx.csv"

  [[ -f "$GT_FILE" && -f "$TRUE_TREE" ]] || continue
  if [[ "$FRESH" == false && -f "$STAT_FILE" ]]; then
    echo "SKIPPING: ${STAT_FILE} exists."
    continue
  fi

  mkdir -p "$OUT_DIR"
  CMD=("${ASTRALX_ROOT}/run-astralx-with-monitor.sh" -i "$GT_FILE" -o "$OUT_FILE" --astralx-root "$ASTRALX_ROOT" --no-notify)
  if [[ "$TIME_MONITOR" == false ]]; then CMD+=(--no-time-monitor); fi
  if [[ "$GPU_MONITOR" == false ]]; then CMD+=(--no-gpu-monitor); fi
  if [[ -n "$ASTRALX_OPTS" ]]; then
    read -r -a EXTRA <<< "$ASTRALX_OPTS"
    CMD+=("${EXTRA[@]}")
  fi

  "${CMD[@]}"

  SIDE_STATS="${OUT_FILE%.tre}_stats.csv"
  RUNNING_TIME=$(awk -F, 'NR==2 {print $4}' "$SIDE_STATS")
  MAX_CPU_MB=$(awk -F, 'NR==2 {print $5}' "$SIDE_STATS")
  MAX_GPU_MB=$(awk -F, 'NR==2 {print $6}' "$SIDE_STATS")
  OPTIMAL_QUARTET_SCORE=$(awk -F, 'NR==2 {print $7}' "$SIDE_STATS")
  RF_RATE="NA"
  if [[ -f "$OUT_FILE" && -f "$TRUE_TREE" ]]; then
    rf_output=$(python3 "${ASTRALX_ROOT}/rf.py" "$OUT_FILE" "$TRUE_TREE" 2>&1) || true
    rf_line=$(echo "$rf_output" | grep -i "Robinson-Foulds distance" | tail -n1 || true)
    if [[ -n "$rf_line" ]]; then
      RF_RATE=$(echo "$rf_line" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -n1 || echo "NA")
    fi
  fi

  echo "alg,replicate,tree_type,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb" > "$STAT_FILE"
  echo "astralx,${REPL},${TREE_TYPE},${RF_RATE},${OPTIMAL_QUARTET_SCORE},${RUNNING_TIME},${MAX_CPU_MB},${MAX_GPU_MB}" >> "$STAT_FILE"
  echo "Saved $STAT_FILE"
done
