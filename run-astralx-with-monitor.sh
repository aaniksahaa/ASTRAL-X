#!/usr/bin/env bash
# run-astralx-with-monitor.sh
# Wrapper for ASTRAL-X that records wall time, CPU RAM, GPU VRAM, and summary
# stats while preserving the existing research-script structure.

set -euo pipefail

NTFY_CHANNEL_NAME="${NTFY_CHANNEL_NAME:-anik-phylo}"

INPUT_FILE=""
OUTPUT_FILE=""
ASTRALX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIME_MONITOR=true
GPU_MONITOR=true
NO_NOTIFY=false
DEBUG=0
ASTRALX_ARGS=()

print_help() {
  cat <<EOF
run-astralx-with-monitor.sh - ASTRAL-X wrapper with performance monitoring

Usage: $0 --input <input_file> --output <output_file> [options]

Required:
  --input, -i           Path to gene trees file
  --output, -o          Path to output species tree file

Optional:
  --astralx-root        Path to ASTRAL-X root directory (default: current directory)
  --stelar-root         Compatibility alias for --astralx-root
  --astralx-opts "..."  Extra ASTRAL-X options passed to run.sh
  --stelar-opts "..."   Compatibility alias for --astralx-opts
  --no-time-monitor     Disable time monitoring
  --no-gpu-monitor      Disable GPU monitoring
  --no-notify, -nn      Disable ntfy notifications
  --debug               Enable shell tracing
  --help, -h            Show this message
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input) INPUT_FILE="$2"; shift 2 ;;
    -o|--output) OUTPUT_FILE="$2"; shift 2 ;;
    --astralx-root|--stelar-root) ASTRALX_ROOT="$2"; shift 2 ;;
    --astralx-opts|--stelar-opts)
      read -r -a TMP_OPTS <<< "$2"
      ASTRALX_ARGS+=("${TMP_OPTS[@]}")
      shift 2
      ;;
    --no-time-monitor) TIME_MONITOR=false; shift ;;
    --no-gpu-monitor) GPU_MONITOR=false; shift ;;
    --no-notify|-nn) NO_NOTIFY=true; shift ;;
    --debug) DEBUG=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    --cpu|--gpu|--rooted|--unrooted|--verify-parse|--verify-hash|--verify-clusters|--verify-partitions|--verify-dp|--verify-weights|-v|-vv|-vvv|-q|--quiet)
      ASTRALX_ARGS+=("$1")
      shift
      ;;
    --search-mode|-t|--threads|-m|--seeds)
      ASTRALX_ARGS+=("$1" "$2")
      shift 2
      ;;
    --xms|--Xms|--xmx|--Xmx|--no-build)
      ASTRALX_ARGS+=("$1")
      if [[ "$1" != "--no-build" ]]; then
        ASTRALX_ARGS+=("$2")
        shift 2
      else
        shift
      fi
      ;;
    --)
      shift
      ASTRALX_ARGS+=("$@")
      break
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  echo "Error: positional arguments are not supported."
  print_help
  exit 1
fi

if [[ -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
  echo "Error: both --input and --output are required."
  exit 1
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

INPUT_FILE="$(realpath "$INPUT_FILE")"
OUTPUT_FILE="$(realpath "$OUTPUT_FILE")"
ASTRALX_ROOT="$(realpath "$ASTRALX_ROOT")"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo -e "${RED}Error: input file '$INPUT_FILE' does not exist.${NC}"
  exit 1
fi
if [[ ! -x "${ASTRALX_ROOT}/run.sh" ]]; then
  echo -e "${RED}Error: run.sh not found or not executable in '$ASTRALX_ROOT'.${NC}"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

TEMP_DIR="$(mktemp -d)"
TIME_TMP="${TEMP_DIR}/astralx_time_err.log"
MON_TMP="${TEMP_DIR}/astralx_gpu_mem.log"
DONE_FILE="${TEMP_DIR}/.astralx_done"

cleanup() {
  rm -f "$DONE_FILE" 2>/dev/null || true
  if [[ -n "${MON_PID:-}" ]]; then
    kill "$MON_PID" 2>/dev/null || true
    wait "$MON_PID" 2>/dev/null || true
  fi
  rm -rf "$TEMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

TIME_CMD=""
if [[ "$TIME_MONITOR" == true ]]; then
  if [[ -x "/usr/bin/time" ]]; then
    TIME_CMD="/usr/bin/time"
  elif command -v time >/dev/null 2>&1; then
    TMP_TEST="$(mktemp)"
    sh -c "command time -v true" 2> "$TMP_TEST" >/dev/null || true
    if grep -qi "Maximum resident set size" "$TMP_TEST" 2>/dev/null; then
      TIME_CMD="$(command -v time)"
    fi
    rm -f "$TMP_TEST"
  fi
  if [[ -z "$TIME_CMD" ]]; then
    echo -e "${YELLOW}Warning: no suitable 'time -v' binary found; continuing without time monitor.${NC}"
    TIME_MONITOR=false
  fi
fi

MON_PID=""
if [[ "$GPU_MONITOR" == true ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  (
    curmax=0
    while true; do
      gpu_val=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{m=0} {v=int($1); if(v>m) m=v} END{print m+0}')
      if [[ -n "$gpu_val" && "$gpu_val" =~ ^[0-9]+$ ]] && (( gpu_val > curmax )); then
        curmax=$gpu_val
      fi
      [[ -f "$DONE_FILE" ]] && break
      sleep 0.2
    done
    echo "$curmax" > "$MON_TMP"
  ) &
  MON_PID=$!
else
  GPU_MONITOR=false
fi

echo "=== ASTRAL-X Monitor Wrapper ==="
echo "Input file:     $INPUT_FILE"
echo "Output file:    $OUTPUT_FILE"
echo "ASTRAL-X root:  $ASTRALX_ROOT"
echo "ASTRAL-X opts:  ${ASTRALX_ARGS[*]:-(defaults)}"
echo "Time monitor:   $TIME_MONITOR"
echo "GPU monitor:    $GPU_MONITOR"
echo "Notifications:  $(if [[ "$NO_NOTIFY" == true ]]; then echo "disabled"; else echo "enabled"; fi)"
echo

START_NS=$(date +%s%N)

ASTRALX_PID=""
if [[ "$TIME_MONITOR" == true && -n "$TIME_CMD" ]]; then
  (
    cd "$ASTRALX_ROOT" && "$TIME_CMD" -v ./run.sh --input "$INPUT_FILE" --output "$OUTPUT_FILE" "${ASTRALX_ARGS[@]}" < /dev/null 2>&1 | tee "$TIME_TMP"
  ) &
  ASTRALX_PID=$!
else
  (
    cd "$ASTRALX_ROOT" && ./run.sh --input "$INPUT_FILE" --output "$OUTPUT_FILE" "${ASTRALX_ARGS[@]}" < /dev/null 2>&1 | tee "$TIME_TMP"
  ) &
  ASTRALX_PID=$!
fi

sleep 0.25
if ! kill -0 "$ASTRALX_PID" >/dev/null 2>&1; then
  echo -e "${RED}Error: ASTRAL-X process failed to start or died immediately.${NC}"
  head -n 200 "$TIME_TMP" 2>/dev/null || true
  touch "$DONE_FILE"
  exit 5
fi

wait "$ASTRALX_PID"
ASTRALX_EXIT_CODE=$?
touch "$DONE_FILE"

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
RUNNING_TIME=$(awk "BEGIN {printf \"%.3f\", ${ELAPSED_MS}/1000}")

if [[ -n "${MON_PID:-}" ]]; then
  wait "$MON_PID" 2>/dev/null || true
fi

MAX_GPU_VAL="NA"
if [[ -f "$MON_TMP" ]]; then
  MAX_GPU_VAL="$(cat "$MON_TMP" 2>/dev/null || echo "NA")"
fi
if [[ "$MAX_GPU_VAL" =~ ^[0-9]+$ ]]; then
  MAX_GPU_MB=$(awk "BEGIN {printf \"%.3f\", ${MAX_GPU_VAL} * 1.024}")
else
  MAX_GPU_MB="NA"
fi

MAX_CPU_MB="NA"
if [[ -f "$TIME_TMP" && -s "$TIME_TMP" ]]; then
  MAX_RSS_KB=$(grep -i "Maximum resident set size" "$TIME_TMP" 2>/dev/null | awk -F: '{gsub(/^[ \t]+/,"",$2); print $2}' | awk '{print int($1)}' | head -n1 || true)
  if [[ -n "${MAX_RSS_KB:-}" && "$MAX_RSS_KB" =~ ^[0-9]+$ ]]; then
    MAX_CPU_MB=$(awk "BEGIN {printf \"%.3f\", ${MAX_RSS_KB}/1024}")
  fi
fi

OPTIMAL_QUARTET_SCORE="NA"
if [[ -f "$TIME_TMP" ]]; then
  SCORE_LINE=$(grep -i "optimal quartet score" "$TIME_TMP" 2>/dev/null | tail -n1 || true)
  if [[ -n "$SCORE_LINE" ]]; then
    OPTIMAL_QUARTET_SCORE=$(echo "$SCORE_LINE" | awk -F'=' '{print $2}' | awk '{print $1}' | tr -d ' ' || echo "NA")
  fi
fi

echo
echo -e "${GREEN}=== ASTRAL-X Execution Summary ===${NC}"
echo "Status:         $(if [[ $ASTRALX_EXIT_CODE -eq 0 ]]; then echo -e "${GREEN}SUCCESS${NC}"; else echo -e "${RED}FAILED (exit code $ASTRALX_EXIT_CODE)${NC}"; fi)"
echo "Running time:   ${RUNNING_TIME}s"
echo "Max CPU RAM:    ${MAX_CPU_MB} MB"
echo "Max GPU VRAM:   ${MAX_GPU_MB} MB"
echo "Quartet score:  ${OPTIMAL_QUARTET_SCORE}"
echo "Output exists:  $(if [[ -f "$OUTPUT_FILE" ]]; then echo "Yes"; else echo "No"; fi)"

STATS_FILE="${OUTPUT_FILE%.*}_stats.csv"
echo "algorithm,input_file,output_file,running_time_s,max_cpu_mb,max_gpu_mb,optimal_quartet_score,exit_code" > "$STATS_FILE"
echo "astral-x,$(basename "$INPUT_FILE"),$(basename "$OUTPUT_FILE"),${RUNNING_TIME},${MAX_CPU_MB},${MAX_GPU_MB},${OPTIMAL_QUARTET_SCORE},${ASTRALX_EXIT_CODE}" >> "$STATS_FILE"
echo "Stats saved to: $STATS_FILE"

if [[ "$NO_NOTIFY" == false ]] && command -v curl >/dev/null 2>&1; then
  STATUS_EMOJI=$(if [[ $ASTRALX_EXIT_CODE -eq 0 ]]; then echo "✅"; else echo "❌"; fi)
  STATUS_TEXT=$(if [[ $ASTRALX_EXIT_CODE -eq 0 ]]; then echo "completed"; else echo "failed (exit $ASTRALX_EXIT_CODE)"; fi)
  curl -s -d "${STATUS_EMOJI} ASTRAL-X ${STATUS_TEXT}

Running time: ${RUNNING_TIME}s
Max CPU RAM: ${MAX_CPU_MB} MB
Max GPU VRAM: ${MAX_GPU_MB} MB
Quartet score: ${OPTIMAL_QUARTET_SCORE}

Input: $(basename "$INPUT_FILE")
Output: $(basename "$OUTPUT_FILE")
Stats: $(basename "$STATS_FILE")" "https://ntfy.sh/${NTFY_CHANNEL_NAME}" >/dev/null 2>&1 || true
fi

exit "$ASTRALX_EXIT_CODE"
