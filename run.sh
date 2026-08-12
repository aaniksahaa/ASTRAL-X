#!/usr/bin/env bash
#
# ASTRAL-X runner
# ===============
# Usage: ./run.sh -i <gene_trees> -o <output> [options]
#
# Core options are forwarded to astralx.Main. This wrapper centralizes the
# working classpath/library-path invocation so higher-level scripts do not need
# to duplicate it.

set -euo pipefail
ORIGINAL_ARGS=("$@")

# Preserve coloured Java output when score-only mode pipes through tee for
# notification parsing. Banner still honours NO_COLOR over FORCE_COLOR.
if [[ -t 1 || -t 2 ]]; then
  export FORCE_COLOR="${FORCE_COLOR:-1}"
fi

ASTRALX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ASTRALX_ROOT}/build"
NATIVE_DIR="${ASTRALX_ROOT}/native"
NTFY_CHANNEL_NAME="${NTFY_CHANNEL_NAME:-anik-phylo-asx}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

INPUT_FILE=""
OUTPUT_FILE=""
LOG_FILE=""
SCORE_SPECIES_TREE=""
XMS="${ASTRALX_XMS:-4g}"
XMX="${ASTRALX_XMX:-128g}"
BUILD_FIRST=true
PROGRAM_ARGS=()
COMPUTE_MODE_SET=false
NO_NOTIFY=false

print_help() {
  cat <<EOF
run.sh - ASTRAL-X wrapper

Usage: $0 --input <gene_trees> [--output <species_tree>] [options]

Required:
  --input, -i        Input gene trees file

Optional:
  --output, -o       Output species tree file
  --log-file FILE    Save run messages to FILE (progress remains terminal-only)
  --score-species-tree, --species-tree, --score, -c
                     Score the supplied species tree and exit
  --cpu              Force CPU mode
  --gpu              Force GPU mode
  --auto             Automatically select CUDA or CPU (default)
  --gpu-strict       Require CUDA; do not fall back to CPU
  --search-space     S1 through S3 (recommended)
  --intersection-method, --im
                     I1 through I4 (recommended)
  --search-mode      local | full
  --weight-intersection-method  prefix-sum | smaller-side-traversal | bitset | simple-tree-walk  (default: prefix-sum)
  --anchor-outgroup  Anchor the DP root at one outgroup taxon (default: on)
  --no-anchor-outgroup  Disable anchored-outgroup DP root/search-space reduction
  --anchor-taxon     Global taxon id to use as the anchor (default: 0)
  --no-prune-search-space  Disable the DP-reachability weight prune (default: on)
  --threads, --num-threads, -t, -T
                     Thread count
  --seeds, -m        Number of hash seeds
  --rooted           Treat input as rooted
  --unrooted         Treat input as unrooted
  --keep-polytomy    Keep input polytomies for native unresolved-quartet scoring
                     (default: deterministic first-pair binary refinement)
  --no-gpu-batch              Disable GPU batching
  --gpu-batch-size            GPU batch size (manual)
  --gpu-batches               Number of GPU batches (manual)
  --gpu-vram-occupancy-factor Fraction of free VRAM to use for batching (default: 0.75)
  --gpu-treewalk-vram-cap-mb  Simple-tree-walk automatic batch scratch cap in MiB (default: 512)
  --gpu-progress-interval     GPU weight-kernel progress update interval, seconds (default: auto)
  --gpu-vram-control-factor   Resident-relative batch sizing override
  --gpu-dist-tile-size        Tile size B for GPU distance matrix kernel
  --gpu-sim-vram-cap-mb       Similarity tree-batch VRAM cap (default: 512 MiB;
                              large-N auto mode may raise it within free VRAM)
  --verify-distance-matrix    Dump distance matrix and exit
  --autocomplete-incomplete-gene-trees  Autocomplete incomplete gene trees before inference
  --consensus-experimental              Enable consensus-based X enrichment (Step A + Step B)
  --stepb-fast-restriction              Enable O(d log d) Step B restriction (default: on)
  --stepb-quadratic-nn-balls            Enable quadratic NN-ball candidate emission (D1, opt-in)
  --stepb-random-leftover-resolution    Enable random leftover-polytomy resolution (D2, opt-in)
  --stepb-process-large-polytomies      Process polytomies of any degree (lift the d≤sizeLimit/31 bar, opt-in)
  --resolve-input-gene-tree-polytomies  Enrich X by resolving input gene-tree polytomies vs the UPGMA guide (opt-in)
  -v|-vv|-vvv        Verbosity
  --xms SIZE         Java min heap (default: ${XMS})
  --xmx SIZE         Java max heap (default: ${XMX})
  --no-build         Skip build.sh before running
  --no-notify, -nn   Disable ntfy notification for score-only mode
  --help, -h         Show this message

Compatibility:
  Positional form './run.sh <input> <output> ...' is also accepted.
EOF
}

if [[ $# -eq 0 ]]; then
  print_help
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

# Backward-compatible positional form: ./run.sh input output [opts...]
if [[ "${1:-}" != -* ]]; then
  INPUT_FILE="$1"
  shift
  if [[ $# -gt 0 && "${1:-}" != -* ]]; then
    OUTPUT_FILE="$1"
    shift
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT_FILE="$2"
      PROGRAM_ARGS+=("-i" "$2")
      shift 2
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      PROGRAM_ARGS+=("-o" "$2")
      shift 2
      ;;
    --log-file)
      [[ $# -ge 2 ]] || { echo -e "${RED}Error: --log-file requires a file path.${NC}"; exit 2; }
      LOG_FILE="$2"
      PROGRAM_ARGS+=("--log-file" "$2")
      shift 2
      ;;
    --auto|--cpu|--gpu|--gpu-strict)
      PROGRAM_ARGS+=("$1")
      COMPUTE_MODE_SET=true
      shift
      ;;
    --score-species-tree|--species-tree|--score|-c)
      SCORE_SPECIES_TREE="$2"
      PROGRAM_ARGS+=("$1" "$2")
      shift 2
      ;;
    --search-space|--intersection-method|--im|--search-mode|-t|-T|--threads|--num-threads|-m|--seeds|--weight-intersection-method|--large-n-score-type|--large-score-type|--anchor-taxon|--gpu-batch-size|--gpu-batches|--gpu-vram-control-factor|--gpu-vram-occupancy-factor|--gpu-treewalk-vram-cap-mb|--gpu-progress-interval|--gpu-dp-state-space-construction-output-cap|--gpu-dp-state-space-progress-time-interval|--gpu-dp-state-space-progress-max-steps|--gpu-dist-tile-size|--gpu-sim-vram-cap-mb|--dump-clusters|--dump-completed-gene-trees|--completion-method|--stepb-restriction)
      PROGRAM_ARGS+=("$1" "$2")
      shift 2
      ;;
    --stepb-fast-restriction)
      PROGRAM_ARGS+=("--stepb-restriction" "dlogd")
      shift
      ;;
    --rooted|--unrooted|--keep-polytomy|--keep-polytomies|--anchor-outgroup|--anchor|--no-anchor-outgroup|--no-anchor|--no-prune-search-space|--no-prune-unreachable|--prune-search-space|--prune-unreachable|--no-gpu-batch|--consensus-experimental|--stepb-quadratic-nn-balls|--stepb-random-leftover-resolution|--stepb-process-large-polytomies|--resolve-input-gene-tree-polytomies|--verify-parse|--verify-hash|--verify-clusters|--verify-partitions|--verify-dp|--verify-weights|--verify-distance-matrix|--verify-similarity-matrix|--verify-upgma|--verify-greedy-consensus|--autocomplete-incomplete-gene-trees|-v|-vv|-vvv|-q|--quiet)
      PROGRAM_ARGS+=("$1")
      shift
      ;;
    --xms|--Xms)
      XMS="$2"
      shift 2
      ;;
    --xmx|--Xmx)
      XMX="$2"
      shift 2
      ;;
    --no-build)
      BUILD_FIRST=false
      shift
      ;;
    --no-notify|-nn)
      NO_NOTIFY=true
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo -e "${RED}Error: Unknown option '$1'.${NC}"
      print_help
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_FILE" ]]; then
  echo -e "${RED}Error: --input is required.${NC}"
  exit 1
fi

INPUT_FILE="$(realpath "$INPUT_FILE")"
if [[ ! -f "$INPUT_FILE" ]]; then
  echo -e "${RED}Error: input file '$INPUT_FILE' does not exist.${NC}"
  exit 1
fi

if [[ -n "$OUTPUT_FILE" ]]; then
  mkdir -p "$(dirname "$OUTPUT_FILE")"
  OUTPUT_FILE="$(realpath "$OUTPUT_FILE")"
fi

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  LOG_FILE="$(realpath "$LOG_FILE")"
  if [[ "$LOG_FILE" == "$INPUT_FILE" || (-n "$OUTPUT_FILE" && "$LOG_FILE" == "$OUTPUT_FILE") ]]; then
    echo -e "${RED}Error: --log-file must differ from the input and output files: $LOG_FILE${NC}"
    exit 2
  fi
fi

# Re-enter once under tee so the log contains the wrapper diagnostics, build
# output, Java output, and native CUDA messages (but not progress repaints).
if [[ -n "$LOG_FILE" && "${ASTRALX_LOG_CAPTURED:-}" != "1" ]]; then
  filter_terminal_log() {
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ "$line" == *$'\r'* ]] && continue
      printf '%s\n' "$line"
    done
  }
  FILTER_DIR="$(mktemp -d "${TMPDIR:-/tmp}/astralx-log-filter.XXXXXX")"
  FILTER_PIPE="${FILTER_DIR}/stream"
  mkfifo "$FILTER_PIPE"
  cleanup_log_filter() {
    rm -f "$FILTER_PIPE"
    rmdir "$FILTER_DIR" 2>/dev/null || true
  }
  trap cleanup_log_filter EXIT
  filter_terminal_log < "$FILTER_PIPE" > "$LOG_FILE" &
  FILTER_PID=$!
  set +e
  ASTRALX_LOG_CAPTURED=1 "${BASH_SOURCE[0]}" "${ORIGINAL_ARGS[@]}" 2>&1 | tee "$FILTER_PIPE"
  PIPE_STATUS=("${PIPESTATUS[@]}")
  wait "$FILTER_PID"
  FILTER_STATUS=$?
  set -e
  if ((PIPE_STATUS[0] != 0)); then exit "${PIPE_STATUS[0]}"; fi
  if ((PIPE_STATUS[1] != 0)); then exit "${PIPE_STATUS[1]}"; fi
  exit "$FILTER_STATUS"
fi

if [[ -n "$SCORE_SPECIES_TREE" ]]; then
  SCORE_SPECIES_TREE="$(realpath "$SCORE_SPECIES_TREE")"
  if [[ ! -f "$SCORE_SPECIES_TREE" ]]; then
    echo -e "${RED}Error: species tree file '$SCORE_SPECIES_TREE' does not exist.${NC}"
    exit 1
  fi
fi

if [[ "$BUILD_FIRST" == true ]]; then
  "${ASTRALX_ROOT}/build.sh"
fi

if [[ ! -f "${BUILD_DIR}/astralx/Main.class" ]]; then
  echo -e "${RED}Error: compiled class not found at ${BUILD_DIR}/astralx/Main.class${NC}"
  exit 1
fi

gpu_available=false
if [[ -f "${NATIVE_DIR}/libastralx_weight.so" && -f "${NATIVE_DIR}/libastralx_dp.so" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    gpu_available=true
  fi
fi

if [[ "$COMPUTE_MODE_SET" == false ]]; then
  PROGRAM_ARGS+=("--auto")
fi

echo "=== ASTRAL-X ==="
echo "Input:       $INPUT_FILE"
if [[ -n "$OUTPUT_FILE" ]]; then
  echo "Output:      $OUTPUT_FILE"
fi
if [[ -n "$SCORE_SPECIES_TREE" ]]; then
  echo "Score tree:  $SCORE_SPECIES_TREE"
fi
echo "Build dir:   $BUILD_DIR"
echo "Native dir:  $NATIVE_DIR"
echo "GPU ready:   $gpu_available"
echo "Java heap:   -Xms${XMS} -Xmx${XMX}"
echo

if [[ -n "$SCORE_SPECIES_TREE" ]]; then
  TMP_LOG="$(mktemp /tmp/astralx_score_only.XXXXXX.log)"
  cleanup_score_log() { rm -f "$TMP_LOG"; }
  trap cleanup_score_log EXIT

  set +e
  java \
    -Xms"${XMS}" -Xmx"${XMX}" \
    -Djava.library.path="${NATIVE_DIR}" \
    -cp "${BUILD_DIR}" \
    astralx.Main \
    "${PROGRAM_ARGS[@]}" 2>&1 | tee "$TMP_LOG"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  SCORE_VALUE="NA"
  SCORE_LINE="$(grep -E 'QUARTET_SCORE:' "$TMP_LOG" | tail -n1 || true)"
  if [[ -n "$SCORE_LINE" ]]; then
    SCORE_VALUE="$(echo "$SCORE_LINE" | awk -F: '{gsub(/^[ \t]+/,"",$2); print $2}' | awk '{print $1}')"
  fi

  if [[ "$NO_NOTIFY" == false ]] && command -v curl >/dev/null 2>&1; then
    STATUS_TEXT="$(if [[ $EXIT_CODE -eq 0 ]]; then echo "completed"; else echo "failed (exit $EXIT_CODE)"; fi)"
    NOTIFY_BODY="ASTRAL-X score-only ${STATUS_TEXT}

Quartet score: ${SCORE_VALUE}
Input: $(basename "$INPUT_FILE")
Species tree: $(basename "$SCORE_SPECIES_TREE")"
    if [[ -n "$OUTPUT_FILE" ]]; then
      NOTIFY_BODY+="
Output: $(basename "$OUTPUT_FILE")"
    fi
    curl -s -d "$NOTIFY_BODY" "https://ntfy.sh/${NTFY_CHANNEL_NAME}" >/dev/null 2>&1 || true
  fi

  exit "$EXIT_CODE"
fi

exec java \
  -Xms"${XMS}" -Xmx"${XMX}" \
  -Djava.library.path="${NATIVE_DIR}" \
  -cp "${BUILD_DIR}" \
  astralx.Main \
  "${PROGRAM_ARGS[@]}"
