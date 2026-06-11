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

ASTRALX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ASTRALX_ROOT}/build"
NATIVE_DIR="${ASTRALX_ROOT}/native"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

INPUT_FILE=""
OUTPUT_FILE=""
XMS="${ASTRALX_XMS:-4g}"
XMX="${ASTRALX_XMX:-128g}"
BUILD_FIRST=true
PROGRAM_ARGS=()
COMPUTE_MODE_SET=false

print_help() {
  cat <<EOF
run.sh - ASTRAL-X wrapper

Usage: $0 --input <gene_trees> [--output <species_tree>] [options]

Required:
  --input, -i        Input gene trees file

Optional:
  --output, -o       Output species tree file
  --cpu              Force CPU mode
  --gpu              Force GPU mode
  --search-mode      local | full
  --weight-intersection-method  prefix-sum | smaller-side-traversal  (default: prefix-sum)
  --threads, -t      Thread count
  --seeds, -m        Number of hash seeds
  --rooted           Treat input as rooted
  --unrooted         Treat input as unrooted
  --no-gpu-batch              Disable GPU batching
  --gpu-batch-size            GPU batch size (manual)
  --gpu-batches               Number of GPU batches (manual)
  --gpu-vram-occupancy-factor Fraction of free VRAM to use for batching (default: 0.75)
  --gpu-vram-control-factor   Resident-relative batch sizing override
  --gpu-dist-tile-size        Tile size B for GPU distance matrix kernel
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
    --cpu|--gpu)
      PROGRAM_ARGS+=("$1")
      COMPUTE_MODE_SET=true
      shift
      ;;
    --search-mode|-t|--threads|-m|--seeds|--weight-intersection-method|--gpu-batch-size|--gpu-batches|--gpu-vram-control-factor|--gpu-vram-occupancy-factor|--gpu-dp-state-space-construction-output-cap|--gpu-dist-tile-size|--dump-completed-gene-trees|--completion-method)
      PROGRAM_ARGS+=("$1" "$2")
      shift 2
      ;;
    --rooted|--unrooted|--no-gpu-batch|--consensus-experimental|--stepb-fast-restriction|--stepb-quadratic-nn-balls|--stepb-random-leftover-resolution|--stepb-process-large-polytomies|--resolve-input-gene-tree-polytomies|--verify-parse|--verify-hash|--verify-clusters|--verify-partitions|--verify-dp|--verify-weights|--verify-distance-matrix|--verify-similarity-matrix|--verify-upgma|--verify-greedy-consensus|--autocomplete-incomplete-gene-trees|-v|-vv|-vvv|-q|--quiet)
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
  if [[ "$gpu_available" == true ]]; then
    PROGRAM_ARGS+=("--gpu")
  else
    PROGRAM_ARGS+=("--cpu")
  fi
fi

echo "=== ASTRAL-X ==="
echo "Input:       $INPUT_FILE"
if [[ -n "$OUTPUT_FILE" ]]; then
  echo "Output:      $OUTPUT_FILE"
fi
echo "Build dir:   $BUILD_DIR"
echo "Native dir:  $NATIVE_DIR"
echo "GPU ready:   $gpu_available"
echo "Java heap:   -Xms${XMS} -Xmx${XMX}"
echo

exec java \
  -Xms"${XMS}" -Xmx"${XMX}" \
  -Djava.library.path="${NATIVE_DIR}" \
  -cp "${BUILD_DIR}" \
  astralx.Main \
  "${PROGRAM_ARGS[@]}"
