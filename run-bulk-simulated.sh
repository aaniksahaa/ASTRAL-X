#!/usr/bin/env bash
# run-bulk-simulated.sh
#
# Runs sim.sh and test-astralx-simulated.sh or test-baseline-simulated.sh
# over all combinations of parameter lists.
#
# Usage:
#   ./run-bulk-simulated.sh -m stelar
#   ./run-bulk-simulated.sh -m aster --base-dir /path/to/research
#   ./run-bulk-simulated.sh -m astral --base-dir /path/to/research
#
# Default base-dir = $HOME/phylogeny

set -euo pipefail

BASE_DIR=""
BASE_DIR_PROVIDED=false
METHOD="astralx"  # default method
FRESH=false
NUM_REPLICATES=1

# Method-specific options (passed through)
ASTER_OPTS=""
ASTER_BIN=""
ASTRAL_OPTS=""
ASTRALX_OPTS_LIST_RAW=""
ASTRAL_XMS=""
ASTRAL_XMX=""
TREEQMC_OPTS=""
WQFM_OPTS=""
SUPERTRIPLETS_OPTS=""
TMC_OPTS=""

print_help() {
  cat <<EOF
run-bulk-simulated.sh

Runs sim.sh and test-astralx-simulated.sh or test-baseline-simulated.sh for all combinations of parameter lists.

Options:
  --method, -m      Method to use: astralx (default: astralx)
  --base-dir, -b    Base directory (optional, passed to sub-scripts if provided)
  --num-replicates, -n  Number of replicates to run (default: 1)
  --fresh           Pass --fresh to sim.sh and test scripts (recreate outputs)
  --opts, --alg-opts       Extra options for one ASTRAL-X simulated setting
  --opts-list, --alg-opts-list
                         Semicolon-separated list of ASTRAL-X option strings to loop over
  --help, -h        Show this message

Examples:
  ./run-bulk-simulated.sh -m astralx
  ./run-bulk-simulated.sh -m astralx --num-replicates 3 --opts "--search-mode full -vv"
  ./run-bulk-simulated.sh -m astralx --num-replicates 3 --opts-list "--search-mode local -vv;--search-mode full -vv"
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method|-m) METHOD="$2"; shift 2 ;;
    --base-dir|-b) BASE_DIR="$2"; BASE_DIR_PROVIDED=true; shift 2 ;;
    --num-replicates|-n) NUM_REPLICATES="$2"; shift 2 ;;
    --opts|--alg-opts|--astralx-opts) ASTRAL_OPTS="$2"; shift 2 ;;
    --opts=*|--alg-opts=*|--astralx-opts=*) ASTRAL_OPTS="${1#*=}"; shift ;;
    --opts-list|--alg-opts-list|--astralx-opts-list) ASTRALX_OPTS_LIST_RAW="$2"; shift 2 ;;
    --opts-list=*|--alg-opts-list=*|--astralx-opts-list=*) ASTRALX_OPTS_LIST_RAW="${1#*=}"; shift ;;
    --fresh) FRESH=true; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

# Validate method
case "$METHOD" in
  astralx|astral-x|stelar) METHOD="astralx" ;;
  *)
    echo "Error: --method must be astralx."
    exit 1
    ;;
esac

# -------------------------------
# parameter lists (EDIT AS NEEDED)
# -------------------------------
# T_LIST=(1000 2000 5000 10000 15000 20000 25000 30000)
# G_LIST=(1000)
# SB_LIST=(0.000001)
# SPMIN_LIST=(50000 100000)
# SPMAX_LIST=(150000 200000 250000 300000)

# T_LIST=(1000)
# G_LIST=(100 200)
# SB_LIST=(0.000001)
# SPMIN_LIST=(50000)
# SPMAX_LIST=(150000)

T_LIST=(100)
G_LIST=(100 200 1000 2500 5000)
SB_LIST=(0.000001)
SPMIN_LIST=(100000)
SPMAX_LIST=(200000)

T_LIST=(7500)
G_LIST=(100 200 1000 2500 5000)
SB_LIST=(0.000001)
SPMIN_LIST=(100000)
SPMAX_LIST=(200000)

# T_LIST=(10)
# G_LIST=(10)
# SB_LIST=(0.000001)
# SPMIN_LIST=(100000)
# SPMAX_LIST=(200000)

T_LIST=(100 200 500)
G_LIST=(1000)
SB_LIST=(0.000001)
SPMIN_LIST=(50000)
SPMAX_LIST=(1000000)

T_LIST=(10 20)
G_LIST=(10)
SB_LIST=(0.000001)
SPMIN_LIST=(50000)
SPMAX_LIST=(1000000)

T_LIST=(1000 2500 5000 7500 10000 25000)
G_LIST=(1000)
SB_LIST=(0.000001)
SPMIN_LIST=(100000)
SPMAX_LIST=(200000)


# T_LIST=(30000 40000)
# G_LIST=(1000)
# SB_LIST=(0.000001)
# SPMIN_LIST=(100000)
# SPMAX_LIST=(150000)

# Number of replicates to run
# NUM_REPLICATES=5  # Now set via --num-replicates flag (default: 1)

# ASTRAL-X setting sweep examples:
# ASTRAL_OPTS="--search-mode full -vv"
# ASTRALX_OPTS_LIST_RAW="--search-mode local -vv;--search-mode full -vv"
# The setting-name encoder ignores verbosity, so these become:
#   search-mode_local
#   search-mode_full

# -------------------------------
# execution
# -------------------------------

# Build base-dir argument if provided
if $BASE_DIR_PROVIDED; then
  BASE_DIR_ARG="--base-dir $BASE_DIR"
  echo "Base dir: $BASE_DIR"
else
  BASE_DIR_ARG=""
  echo "Base dir: (not specified, scripts will use their defaults)"
fi

# Build fresh argument if provided
if $FRESH; then
  FRESH_ARG="--fresh"
  echo "Fresh:    yes"
else
  FRESH_ARG=""
  echo "Fresh:    no"
fi
echo "Method:   $METHOD"
echo "Replicates: $NUM_REPLICATES"

ASTRALX_OPTS_LIST=()
if [[ -n "$ASTRALX_OPTS_LIST_RAW" ]]; then
  IFS=';' read -r -a raw_opts_list <<< "$ASTRALX_OPTS_LIST_RAW"
  for opts in "${raw_opts_list[@]}"; do
    opts="$(echo "$opts" | sed 's/^ *//;s/ *$//')"
    [[ -n "$opts" ]] && ASTRALX_OPTS_LIST+=("$opts")
  done
fi
if [[ ${#ASTRALX_OPTS_LIST[@]} -eq 0 ]]; then
  ASTRALX_OPTS_LIST+=("${ASTRAL_OPTS}")
fi

echo "Starting bulk runs..."

for t in "${T_LIST[@]}"; do
  for g in "${G_LIST[@]}"; do
    for sb in "${SB_LIST[@]}"; do
      for spmin in "${SPMIN_LIST[@]}"; do
        for spmax in "${SPMAX_LIST[@]}"; do

          echo ">>> Running: t=$t g=$g sb=$sb spmin=$spmin spmax=$spmax (method=$METHOD)"
          
          ./sim.sh -rs $NUM_REPLICATES $BASE_DIR_ARG -t "$t" -g "$g" --sb "$sb" --spmin "$spmin" --spmax "$spmax" --fresh
          
          # Run replicates
          for ((i=1; i<=NUM_REPLICATES; i++)); do
            echo "  Running replicate R$i with $METHOD"
            
            for ASTRALX_OPTS_ITEM in "${ASTRALX_OPTS_LIST[@]}"; do
              TEST_CMD=(./test-astralx-simulated.sh -r "R$i" $BASE_DIR_ARG -t "$t" -g "$g" --sb "$sb" --spmin "$spmin" --spmax "$spmax" $FRESH_ARG)
              if [[ -n "$ASTRALX_OPTS_ITEM" ]]; then
                TEST_CMD+=(--opts "$ASTRALX_OPTS_ITEM")
              fi
              "${TEST_CMD[@]}"
            done
          done

        done
      done
    done
  done
done

echo "All runs finished."














T_LIST=(1000)
G_LIST=(1000 2500 5000 7500 10000 25000)
SB_LIST=(0.000001)
SPMIN_LIST=(100000)
SPMAX_LIST=(200000)












echo "Starting bulk runs... phase 2"

for t in "${T_LIST[@]}"; do
  for g in "${G_LIST[@]}"; do
    for sb in "${SB_LIST[@]}"; do
      for spmin in "${SPMIN_LIST[@]}"; do
        for spmax in "${SPMAX_LIST[@]}"; do

          echo ">>> Running: t=$t g=$g sb=$sb spmin=$spmin spmax=$spmax (method=$METHOD)"
          
          ./sim.sh -rs $NUM_REPLICATES $BASE_DIR_ARG -t "$t" -g "$g" --sb "$sb" --spmin "$spmin" --spmax "$spmax" --fresh
          
          # Run replicates
          for ((i=1; i<=NUM_REPLICATES; i++)); do
            echo "  Running replicate R$i with $METHOD"
            
            for ASTRALX_OPTS_ITEM in "${ASTRALX_OPTS_LIST[@]}"; do
              TEST_CMD=(./test-astralx-simulated.sh -r "R$i" $BASE_DIR_ARG -t "$t" -g "$g" --sb "$sb" --spmin "$spmin" --spmax "$spmax" $FRESH_ARG)
              if [[ -n "$ASTRALX_OPTS_ITEM" ]]; then
                TEST_CMD+=(--opts "$ASTRALX_OPTS_ITEM")
              fi
              "${TEST_CMD[@]}"
            done
          done

        done
      done
    done
  done
done

echo "All runs finished."
