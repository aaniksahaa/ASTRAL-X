#!/usr/bin/env bash
# test-astralx-simulated.sh
# Historical filename preserved; this now runs ASTRAL-X on a simulated dataset
# replicate and records the usual research statistics.

set -euo pipefail

# Propagate terminal color preference to Java subprocesses even when stderr is
# piped through tee further down the call chain.
[[ -t 1 || -t 2 ]] && export FORCE_COLOR=1

NTFY_CHANNEL_NAME="${NTFY_CHANNEL_NAME:-anik-phylo}"

TAXA_NUM=""
GENE_TREES=""
REPLICATE="R1"
BASE_DIR=".."
SIMPHY_DIR=""
SIMPHY_DIR_SET=false
SIMPHY_DATA_DIR=""
SIMPHY_DATA_DIR_SET=false
ASTRALX_ROOT=""
ASTRALX_ROOT_SET=false
SB="0.000001"
SPMIN="500000"
SPMAX="1500000"
USE_LEGACY_LAYOUT=false
ASTRALX_OPTS="--search-mode full -vv"
FRESH=false
INCOMPLETE=false
TIME_MONITOR=true
GPU_MONITOR=true
NO_NOTIFY=false
DEBUG=0

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
  local i search_mode_val=""

  if [[ -z "${raw// }" ]]; then
    printf 'default'
    return
  fi

  read -r -a tokens <<< "$raw"
  i=0
  while (( i < ${#tokens[@]} )); do
    if [[ "${tokens[$i]}" == "--search-mode" ]] && (( i + 1 < ${#tokens[@]} )); then
      search_mode_val="${tokens[$((i + 1))]}"
      ((i+=2))
    else
      ((i+=1))
    fi
  done

  if [[ -n "$search_mode_val" ]]; then
    printf 'search-mode_%s' "$(sanitize_setting_part "$search_mode_val")"
  else
    printf 'default'
  fi
}

print_help() {
  cat <<EOF
test-astralx-simulated.sh

Required:
  --taxa_num, -t       Number of taxa
  --gene_trees, -g     Number of gene trees

Optional:
  --replicate, -r      Replicate name (default: ${REPLICATE})
  --base-dir, -b       Base directory (default: ${BASE_DIR})
  --simphy-dir         Path to simphy dir
  --simphy-data-dir    Custom path to simphy/data root
  --astralx-root       Path to ASTRAL-X root
  --stelar-root        Compatibility alias for --astralx-root
  --opts, --alg-opts   Extra args for the selected algorithm run (default: "${ASTRALX_OPTS}")
  --astralx-opts       Compatibility alias for --opts
  --stelar-opts        Compatibility alias for --opts
  --sb                 Substitution/birthrate parameter
  --spmin              Population size minimum
  --spmax              Population size maximum
  --use-legacy-layout  Use legacy simphy layout
  --incomplete         Use the incomplete-tree variant of the dataset
                       (appends _incomplete to the dataset directory name;
                        generate with sim_incomplete.sh first)
  If the expected simulated dataset is missing, this script will first invoke
  ./sim.sh with matching parameters to generate the required replicate.
  --fresh              Force rerun even if stat-astralx.csv exists
  --no-time-monitor    Disable time monitoring
  --no-gpu-monitor     Disable GPU monitoring
  --no-notify, -nn     Disable ntfy notifications
  --debug              Enable shell tracing

Examples:
  ./test-astralx-simulated.sh -t 100 -g 100 -r R1 --fresh
  ./test-astralx-simulated.sh -t 100 -g 100 -r R1 --fresh --opts "--search-mode local -vv"
  ./test-astralx-simulated.sh -t 100 -g 100 -r R1 --fresh --opts "--search-mode full -vv"
  Verbosity is ignored when constructing the setting name.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --taxa_num|-t) TAXA_NUM="$2"; shift 2 ;;
    --gene_trees|-g) GENE_TREES="$2"; shift 2 ;;
    --replicate|-r) REPLICATE="$2"; shift 2 ;;
    --simphy-dir) SIMPHY_DIR="$2"; SIMPHY_DIR_SET=true; shift 2 ;;
    --simphy-data-dir) SIMPHY_DATA_DIR="$2"; SIMPHY_DATA_DIR_SET=true; shift 2 ;;
    --astralx-root|--stelar-root) ASTRALX_ROOT="$2"; ASTRALX_ROOT_SET=true; shift 2 ;;
    --opts|--alg-opts|--astralx-opts|--stelar-opts) ASTRALX_OPTS="$2"; shift 2 ;;
    --base-dir|-b) BASE_DIR="$2"; shift 2 ;;
    --sb) SB="$2"; shift 2 ;;
    --spmin) SPMIN="$2"; shift 2 ;;
    --spmax) SPMAX="$2"; shift 2 ;;
    --use-legacy-layout) USE_LEGACY_LAYOUT=true; shift ;;
    --incomplete) INCOMPLETE=true; shift ;;
    --fresh) FRESH=true; shift ;;
    --no-time-monitor) TIME_MONITOR=false; shift ;;
    --no-gpu-monitor) GPU_MONITOR=false; shift ;;
    --no-notify|-nn) NO_NOTIFY=true; shift ;;
    --debug) DEBUG=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

if [[ -z "$TAXA_NUM" || -z "$GENE_TREES" ]]; then
  echo "Error: --taxa_num and --gene_trees are required."
  exit 2
fi

if [[ "$SIMPHY_DIR_SET" == false ]]; then
  SIMPHY_DIR="./simphy"
fi
if [[ "$ASTRALX_ROOT_SET" == false ]]; then
  ASTRALX_ROOT="."
fi

SETTING_NAME="$(build_setting_name_from_opts "$ASTRALX_OPTS")"

PAIR="${TAXA_NUM}_${GENE_TREES}"
if [[ "$SIMPHY_DATA_DIR_SET" == true ]]; then
  if [[ "$USE_LEGACY_LAYOUT" == true ]]; then
    SIMPHY_RUN_DIR="${SIMPHY_DATA_DIR%/}/${PAIR}/${REPLICATE}"
  else
    SIMPHY_RUN_DIR="${SIMPHY_DATA_DIR%/}/t_${TAXA_NUM}_g_${GENE_TREES}_sb_${SB}_spmin_${SPMIN}_spmax_${SPMAX}/${REPLICATE}"
  fi
else
  if [[ "$USE_LEGACY_LAYOUT" == true ]]; then
    SIMPHY_RUN_DIR="${SIMPHY_DIR%/}/data/${PAIR}/${REPLICATE}"
  else
    SIMPHY_RUN_DIR="${SIMPHY_DIR%/}/data/t_${TAXA_NUM}_g_${GENE_TREES}_sb_${SB}_spmin_${SPMIN}_spmax_${SPMAX}/${REPLICATE}"
  fi
fi

# When --incomplete is set, the dataset lives in the _incomplete variant directory.
# e.g. simphy/data/t_100_g_100_sb_.../R1  →  simphy/data/t_100_g_100_sb_..._incomplete/R1
if [[ "$INCOMPLETE" == true ]]; then
  _REPL_BASE="$(basename "$SIMPHY_RUN_DIR")"
  _DATASET_DIR="$(dirname "$SIMPHY_RUN_DIR")"
  SIMPHY_RUN_DIR="${_DATASET_DIR}_incomplete/${_REPL_BASE}"
fi

ALL_GT_FILE="${SIMPHY_RUN_DIR%/}/all_gt.tre"
TRUE_SPECIES_TREE="${SIMPHY_RUN_DIR%/}/s_tree.trees"
RESULTS_DIR="${SIMPHY_RUN_DIR%/}/astralx_outputs/${SETTING_NAME}"
STAT_FILE="${RESULTS_DIR%/}/stat-astralx.csv"
LOCK_FILE="${RESULTS_DIR%/}/.astralx.lock"
OUT_ASTRALX="${RESULTS_DIR%/}/out-astralx.tre"
RUN_LOG="${RESULTS_DIR%/}/.astralx_run.log"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

if [[ "$FRESH" == false && -f "$STAT_FILE" ]]; then
  echo "SKIPPING: ${STAT_FILE} already exists. Use --fresh to force rerun."
  exit 0
fi

if [[ ! -f "$ALL_GT_FILE" ]]; then
  if [[ "$USE_LEGACY_LAYOUT" == true ]]; then
    echo "Error: gene-tree file not found at $ALL_GT_FILE"
    echo "Automatic simulation bootstrap is not supported with --use-legacy-layout."
    exit 6
  fi

  if [[ "$INCOMPLETE" == true ]]; then
    echo "Incomplete gene-tree file not found at $ALL_GT_FILE"
    echo "==> Bootstrapping missing incomplete dataset via ./sim_incomplete.sh"

    REPLICATE_COUNT=1
    if [[ "$REPLICATE" =~ ^R([0-9]+)$ ]]; then
      REPLICATE_COUNT="${BASH_REMATCH[1]}"
    fi

    SIM_INC_CMD=(./sim_incomplete.sh -t "$TAXA_NUM" -g "$GENE_TREES" -rs "$REPLICATE_COUNT" --sb "$SB" --spmin "$SPMIN" --spmax "$SPMAX")
    if [[ "$SIMPHY_DIR_SET" == true ]];      then SIM_INC_CMD+=(--simphy-dir      "$SIMPHY_DIR");      fi
    if [[ "$SIMPHY_DATA_DIR_SET" == true ]]; then SIM_INC_CMD+=(--simphy-data-dir "$SIMPHY_DATA_DIR"); fi
    if [[ "$FRESH" == true ]];               then SIM_INC_CMD+=(--fresh-inc);                          fi

    "${SIM_INC_CMD[@]}"

    if [[ ! -f "$ALL_GT_FILE" ]]; then
      echo "Error: bootstrap completed but incomplete gene-tree file is still missing at $ALL_GT_FILE"
      exit 6
    fi
  else
    echo "Gene-tree file not found at $ALL_GT_FILE"
    echo "==> Bootstrapping missing simulated dataset via ./sim.sh"

    REPLICATE_COUNT=1
    if [[ "$REPLICATE" =~ ^R([0-9]+)$ ]]; then
      REPLICATE_COUNT="${BASH_REMATCH[1]}"
    elif [[ "$REPLICATE" =~ ^[0-9]+$ ]]; then
      REPLICATE_COUNT="$REPLICATE"
      REPLICATE="R${REPLICATE}"
      SIMPHY_RUN_DIR="${SIMPHY_RUN_DIR%/*}/R${REPLICATE_COUNT}"
      ALL_GT_FILE="${SIMPHY_RUN_DIR%/}/all_gt.tre"
      TRUE_SPECIES_TREE="${SIMPHY_RUN_DIR%/}/s_tree.trees"
      RESULTS_DIR="${SIMPHY_RUN_DIR%/}/astralx_outputs/${SETTING_NAME}"
      STAT_FILE="${RESULTS_DIR%/}/stat-astralx.csv"
      LOCK_FILE="${RESULTS_DIR%/}/.astralx.lock"
      OUT_ASTRALX="${RESULTS_DIR%/}/out-astralx.tre"
      RUN_LOG="${RESULTS_DIR%/}/.astralx_run.log"
    fi

    SIM_CMD=(./sim.sh -t "$TAXA_NUM" -g "$GENE_TREES" -r "$REPLICATE" -rs "$REPLICATE_COUNT" --sb "$SB" --spmin "$SPMIN" --spmax "$SPMAX")
    if [[ "$SIMPHY_DIR_SET" == true ]];      then SIM_CMD+=(--simphy-dir      "$SIMPHY_DIR");      fi
    if [[ "$SIMPHY_DATA_DIR_SET" == true ]]; then SIM_CMD+=(--simphy-data-dir "$SIMPHY_DATA_DIR"); fi
    if [[ "$FRESH" == true ]];               then SIM_CMD+=(--fresh);                              fi

    "${SIM_CMD[@]}"

    if [[ ! -f "$ALL_GT_FILE" ]]; then
      echo "Error: dataset bootstrap completed but gene-tree file is still missing at $ALL_GT_FILE"
      exit 6
    fi
  fi
fi

mkdir -p "${RESULTS_DIR%/}"
rm -f "$LOCK_FILE" "$RUN_LOG"
touch "$LOCK_FILE"

echo "Parameters:"
echo "  taxa_num:       $TAXA_NUM"
echo "  gene_trees:     $GENE_TREES"
echo "  replicate:      $REPLICATE"
echo "  setting:        $SETTING_NAME"
echo "  simphy run dir: $SIMPHY_RUN_DIR"
echo "  results dir:    $RESULTS_DIR"
echo "  output tree:    $OUT_ASTRALX"
echo "  stat file:      $STAT_FILE"
echo

CMD=(./run-astralx-with-monitor.sh -i "$ALL_GT_FILE" -o "$OUT_ASTRALX" --astralx-root "$ASTRALX_ROOT" --no-notify)
if [[ "$TIME_MONITOR" == false ]]; then CMD+=(--no-time-monitor); fi
if [[ "$GPU_MONITOR" == false ]]; then CMD+=(--no-gpu-monitor); fi
if [[ "$DEBUG" == 1 ]]; then CMD+=(--debug); fi
if [[ -n "$ASTRALX_OPTS" ]]; then
  CMD+=(--opts "$ASTRALX_OPTS")
fi

echo "==> Running ASTRAL-X"
set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
ASTRALX_EXIT_CODE=${PIPESTATUS[0]}
set -e

RUNNING_TIME="NA"
MAX_CPU_MB="NA"
MAX_GPU_MB="NA"
OPTIMAL_QUARTET_SCORE="NA"

STATS_SIDE_FILE="${OUT_ASTRALX%.tre}_stats.csv"
if [[ -f "$STATS_SIDE_FILE" ]]; then
  RUNNING_TIME=$(awk -F, 'NR==2 {print $4}' "$STATS_SIDE_FILE")
  MAX_CPU_MB=$(awk -F, 'NR==2 {print $5}' "$STATS_SIDE_FILE")
  MAX_GPU_MB=$(awk -F, 'NR==2 {print $6}' "$STATS_SIDE_FILE")
  OPTIMAL_QUARTET_SCORE=$(awk -F, 'NR==2 {print $7}' "$STATS_SIDE_FILE")
fi

RF_RATE="NA"
if [[ -f "$OUT_ASTRALX" && -f "$TRUE_SPECIES_TREE" ]]; then
  rf_output=$(python3 ./rf.py "$OUT_ASTRALX" "$TRUE_SPECIES_TREE" 2>&1) || true
  rf_line=$(echo "$rf_output" | grep -i "Robinson-Foulds distance" | tail -n1 || true)
  if [[ -n "$rf_line" ]]; then
    RF_RATE=$(echo "$rf_line" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -n1 || echo "NA")
  fi
fi

echo "alg,setting,num-taxa,gene-trees,replicate,sb,spmin,spmax,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb" > "$STAT_FILE"
echo "astralx,${SETTING_NAME},${TAXA_NUM},${GENE_TREES},${REPLICATE},${SB},${SPMIN},${SPMAX},${RF_RATE},${OPTIMAL_QUARTET_SCORE},${RUNNING_TIME},${MAX_CPU_MB},${MAX_GPU_MB}" >> "$STAT_FILE"

if [[ "$ASTRALX_EXIT_CODE" -ne 0 ]]; then
  rm -f "$LOCK_FILE"
else
  touch "$LOCK_FILE"
fi

echo
echo "ASTRAL-X finished in ${RUNNING_TIME}s (exit code ${ASTRALX_EXIT_CODE})"
echo "RF rate: ${RF_RATE}"
echo "Quartet score: ${OPTIMAL_QUARTET_SCORE}"
echo "Max CPU RAM (MB): ${MAX_CPU_MB}"
echo "Max GPU VRAM (MB): ${MAX_GPU_MB}"
echo "Wrote stats to $STAT_FILE"

if [[ "$NO_NOTIFY" == false ]] && command -v curl >/dev/null 2>&1; then
  STATUS_EMOJI=$(if [[ $ASTRALX_EXIT_CODE -eq 0 ]]; then echo "✅"; else echo "❌"; fi)
  STATUS_TEXT=$(if [[ $ASTRALX_EXIT_CODE -eq 0 ]]; then echo "completed"; else echo "failed (exit $ASTRALX_EXIT_CODE)"; fi)
  curl -s -d "${STATUS_EMOJI} ASTRAL-X ${STATUS_TEXT} for ${TAXA_NUM} taxa and ${GENE_TREES} gene trees

RF Rate: ${RF_RATE}
Quartet score: ${OPTIMAL_QUARTET_SCORE}
Running time: ${RUNNING_TIME}s
Max CPU RAM: ${MAX_CPU_MB} MB
Max GPU VRAM: ${MAX_GPU_MB} MB

Stats: ${STAT_FILE}" "https://ntfy.sh/${NTFY_CHANNEL_NAME}" >/dev/null 2>&1 || true
fi

exit "$ASTRALX_EXIT_CODE"
