#!/usr/bin/env bash
# test-astralx-simulated.sh
# Historical filename preserved; this now runs ASTRAL-X on a simulated dataset
# replicate and records the usual research statistics.

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_ROOT}/scripts/phylogeny-data-dir.sh"
source "${SCRIPT_ROOT}/scripts/simphy-outputs-dir.sh"

# Propagate terminal color preference to Java subprocesses even when stderr is
# piped through tee further down the call chain.
[[ -t 1 || -t 2 ]] && export FORCE_COLOR=1

NTFY_CHANNEL_NAME="${NTFY_CHANNEL_NAME:-anik-phylo-asx}"

# Exact invocation of this script, appended to the run's command record.
SCRIPT_ARGV=("$0" "$@")

TAXA_NUM=""
GENE_TREES=""
REPLICATE="R1"
BASE_DIR="$(dirname "$SCRIPT_ROOT")"
BASE_DIR_SET=false
SIMPHY_DIR=""
SIMPHY_DIR_SET=false
SIMPHY_DATA_DIR=""
SIMPHY_OUTPUTS_DIR=""
OUTPUTS_MIRROR=true
ASTRALX_ROOT=""
ASTRALX_ROOT_SET=false
SB="0.000001"
SPMIN="500000"
SPMAX="1500000"
USE_LEGACY_LAYOUT=false
ASTRALX_OPTS="--search-space S2 -vv"
FRESH=false
INCOMPLETE=false
TIME_MONITOR=true
GPU_MONITOR=true
NO_NOTIFY=false
DEBUG=0

source "${SCRIPT_ROOT}/experiment-setting-name.sh"

# Extract the canonical intersection method from an opts string.
# Returns 'prefix-sum' (the default) when not specified.
extract_weight_method_from_opts() {
  local raw="$1"
  local -a tokens=()
  local i wim_val=""
  read -r -a tokens <<< "$raw"
  i=0
  while (( i < ${#tokens[@]} )); do
    case "${tokens[$i]}" in
      --weight-intersection-method|--intersection-method|--im)
        if (( i + 1 < ${#tokens[@]} )); then
          wim_val="${tokens[$((i + 1))]}"
          ((i+=2))
        else
          ((i+=1))
        fi
        ;;
      --weight-intersection-method=*|--intersection-method=*|--im=*)
        wim_val="${tokens[$i]#*=}"
        ((i+=1))
        ;;
      *) ((i+=1)) ;;
    esac
  done
  case "${wim_val,,}" in
    ""|i2|2|prefix-sum|prefix_sum|prefixsum|prefix)                               printf 'prefix-sum' ;;
    i1|1|smaller-side-traversal|smaller_side_traversal|smaller-side|smallerside|legacy) printf 'smaller-side-traversal' ;;
    i3|3|simple-tree-walk|simple_tree_walk|tree-walk)                              printf 'simple-tree-walk' ;;
    i4|4|bitset)                                                                   printf 'bitset' ;;
    *)                                                                             printf '%s' "$wim_val" ;;
  esac
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
  --simphy-outputs-dir Reproducibility mirror root for the small run outputs
                       (default: derived from the resolved SimPhy data root;
                        ".../simphy/data" mirrors into ".../outputs/simphy")
  --no-outputs-mirror  Do not copy results into the outputs mirror
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
  Results remain in <data>/<dataset>/<replicate>/astralx_outputs/<setting>
  and are also mirrored with the dataset's SimPhy .command/.params files to
  <outputs>/astralx_outputs/<dataset>/<replicate>/<setting>.
  --no-time-monitor    Disable time monitoring
  --no-gpu-monitor     Disable GPU monitoring
  --no-notify, -nn     Disable ntfy notifications
  --debug              Enable shell tracing

Examples:
  ./test-astralx-simulated.sh -t 100 -g 100 -r R1 --fresh
  ./test-astralx-simulated.sh -t 100 -g 100 -r R1 --fresh --opts "--search-space S1 --intersection-method I2 -vv"
  The example setting is named search-space_S1__intersection-method_I2.
  Verbosity is ignored; other meaningful options are appended to the name.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --taxa_num|-t) TAXA_NUM="$2"; shift 2 ;;
    --gene_trees|-g) GENE_TREES="$2"; shift 2 ;;
    --replicate|-r) REPLICATE="$2"; shift 2 ;;
    --simphy-dir) SIMPHY_DIR="$2"; SIMPHY_DIR_SET=true; shift 2 ;;
    --simphy-data-dir) SIMPHY_DATA_DIR="$2"; shift 2 ;;
    --simphy-outputs-dir) SIMPHY_OUTPUTS_DIR="$2"; shift 2 ;;
    --no-outputs-mirror) OUTPUTS_MIRROR=false; shift ;;
    --astralx-root|--stelar-root) ASTRALX_ROOT="$2"; ASTRALX_ROOT_SET=true; shift 2 ;;
    --opts|--alg-opts|--astralx-opts|--stelar-opts) ASTRALX_OPTS="$2"; shift 2 ;;
    --base-dir|-b) BASE_DIR="$2"; BASE_DIR_SET=true; shift 2 ;;
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
  if [[ "$BASE_DIR_SET" == true ]]; then
    SIMPHY_DIR="${BASE_DIR%/}/ASTRAL-X/simphy"
  else
    SIMPHY_DIR="${SCRIPT_ROOT}/simphy"
  fi
fi
if [[ "$ASTRALX_ROOT_SET" == false ]]; then
  ASTRALX_ROOT="$SCRIPT_ROOT"
fi
SIMPHY_DIR="$(realpath "$SIMPHY_DIR")"
SIMPHY_DATA_DIR="$(astralx_resolve_simphy_data_dir "$SIMPHY_DATA_DIR" "${SIMPHY_DIR%/}/data")"
if [[ "$OUTPUTS_MIRROR" == true ]]; then
  SIMPHY_OUTPUTS_DIR="$(astralx_prepare_simphy_outputs_dir "$SIMPHY_OUTPUTS_DIR" "$SIMPHY_DATA_DIR")"
else
  SIMPHY_OUTPUTS_DIR="(disabled)"
fi
ASTRALX_ROOT="$(realpath "$ASTRALX_ROOT")"
PYTHON_BIN="${ASTRALX_PYTHON:-${ASTRALX_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python3"

# Copy the current results directory into the reproducibility mirror. A mirror
# problem is reported loudly but never changes the run's own exit status.
mirror_results_dir() {
  local mirrored
  [[ "$OUTPUTS_MIRROR" == true ]] || return 0
  [[ -d "$RESULTS_DIR" ]] || return 0
  if mirrored="$(astralx_mirror_simulated_results "$SIMPHY_DATA_DIR" "$SIMPHY_OUTPUTS_DIR" "$RESULTS_DIR")"; then
    echo "Mirrored outputs to: $mirrored"
  else
    echo "WARNING: outputs mirror was not updated for $RESULTS_DIR" >&2
  fi
}

SETTING_NAME="$(build_setting_name_from_opts "$ASTRALX_OPTS")"
WEIGHT_METHOD="$(extract_weight_method_from_opts "$ASTRALX_OPTS")"

PAIR="${TAXA_NUM}_${GENE_TREES}"
if [[ "$USE_LEGACY_LAYOUT" == true ]]; then
  SIMPHY_RUN_DIR="${SIMPHY_DATA_DIR%/}/${PAIR}/${REPLICATE}"
else
  SIMPHY_RUN_DIR="${SIMPHY_DATA_DIR%/}/t_${TAXA_NUM}_g_${GENE_TREES}_sb_${SB}_spmin_${SPMIN}_spmax_${SPMAX}/${REPLICATE}"
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
SUCCESS_FILE="${RESULTS_DIR%/}/.astralx.success"
OUT_ASTRALX="${RESULTS_DIR%/}/out-astralx.tre"
RUN_LOG="${RESULTS_DIR%/}/.astralx_run.log"
STATS_SIDE_FILE="${OUT_ASTRALX%.tre}_stats.csv"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

if [[ "$FRESH" == false && -f "$STAT_FILE" ]]; then
  PREVIOUS_EXIT=""
  if [[ -f "$STATS_SIDE_FILE" ]]; then
    PREVIOUS_EXIT=$(awk -F, 'NR==2 {print $9}' "$STATS_SIDE_FILE")
  fi
  if [[ -f "$OUT_ASTRALX" && ( -f "$SUCCESS_FILE" || "$PREVIOUS_EXIT" == "0" ) ]]; then
    echo "SKIPPING: successful output already exists at ${OUT_ASTRALX}. Use --fresh to force rerun."
    # Keep the reproducibility mirror complete even for runs finished earlier.
    mirror_results_dir
    exit 0
  fi
  echo "Previous statistics exist but no successful output was recorded; rerunning."
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

    SIM_INC_CMD=("${ASTRALX_ROOT}/sim_incomplete.sh" -t "$TAXA_NUM" -g "$GENE_TREES" -rs "$REPLICATE_COUNT" --sb "$SB" --spmin "$SPMIN" --spmax "$SPMAX")
    if [[ "$SIMPHY_DIR_SET" == true ]];      then SIM_INC_CMD+=(--simphy-dir      "$SIMPHY_DIR");      fi
    SIM_INC_CMD+=(--simphy-data-dir "$SIMPHY_DATA_DIR")
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
      SUCCESS_FILE="${RESULTS_DIR%/}/.astralx.success"
      OUT_ASTRALX="${RESULTS_DIR%/}/out-astralx.tre"
      RUN_LOG="${RESULTS_DIR%/}/.astralx_run.log"
      STATS_SIDE_FILE="${OUT_ASTRALX%.tre}_stats.csv"
    fi

    SIM_CMD=("${ASTRALX_ROOT}/sim.sh" -t "$TAXA_NUM" -g "$GENE_TREES" -r "$REPLICATE" -rs "$REPLICATE_COUNT" --sb "$SB" --spmin "$SPMIN" --spmax "$SPMAX")
    if [[ "$SIMPHY_DIR_SET" == true ]];      then SIM_CMD+=(--simphy-dir      "$SIMPHY_DIR");      fi
    SIM_CMD+=(--simphy-data-dir "$SIMPHY_DATA_DIR")
    if [[ "$FRESH" == true ]];               then SIM_CMD+=(--fresh);                              fi

    "${SIM_CMD[@]}"

    if [[ ! -f "$ALL_GT_FILE" ]]; then
      echo "Error: dataset bootstrap completed but gene-tree file is still missing at $ALL_GT_FILE"
      exit 6
    fi
  fi
fi

mkdir -p "${RESULTS_DIR%/}"
rm -f "$LOCK_FILE" "$SUCCESS_FILE" "$RUN_LOG" "$OUT_ASTRALX"
touch "$LOCK_FILE"

echo "Parameters:"
echo "  taxa_num:       $TAXA_NUM"
echo "  gene_trees:     $GENE_TREES"
echo "  replicate:      $REPLICATE"
echo "  setting:        $SETTING_NAME"
echo "  simphy run dir: $SIMPHY_RUN_DIR"
echo "  results dir:    $RESULTS_DIR"
echo "  outputs mirror: $SIMPHY_OUTPUTS_DIR"
echo "  output tree:    $OUT_ASTRALX"
echo "  stat file:      $STAT_FILE"
echo

CMD=("${ASTRALX_ROOT}/run-astralx-with-monitor.sh" -i "$ALL_GT_FILE" -o "$OUT_ASTRALX" --astralx-root "$ASTRALX_ROOT" --no-notify)
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

if [[ -f "$STATS_SIDE_FILE" ]]; then
  RUNNING_TIME=$(awk -F, 'NR==2 {print $4}' "$STATS_SIDE_FILE")
  MAX_CPU_MB=$(awk -F, 'NR==2 {print $5}' "$STATS_SIDE_FILE")
  MAX_GPU_MB=$(awk -F, 'NR==2 {print $6}' "$STATS_SIDE_FILE")
  OPTIMAL_QUARTET_SCORE=$(awk -F, 'NR==2 {print $7}' "$STATS_SIDE_FILE")
fi

RF_RATE="NA"
if [[ -f "$OUT_ASTRALX" && -f "$TRUE_SPECIES_TREE" ]]; then
  rf_output=$("$PYTHON_BIN" "${ASTRALX_ROOT}/rf.py" "$OUT_ASTRALX" "$TRUE_SPECIES_TREE" 2>&1) || true
  rf_line=$(echo "$rf_output" | grep -i "Robinson-Foulds distance" | tail -n1 || true)
  if [[ -n "$rf_line" ]]; then
    RF_RATE=$(echo "$rf_line" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -n1 || echo "NA")
  fi
fi

echo "alg,setting,num-taxa,gene-trees,replicate,sb,spmin,spmax,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb" > "$STAT_FILE"
echo "astralx,${SETTING_NAME},${TAXA_NUM},${GENE_TREES},${REPLICATE},${SB},${SPMIN},${SPMAX},${RF_RATE},${OPTIMAL_QUARTET_SCORE},${RUNNING_TIME},${MAX_CPU_MB},${MAX_GPU_MB}" >> "$STAT_FILE"

if [[ "$ASTRALX_EXIT_CODE" -ne 0 ]]; then
  rm -f "$LOCK_FILE" "$SUCCESS_FILE"
else
  touch "$LOCK_FILE"
  SUCCESS_TMP="${SUCCESS_FILE}.tmp.$$"
  printf 'exit_code=0\noutput=%s\n' "$OUT_ASTRALX" > "$SUCCESS_TMP"
  mv -f "$SUCCESS_TMP" "$SUCCESS_FILE"
fi

# Complete the command record (out-astralx.command, written by the wrapper with
# the exact run.sh invocation) with the outer commands that produced this run.
COMMAND_FILE="${OUT_ASTRALX%.*}.command"
{
  echo "# --- simulated-run context (test-astralx-simulated.sh) ---"
  echo "# dataset:      $(basename "$(dirname "$SIMPHY_RUN_DIR")")"
  echo "# replicate:    $REPLICATE"
  echo "# setting:      $SETTING_NAME"
  echo "# true tree:    $TRUE_SPECIES_TREE"
  echo "# rf_rate:      $RF_RATE"
  printf '# invoked as:  '
  printf ' %q' "${SCRIPT_ARGV[@]}"
  printf '\n'
  printf '# wrapper cmd: '
  printf ' %q' "${CMD[@]}"
  printf '\n'
} >> "$COMMAND_FILE" 2>/dev/null || echo "Warning: could not append to command record $COMMAND_FILE" >&2

mirror_results_dir

echo
echo "ASTRAL-X finished in ${RUNNING_TIME}s (exit code ${ASTRALX_EXIT_CODE})"
echo "Weight method: ${WEIGHT_METHOD}"
echo "RF rate: ${RF_RATE}"
echo "Quartet score: ${OPTIMAL_QUARTET_SCORE}"
echo "Max CPU RAM (MB): ${MAX_CPU_MB}"
echo "Max GPU VRAM (MB): ${MAX_GPU_MB}"
echo "Wrote stats to $STAT_FILE"

if [[ "$NO_NOTIFY" == false ]] && command -v curl >/dev/null 2>&1; then
  STATUS_EMOJI=$(if [[ $ASTRALX_EXIT_CODE -eq 0 ]]; then echo "✅"; else echo "❌"; fi)
  STATUS_TEXT=$(if [[ $ASTRALX_EXIT_CODE -eq 0 ]]; then echo "completed"; else echo "failed (exit $ASTRALX_EXIT_CODE)"; fi)
  CSV_HEADER="alg,setting,num-taxa,gene-trees,replicate,sb,spmin,spmax,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb"
  CSV_ROW="astralx,${SETTING_NAME},${TAXA_NUM},${GENE_TREES},${REPLICATE},${SB},${SPMIN},${SPMAX},${RF_RATE},${OPTIMAL_QUARTET_SCORE},${RUNNING_TIME},${MAX_CPU_MB},${MAX_GPU_MB}"
  curl -s -d "${STATUS_EMOJI} ASTRAL-X ${STATUS_TEXT} for ${TAXA_NUM} taxa and ${GENE_TREES} gene trees

Weight method: ${WEIGHT_METHOD}
RF Rate: ${RF_RATE}
Quartet score: ${OPTIMAL_QUARTET_SCORE}
Running time: ${RUNNING_TIME}s
Max CPU RAM: ${MAX_CPU_MB} MB
Max GPU VRAM: ${MAX_GPU_MB} MB

${CSV_HEADER}
${CSV_ROW}

Stats: ${STAT_FILE}" "https://ntfy.sh/${NTFY_CHANNEL_NAME}" >/dev/null 2>&1 || true
fi

exit "$ASTRALX_EXIT_CODE"
