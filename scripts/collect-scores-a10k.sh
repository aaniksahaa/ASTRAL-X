#!/usr/bin/env bash
# collect-scores-a10k.sh
# Aggregates per-replicate ASTRAL-X run stats from the A10K layout.

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_ROOT}/scripts/a10k-outputs-dir.sh"

DATA_DIR=""
START_REP=""
END_REP=""
OUTPUTS_DIR=""
OUTPUTS_MIRROR=true

print_help() {
  cat <<EOF
collect-scores-a10k.sh

Usage: $0 --data-dir <dir> --start-rep <N> --end-rep <M> [--outputs-dir <dir>] [--no-outputs-mirror]

Merges every 10k-simphy/R<n>/astralx_outputs/**/stat-astralx.csv into
<dir>/a10k_astralx_scores_merged.csv and copies that CSV into the outputs
mirror (default: <parent>/outputs/<dataset>, e.g.
\$PHYLOGENY_DATA_DIR/10k-astral-dataset -> \$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --start-rep|-sr) START_REP="$2"; shift 2 ;;
    --end-rep|-er) END_REP="$2"; shift 2 ;;
    --outputs-dir|--a10k-outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --outputs-dir=*|--a10k-outputs-dir=*) OUTPUTS_DIR="${1#*=}"; shift ;;
    --no-outputs-mirror) OUTPUTS_MIRROR=false; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" || -z "$START_REP" || -z "$END_REP" ]]; then
  echo "Error: --data-dir, --start-rep and --end-rep are required."
  exit 2
fi

DATA_DIR="$(realpath "$DATA_DIR")"
MERGED_CSV="${DATA_DIR}/${ASTRALX_A10K_MERGED_CSV_NAME}"
echo "alg,setting,replicate,tree_type,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb" > "$MERGED_CSV"

for i in $(seq "$START_REP" "$END_REP"); do
  while IFS= read -r -d '' stat_file; do
    tail -n +2 "$stat_file" >> "$MERGED_CSV"
  done < <(find "${DATA_DIR}/10k-simphy/R${i}/astralx_outputs" -type f -name 'stat-astralx.csv' -print0 2>/dev/null | sort -z)
done

echo "Merged A10K ASTRAL-X stats saved to: $MERGED_CSV"

# Keep the reproducibility mirror in step with the merged CSV. A mirror problem
# is reported but never fails the collection.
if [[ "$OUTPUTS_MIRROR" == true ]]; then
  if OUTPUTS_DIR="$(astralx_prepare_a10k_outputs_dir "$OUTPUTS_DIR" "$DATA_DIR")" &&
     astralx_mirror_a10k_merged_csv "$DATA_DIR" "$OUTPUTS_DIR"; then
    echo "Mirrored merged stats to: ${OUTPUTS_DIR}/${ASTRALX_A10K_MERGED_CSV_NAME}"
  else
    echo "WARNING: outputs mirror was not updated with the merged CSV" >&2
  fi
fi
