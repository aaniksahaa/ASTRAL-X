#!/usr/bin/env bash
# collect-scores-a10k.sh
# Aggregates per-replicate ASTRAL-X run stats from the A10K layout.

set -euo pipefail

DATA_DIR=""
START_REP=""
END_REP=""

print_help() {
  cat <<EOF
collect-scores-a10k.sh

Usage: $0 --data-dir <dir> --start-rep <N> --end-rep <M>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --start-rep|-sr) START_REP="$2"; shift 2 ;;
    --end-rep|-er) END_REP="$2"; shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" || -z "$START_REP" || -z "$END_REP" ]]; then
  echo "Error: --data-dir, --start-rep and --end-rep are required."
  exit 2
fi

DATA_DIR="$(realpath "$DATA_DIR")"
MERGED_CSV="${DATA_DIR}/a10k_astralx_scores_merged.csv"
echo "alg,setting,replicate,tree_type,rf-rate,optimal-quartet-score,running-time-s,max-cpu-mb,max-gpu-mb" > "$MERGED_CSV"

for i in $(seq "$START_REP" "$END_REP"); do
  while IFS= read -r -d '' stat_file; do
    tail -n +2 "$stat_file" >> "$MERGED_CSV"
  done < <(find "${DATA_DIR}/10k-simphy/R${i}/astralx_outputs" -type f -name 'stat-astralx.csv' -print0 2>/dev/null | sort -z)
done

echo "Merged A10K ASTRAL-X stats saved to: $MERGED_CSV"
