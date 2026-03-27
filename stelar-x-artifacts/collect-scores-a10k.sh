#!/usr/bin/env bash
# collect-scores-a10k.sh
# Collects Quartet and Triplet scores (raw and normalized) for A10K dataset.

set -euo pipefail

# --- Configuration ---
DATA_DIR=""
START_REP=""
END_REP=""
FRESH=false
STELAR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Helper Functions ---

print_help() {
  cat <<EOF
collect-scores-a10k.sh - Collect Quartet and Triplet Scores

Usage: $0 --data-dir <dir> --start-rep <N> --end-rep <M> [options]

Required:
  --data-dir           Path to dataset root (e.g., /.../10k-astral-dataset)
  --start-rep, -sr     Start replicate number
  --end-rep, -er       End replicate number

Optional:
  --fresh              Force score recalculation even if stat file exists
  --help, -h           Show this message
EOF
}

# Parse Arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --start-rep|-sr) START_REP="$2"; shift 2 ;;
    --end-rep|-er) END_REP="$2"; shift 2 ;;
    --fresh) FRESH=true; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" || -z "$START_REP" || -z "$END_REP" ]]; then
  echo "Error: --data-dir, --start-rep, and --end-rep are required."
  print_help
  exit 1
fi

DATA_DIR=$(realpath "$DATA_DIR")
SIMPHY_DIR="${DATA_DIR%/}/10k-simphy"
ASTRAL_OUT_DIR_ROOT="${DATA_DIR%/}/10k-astral-outputs"

if [[ ! -d "$SIMPHY_DIR" ]]; then
  echo "Error: 10k-simphy dir not found at $SIMPHY_DIR"
  exit 1
fi

echo "=== A10K Score Collection ==="
echo "Data Dir:  $DATA_DIR"
echo "Replicates: R${START_REP} to R${END_REP}"
echo "Fresh:      $FRESH"
echo "============================="

# Function to calculate normalized triplet score
calc_norm_triplet_score() {
    local score="$1"
    local taxa="$2"
    local trees="$3"
    
    if [[ -z "$score" || -z "$taxa" || -z "$trees" || "$score" == "NA" ]]; then
        echo "NA"
        return
    fi
    
    # Remove decimals
    local n=$(echo "$taxa" | awk '{print int($1)}')
    local k=$(echo "$trees" | awk '{print int($1)}')
    
    if [[ "$n" -ge 3 && "$k" -gt 0 ]]; then
        # nC3 = n * (n-1) * (n-2) / 6
        local trips=$(echo "$n * ($n - 1) * ($n - 2) / 6" | bc)
        # Norm = Score / (k * trips)
        awk -v score="$score" -v k="$k" -v trips="$trips" 'BEGIN { printf "%.6f", score / (k * trips) }'
    else
        echo "NA"
    fi
}

# --- Main Loop ---

for i in $(seq "$START_REP" "$END_REP"); do
    REPL="R${i}"
    REPL_SIMPHY_DIR="${SIMPHY_DIR}/${REPL}"
    REPL_ASTRAL_DIR="${ASTRAL_OUT_DIR_ROOT}/${REPL}" # For ASTRAL trees
    
    if [[ ! -d "$REPL_SIMPHY_DIR" ]]; then
        echo "Skipping missing replicate: $REPL"
        continue
    fi
    
    echo "Processing $REPL..."

    # Define algorithms and tree types to process
    # Format: "ALG:TREE_TYPE"
    # ASTRAL trees are in 10k-astral-outputs/R?/estimatedgenetre.gtr.tre-gpu241GPU (example)
    # STELAR trees are in 10k-simphy/R?/stelar_outputs/{estimated,true}/out-stelar.tre
    
    VARIANTS=("stelar:estimated" "stelar:true" "astral:estimated" "astral:true")
    
    for VARIANT in "${VARIANTS[@]}"; do
        ALG="${VARIANT%%:*}"
        TREE_TYPE="${VARIANT##*:}"
        
        # 1. Locate Species Tree
        SPECIES_TREE=""
        BASE_OUT_DIR="" # Where to save stats
        
        if [[ "$ALG" == "stelar" ]]; then
            BASE_OUT_DIR="${REPL_SIMPHY_DIR}/stelar_outputs/${TREE_TYPE}"
            SPECIES_TREE="${BASE_OUT_DIR}/out-stelar.tre"
        elif [[ "$ALG" == "astral" ]]; then
             # ASTRAL location logic is a bit tricky based on user example.
             # "data/10k-astral-dataset/10k-astral-outputs/R1/estimatedgenetre.gtr.tre-gpu241GPU"
             # If TREE_TYPE is true, it might be different. But usually ASTRAL is run on estimated.
             # Let's assume standard ASTRAL output names if they exist.
             if [[ "$TREE_TYPE" == "estimated" ]]; then
                 # Try to find the file. It seems variable.
                 # User example: estimatedgenetre.gtr.tre-gpu241GPU
                 # We will look for *.tre (excluding cleaned ones)
                 SPECIES_TREE=$(find "$REPL_ASTRAL_DIR" -maxdepth 1 -name "*estimatedgenetre*" ! -name "*-cleaned*" ! -name "*.log" | head -n1 || true)
                 BASE_OUT_DIR="$REPL_ASTRAL_DIR"
             else
                 # "true" ASTRAL might not exist in standard dataset, but let's check
                 SPECIES_TREE="" 
             fi
        fi
        
        if [[ -z "$SPECIES_TREE" || ! -f "$SPECIES_TREE" ]]; then
            # Silent skip if file doesn't exist (maybe not run yet)
            continue
        fi
        
        # Ensure output dir exists for stats
        mkdir -p "$BASE_OUT_DIR"
        STAT_FILE="${BASE_OUT_DIR}/stat-${ALG}-score.csv"
        
        if [[ "$FRESH" == "false" && -f "$STAT_FILE" ]]; then
            echo "  [$ALG-$TREE_TYPE] Stats exist. Skipping."
            continue
        fi
        
        # 2. Locate Gene Trees
        GENE_TREES=""
        if [[ "$TREE_TYPE" == "estimated" ]]; then
            GENE_TREES="${REPL_SIMPHY_DIR}/estimatedgenetrees/estimatedgenetrees.tre"
            # Prefer rooted for Triplet Score if available
            GENE_TREES_ROOTED="${REPL_SIMPHY_DIR}/estimatedgenetrees/estimatedgenetrees.rooted.tre"
        else
            GENE_TREES="${REPL_SIMPHY_DIR}/truegenetrees"
            GENE_TREES_ROOTED="$GENE_TREES" # True trees usually don't need explicit rooting file diff
        fi
        
        if [[ ! -f "$GENE_TREES" ]]; then
            echo "  [$ALG-$TREE_TYPE] Gene trees not found. Skipping."
            continue
        fi
        
        echo "  [$ALG-$TREE_TYPE] Calculating scores..."
        
        # --- Triplet Score ---
        T_SCORE="NA"
        T_NORM_SCORE="NA"
        
        # Prepare Species Tree for Triplet Scoring (Clean if ASTRAL)
        ST_FOR_TRIPLET="$SPECIES_TREE"
        CLEAN_TEMP=""
        
        if [[ "$ALG" == "astral" ]]; then
            CLEAN_TEMP="${SPECIES_TREE}-cleaned-temp.tre"
            python3 "${STELAR_ROOT}/clean.py" -i "$SPECIES_TREE" -o "$CLEAN_TEMP" >/dev/null 2>&1
            ST_FOR_TRIPLET="$CLEAN_TEMP"
        fi
        
        # Use rooted gene trees for estimated, if available
        GT_FOR_TRIPLET="$GENE_TREES"
        if [[ "$TREE_TYPE" == "estimated" && -f "$GENE_TREES_ROOTED" ]]; then
            GT_FOR_TRIPLET="$GENE_TREES_ROOTED"
        fi
        
        T_LOG=$(mktemp)
        "${STELAR_ROOT}/get_triplet_score_stelar.sh" -i "$GT_FOR_TRIPLET" -st "$ST_FOR_TRIPLET" 2>&1 | tee "$T_LOG" || true
        
        # Parse Triplet Scores
        # OPTIMAL_TRIPLET_SCORE: 12345.0
        T_SCORE=$(grep "OPTIMAL_TRIPLET_SCORE:" "$T_LOG" | tail -n1 | awk -F: '{print $2}' | tr -d ' ' || echo "NA")
        
        # Calc Normalized
        # Need Taxa and Trees count
        TAXA_CNT=$(grep "Real taxa count (dataset):" "$T_LOG" | tail -n1 | awk -F: '{print $2}' | tr -d ' ' || echo "NA")
        TREES_CNT=$(grep "Trees:" "$T_LOG" | head -n1 | awk -F: '{print $2}' | tr -d ' ' || echo "NA")
        
        if [[ "$T_SCORE" != "NA" && "$TAXA_CNT" != "NA" && "$TREES_CNT" != "NA" ]]; then
            T_NORM_SCORE=$(calc_norm_triplet_score "$T_SCORE" "$TAXA_CNT" "$TREES_CNT")
        fi
        
        rm -f "$T_LOG"
        if [[ -n "$CLEAN_TEMP" && -f "$CLEAN_TEMP" ]]; then
            rm -f "$CLEAN_TEMP"
        fi

        # --- Quartet Score ---
        Q_SCORE="NA"
        Q_NORM_SCORE="NA"
        
        Q_LOG=$(mktemp)
        "${STELAR_ROOT}/get_quartet_score_astral.sh" -i "$GENE_TREES" -st "$SPECIES_TREE" 2>&1 | tee "$Q_LOG" || true
        
        # Parse Quartet Scores
        # "Final quartet score is: 12345"
        # "Final normalized quartet score is: 0.12345"
        Q_SCORE=$(grep "Final quartet score is" "$Q_LOG" | awk '{print $NF}' || echo "NA")
        Q_NORM_SCORE=$(grep "Final normalized quartet score is" "$Q_LOG" | awk '{print $NF}' || echo "NA")
        rm -f "$Q_LOG"
        
        # Write CSV
        echo "dataset,alg,replicate,tree_type,triplet_score,norm_triplet_score,quartet_score,norm_quartet_score" > "$STAT_FILE"
        echo "a10k,${ALG},${REPL},${TREE_TYPE},${T_SCORE},${T_NORM_SCORE},${Q_SCORE},${Q_NORM_SCORE}" >> "$STAT_FILE"
        echo "    -> Saved to $STAT_FILE"
        
    done
done

echo "=== Aggregating Results ==="
MERGED_CSV="${DATA_DIR}/a10k_scores_merged.csv"
echo "dataset,alg,replicate,tree_type,triplet_score,norm_triplet_score,quartet_score,norm_quartet_score" > "$MERGED_CSV"

# Find all stat-*-score.csv files in data dir and append (skipping headers)
find "$DATA_DIR" -name "stat-*-score.csv" -print0 | sort -z | xargs -0 -I {} tail -n +2 {} >> "$MERGED_CSV"

echo "Merged scores saved to: $MERGED_CSV"
echo "Done."
