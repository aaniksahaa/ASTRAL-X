#!/bin/bash

# get_quartet_score_astral.sh
# Script to calculate quartet score using ASTRAL
# Usage: ./get_quartet_score_astral.sh -i <gene_tree_file> -st <species_tree_file> [-o <output_file>]

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Default values
INPUT_FILE=""
SPECIES_TREE=""
OUTPUT_FILE="/dev/null" 

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -st|--species-tree|-s|-q)
            SPECIES_TREE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -i <gene_tree_file> -st <species_tree_file> [-o <output_file>]"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$INPUT_FILE" || -z "$SPECIES_TREE" ]]; then
    echo -e "${RED}Error: Missing required arguments.${NC}"
    echo "Usage: $0 -i <gene_tree_file> -st <species_tree_file> [-o <output_file>]"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' does not exist.${NC}"
    exit 1
fi

if [[ ! -f "$SPECIES_TREE" ]]; then
    echo -e "${RED}Error: Species tree file '$SPECIES_TREE' does not exist.${NC}"
    exit 1
fi

# Get absolute paths (required because run_astral.sh changes directory)
INPUT_FILE=$(realpath "$INPUT_FILE")
SPECIES_TREE=$(realpath "$SPECIES_TREE")
OUTPUT_FILE=$(realpath -m "$OUTPUT_FILE") # -m to allow missing file

# Locate ASTRAL runner script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTRAL_RUNNER="${SCRIPT_DIR}/baselines/ASTRAL/run_astral.sh"

if [[ ! -f "$ASTRAL_RUNNER" ]]; then
    echo -e "${RED}Error: ASTRAL runner script not found at $ASTRAL_RUNNER${NC}"
    exit 1
fi

echo -e "${GREEN}Running ASTRAL scoring...${NC}"
echo "Gene Trees:   $INPUT_FILE"
echo "Species Tree: $SPECIES_TREE"
echo "Output Log:   (stdout/stderr)"
echo "Output Tree:  $OUTPUT_FILE"
echo "================================================="

# Execute ASTRAL
# We use -q option of ASTRAL to score the species tree
# NOTE: run_astral.sh expects to be run from its own directory
RUNNER_DIR=$(dirname "$ASTRAL_RUNNER")
TEMP_LOG=$(mktemp)

pushd "$RUNNER_DIR" > /dev/null
# Run and pipe to tee to show output + save to file
./run_astral.sh -i "$INPUT_FILE" -q "$SPECIES_TREE" -o "$OUTPUT_FILE" 2>&1 | tee "$TEMP_LOG"
EXIT_CODE=${PIPESTATUS[0]}
popd > /dev/null

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}Scoring completed successfully.${NC}"
    echo "================================================="
    echo -e "${GREEN}SUMMARY SCORES:${NC}"
    grep "Number of quartet trees in the gene trees" "$TEMP_LOG"
    grep "Final quartet score is" "$TEMP_LOG"
    grep "Final normalized quartet score is" "$TEMP_LOG"
    echo "================================================="
else
    echo -e "${RED}Scoring failed with exit code $EXIT_CODE.${NC}"
fi

rm -f "$TEMP_LOG"
exit $EXIT_CODE
