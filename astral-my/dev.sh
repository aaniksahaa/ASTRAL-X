#!/usr/bin/env bash
# dev.sh — Compile ASTRAL-MP from source and optionally run it.
# Run from the project root (astral-my/).
#
# Usage:
#   ./dev.sh                               compile only
#   ./dev.sh -i test.tre -o out.tre -C     compile then run
#   ./dev.sh --run-only -i test.tre -C     skip compile, run immediately
#   ./dev.sh --compile-only                only compile
#   ./dev.sh --help
#
# Any flags not listed above are forwarded directly to ASTRAL.
# File paths (-i, -o, etc.) can be relative — they are resolved automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASTRAL_DIR="${SCRIPT_DIR}/ASTRAL"

COMPILE_ONLY=false
RUN_ONLY=false
ASTRAL_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [--compile-only | --run-only] [-i INPUT] [-o OUTPUT] [astral-options...]

  (no mode flag)   compile from source, then run if -i is given
  --compile-only   only compile
  --run-only       skip compilation, run immediately
  --help / -h      this message

ASTRAL options are passed through unchanged (file paths resolved automatically):
  -i FILE    Input gene trees (required to trigger a run)
  -o FILE    Output species tree
  -C         CPU-only mode
  -T N       CPU threads
  -G LIST    GPU indices (comma-separated)
  -t N       Branch annotation level

Examples:
  $0 -i test.tre -o out.tre -C
  $0 -i test.tre -o out.tre -T 8
  $0 --compile-only
  $0 --run-only -i test.tre -o out.tre -C
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compile-only) COMPILE_ONLY=true;  shift ;;
        --run-only)     RUN_ONLY=true;      shift ;;
        --help|-h)      usage; exit 0 ;;
        *)              ASTRAL_ARGS+=("$1"); shift ;;
    esac
done

# ── Compile ─────────────────────────────────────────────────────────────────────
if [[ "$RUN_ONLY" = false ]]; then
    echo "=== Compiling ASTRAL-MP from source ==="
    (cd "${ASTRAL_DIR}" && bash compile_astral.sh)
    echo ""
fi

# ── Run ─────────────────────────────────────────────────────────────────────────
if [[ "$COMPILE_ONLY" = false ]]; then
    if [[ ${#ASTRAL_ARGS[@]} -gt 0 ]]; then
        echo "=== Running ASTRAL-MP ==="
        bash "${ASTRAL_DIR}/run_astral.sh" "${ASTRAL_ARGS[@]}"
    else
        echo "(No ASTRAL arguments given — compile-only.)"
        echo "To run: $0 -i <input.tre> -o <output.tre> [-C] ..."
    fi
fi
