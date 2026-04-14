#!/bin/bash
# run_astral.sh — Run ASTRAL-MP from compiled source classes.
# Must be called from the ASTRAL/ root (or via dev.sh from the project root).
#
# Usage:
#   ./run_astral.sh -i INPUT.tre [-o OUTPUT.tre] [other ASTRAL options]
#
# Common ASTRAL options:
#   -i FILE    Input gene trees (required)
#   -o FILE    Output species tree
#   -C         CPU-only mode (no GPU)
#   -T N       Number of CPU threads
#   -G LIST    GPU indices, comma-separated
#   -t N       Branch annotation level (0-4, 8, 10, 16, 32)
#   -q FILE    Score a provided species tree and exit
#   -b FILE    Multi-locus bootstrap file list
#   -e FILE    Extra gene trees to enrich cluster set
#   -f FILE    Extra species trees to enrich cluster set
#   -a FILE    Taxon name mapping file
#
# Memory tuning (env vars or flags):
#   ASTRAL_XMS / ASTRAL_XMX   (default: 4g / 128g)
#   --xms SIZE / --xmx SIZE
#   --java-mem "OPTS"         Override all JVM memory options directly

set -euo pipefail

ASTRAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
MAIN_DIR="${ASTRAL_ROOT}/main"
LIB_DIR="${ASTRAL_ROOT}/lib"

MAIN_JAR="${LIB_DIR}/main.jar"
COLT_JAR="${LIB_DIR}/colt.jar"
JSAP_JAR="${LIB_DIR}/JSAP-2.1.jar"
JOCL_JAR="${LIB_DIR}/jocl-2.0.0.jar"
CLASSPATH=".:${MAIN_JAR}:${COLT_JAR}:${JSAP_JAR}:${JOCL_JAR}"

XMS="${ASTRAL_XMS:-4g}"
XMX="${ASTRAL_XMX:-128g}"
JAVA_MEM_OVERRIDE=""
# Extra JVM options (e.g. -Dastralmp.sim.dump=/tmp/sim.txt)
ASTRAL_JVM_OPTS="${ASTRAL_JVM_OPTS:-}"

# ── Validate environment ────────────────────────────────────────────────────────
if [[ ! -f "${MAIN_DIR}/phylonet/coalescent/CommandLine.class" ]]; then
    echo "Error: compiled classes not found. Run compile_astral.sh first:"
    echo "  cd ${ASTRAL_ROOT} && bash compile_astral.sh"
    exit 1
fi

for jar in "${MAIN_JAR}" "${COLT_JAR}" "${JSAP_JAR}" "${JOCL_JAR}"; do
    if [[ ! -f "$jar" ]]; then
        echo "Error: required JAR not found: $jar"
        exit 1
    fi
done

# ── Show help if no arguments ───────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 -i INPUT.tre [-o OUTPUT.tre] [options]"
    echo "Run with -h for full ASTRAL help."
    exit 0
fi

# ── Parse: resolve file paths and extract JVM options ──────────────────────────
# File flags: ASTRAL passes these directly to java.io.File, which resolves
# relative to the JVM working directory. Since we cd into main/ before running,
# any relative path breaks unless we absolutise it here.
FILE_FLAGS=(-i -o -q -b -e -f -a)

ASTRAL_ARGS=()
i=0
args=("$@")
n=${#args[@]}

while [[ $i -lt $n ]]; do
    arg="${args[$i]}"
    case "$arg" in
        --java-mem)
            JAVA_MEM_OVERRIDE="${args[$((i+1))]}"
            i=$((i+2))
            ;;
        --xms|--Xms)
            XMS="${args[$((i+1))]}"
            i=$((i+2))
            ;;
        --xmx|--Xmx)
            XMX="${args[$((i+1))]}"
            i=$((i+2))
            ;;
        --remove-bipartitions)
            ASTRAL_ARGS+=("$arg")
            # next arg is a file path
            if [[ $((i+1)) -lt $n ]]; then
                ASTRAL_ARGS+=("$(realpath -m "${args[$((i+1))]}")")
                i=$((i+2))
            else
                i=$((i+1))
            fi
            ;;
        *)
            # Check if this is a known file flag (-i, -o, etc.)
            is_file_flag=false
            for ff in "${FILE_FLAGS[@]}"; do
                if [[ "$arg" == "$ff" ]]; then
                    is_file_flag=true
                    break
                fi
            done

            if [[ "$is_file_flag" == true && $((i+1)) -lt $n ]]; then
                ASTRAL_ARGS+=("$arg")
                ASTRAL_ARGS+=("$(realpath -m "${args[$((i+1))]}")")
                i=$((i+2))
            else
                ASTRAL_ARGS+=("$arg")
                i=$((i+1))
            fi
            ;;
    esac
done

# ── Build JVM memory opts ───────────────────────────────────────────────────────
if [[ -n "$JAVA_MEM_OVERRIDE" ]]; then
    JAVA_MEM_OPTS="$JAVA_MEM_OVERRIDE"
else
    JAVA_MEM_OPTS="-Xms${XMS} -Xmx${XMX}"
fi

# ── Run ─────────────────────────────────────────────────────────────────────────
echo "ASTRAL-MP | root: ${ASTRAL_ROOT}"
echo "JVM mem:   ${JAVA_MEM_OPTS}"
echo "Args:      ${ASTRAL_ARGS[*]}"
echo "---"

cd "${MAIN_DIR}"
exec java ${JAVA_MEM_OPTS} ${ASTRAL_JVM_OPTS} \
    -classpath "${CLASSPATH}" \
    phylonet.coalescent.CommandLine \
    "${ASTRAL_ARGS[@]}"
