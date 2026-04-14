#!/bin/bash
# Build all CUDA JNI shared libraries for ASTRAL-X.
#   native/libastralx_weight.so  -- GPU weight calculation kernel
#   native/libastralx_dp.so      -- GPU cross-tree DP transition search kernel
#   native/libastralx_dist.so    -- GPU distance matrix kernel (Euler tour + RMQ)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"

# GPU architecture.  Override with e.g.  CUDA_ARCH=sm_80 ./build_native.sh
CUDA_ARCH="${CUDA_ARCH:-sm_86}"

mkdir -p "$ROOT/native"

NVCC_FLAGS=(
  -arch="${CUDA_ARCH}"
  -O3
  -Xcompiler '-fPIC'
  --shared
  -I"${JAVA_HOME}/include"
  -I"${JAVA_HOME}/include/linux"
)

echo "=== Building ASTRAL-X native GPU libraries ==="
echo "  JDK         : $JAVA_HOME"
echo "  CUDA arch   : $CUDA_ARCH"

# ── Weight kernel ─────────────────────────────────────────────────────────────
SRC_W="$ROOT/src/native/astralx_weight.cu"
OUT_W="$ROOT/native/libastralx_weight.so"
echo "  Building    : $SRC_W  ->  $OUT_W"
nvcc "${NVCC_FLAGS[@]}" -o "$OUT_W" "$SRC_W"
echo "  OK"

# ── DP cross-tree search kernel ───────────────────────────────────────────────
SRC_DP="$ROOT/src/native/astralx_dp.cu"
OUT_DP="$ROOT/native/libastralx_dp.so"
echo "  Building    : $SRC_DP  ->  $OUT_DP"
nvcc "${NVCC_FLAGS[@]}" -o "$OUT_DP" "$SRC_DP"
echo "  OK"

# ── Distance matrix kernel ────────────────────────────────────────────────────
SRC_DM="$ROOT/src/native/astralx_dist.cu"
OUT_DM="$ROOT/native/libastralx_dist.so"
echo "  Building    : $SRC_DM  ->  $OUT_DM"
nvcc "${NVCC_FLAGS[@]}" -o "$OUT_DM" "$SRC_DM"
echo "  OK"

# ── Similarity matrix kernel ──────────────────────────────────────────────────
SRC_SIM="$ROOT/src/native/astralx_similarity.cu"
OUT_SIM="$ROOT/native/libastralx_sim.so"
echo "  Building    : $SRC_SIM  ->  $OUT_SIM"
nvcc "${NVCC_FLAGS[@]}" -o "$OUT_SIM" "$SRC_SIM"
echo "  OK"

echo "=== Native build complete ==="
echo "Run with:"
echo "  java -Djava.library.path=native -cp build astralx.Main -i <input.tre> --gpu --search-mode full -vv"
