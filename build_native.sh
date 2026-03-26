#!/bin/bash
# Build the CUDA JNI shared library for ASTRAL-X GPU weight calculation.
# Output: native/libastralx_weight.so
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src/native/astralx_weight.cu"
OUT="$ROOT/native/libastralx_weight.so"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"

mkdir -p "$ROOT/native"

echo "=== Building ASTRAL-X native GPU library ==="
echo "  Source : $SRC"
echo "  Output : $OUT"
echo "  JDK    : $JAVA_HOME"

nvcc \
  -arch=sm_86 \
  -O3 \
  -Xcompiler '-fPIC' \
  -I"${JAVA_HOME}/include" \
  -I"${JAVA_HOME}/include/linux" \
  --shared \
  -o "$OUT" \
  "$SRC"

echo "=== Native build OK -> $OUT ==="
echo "Run with: java -Djava.library.path=native -cp build astralx.Main -i <input.tre> --gpu -vv"
