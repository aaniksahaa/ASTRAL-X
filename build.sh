#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$PROJECT_DIR/src"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Building ASTRAL-X ==="

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Compile Java
echo "Compiling Java sources..."
find "$SRC_DIR" -name "*.java" > /tmp/astralx_sources.txt
javac -d "$BUILD_DIR" -sourcepath "$SRC_DIR" @/tmp/astralx_sources.txt
rm /tmp/astralx_sources.txt

echo "Build successful: $BUILD_DIR"
echo ""
echo "Run with:"
echo "  java -cp $BUILD_DIR astralx.Main -i <input.tre> -o <output.tre> -v"
