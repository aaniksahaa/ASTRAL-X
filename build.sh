#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
BUILD="$ROOT/build"

echo "=== Building ASTRAL-X ==="
rm -rf "$BUILD"
mkdir -p "$BUILD"

find "$SRC" -name "*.java" > /tmp/astralx_src.txt
javac -d "$BUILD" -sourcepath "$SRC" @/tmp/astralx_src.txt
rm /tmp/astralx_src.txt

echo "Build OK -> $BUILD"
echo "Run: java -cp $BUILD astralx.Main -i <input.tre> -vv"
