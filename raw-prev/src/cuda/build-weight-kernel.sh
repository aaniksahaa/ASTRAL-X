#!/usr/bin/env bash
set -euo pipefail

# Build the CUDA GPU weight precompute runner.
# Output binary: ./src/cuda/astralx_weight_precompute

ARCH="${1:-sm_86}"
nvcc -O3 -arch="${ARCH}" src/cuda/astralx_weight_precompute.cu -o src/cuda/astralx_weight_precompute
nvcc -O3 -arch="${ARCH}" src/cuda/astralx_search_lookup.cu -o src/cuda/astralx_search_lookup
nvcc -O3 -arch="${ARCH}" src/cuda/astralx_search_space_build.cu -o src/cuda/astralx_search_space_build
echo "Built src/cuda/astralx_weight_precompute with arch=${ARCH}"
echo "Built src/cuda/astralx_search_lookup with arch=${ARCH}"
echo "Built src/cuda/astralx_search_space_build with arch=${ARCH}"
