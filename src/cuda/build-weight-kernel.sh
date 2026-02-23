#!/usr/bin/env bash
set -euo pipefail

# Build the CUDA GPU weight precompute runner.
# Output binary: ./src/cuda/astralx_weight_precompute

ARCH="${1:-sm_86}"
nvcc -O3 -arch="${ARCH}" src/cuda/astralx_weight_precompute.cu -o src/cuda/astralx_weight_precompute
echo "Built src/cuda/astralx_weight_precompute with arch=${ARCH}"
