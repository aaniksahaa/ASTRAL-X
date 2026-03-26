#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: ./run-astralx.sh -i <input.tre> -o <output.newick> [--intersection wavelet|cpu] [--weight-mode gpu|cpu] [--unrooted] [-m N] [--seed S]"
  exit 1
fi

# Compile sources (works with or without ripgrep)
if command -v rg >/dev/null 2>&1; then
  mapfile -t JAVA_FILES < <(rg --files -g '*.java' src/astralx)
else
  mapfile -t JAVA_FILES < <(find src/astralx -type f -name '*.java' | sort)
fi

if [[ ${#JAVA_FILES[@]} -eq 0 ]]; then
  echo "No Java source files found under src/astralx"
  exit 1
fi

javac "${JAVA_FILES[@]}"

# Try building GPU runner if not present and nvcc exists.
if [[ ! -x src/cuda/astralx_weight_precompute ]] && command -v nvcc >/dev/null 2>&1; then
  echo "GPU runner not found. Attempting to build src/cuda/astralx_weight_precompute ..."
  set +e
  ./src/cuda/build-weight-kernel.sh "${ASTRALX_CUDA_ARCH:-sm_86}"
  BUILD_RC=$?
  set -e
  if [[ ${BUILD_RC} -ne 0 ]]; then
    echo "GPU runner build failed; Java layer will use CPU fallback."
  fi
fi

# Forward args to main
java -cp src astralx.Main "$@"
