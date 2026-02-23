# ASTRAL-X Minimal Codebase (Java + CUDA Hooks)

This `src` directory contains a clean, modular implementation of the workflow described in `prompt-astral-x.md`.

## Implemented pipeline

1. Parse rooted binary Newick gene trees.
2. Preprocess each tree into:
   - postorder leaf arrays padded with `-1`
   - per-taxon position maps
   - subtree leaf ranges
3. Build seeded taxon hashes (`SplitMix64`) with `m` replicates.
4. Build per-tree prefix hash scans (sum and XOR modulo `2^64`).
5. Extract clusters (`X`) as range/complement descriptors with hash-based deduplication and frequencies.
6. Add global super-complements (`S - A`) for every cluster.
7. Extract and deduplicate gene-tree tripartitions with frequencies.
8. Build DP search space: for each cluster `A`, find `B|C` such that `B U C = A`, `B ∩ C = ∅`.
9. Precompute candidate bipartition weights (GPU-first with CPU-parallel fallback).
10. Run inference DP and output an inferred species tree in Newick.

## Layout

- `astralx/Main.java` : end-to-end orchestrator.
- `astralx/parse/*` : Newick parsing and loading.
- `astralx/preprocess/*` : postorder arrays, positions, ranges.
- `astralx/hash/*` : seeded taxon hashes and prefix hash index.
- `astralx/cluster/*` : cluster representation, extraction, dedup table.
- `astralx/partition/*` : generic partition model and unique tripartition table.
- `astralx/dp/*` : search-space builder and inference DP.
- `astralx/weight/*` : weight calculator and GPU runner bridge.
- `cuda/*.cu` : CUDA kernels/runners.

## Build Java

```bash
javac $(find src/astralx -type f -name '*.java' | sort | tr '\n' ' ')
```

## Build GPU runner

```bash
./src/cuda/build-weight-kernel.sh sm_86
```

(Use your architecture if different, e.g. `sm_89`.)

## Run

```bash
java -cp src astralx.Main -i gene_trees.tre -o species_tree.newick
```

Options:

- `--unrooted` : also register unrooted-style complements during extraction.
- `-m`, `--hash-replicates` : number of hash replicates (default `4`).
- `--seed` : random seed for replicate hashes.
- `--intersection wavelet|cpu` : intersection backend (default `wavelet`; CPU fallback is available).
- `--weight-mode gpu|cpu` : weight precompute backend (default `gpu`; if GPU runner/device is unavailable, falls back to CPU-parallel).

## Notes

- Implemented for rooted binary input trees.
- Polytomy-specific partition handling is intentionally left as a planned extension point.
- Per-tree wavelet-matrix intersection acceleration is implemented in Java (`PerTreeWaveletIntersectionCounter`) with CPU fallback.
- GPU weight precompute uses `src/cuda/astralx_weight_precompute` via subprocess and temporary flat buffers.
