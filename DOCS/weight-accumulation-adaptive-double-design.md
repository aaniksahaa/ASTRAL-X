# Adaptive `long` → `double` Weight Accumulation (Large-`n` Overflow Fix)

## Problem

ASTRAL-X scores each candidate bipartition split by summing, over every
gene-tree tripartition, `frequency · 2·QI`, where `QI` is the quartet-intersection
score of the 3×3 intersection matrix. Both the per-split score and the DP total
were accumulated as signed 64-bit integers (`long` in Java, `long long` in CUDA).

For large taxon sets these values **overflow `long`**:

```
per-split 2·score  ≈  genes · C(n,4) · 2  ≈  genes · n⁴ / 12
```

| n       | genes | est. max 2·score | fits in long (max 9.2e18)? |
|---------|-------|------------------|----------------------------|
| 1,000   | 1000  | 8.3e13           | yes                        |
| 10,000  | 1000  | 8.3e17           | yes                        |
| ~18,000 | 1000  | ~9e18            | threshold                  |
| 50,000  | 1000  | 5.2e20           | **NO → wraps negative**    |
| 100,000 | 1000  | 8.3e21           | **NO**                     |

The observed symptom was a **negative "optimal quartet score"** at n=50,000
(two's-complement wraparound). The overflow is end-to-end: the CUDA accumulator,
the JNI transport, the `WeightTable` score map, and the `long`-based inference DP
all overflow.

## Decision: adaptive switch to `double`

When the estimated maximum score would exceed the long-safe range, the entire
scoring + DP pipeline switches from exact 64-bit integers to 64-bit floating
point (`double`). Below the threshold, the exact-integer path is **unchanged**.

### Why `double` (and not int128 / scaling)

- **`double`**: 64-bit (no memory change), ~2⁻⁵³ ≈ 1e-16 relative precision, no
  overflow until ~1e308. The rounding error is **topologically irrelevant** — a
  DP decision only flips when two competing resolutions' totals differ by less
  than ~1e-8 relative, i.e. at statistically-tied/unsupported nodes whose exact
  answer is itself arbitrary. GPU FP64 throughput is lower, but at large `n` the
  weight kernel is memory-bound (bottleneck = loading the orderings/invIndex
  prefix data), so the arithmetic cost is largely absorbed.
- **int128 (rejected)**: exact, but requires emulated 128-bit in the innermost
  kernel loop, `(hi,lo)` JNI transport, an `Int128` type threaded through the DP
  hot loop (no Java operator overloading → 1.5–3× slower DP), and pushes a
  `> 2⁶³` integer into the CSV that overflows int64 again in downstream
  pandas/numpy analysis. Its only advantage (exactness) does not change the trees.
- **scaling to long (rejected)**: marginal headroom; reported score becomes a
  scaled quantity.

See the conversation rationale; this doc records the chosen `double` design.

## Threshold

`WeightTable.needsDoubleAccumulation(n, numGenes)`:

```
estMaxTwoScore = numGenes · n⁴ / 12          // ≈ max per-split doubled score
longSafe       = Long.MAX_VALUE / 8          // ≈ 1.153e18, 8× margin for
                                             //   intermediate freq·2QI and partial sums
useDouble = estMaxTwoScore > longSafe
```

For `genes = 1000` this switches at `n ≈ 10,800` (comfortably below the hard
overflow at `n ≈ 18,000`, with margin to spare). The estimate is computed in
`double` to avoid overflow in the check itself.

**Overrides for testing:**
- `ASTRALX_WEIGHT_FORCE_DOUBLE=1` — force the double path.
- `ASTRALX_WEIGHT_FORCE_LONG=1` — force the exact-integer path.

## How the data type is logged

The active numeric type is stated explicitly, at three points:

1. **Phase 6 decision line** (`WeightTable`):
   ```
   Weight accumulation: DOUBLE (64-bit floating point, ~15-16 significant digits)
     [taxa=50000, genes=1000, est. max 2·score ≈ 5.21e+20 exceeds long-safe 1.15e+18]
     — switched to avoid 64-bit integer overflow; scores are approximate but
       topologically equivalent.
   ```
   or, below the threshold:
   ```
   Weight accumulation: LONG (exact 64-bit integer)
     [taxa=1000, genes=1000, est. max 2·score ≈ 8.33e+13 within long-safe 1.15e+18].
   ```
2. **Native kernel line** (`stderr`): `[ASTRAL-X GPU] weight accumulator: DOUBLE …` / `LONG …`.
3. **Weight-table summary**: `… splits scored [DOUBLE] …` / `[LONG] …`.
4. **Inference score line**: `Inference DP: optimal quartet score = … [double]` / `[long]`.

## Implementation map

| Layer | File | Change |
|-------|------|--------|
| Threshold + decision logging | `src/astralx/weight/WeightTable.java` | `needsDoubleAccumulation`, `estimatedMaxTwoScore`, `longSafeBound`, `logAccumulationDecision` |
| Score storage | `WeightTable.java` | parallel `scoresD` map + `maxScoreD`/`totalScoreD`; `useDouble` flag; `getScoreD`/`getMaxScoreD`/`getTotalScoreD`/`isDouble` accessors |
| CPU scoring | `WeightTable.java` | `computeScoreD` + `computeTwoQIDouble` (double mirrors of the exact methods); `computeScoresCPU` branches |
| GPU transport decode | `WeightTable.java` | `unpackTwoScores` — long verbatim, or `Double.longBitsToDouble` in double mode |
| JNI signatures | `src/astralx/gpu/GPUWeightCalculator.java` | added `boolean useDouble` to both native methods |
| CUDA kernels | `src/native/astralx_weight.cu` | `scoreSplit<ACC>`, `computeWeightsKernel<GLOBAL,ACC>`, `computeWeightsSmallerSideKernel<ACC>`; `storeTwoScore` overloads; `jboolean` param on both JNI entry points; launch/attribute/occupancy branched on `useDouble` |
| Inference DP | `src/astralx/dp/Inference.java` | parallel `solveD` + `dpMemoD`; `run` branches on `weightTable.isDouble()`; type-tagged score log |
| Verifier | `src/astralx/Phase6Verifier.java` | prints score type; skips the long-map per-split scan in double mode |

### Transport trick (single JNI return type)

The native kernels still return `long[]` (one slot per split = `2·score`). In
LONG mode the slot is the exact integer. In DOUBLE mode the kernel stores the
IEEE-754 **bit pattern** of the floating-point `2·score`
(`__double_as_longlong`), which Java decodes with `Double.longBitsToDouble`.
This keeps one JNI signature and avoids `(hi,lo)` array packing. `0LL` is the bit
pattern of `+0.0`, so the "invalid split → 0" path is correct in both modes.

### Why the exact-integer path is byte-identical below threshold

The `long` accumulators, the `scores` map, `solve`/`dpMemo`, and the kernel's
`long long` instantiation are all untouched. `useDouble` is `false` below the
threshold, so all existing (sub-threshold) runs are bit-for-bit unchanged. The
`double` path is a strict addition (parallel methods, separate maps).

## Validation

On the 48-taxon / 500-gene set, LONG (default) and forced-DOUBLE
(`ASTRALX_WEIGHT_FORCE_DOUBLE=1`) were compared across all code paths:

| Path | LONG score | DOUBLE score | trees |
|------|-----------|--------------|-------|
| prefix-sum GPU      | 134991678 | 134991678 | byte-identical |
| smaller-side GPU    | 134991678 | 134991678 | byte-identical |
| CPU                 | 134991678 | 134991678 | byte-identical |

At this size `double` represents the integer scores exactly (`< 2⁵³`), so the
match is exact. At very large `n` the double score carries ~1e-8 worst-case
relative rounding (realistically ~1e-12 to 1e-10 due to balanced block
reductions), which does not affect tree topology.

## Accuracy notes (large n)

- **Topology (RF rate)**: effectively unaffected. Only edges where competing
  resolutions are within ~1e-8 relative could flip — these are statistically
  unsupported edges whose exact resolution is already arbitrary.
- **Reported quartet score**: becomes floating-point above the threshold.
  Accurate to ~8 significant figures worst case (more in practice). The run
  scripts parse `optimal quartet score = <value>` and store the string verbatim;
  `%.0f` keeps it a plain number for the CSV.
- **Determinism**: preserved — the GPU block reduction order and the
  single-threaded DP are deterministic for fixed input.
