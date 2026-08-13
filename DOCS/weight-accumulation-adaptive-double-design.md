# Adaptive Large-`n` Weight Accumulation (LONG → INT128 / DOUBLE)

## Problem

ASTRAL-X scores each candidate bipartition split by summing, over every
gene-tree tripartition, `frequency · 2·QI`. Both the per-split score and the DP
total were accumulated as signed 64-bit integers (`long` / `long long`). For
large taxon sets these **overflow `long`**:

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

Observed symptom: a **negative "optimal quartet score"** at n=50,000
(two's-complement wraparound). The overflow is end-to-end — CUDA accumulator,
JNI transport, the `WeightTable` score map, and the `long`-based inference DP.

## Decision: keep LONG below threshold; above it use DOUBLE (default) or INT128

When the estimated maximum score would exceed the long-safe range, the whole
scoring + DP pipeline switches to a wider type. Two options, selected by
`--large-n-score-type`:

- **DOUBLE (default)** — 64-bit floating point. Faster in practice on consumer
  GPUs despite the 1/64 FP64 throttle, because INT128 emulation requires more
  instructions per multiply and 2× JNI transport bandwidth.
- **INT128** — exact 128-bit integers, emulated from `__umul64hi` + carries.
  Use when exact integer scores are required (`--large-n-score-type int128`).

Below the threshold, scores are always exact LONG and the existing code path is
**byte-for-byte unchanged**.

### Why DOUBLE is the default (measured performance)

The INT128 emulation turned out slower in practice due to three compounding factors:

- **More PTX instructions per multiply** — `__umul64hi` + add-with-carry is
  several instructions vs one FP64 multiply, even counting the 1/64 throttle.
- **2× JNI transport bandwidth** — INT128 returns 2 longs per split; the host
  buffer, `cudaMemcpy`, and Java unpack step all double in size.
- **Higher register pressure** — the `I128` struct needs two registers per
  accumulator; on a register-heavy reduction kernel this reduces occupancy and
  limits latency hiding.

Empirically the DOUBLE path finishes faster at large n on consumer hardware
(RTX 4090). INT128 remains available for exact-score use cases.

### Why not int128 *everywhere*

Below the threshold `long` is exact and fastest; using int128 there would only
add register pressure for no benefit. Hence the adaptive gate.

## Threshold

`WeightTable.needsDoubleAccumulation(n, numGenes)` (the name predates the int128
option; it gates *any* widening):

```
estMaxTwoScore = numGenes · n⁴ / 12          // ≈ max per-split doubled score
longSafe       = Long.MAX_VALUE / 8          // ≈ 1.153e18, 8× margin for
                                             //   intermediate freq·2QI and partial sums
widen = estMaxTwoScore > longSafe            // → INT128 (default) or DOUBLE
```

For `genes = 1000` this switches at `n ≈ 10,800` (below the hard overflow at
`n ≈ 18,000`, with margin). Computed in `double` to avoid overflow in the check.

**Overrides for testing:** `ASTRALX_WEIGHT_FORCE_DOUBLE=1` (force widen) /
`ASTRALX_WEIGHT_FORCE_LONG=1` (force exact long). The *widened* type is then
INT128 or DOUBLE per `--large-n-score-type`.

## CLI flag

```
--large-n-score-type double    # default — faster in practice on consumer GPUs
--large-n-score-type int128   # exact integer scores (slower; use when needed)
```
(`--large-score-type` is an accepted alias.)

## How the data type is logged

Stated explicitly at multiple points:

1. **Phase-6 decision line** (`WeightTable`), e.g.:
   ```
   Weight accumulation: INT128 (exact 128-bit integer)  [taxa=50000, genes=1000,
     est. max 2·score ≈ 5.21e+20 exceeds long-safe 1.15e+18]  — switched to avoid
     64-bit integer overflow; scores remain exact (full-rate integer math, no FP64
     penalty).  Override with --large-n-score-type double.
   ```
   (DOUBLE and LONG produce analogous lines.)
2. **Native kernel line** (`stderr`): `[ASTRAL-X GPU] weight accumulator: INT128 …` / `DOUBLE …` / `LONG …`.
3. **Weight-table summary**: `… splits scored [INT128] …` / `[DOUBLE]` / `[LONG]`.
4. **Inference objective line**: `Inference DP: optimization-objective quartet score = … [int128]` / `[double]` / `[long]`.

## Implementation map

| Layer | File | Change |
|-------|------|--------|
| 128-bit value type (Java) | `src/astralx/util/Int128.java` | immutable {hi signed, lo unsigned}; `add`, `compareTo`, `halve`, `mulLong` (via `Math.multiplyHigh`), `mulScalar`, `toDouble`, `toString` (BigInteger) |
| Config flag | `src/astralx/Config.java` | `enum LargeScoreType {INT128, DOUBLE}` (default INT128) + getter/setter |
| CLI | `src/astralx/Main.java` | `--large-n-score-type int128\|double` |
| Threshold + 3-way decision + logging | `src/astralx/weight/WeightTable.java` | `Mode {LONG,DOUBLE,INT128}`; `needsDoubleAccumulation`, `nativeScoreMode`, `logAccumulationDecision` |
| Score storage | `WeightTable.java` | parallel `scores`/`scoresD`/`scoresI` maps + per-mode max/total; accessors `getScore`/`getScoreD`/`getScoreI`, `getMode`/`isDouble`/`isInt128` |
| CPU scoring | `WeightTable.java` | `computeScoreI` + `computeTwoQIInt128` (exact 128-bit), alongside long/double variants |
| GPU transport decode | `WeightTable.java` | `unpackTwoScores` — long verbatim / `longBitsToDouble` / `(lo,hi)`→`Int128.halve()` |
| JNI signatures | `src/astralx/gpu/GPUWeightCalculator.java` | `int scoreMode` (0=LONG,1=DOUBLE,2=INT128) on both natives |
| CUDA kernels | `src/native/astralx_weight.cu` | device `I128` struct + helpers (`i128_add`, `i128_mul_u64` via `__umul64hi`, `i128_mul_scalar`); `scoreSplitI128` + `computeWeightsKernelI128<GLOBAL>`; `computeWeightsSmallerSideKernelI128`; `scoreMode` dispatch, 2-wide INT128 transport, scaled batch buffers/memcpy/result arrays |
| Inference DP | `src/astralx/dp/Inference.java` | `solveI` + `dpMemoI` (Int128); `run` 3-way branch; type-tagged score log |
| Verifier | `src/astralx/Phase6Verifier.java` | prints `getMode()`; skips the long-map scan in non-LONG modes |

### Transport formats (single JNI return type)

The native kernels return `long[]`:
- **LONG**: `numSplits` slots, each the exact integer `2·score`.
- **DOUBLE**: `numSplits` slots, each the IEEE-754 bit pattern of `2·score`
  (`__double_as_longlong`; decode with `Double.longBitsToDouble`).
- **INT128**: `2·numSplits` slots — `[2i]` = low (unsigned), `[2i+1]` = high
  (signed). Java rebuilds `new Int128(hi, lo).halve()` (2·score → score).

`0LL` is the bit pattern of `+0.0` and the zero Int128 low word, so the
"invalid split → 0" path is correct in all three modes.

### GPU int128 magnitude budget (why 128 bits suffices and where the wide ops live)

```
ai·bj·ck      ≤ ~2^51   (fits signed 64-bit; computed as a plain long long)
(ai·bj·ck)·su  → ~2^70  (one 64×64→128 via __umul64hi → I128)
2·QI = Σ6 terms, freq·2·QI (i128_mul_scalar), per-split block sum  → all < 2^96
```

So the only 128-bit work per node is six `__umul64hi` + adds and one scalar
multiply — minimal, on full-rate integer units. Even at n=200K / genes=1e4 the
DP total stays far under 2^127.

### Why the exact-LONG path is byte-identical below threshold

`mode == LONG` keeps the original `long` accumulators, `scores` map,
`solve`/`dpMemo`, and the kernel's `long long` instantiation untouched. The
DOUBLE and INT128 paths are strict additions (separate methods, maps, kernels).

## Validation

48-taxon / 500-gene set, all paths, forcing the widened modes
(`ASTRALX_WEIGHT_FORCE_DOUBLE=1` + `--large-n-score-type …`):

| Path | LONG | DOUBLE | INT128 | trees |
|------|------|--------|--------|-------|
| prefix-sum GPU   | 134991678 | 134991678 | 134991678 | byte-identical |
| smaller-side GPU | 134991678 | —         | 134991678 | byte-identical |
| CPU              | 134991678 | 134991678 | 134991678 | byte-identical |

Default over-threshold selects INT128; a normal (sub-threshold) run stays LONG
and is unchanged. At this size all three types are exact (values `< 2^53`), so
the agreement is exact. At very large n, INT128 stays exact while DOUBLE carries
~1e-8 worst-case relative rounding (topology-irrelevant).

## Accuracy / performance notes (large n)

- **INT128**: exact integer scores, no overflow (128 bits ≫ the ~10²⁸ DP total
  even at n=200K). Fast on consumer GPUs (full-rate integer math); modest extra
  register pressure vs LONG.
- **DOUBLE**: ~1e-8 worst-case relative rounding (does not change topology), but
  the reported score becomes floating-point **and** can exceed 2⁶³ — so reading
  it into int64-typed pandas/numpy columns can itself overflow. INT128 prints an
  exact decimal string (also `>2⁶³`), so downstream analysis must treat the
  quartet-score column as a big integer / string, not int64.
- **Determinism**: preserved — GPU block-reduction order and the single-threaded
  DP are deterministic for fixed input.
