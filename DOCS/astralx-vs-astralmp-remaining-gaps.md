# ASTRAL-X vs ASTRAL-MP — Deep Remaining Differences (all ASTRAL-X features ON)

> **Status**: analysis only, no code change.
> **Premise**: assume ASTRAL-X is run with *everything* enabled — `--search-mode full`,
> `--consensus-experimental`, `--autocomplete…`, etc. The question is what genuine
> **algorithmic / detail** differences from ASTRAL-MP *still* remain, beyond config.
> **Method**: read ASTRAL-MP's `WQDataCollection` enrichment path to the leaf level
> (`addExtraBipartitionByHeuristicsLoop` → `sampleAndResolve` → `resolveLinearly` /
> `resolveByDistance`, and `AbstractMatrix.getQuadraticBitsets`) and diff against ASTRAL-X
> `greedy/PolytomyResolver` (Step A / Step B), `EmissionBridge`, `UPGMAClusterer`, `weight/`,
> `tree/TreeParser`.

A "difference in inference" is one of two things — keep them separate:
- **Different quartet score** ⇒ a real search-space (X) or scoring gap → §1, §2.
- **Same score, different topology** ⇒ tie-breaking / nondeterminism → §3.

---

## 1. Deep differences *inside* the polytomy enrichment (the "oversimplified consensus")

Even with `--consensus-experimental`, ASTRAL-X's per-polytomy resolution (`PolytomyResolver`
Step A + Step B) is a **proper subset** of ASTRAL-MP's `addExtraBipartitionByHeuristicsLoop`.
Per polytomy, ASTRAL-MP does (WQDataCollection.java:1206–1260):

1. **Distance pre-resolution** — `speciesMatrix.resolvePolytomy(arms)` = UPGMA on the arm
   groups, add its bipartitions. *(≈ ASTRAL-X Step A.)*
2. **`GREEDY_ADDITION_DEFAULT_RUNS (+adaptive k)` rounds** of `sampleAndResolve`, each round:
   a. `resolveLinearly` — sample one taxon/arm, count induced gene-tree bipartitions, sort by
      frequency, greedily build a tree by LCA insertion, add each accepted bipartition (expanded
      back to arms). *(≈ ASTRAL-X Step B mini-greedy.)*
   b. **`resolveByDistance`** — build the **induced similarity matrix on the sampled reps** and add:
      - `inferTreeBitsets()` (UPGMA on the reps), *(≈ ASTRAL-X `stepBResolveByDistance`)* **and**
      - **`getQuadraticBitsets()`** when `quartetAddition` is set.

### The three concrete things ASTRAL-X omits here

**D1 — Quadratic "nearest-neighbour ball" bitsets — ✅ IMPLEMENTED** (was likely #1 culprit).
`PolytomyResolver.emitQuadraticBalls` now reproduces `getQuadraticBitsets` on the induced
rep matrix: for each arm, the nested k-NN balls (rep-to-rep similarity, descending, index
tie-break) → each expanded to an arm-union **multi-range cluster** via the validated
`emitInducedSplit`. Gated exactly like ASTRAL-MP — `thresholdIndex < 3 && round < STEPB_DEFAULT_RUNS`
(the `childCount ≤ polytomySizeLimit` clause is already enforced by the polytomy pool + the
d≤31 cap). Reuses the `d×d` induced matrix already built in `stepBResolveByDistance` and the
existing multi-range emission path, so generation is `O(d² log d)` per (polytomy, round),
d≤31 — bounded and cheaper per candidate than ASTRAL-MP. Validated: 13/13 regression; all
emissions (with the similarity matrix on, so the balls fire) pass size + ASTRAL-MP
signature-fidelity checks; full GPU pipeline runs end-to-end with the enriched X.
*Original description (for reference):*
`resolveByDistance` adds `inducedMatrix.getQuadraticBitsets()` whenever
`quadratic = (SLOW || (th < GREEDY_DIST_ADDITTION_LAST_THRESHOLD_INDX && j < GREEDY_ADDITION_DEFAULT_RUNS))
&& childCount ≤ polytomySizeLimit` (loop at :1245). With `GREEDY_DIST_ADDITTION_LAST_THRESHOLD_INDX = 3`,
this fires **by default** (no SLOW needed) for the first three thresholds.
`getQuadraticBitsets` (AbstractMatrix.java:98) emits, **for every taxon, the nested prefixes of
its similarity-sorted neighbour list** — i.e. the k-nearest-neighbour cluster for every k — an
`O(reps²)` family of distance-derived candidate clusters.
ASTRAL-X's `stepBResolveByDistance` adds **only** the induced UPGMA tree (`inferTreeBitsets`);
it has **no `getQuadraticBitsets` equivalent anywhere**. → ASTRAL-X's X is missing a whole
distance-derived candidate family around every polytomy.

**D2 — Random resolution of leftover multifurcations — ✅ IMPLEMENTED (opt-in).**
`MiniGreedyBuilder.resolveLeftoverPolytomiesRandomly` + `stepBRound` step (6b): after the
sampled mini-greedy build, for any node still with ≥3 children (incl. the virtual root), add
the complement "rest" and **randomly pair-merge** until two remain, emitting each intermediate
union via the validated `emitInducedSplit`. Gated like ASTRAL-MP on "the round accepted ≥1
cluster" (`anyAccepted`) and behind the opt-in flag `--stepb-random-leftover-resolution`.
Validated: emissions pass size + signature-fidelity; runs on CPU + GPU. *Original description:*
After the greedy build, ASTRAL-MP (resolveLinearly :1441–1473) walks the sampled greedy tree and,
for any node still with ≥3 children, **randomly pairs children** and adds the resulting unions to X
(`forceresolution || added`). This injects extra, randomly-resolved candidate bipartitions.
ASTRAL-X's `MiniGreedyBuilder` emits only the **accepted laminar** clusters — it does **not**
randomly resolve the leftover polytomies of the sampled tree. → another missing candidate set.

**D3 — Adaptive-round criterion differs.**
ASTRAL-MP counts a round "productive" iff it added a bipartition with `freq/|trees| ≥
GREEDY_ADDITION_MIN_RATIO (0.01)` **and** `freq > GREEDY_ADDITION_MIN_FREQ (5)` (:1421–1426);
each productive round grants `+2` rounds up to `GREEDY_ADDITION_MAX (100)`.
ASTRAL-X grants the bonus when a round adds `≥ STEPB_MIN_FREQ (5)` **new signatures** to the buffer
(local-novelty). Different trigger ⇒ different number of rounds ⇒ different emitted set
(compounds with the nondeterminism in §3).

### Smaller detail to verify (probably matched)
- **Step A exactness**: ASTRAL-MP's `resolveByUPGMA(arms, original=true)` vs ASTRAL-X's `g×g`
  group-similarity `MiniUPGMA`. Both UPGMA on averaged similarity; verify the `original=true`
  semantics (it seeds the dendrogram with the arm clusters themselves) and the linkage/tie-break
  match, else the emitted dendrogram bipartitions differ.

---

## 2. Deep differences in X construction *outside* the polytomy enrichment

**D4 — Global `getQuadraticBitsets` (`addExtraBipartitionByDistance`, SLOW / addExtra≥2 only).**
ASTRAL-MP in SLOW mode adds `speciesMatrix.getQuadraticBitsets()` over **all** species — the
nearest-neighbour balls around every taxon — straight into X (WQDataCollection.java:1039–1046).
ASTRAL-X has no equivalent. *Only matters if ASTRAL-MP is run in SLOW (`-x`/addExtra≥2);* if the
ASTRAL-MP baseline is default (addExtra=1), this one is moot — but **D1 (the per-polytomy quadratic)
is on by default and is the bigger one.**

**D5 — UPGMA species-tree `ST` exactness.**
ASTRAL-MP's `ST = buildTreeFromClusters(speciesMatrix.inferTreeBitsets())` = its own UPGMA; its
bipartitions enter X (steps 6 & 8). ASTRAL-X adds a UPGMA guide tree (`UPGMAClusterer`) — but the
**linkage rule and tie-breaking** of `UPGMAClusterer.build` vs ASTRAL-MP's `SimilarityMatrix.UPGMA`
may not be identical → a slightly different `ST` → different guide bipartitions. Worth a direct
tree-vs-tree diff on a shared similarity matrix.

**D6 — Which trees feed the greedy consensus + `secondRoundSampling`.**
For **single-individual** data ASTRAL-MP's `allGreedies[gt] = [relabelled gene tree]` and
`secondRoundSampling = 1`, so its enrichment input ≈ ASTRAL-X's (gene trees directly) — **matched**.
For **multi-individual** data they diverge sharply (see D8).

---

## 3. Scoring / search / determinism (deep)

**D7 — Gene-tree polytomies (d-partition QI).** *Feature-independent; cannot be turned on.*
`TreeParser` rejects any non-root multifurcation. ASTRAL-MP scores a multifurcating gene-tree node
as a `Polytomy` d-partition. If the input gene trees are **not fully binary** (estimated/BS-collapsed
trees usually aren't), ASTRAL-X either can't run or forces an arbitrary resolution → a genuinely
different quartet signal. Designed in `polytomy-design.md`; **the largest gap whenever inputs aren't
binary.**

**D8 — Multiple individuals per species.** *Feature-independent.* ASTRAL-X has no `SpeciesMapper` /
`gtToSt` / single-individual sampling. On multi-allele datasets it cannot reproduce ASTRAL-MP at all;
on single-individual datasets this is a non-issue.

**D9 — Search-space equivalence (full mode).** With the same X, ASTRAL-X Mode 1 + Mode 2 should
enumerate the same `A→B|R` (both halves in X) splits as ASTRAL-MP's X-constrained DP. This is
*believed equivalent* but unverified — worth a differential check: same X ⇒ identical transition set
⇒ identical optimum. If X differs (D1–D6), the optima differ regardless.

**D10 — Tie-breaking & nondeterminism.** Equal optimal score reached by different topologies is
**benign** and expected. ASTRAL-X's consensus sampling is itself nondeterministic across runs
(parallel Step B + GPU similarity float-ordering; see `multi-range-cluster-design.md`). So even
two ASTRAL-X runs can differ. **Always compare the quartet *score* first.**

---

## 4. What's confirmed matched (no gap)

- Gene-tree **completion** (four-point / similarity) — previously simplified, now matched.
- **Binary QI weight** (2·QI, LONG/DOUBLE/INT128, CPU + full-GPU incl. multi-range clusters).
- **Greedy-consensus laminar construction** (7 thresholds) and the **mini-greedy** core of Step B,
  plus multi-range emission into X (validated for signature fidelity, GPU==CPU).
- **O(d log d) Step B restriction.**
- The **DP search machinery** itself (given a fixed X).

---

## 5. Ranked suspects for the residual *score* difference (single-individual, all features on)

1. **D1 — per-polytomy quadratic `getQuadraticBitsets`** (default-on for low thresholds): a whole
   nearest-neighbour candidate family ASTRAL-X never adds. **Most likely.**
2. **D2 — random leftover-polytomy resolution** in `resolveLinearly`: extra candidate bipartitions
   ASTRAL-X doesn't emit.
3. **D7 — gene-tree polytomies** if the input trees are not fully binary (then this dominates).
4. **D5 — UPGMA `ST` exactness** and **D3 — adaptive-round criterion**: shift the emitted set.
5. **D4 — global quadratic bitsets** only if the ASTRAL-MP baseline is SLOW (`-x`).

All of D1–D5 make ASTRAL-X's **X a subset** of ASTRAL-MP's, so ASTRAL-X's optimum score is
**≤** ASTRAL-MP's — the classic "we're missing candidate bipartitions" signature.

---

## 6. Cheap empirical localization (no code)

- Log the **final quartet score** from both tools on a shared dataset. Equal ⇒ §3 tie-break only;
  ASTRAL-X lower ⇒ a real X gap (D1–D5/D7).
- `--dump-clusters` (ASTRAL-X) vs an ASTRAL-MP cluster dump → count/diff X. The **size gap** and
  *which* clusters are missing pinpoints D1 (NN-ball shapes) vs D2 (random unions) vs D5 (ST).
- Confirm the gene trees are **fully binary** (else D7) and **single-individual** (else D8).
- Re-run ASTRAL-X twice → if its *own* result varies, factor out D10 before chasing the rest.
