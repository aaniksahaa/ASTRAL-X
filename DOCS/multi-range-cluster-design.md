# Multi-Range Cluster Support in ASTRAL-X — Design Exploration

> **Status**: exploratory design. Not yet scheduled. This documents what multi-range
> clusters would enable, the (surprisingly small) core change, and the real costs/risks,
> so we can decide whether/when to build it.
>
> **Related**: [polytomy-design.md](polytomy-design.md) §5 (search-space enrichment by
> sampling) is the motivating use case; the consensus → cluster pipeline is the other.

---

## 0. The one-sentence version

A cluster is a *set of taxa*; today we can only represent sets that are **one contiguous
range** (or its complement) in some exemplar tree's postorder. Multi-range support lets a
cluster's exemplar be a **union of disjoint ranges** `[l₁,r₁) ∪ … ∪ [l_d,r_d)`, which lets
us materialize *arbitrary* taxon sets — at the cost that every per-cluster operation
(hashing, intersection, membership) walks `d` ranges instead of 1.

---

## 1. Motivation — what forces this

Two heuristics both want to add taxon sets to X that are **not** any single tree's subtree:

1. **Polytomy sampling enrichment** ([polytomy-design.md](polytomy-design.md) §5).
   ASTRAL-MP resolves a polytomy's arms by sampling and expanding restricted bipartitions
   back to *unions of arms* (e.g. `arm₀ ∪ arm₂`). In a tree's postorder the arms are
   contiguous but **non-adjacent**, so their union is multi-range.

2. **Consensus → cluster extraction.** When we resolve polytomies in our own
   greedy-consensus trees (`greedy/`), the resolutions produce taxon sets that need not be
   contiguous in any input gene tree's ordering.

**A note on Mode 2 (what multi-range does NOT change).** It is tempting to think multi-range
"enriches" Mode 2 (`DPTable.addCrossTreeTransitions`), but it does not. Mode 2 is already
*complete over X*: for every cluster `A` it finds every split `A → B | (A\B)` with both
halves in X. Whether `A\B` is "in X" is a hash lookup of the residual *set*, which depends
only on whether some tree realizes that set contiguously — **independent of how `A` or `B`
are represented**. So multi-range adds zero new splits to Mode 2 over a fixed X. Its only
effect on the DP is *indirect*: if an enrichment step (§5.4) adds new (possibly
non-contiguous) clusters *into* X, Mode 2 — being hash-based — consumes them transparently
with no code change. Materializing residuals that *no tree supports* on purpose would be a
departure from ASTRAL's "X = tree-derived bipartitions" design (and can blow up DP cost),
not a free win; the DP dropping non-X residuals today is correct by design.

---

## 2. The core enabling fact: cluster identity is already range-agnostic

`ClusterHash` (cluster/ClusterHash.java) stores raw per-seed `sums[]` / `xors[]` of the
taxon hashes in the set, with the documented arithmetic:

```
sum(A ∪ B) = sum(A) + sum(B)        xor(A ∪ B) = xor(A) ^ xor(B)   (disjoint A,B)
sum(A \ B) = sum(A) − sum(B)        xor(A \ B) = xor(A) ^ xor(B)   (B ⊆ A)
```

Consequences that make multi-range cheap to bolt on:

- **A multi-range cluster has the *same hash* as any other representation of the same set.**
  So deduplication (`ClusterTable`, `DPTable` transitions), `equals()`, and the whole DP —
  all hash-based — are **completely unaffected**. The range list is only ever a *witness*
  used to recompute hashes/intersections; it is never part of identity.
- **Building a multi-range hash is just disjoint union:**
  `sums[s] = Σⱼ rangeSum(ti, s, lⱼ, rⱼ)`, `xors[s] = ⊕ⱼ rangeXor(ti, s, lⱼ, rⱼ)`.
  No new prefix machinery — just call the existing `PrefixHashArrays.rangeSum/rangeXor`
  once per range and combine. `O(d·m)`.

This is the same insight that made polytomy scoring cheap: the expensive, identity-defining
layer (`ClusterHash`) doesn't care about range structure.

### 2.1 The hash is the abstraction boundary — multi-range slots in *underneath* it

The strongest framing: `ClusterHash` is an intermediate identity layer between "a set of
taxa" and "how that set is laid out as ranges." Everything above it operates on hashes and
is **range-blind**; the exemplar lives below it as a recompute-witness. Concretely, the
Mode 2 residual+lookup ([DPTable.java:193-201](../src/astralx/dp/DPTable.java#L193)) is
unchanged by multi-range:
- `ClusterHash.residual(A, B)` is pure hash arithmetic — no ranges. Valid for *any* A, B
  regardless of representation.
- `clusterTable.contains(residual)` is a hash lookup. It may now return an `Entry` whose
  exemplar is multi-range — which is fine, because the exemplar is only consumed later by
  the (generalized) intersection/scoring code.
- `equals()` compares the full `sums[]`/`xors[]` fingerprint (set-based, not range-based),
  so collisions stay range-agnostic and a multi-range vs single-range witness of the same
  set always compare equal. Multi-range cannot weaken the identity guarantee at any level.

Two corollaries worth stating:
- **Multi-range is a *fallback* representation, never a penalty.** One `Entry` per hash, so
  if a set is realizable as a contiguous subtree in *any* tree, that cheap single-range
  exemplar is what's stored; a multi-range witness is materialized only for sets contiguous
  in *no* tree. (Registration rule: prefer the lower range-count exemplar.) Existing
  single-range clusters keep today's fast path bit-for-bit.
- **Residual+lookup is passive.** Mode 2 never *creates* clusters — it only connects
  existing ones. It will find a multi-range entry only if an enrichment step (§5.4) has
  registered one. The mechanism is untouched; only the *contents* of X change.

---

## 3. What a multi-range cluster is (and the complement subtlety)

**Representation.** A cluster's exemplar becomes:
```
treeIndex
int[] los, int[] ros     // d disjoint ranges [los[j], ros[j]) in tree treeIndex's postorder
boolean complement       // KEEP this flag (see below)
int size
```
The current `Cluster` (single `left,right,complement`) is the special case `d == 1`.

**Complement stays a flag — do NOT try to express it as ranges.** A complement cluster is
`C = S \ (⋃ⱼ rangeⱼ)` where `S` is *all n* taxa. For an **incomplete** exemplar tree, `C`
contains taxa with no position in that tree, so it cannot be written as ranges *in that
tree*. But it never needs to be: every intersection uses the identity
```
|C ∩ M| = |M| − |(⋃ⱼ rangeⱼ) ∩ M| = |M| − Σⱼ |rangeⱼ ∩ M|
```
(valid because `M ⊆ Lg ⊆ S`). So **complement is orthogonal to multi-range**: multi-range
generalizes only the *positive* part `⋃ⱼ rangeⱼ`; the existing `cComp ? size−core : core`
trick carries over with `core = Σⱼ coreIntersect(rangeⱼ)`. This resolves the
"out-of-tree taxa" worry cleanly — no need to require complete exemplars.

---

## 4. Range compression (the interval-merge routine)

When a set's ranges are first computed in an exemplar's postorder, **sort by left endpoint
and merge overlapping/adjacent ranges** (classic interval merge, `O(d log d)`):
```
sort ranges by lo
out = []
for (lo,hi) in sorted:
    if out nonempty and lo <= out.last.hi:   # overlap or touch
        out.last.hi = max(out.last.hi, hi)
    else:
        out.append((lo,hi))
```
Why it matters:
- A connected subtree compresses to **1** range; a complement to ≤ 2 (handled by the flag).
  So compression guarantees we never pay multi-range cost for sets that are "really"
  contiguous — the common case stays `d == 1`.
- It bounds and canonicalizes the witness, keeping per-cluster cost `O(d_min)` where
  `d_min` = number of contiguous *runs* of the set in that ordering.

**Exemplar choice is a free optimization knob.** The run count depends on the ordering:
a set may be 1 run in tree T but 5 runs in tree T′. A set extracted *as a subtree of T* is
1 run in T (pick T → optimal). For synthesized sets (sampling), try a few candidate trees
(e.g. the completed trees) and keep the one with fewest runs. Note this does **not** affect
identity (same hash regardless) — only performance.

---

## 5. Layer-by-layer impact

| Layer | Impact | Notes |
|-------|--------|-------|
| `ClusterHash` | **none** | already set-based; identity/dedup/`equals` unchanged |
| `PrefixHashArrays` | **none** | reuse `rangeSum/rangeXor` per range and combine |
| `Cluster` | **generalize** | `left,right` → `los[],ros[]`; `d==1` is today's fast case |
| `IntersectionCounter` | **generalize** | membership / core-count over `d` ranges (below) |
| `WeightTable` (CPU) | **moderate** | only the candidate A/B sides can be multi-range; the gene-tree partition side stays contiguous |
| `astralx_weight.cu` prefix-sum | **moderate** | `O(d)` only in `buildPrefix`, then amortized; node loop stays O(1) — scales best under multi-range |
| `astralx_weight.cu` smaller-side | **moderate** | one `ssCoreIntersect` per range (walk-smaller preserved); `O(d)` paid **per part**, not amortized |
| `GPUWeightCalculator` / packers | **moderate** | split sides become a range-CSR (two-tier: keep 10-int fast path for `d==1`) |
| `DPTable` | **policy only** | mechanism transparent (hash-based); *which* multi-range sets to add is a separate decision |
| `Inference` | **none** | operates on hashes only |
| Verifiers, `toString`, GPU layout plumbing | **small** | assume single range today |

### 5.1 IntersectionCounter

`coreIntersect` currently walks the smaller of two ranges and tests membership via the
other tree's `positionMap`. Generalized to a multi-range cluster `C = ⋃ⱼ [lⱼ,rⱼ)` in tree
`tC`, intersected with a contiguous gene-tree range `M = [loGT,hiGT)` in `tGT`:

- **Walk the gene-tree range, test membership in C:** for each taxon in `M`, look up its
  position `pos` in `tC` and test `pos ∈ ⋃ⱼ[lⱼ,rⱼ)`. With ranges sorted (from compression),
  that test is `O(log d)` (binary search) or `O(d)` (linear; fine for small d).
  Cost `O(|M| · log d)`.
- **Or walk C's ranges, test membership in M:** `Σⱼ` over `[lⱼ,rⱼ)`, each taxon looked up in
  `tGT.positionMap`, `O(1)` test. Cost `O(|C|)`.

Pick the cheaper, exactly as today. Complement via the `|M| − core` subtract trick (§3).

### 5.2 GPU prefix-sum kernel — overhead is paid once per tree, then amortized

First, the framing the intuition gets right: **a multi-range cluster is still just a union
of ranges**, so every quantity we computed for a single range becomes a sum over `d`
ranges — same operations, same results, `O(d)` factor. Nothing about the algorithm
changes; it widens.

What's special about the prefix-sum kernel is *where* that `O(d)` lands. `buildPrefix`
produces `pA[p] = #leaves among the first p (of gene tree g) that are in cluster A`. The
only change is the per-leaf membership test:
```c
posA = invIndex[aTree*numTaxa + t];          // t's position in A's exemplar tree
inA  = (posA ∈ ⋃ⱼ [lⱼ, rⱼ)) XOR aComp;       // was: (posA ∈ [aLo,aHi)) XOR aComp
```
With A's ranges sorted (from compression), that test is `O(log d)` (binary search) or
`O(d)` (linear; fine for small d). So `buildPrefix` goes from `O(L)` to `O(L·log d)`.

**The hot node loop does not change at all.** Once `pA[]`/`pB[]` exist, every tripartition
intersection is still `a0 = pA[mid] − pA[lo]`, `O(1)` — *regardless of how diffuse A is*.
The prefix arrays have already absorbed the range structure. And the row sum is free:
`lgA = pA[L]` already counts membership correctly for a multi-range A.

So the `O(d·log d)` multi-range cost is paid **once per (split, tree) at prefix-build and
amortized across all of that tree's internal nodes**. A tree with `~L` nodes spreads the
`O(L·log d)` build over `~L` node evaluations → `O(log d)` extra per node, vanishing next to
the per-node arithmetic. This is the exact same "localize the cost to prefix-build"
property that made polytomy scoring cheap.

Other invariants kept:
- **No extra working memory.** `pA`/`pB` are still one count per leaf (sized by
  `maxLeafCount`); multi-range adds nothing to shared/global prefix memory. The only new
  resident data is the per-split range list.
- **A is still anchored to ONE exemplar tree** — multi-range means several ranges *within*
  that one tree's postorder, so `aTree`/`invIndex` addressing is unchanged. (Synthesized
  multi-range clusters use a *complete* exemplar so all their taxa have positions; §3.)
- Only the candidate A/B split sides can be multi-range; the gene-tree partition intervals
  `[lo,mid,hi]` stay contiguous.

Data layout: A/B sides become a small range-CSR per split. Keep a **two-tier** layout
(the polytomy doc's pattern): single-range splits stay on the existing 10-int fast path
bit-identically; only multi-range splits carry a CSR offset and use the generalized
membership test.

### 5.3 GPU smaller-side kernel — same benefit per range, but NOT amortized

Here too the operation just widens to a union of ranges. For `a0 = |M1 ∩ A|` with a
multi-range A, do one `ssCoreIntersect` **per range** and sum:
```c
a0 = 0;
for (j = 0; j < dA; j++)
    a0 += ssCoreIntersect(tGT, m1Lo, m1Hi, aTree, l[j], r[j]);   // walks the smaller side, per range
a0 = aComp ? (sz1 - a0) : a0;
```
The "walk the smaller of the two ranges" benefit is **preserved per range** — each
`ssCoreIntersect(M, [lⱼ,rⱼ))` still iterates `min(|M|, rⱼ−lⱼ)`. Total per intersection is
`Σⱼ min(|M|, |rangeⱼ|)`, i.e. the same work as before times the `O(d)` factor the intuition
predicts. The four core counts `a0,a1,b0,b1` and the row sums `lgA,lgB` all widen the same
way. Complement via the subtract trick (§3).

**The asymmetry to be aware of.** The smaller-side kernel has *no per-tree reuse* — it
streams parts with zero per-thread state — so the `O(d)` is paid **on every part** that
references a multi-range cluster, not amortized. Contrast the prefix-sum kernel, which pays
multi-range cost once per tree and reuses `pA`/`pB` across all that tree's nodes. Practical
consequence:

> If multi-range clusters become common, the **prefix-sum** kernel scales better under
> multi-range (build-once, reuse) while the **smaller-side** kernel pays the `O(d)` per
> part. For the expected case (few multi-range clusters, small `d` after compression) both
> are fine; but a workload heavy in diffuse clusters is an argument to prefer prefix-sum.

Data: a variable-length range descriptor per split side (CSR), again two-tier so `d==1`
splits run the existing 9-int/10-int path unchanged.

### 5.4 DPTable / search-space policy

The DP is hash-based, so splitting a multi-range cluster into multi-range halves is
transparent — `Inference` and the transition maps need no change. The real question is
**policy**: multi-range *enables* materializing arbitrary sets, but we should add them only
where a heuristic explicitly proposes them (sampling, consensus resolution), not
blanket-materialize every non-contiguous residual — otherwise X (and DP cost, which is
`O(#clusters × #splits)`) can blow up. Multi-range is the *mechanism*; the *amount* added is
a deliberate, separate tuning decision.

### 5.0 IMPLEMENTED (current): full two-tier range-CSR — multi-range scored 100% on GPU

The shipped GPU implementation is the **full two-tier range-CSR** (§5.1–§5.5), not a CPU
hybrid: multi-range split sides are scored entirely on the GPU.

- **Per-split range descriptor** `splitRangeMeta[i*4]={aRngOff,aRngCnt,bRngOff,bRngCnt}`
  (offsets in pairs) + a **resident flat** `rangeData` of `[lo,hi]` pairs. A single-range
  side has `cnt==0` → the kernel uses the split's `[lo,hi)` — **byte-identical** fast path.
  Built by `WeightTable.buildSplitRangeData`; batched alongside the splits, `rangeData`
  uploaded once.
- **Prefix-sum kernel**: `buildPrefix` membership becomes "pos in any of `cnt` ranges"
  when `cnt>0` (§5.2); the node loop is unchanged.
- **Smaller-side kernel**: `ssIntersectSide`/`ssRowSum` sum `ssCoreIntersect` over the
  side's ranges when `cnt>0` (§5.3).
- Both numeric accumulators (LONG/DOUBLE; INT128 shares the same membership code).

**Validated.** 13/13 GPU regression bit-identical (single-range packs `cnt==0`). On an
identical X with 36 multi-range clusters / **103 multi-range splits**, both kernels
(prefix-sum and smaller-side) score **bit-identical to CPU** — 3248 splits, 0 mismatches,
in LONG and DOUBLE modes (`WeightKernelCheck`). No CPU correction remains; the only CPU
fallback is the pre-existing "GPU infeasible → full CPU" path.

> The earlier CPU/GPU **hybrid** (zero kernel change, multi-range splits corrected on CPU)
> was the interim delivery; it is now superseded by this full-GPU path.

### 5.5 Interaction with polytomy — does the CSR layout get a "double blow"?

Concern: if we ship **both** polytomy and multi-range, the gene-tree partition (M) side and
the candidate split (A/B) side *both* become variable — does every GPU structure turn
variable and combinatorial?

**The two variabilities are on orthogonal axes:**

| Feature | Variable side | What varies |
|---|---|---|
| Polytomy | partition **M** side | *count* of parts `M₁…M_dM`; each part still **one contiguous range** → poly-partition CSR (boundary list per node) |
| Multi-range | split **A/B** side | *count* of ranges per cluster → split-range CSR |

Different structures, different sides. Whether they collide depends on whether a kernel
consumes them in the same place.

**Prefix-sum kernel — NO double blow (axes consumed in separate phases):**
- `buildPrefix` consumes *only* the A/B axis (per-leaf membership in A → `pA`/`pB`).
  Polytomy plays no role.
- The node loop consumes *only* the M axis (prefix differences at M's boundaries, O(d_M)).
  Multi-range plays no role — `pA`/`pB` have already "forgotten" A's layout.
- So you need `buildPrefix` to handle single-OR-multi-range (one branch) and the node loop
  to handle binary-OR-poly (one branch): **2 + 2 independent branches, not a 2×2 matrix.**
  No shared-memory growth (`pA`/`pB` stay one count per leaf). → strong reason to make
  prefix-sum the primary kernel once both features coexist.

**Smaller-side kernel — the one real interaction:**
- Both variabilities meet in the inner intersection. Getting all `aᵢ = |Mᵢ ∩ A|` for a
  polytomous part (`d_M` child ranges) against a multi-range A (`d_A` ranges) is naively
  `O(d_M · d_A)` sub-walks — the "double blow."
- Avoidable by **binning**: walk one side once, binary-search the other's structure to bucket
  each taxon (e.g. walk A's taxa, find gene-tree position, binary-search which `Mᵢ` interval
  it lands in, increment `aᵢ`). `O((|A|+|M|)·log d)` instead of the product, at the cost of a
  small per-thread `aᵢ[]` accumulator (size `d_M`) and more kernel logic.
- It is **doubly rare** (needs a polytomous node *and* a multi-range candidate on the same
  pair), so accepting the naive cost and benchmarking first is defensible.

**Two things defuse the worry:**
1. **Two-tier on *each* axis.** `(binary M)` and `(single-range A,B)` stay the bit-identical
   fast path; escalate to a variable structure only on the axis that needs it. The common
   case touches neither.
2. **Sequencing.** Polytomy *scoring* (the near-term need) ships with single-range splits
   (`d_A ≡ 1`) → no interaction on either kernel. Multi-range is later/optional (§5.4). So the
   double blow, if it ever appears, is a Phase-2 concern — and even then it's clean on
   prefix-sum.

---

## 6. What it enables

- **Polytomy sampling enrichment** ([polytomy-design.md](polytomy-design.md) §5): add the
  arm-union bipartitions ASTRAL-MP produces. Removes the "non-contiguous, can't represent"
  blocker noted there.
- **Consensus-resolution clusters**: feed resolutions of our own greedy-consensus polytomies
  into X without needing them contiguous in an input tree.
- **Transparent DP/Mode 2 integration** (a property, not a new capability): because the DP
  and Mode 2 are hash-based, any enrichment-added cluster — single- or multi-range — is
  consumed with no code change. Mode 2 is *already* complete over X; multi-range does not
  add splits there (see the §1 note), it only lets enrichment put more sets *into* X.
- **Unified representation**: subtree, complement, and synthesized sets all become "a list
  of ranges (+ complement flag)" — one code path, with `d==1` as the fast special case.

---

## 7. Challenges / risks

1. **Performance.** Membership/intersection go from `O(1)`/`O(range)` to `O(d)`/`O(log d)`
   factors. Mitigation: aggressive range compression (§4) keeps `d` = #runs small; two-tier
   fast path keeps the dominant single-range case at today's speed; the prefix-sum kernel
   confines the cost to prefix-build. **Kernel asymmetry (§5.2/§5.3):** prefix-sum amortizes
   the `O(d)` across a tree's nodes (build once, reuse `pA`/`pB`); smaller-side pays it per
   part (no reuse). A workload heavy in diffuse multi-range clusters favours prefix-sum.
2. **GPU layout complexity.** Both kernels need a variable-length range-CSR for split sides,
   plus the membership search. This is the most code. The polytomy two-tier CSR is the
   template to follow.
3. **Search-space blow-up.** If multi-range clusters are added liberally, `#clusters` grows
   and DP cost `O(#clusters × #splits)` grows with it. Must be added by conservative policy
   only (see §5.4).
4. **Exemplar selection.** Need a routine to compute a set's ranges in a chosen tree and
   pick a low-run exemplar. Doesn't affect correctness (hash is invariant), only speed.
5. **Plumbing that assumes single range.** `Cluster.toString`, GPU packers, any verifier
   that prints `[left,right)`. Mechanical but must be swept (cf. the polytomy doc's
   undercount of `Partition` accessors — grep, don't guess).
6. **No identity/canonicalization risk.** Explicitly *not* a problem: dedup and `equals` are
   hash/set-based, so two range-orderings of the same set already collapse. Compression is
   for performance, not correctness.

---

## 8. Recommendation & phased plan

Multi-range is a **bigger, optional** capability than polytomy scoring — and polytomy
*scoring* does **not** need it (the d-partition QI works without it; see polytomy-design.md
§5). So sequence it **after** polytomy scoring lands, and only if we decide the enrichment
heuristics are worth it.

Phased, regression-safe:

1. **Representation + compression (CPU only).** Generalize `Cluster` to a range list with a
   `d==1` fast path; add the interval-merge routine; multi-range hash builder. Validate:
   building a multi-range hash of a set equals the single-range hash of the same set
   (random sets, bit-identical).
2. **IntersectionCounter (CPU).** Generalize core-count + complement trick. Validate against
   the single-range path on `d==1` (bit-identical) and against a brute-force set
   intersection for `d>1`.
3. **WeightTable CPU.** Allow multi-range A/B sides. Validate: existing binary/polytomy
   tests bit-identical (all clusters still `d==1`).
4. **First consumer.** Wire in *one* heuristic that produces multi-range clusters (consensus
   resolution or polytomy sampling) behind a flag; measure search-space growth and accuracy.
   The concrete, already-half-built first consumer is the **consensus polytomy emission → X
   bridge** — the resolver already emits multi-range descriptors (`greedy/MultiRange`) into an
   `EmissionBuffer`; only the exemplar synthesis is missing. See
   [consensus-emission-and-restriction-optimization.md](consensus-emission-and-restriction-optimization.md) §2.
5. **GPU (both kernels).** Two-tier range-CSR + generalized membership; validate
   `d==1` bit-identical and `d>1` against CPU.

Open questions to settle before building:
- Which heuristic is the first/most valuable consumer — consensus resolution or §5 sampling?
- What is the policy cap on how many multi-range clusters enter X (to bound DP cost)?
- Is the accuracy gain from the enrichment worth the kernel complexity, measured on real
  data? (Profile before committing to the GPU work.)
