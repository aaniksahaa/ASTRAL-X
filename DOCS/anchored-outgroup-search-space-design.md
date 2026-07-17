# Anchored-Outgroup DP Search Space — Design & Plan

> Status: **IMPLEMENTED** (2026-07-17), enabled by default; disable with `--no-anchor-outgroup`,
> in both **LOCAL and FULL** search modes. Local mode was enabled after making
> tree-local rotation emission complete at leaves: a leaf has no Type-1 split, but
> it induces the valid Type-2 resolution of its complement. Both anchoring layers
> landed and were equality-tested: Layer 1 (single anchored root split,
> `DPTable.applyAnchoredRoot`) and Layer 2 (anchor-free `ClusterTable` registration
> `A`, with-anchor-parent transition skipping `B`, anchor-free emission registration
> `C`). Verified: anchored LOCAL == unanchored LOCAL on TC1–TC16 (including true-local
> polytomy runs), 120 randomized complete/incomplete datasets, and the bundled
> 37-taxon dataset; on the latter, clusters 1200→601 and scored local splits
> 1696→1039 at identical score 23,458,838. Also verified: non-consensus anchored
> full == independent oracle (16/16 GPU+CPU); default unchanged; consensus anchored ==
> unanchored on all 16 TCs; 37-taxon full clusters 1200→601, cross-tree 1301→130 ms,
> scored splits 8433→2612, score identical. The anchored-vs-unanchored score match is
> the correctness gate on any new dataset.
>
> ---
>
> Original plan below. Makes the whole DP
> search space *natively* anchored at one fixed outgroup taxon, so that only the
> anchor-free orientation of every candidate cluster is ever built. This is an exact,
> result-preserving change for unrooted species-tree inference. It supersedes the
> "build both orientations everywhere + all root splits" assumption that currently
> runs through `ClusterTable`, `DPTable`, and the consensus emission bridge. The DP
> (`Inference`), the weight step (`WeightTable` + all four intersection methods), and
> the gene-tree tripartition extraction (`PartitionTable`) are **unchanged** — they
> auto-conform. Gate the rollout on an anchored-vs-unanchored score-equality check.

---

## 1. Motivation

ASTRAL-X infers an **unrooted** species tree (`Config.treatAsUnrooted = true`). Quartet
score is a function of unrooted topology only. Yet the current pipeline builds the DP as
if every rooting were distinct:

- `ClusterTable` registers **both** orientations of every gene-tree bipartition:
  `sub(u)` **and** the super-complement `S\sub(u)` (`ClusterTable.java:15-19`, `:125`, `:128-131`).
- `DPTable.searchRootTransitions` adds **every** complementary root split
  `S → (B, S\B)` for all `B` with `S\B ∈ X` (`DPTable.java:313`).
- Cross-tree ("full" mode) transitions are then enumerated for **every** cluster
  (`DPTable.addCrossTreeCPU`, `:190`), i.e. `O(N²)` over the full — doubled — cluster set.

The inference DP itself (`Inference.solve`, `:106-142`) is top-down and memoized from the
root, and only scores splits of clusters it reaches. But because **all** root
orientations exist, both orientations of nearly every bipartition become reachable. This
was confirmed empirically: the reachability pre-filter (`DPTable.reachableClusters`, added
2026-07-17) reports **1164/1164 clusters reachable = 100%** on the bundled 37-taxon full
run — it prunes nothing, precisely because every rooting is reachable.

The redundant with-anchor orientations are pure duplicate work. On the extended-avian
scale (n = 353, k ≈ 63k) they roughly **double** the candidate splits scored (and
quadruple the `O(N²)` cross-tree pair search), which is a large fraction of the weight
runtime.

## 2. The Core Idea

Fix one **anchor** taxon (the "outgroup"), e.g. taxon id 0. Represent every candidate
species cluster by its **anchor-free** side only — the side not containing the anchor.
The DP is implicitly rooted on the anchor's pendant edge:

```
S  →  ({anchor} | S\{anchor})        (the ONLY root split; weight 0; never scored)
```

and then recurses entirely within the sublattice of clusters `⊆ S\{anchor}`.

### 2.1 Why it is exact (the theorem)

Any unrooted binary tree on `S` can be rooted on the anchor's pendant edge without
changing its unrooted topology, its internal bipartitions, or its quartet score. Under
that rooting:

- Every internal edge's bipartition `(D | S\D)` has a **unique** anchor-free side
  (the anchor lies on exactly one side), which becomes a subtree = a cluster `⊆ S\{anchor}`.
- The top split `{anchor} | S\{anchor}` has an **empty** third side, so its quartet
  weight is exactly 0 — in `WeightTable` terms the `C`-column of the 3×3 matrix is all
  zeros, so `2·QI = 0`.

Hence the set of representable unrooted trees is unchanged, and the maximum quartet score
(and the argmax topology) is identical. This is **not** an approximation or a heuristic
prune of possible trees — it removes only redundant re-rootings.

### 2.2 Why it auto-propagates (the key architectural fact)

Everything downstream of cluster construction only ever consumes *whichever clusters and
splits exist*, and never needs a with-anchor cluster:

- **DP** (`Inference.solve`): takes a cluster `C`, looks up its splits `(A, C\A)`,
  recurses into `A` and `C\A`. If `C ⊆ S\{anchor}` then `A, C\A ⊆ C ⊆ S\{anchor}` too —
  the DP can never leave the anchor-free sublattice on its own.
- **Weight of a split** `(A, B)` under parent `C = A∪B`: the tripartition is
  `(A | B | S\C)`. The with-anchor third side `S\C` is **never materialized as a
  cluster** — it is *derived arithmetically* over each gene tree's leaf set
  (`c_i = |M_i| − a_i − b_i = |M_i ∩ (S\C)|`, `WeightTable.computeScore`). Only the two
  anchor-free halves `A, B` are looked up.
- **Reconstruction** (`Inference.buildNewick`): walks `bestSplits`, all anchor-free.

Therefore, if the invariant is enforced **at cluster registration**, the DP, the weight
step, all four weight-intersection methods (prefix-sum / smaller-side / bitset /
simple-tree-walk), and reconstruction conform automatically with **no changes**. This is
why baking the notion in is a genuine simplification, not a bandage.

## 3. The Invariant

> **X (the candidate cluster set) contains only clusters that do NOT contain the anchor
> taxon.** Each bipartition is stored exactly once, by its anchor-free side. The root
> `S` has exactly one transition, `({anchor}, S\{anchor})`, whose weight is 0 and which
> is never scored. `{anchor}` is a size-1 base case; `S\{anchor}` is the DP's true entry
> point.

Because the anchor is in `S`, it lies on exactly one side of every bipartition, so the
anchor-free side is always unique and well-defined — no "canonical smaller side"
guesswork is needed.

## 4. What Changes (localized) and What Does Not

### 4.1 Changes

1. **`ClusterTable` registration.** Where it currently registers `sub(u)` *and*
   `S\sub(u)`, register only the **anchor-free** one:
   - if `sub(u)` contains the anchor → register the super-complement `S\sub(u)`;
   - else → register `sub(u)`.
   Halves `|X|`.

2. **Consensus emission bridge (`EmissionBridge`).** Register each emission's
   **anchor-free** orientation by the same rule (multi-range + `complement` flag already
   express it; the weight walker already handles complement multi-range —
   `WeightTable.buildClusterBitsInto`). This *retires* the emission orientation bug
   (§6) by construction.

3. **`DPTable` root.** `searchRootTransitions` emits the single split
   `({anchor}, S\{anchor})`; ensure local (Mode 1) extraction and cross-tree (Mode 2)
   never inject other root resolutions. Explicitly register `S\{anchor}` and the
   `{anchor}` hash at setup so the entry point never depends on incidental appearance.

### 4.2 Unchanged (auto-conform)

- `PartitionTable` (gene-tree tripartitions are the *data*; anchoring only changes the
  species-tree *candidates*).
- `DPTable.addCrossTreeCPU/GPU` — same code, now iterating the halved cluster set, so it
  self-reduces to `O((N/2)²) ≈ ¼` the pairs. The `contains(residual)` test still works:
  for an anchor-free parent `A`, every `B, A\B ⊆ A` is anchor-free, so residuals are
  always anchor-free and found in the halved table.
- `WeightTable` and all four intersection methods.
- `Inference` (`solve`/`solveD`/`solveI`, `buildNewick`).
- `DPTable.reachableClusters` (the existing reachability pre-filter) — kept as a
  complementary safety net + diagnostic; it may still trim a few genuinely-orphan
  anchor-free clusters. Should now report ≈100% of the *halved* set.

## 5. Correctness Edge Cases (must be nailed)

### 5.1 The `clusterContainsAnchor` predicate — complement-aware, absence-tolerant

Computed over the **global species set `S`** using the anchor's global taxon id. It must
account for the `complement` flag and for the anchor being **absent from the cluster's
exemplar tree** (`positionMap[anchor] == −1`):

- **Positive single range `[lo,hi)`** in exemplar `t`: contains anchor iff
  `t.positionMap[anchor] ∈ [lo,hi)`. Anchor absent from `t` ⇒ **not** contained.
- **Complement `S\[lo,hi)`**: contains anchor iff `NOT(t.positionMap[anchor] ∈ [lo,hi))`.
  Anchor absent from `t` ⇒ **is** contained (it is in `S`, not in the excluded range).
  *This is the case a naive range-membership check gets wrong.*
- **Multi-range**: union-of-ranges membership, then flip if complemented.

### 5.2 Anchor absent from some gene trees is a non-issue for weighting

The anchor only ever lives in the derived third side `S\C`, computed over each gene
tree's own `Lg` (`c_i = |M_i| − a_i − b_i`). A tree missing the anchor simply never has
it in any `M_i`, so it is never counted — no special handling, and the anchored weight of
any given split is **bit-identical** to the unanchored weight of the same split.
(Anchoring changes *which* splits exist, never how one is scored.)

### 5.3 Anchor choice

- Any fixed leaf is exact. Default: taxon id 0.
- Do **not** require the anchor to be present in all/most gene trees — an incomplete
  dataset may have no universal taxon. Optional robustness: pick the taxon present in the
  most gene trees (a cheap per-taxon presence count); changes nothing about correctness or
  the ~½ reduction, only makes exemplar bookkeeping uniform.
- Multi-individual datasets: ASTRAL-MP anchors on the *smallest X-cluster containing
  individual 0's whole species* (`WQComputeMinCostTaskProducer.java:52`). Single-leaf
  anchoring is exact for the current single-individual model; revisit if/when
  multi-individual is supported.

### 5.4 `S\{anchor}` must exist in X

It is the anchored DP's real entry point (the anchor-free super-complement of the anchor
singleton). Guaranteed present as long as the anchor appears in ≥1 gene tree (it must,
being a taxon), but add an **explicit safeguard** that registers `S\{anchor}` and the
`{anchor}` hash at anchored-root setup.

## 6. Relationship to the Consensus Emission Orientation Bug

Prior analysis (see also `consensus-emission-and-restriction-optimization.md`) noted that
synthesized consensus/multi-range emissions retain only one canonical side. If that side
contains the anchor, the anchor-free side the anchored DP needs is missing → anchored DP
could return a **sub-optimal** score.

Under this design the bug **cannot occur**: the emission rule (§4.1.2) always keeps the
anchor-free orientation. The former special case dissolves into the single global
invariant — "heal the wound" rather than "add a bandage."

## 7. Expected Savings

| Quantity                         | Today (all orientations) | Anchored            |
|----------------------------------|--------------------------|---------------------|
| Candidate clusters `|X|`         | `N`                      | `≈ N/2`             |
| Cross-tree pair search (Mode 2)  | `O(N²)`                  | `O((N/2)²) ≈ N²/4`  |
| Candidate splits scored (weight) | `M`                      | `≈ M/2 … M/3`       |
| Optimal quartet score            | exact                    | **identical**       |

Unlike the reachability pre-filter (which only trims the *weight* step after the fact),
native anchoring shrinks Phases 3–5 (cluster build + cross-tree transitions) as well as
Phase 6 (weight). The ½–⅔ range on scored splits reflects that some anchor-free clusters
were only reachable via with-anchor paths and drop too.

## 8. Rollout Plan (safe, phased)

1. **Flag.** Add `--anchor-outgroup`, then make anchored outgroup the default after
   local/full equivalence testing. Use `--no-anchor-outgroup` to restore the unanchored
   search space. Optionally `--anchor-taxon <id|name>` to override the default anchor.
2. **Implement §4.1 behind the flag**, in order:
   a. `clusterContainsAnchor` predicate (§5.1) + unit-check on positive / complement /
      multi-range / anchor-absent cases.
   b. `ClusterTable` anchor-free registration + explicit `S\{anchor}` / `{anchor}`
      registration.
   c. `DPTable` single anchored root split; verify no other root resolutions leak in.
   d. Emission-bridge anchor-free registration.
3. **Gate: anchored-vs-unanchored score equality.** The anchored optimal quartet score
   MUST equal the unanchored score **bit-for-bit** on TC1–TC16 (local + full) and the
   37-taxon set, and on at least one `--consensus-experimental` run (exercises §4.1.2).
   The theorem guarantees equality; any mismatch is a localized bug (a wrong-orientation
   drop) and points straight at the registration rule.
4. **Diagnostics per run** (log): anchor id + presence count; `|X|` before/after; Mode-2
   pairs before/after; splits scored before/after; the anchored-vs-unanchored score check.
5. **Flip default on** once equality holds everywhere; keep the flag for A/B.

## 9. Risk & Audit Checklist

- [ ] `clusterContainsAnchor` is complement-aware and treats `positionMap[anchor] == −1`
      correctly (§5.1). *(Highest-risk line in the change.)*
- [ ] No code path assumes both orientations of a bipartition exist. Structurally none
      does (§2.2), but audit any `contains(S\C)` / complement lookups outside the DP root.
- [ ] `S\{anchor}` and `{anchor}` are registered even when the anchor is a spotty taxon.
- [ ] Cross-tree residual `contains()` still resolves within the halved table.
- [ ] Reconstruction emits a valid tree; RF comparison treats output as unrooted
      (anchor as an arbitrary outgroup is fine).
- [ ] `--consensus-experimental` run passes the score-equality gate.
- [ ] Reachability pre-filter still runs (should read ≈100% of the halved set).

## 10. Summary

The with-anchor cluster orientations are dead weight that only the artificial root ever
touched, and the with-anchor tripartition side is *derived, not stored*. Enforcing the
single invariant — "every candidate cluster is anchor-free" — at registration makes the
entire downstream pipeline conform automatically, halves the search space and quarters the
cross-tree search, and retires the consensus orientation bug by construction — all while
keeping the exact same quartet score, verifiable by a bit-for-bit anchored-vs-unanchored
comparison.
