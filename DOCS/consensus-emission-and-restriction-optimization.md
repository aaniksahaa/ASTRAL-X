# Consensus Polytomy Emission → X, and the O(d) Restriction Optimization

> **Status**: analysis + plan for the two open items in the consensus/polytomy pipeline.
> **Companion docs**: [astral-x-consensus-design-prospective.md](astral-x-consensus-design-prospective.md)
> (the full Part I/II design), [multi-range-cluster-design.md](multi-range-cluster-design.md)
> (the cluster-side mechanism this depends on).

This doc addresses the two places the consensus pipeline currently stops short:
1. **The emission → X bridge** (the "stuck" point): resolved-polytomy bipartitions are
   produced but not yet usable by the DP. Multi-range clusters are the missing mechanism.
2. **The O(d) restriction optimization**: Step B walks each gene tree in `O(n)`; it can be
   `O(d log d)` per restriction with LCA preprocessing the codebase already has.

---

## 1. Where the pipeline actually stands (grounded)

Implemented and working (Part I + Part II of the prospective design):
- One-pass incremental laminar greedy consensus + 7 threshold snapshots (`GreedyConsensus`,
  `LaminarForest`, `LaminarBuilder`).
- Per-polytomy resolution: **Step A** (UPGMA on the group-similarity matrix) and **Step B**
  (`sampleAndResolve`: per-round rep sampling, induced splits, mini-greedy, resolve-by-distance)
  — `PolytomyResolver.stepA/stepB`.
- Multi-range emission descriptors: `greedy/MultiRange` (union of disjoint ranges into the
  consensus tree's `aCons`) + `ConsensusTree.combineDisjointSigma1/2` (O(m) disjoint-union
  signatures). Each resolution emits an `EmittedBipartition{signature, MultiRange, size, …}`
  into a thread-safe `EmissionBuffer`.

Not yet wired (the two open items):
- **Emission → X.** `EmissionBuffer` javadoc: *"Phase 5 integration of these emissions into
  the global ClusterTable (with exemplars, either by gene-tree lookup or by synthesizing
  multi-range exemplars) happens in a separate later pass."* `Main.java:201-205` gates the
  whole phase behind `--consensus-experimental` and states emission to X *"is not yet wired up."*
- **O(d) restriction.** `PolytomyResolver.stepB` comment: *"Per-tree walk is O(n) … An
  O(d log n) variant via marked-ancestor walks or precomputed LCA is a future optimization."*

---

## 2. The stuck point: emission → X needs multi-range exemplars

### 2.1 What an emission *is*, and why it can't enter X yet

Each `EmittedBipartition` carries:
- `signature` — a `ClusterHash` (set identity), computed from the consensus tree's prefix
  scans; **already deduplicates across gene-tree-derived clusters** (same set → same hash).
- `canonicalSide` — a `MultiRange` (the smaller side, as disjoint ranges into the consensus
  tree's `aCons`).

The DP weight calc, however, scores a candidate cluster by **intersecting it against gene-tree
partitions**, and for that it needs an *exemplar* it can walk — today a single contiguous range
(or complement) in some tree (`cluster/Cluster`). The emitted side is, in general, **contiguous
in no gene tree** (it's a union of polytomy arms). So there is no single-range exemplar to store
→ the emission cannot become a scorable cluster. That is precisely the stall.

### 2.2 Multi-range clusters are the fix — and the emission already speaks that language

Two ways to give an emission a usable exemplar:

1. **Gene-tree lookup (free, no multi-range needed).** If the emitted set's signature already
   matches a cluster in X (some gene tree realizes it contiguously), reuse that single-range
   exemplar. This is the cheap common case — a hash lookup. (Per multi-range-cluster-design.md
   §2.1: identity is the hash; representation is just a witness, so prefer the contiguous one.)

2. **Synthesize a multi-range exemplar (the genuinely new sets).** For a set contiguous in no
   tree, anchor a multi-range exemplar on the **consensus tree** — which is a *complete* tree
   over all `n` taxa (`aCons`), so any set and its complement are expressible as ranges there.
   The emitted `MultiRange` already *is* this descriptor. This is exactly the multi-range
   cluster of [multi-range-cluster-design.md](multi-range-cluster-design.md), with the consensus
   tree as the (complete) exemplar — satisfying that doc's §3 "complete exemplar" requirement.

So the bridge is: **for each `EmittedBipartition`, try (1); else materialize (2).** Everything
downstream (dedup, DP, Mode 2) is hash-based and unaffected (multi-range-cluster-design.md §2.1).

### 2.3 The one concrete integration gap

The weight `IntersectionCounter` consumes a `Tree` (with `postorderArray` + `positionMap`). A
`ConsensusTree` exposes `aCons` (position → taxon) but **not** an inverse `positionMap`
(taxon → position). To use the consensus tree as a weight-calc exemplar we must build that
inverse once per snapshot (`O(n)`), or adapt the multi-range membership test to use it. Minor,
but it must be on the checklist. (Only the snapshot trees that actually contribute a synthesized
exemplar need it.)

### 2.4 Cost note

This bridge is `O(total emissions)` for the gene-tree-lookup case (a hash probe each) plus, for
synthesized clusters, the multi-range hash is *already computed* (it's the signature). The real
cost of multi-range clusters is paid later, in scoring — analyzed in
multi-range-cluster-design.md §5 (CPU) and §5.2/§5.3 (GPU). Polytomy-resolution emissions are
typically a small fraction of X, so the synthesized-exemplar count is bounded.

---

## 3. The O(d) restriction optimization (Step B's hot loop)

### 3.1 What Step B does today — and the actual cost

`stepBRound` (one sampling round of one polytomy):
1. Pick one representative taxon per group → `reps[0..d-1]`  (`O(d)`).
2. **For each gene tree** `collectGeneTreeBitmaps` → `walkCollect`: a **full postorder walk of
   the entire gene tree** (`O(n)`), propagating a `d`-bit rep-membership bitmap and emitting a
   bitmap at each qualifying internal node. This mirrors ASTRAL-MP's `Utils.getBitsets`.
3. Aggregate bitmaps into frequency counts; sort; mini-greedy laminar build; emit accepted
   splits.

The dominant term is step 2: **`O(n)` per gene tree, per round, per polytomy.** Over the whole
phase:
```
Step B  ≈  O( P · R̄ · k · n )
```
where `P` = #polytomies across the 7 snapshots, `R̄` = average adaptive rounds (10–100),
`k` = #gene trees, `n` = #taxa. **This is the most expensive part of the consensus phase** — and
the `n` factor is wasteful, because only `d` reps matter (`d ≤ 31` today, capped by the int
bitmap; `PolytomyResolver.stepB` returns early for `d > 31`).

### 3.2 The correct O(d log d) primitive — and a correction to design §8.4

> **Correction.** Design §8.4 claims the restriction is "sort the `d` positions, sweep with a
> stack — `O(d)`." **Postorder positions alone do not determine the induced topology** — two
> different trees can place the same `d` leaves in the same postorder order. You also need the
> **LCA depths** of consecutive leaves. The right primitive is the *auxiliary tree* (a.k.a.
> *virtual tree* / induced/restricted subtree), which needs an LCA structure.

**Auxiliary-tree construction** of the topology induced on `d` chosen leaves of gene tree `g`:
1. Map each rep to its leaf via `g.positionMap` (`O(d)`).
2. Sort the reps by **Euler first-occurrence** order (`O(d log d)`).
3. For each adjacent pair in that order, compute `LCA` and its **depth** (`O(1)` each via the
   sparse-table RMQ — `O(d)` total).
4. Build the induced tree with a depth-keyed stack over the sorted reps + adjacent LCAs
   (`O(d)`). The induced internal nodes *are* the distinct adjacent-pair LCAs.

Total **`O(d log d)` per (tree, round)** after a **one-time `O(n)` Euler+RMQ preprocess per
tree** (`O(nk)` once for all gene trees). The `log d` comes *solely* from step 2's sort
(putting the reps into the tree's leaf order); steps 3–4 are `O(d)` precisely because LCA is
`O(1)` via the sparse table. Strict `O(d)` would require the reps pre-sorted in this tree's
order (not available per-tree), so the sort stays — but it is irrelevant in practice: `d` is
tiny (`≤ 31`, or `≤ √(50+25n)`), so `log d` is a ~5× constant, and **both `O(d)` and
`O(d log d)` are free of the `n` term, which is the entire win.** **This infrastructure already exists**:
`completion/EulerTourBuilder` builds exactly the Euler tour + sparse-table RMQ for `O(1)` LCA
(currently used by the similarity/distance matrices). It can be reused verbatim.

### 3.3 The multiplicity question — turned out to be a NON-issue (verified)

An earlier worry was that `walkCollect` might emit a bitmap at *every* enclosing internal node
(creating frequency multiplicities the induced tree would lose). Reading the actual code settles
it: `walkCollect` emits **only at binary MERGE nodes** (`legit == 2` — both children carry ≥1
present rep). Those merge nodes are *exactly* the internal nodes of the gene tree restricted to
the present reps, each producing its clade **once**. A "pass-through" node (reps on only one
child) does not emit. So the induced-tree enumeration is **exactly faithful** — no multiplicity
recovery is needed.

**Status: IMPLEMENTED and validated.** `PolytomyResolver.collectGeneTreeBitmapsFast` enumerates
the induced clades via a min-split recursion over consecutive-rep LCA depths (the Cartesian tree
on separator depths), applying the same `2 ≤ sz ≤ d-2` filter and the same gene-tree-root skip
(a merge at Euler depth 0). Selected by `--stepb-restriction dlogd|n` (**default dlogd**); the
O(n) walk remains as the `n` fallback. Validated: **222,000 per-tree checks** comparing fast vs
slow bitmap multisets across 4 inputs (0 mismatches), and end-to-end the two routes give
**identical** emission counts, weight totalScore, and inference score. The per-tree Euler+RMQ
(`EulerTourBuilder.build`) is built once per gene tree and shared read-only across all parallel
tasks.

### 3.4 Complexity: before vs after

| | Per (polytomy, round, tree) | Whole Step B | One-time |
|---|---|---|---|
| **Current** | `O(n)` full walk | `O(P · R̄ · k · n)` | — |
| **Optimized** | `O(d log d)` auxiliary tree | `O(P · R̄ · k · d log d)` | `O(nk)` Euler+RMQ preprocess |

With `d ≤ 31` (today's cap), `d log d ≈ 150`, independent of `n`. So the per-restriction win
grows with `n`: negligible at `n≈100`, ~`n/150`× at large `n` (e.g. ≈ 65× at `n=10⁴`). Since
Step B is the phase's dominant cost and the `O(n)` term is the part that scales badly, this is
the highest-value optimization in the consensus pipeline at scale. (At small `n` the full walk is
genuinely fine — hence it was a reasonable first cut.)

### 3.5 Caveats / interactions

- **The `k` factor stays.** This optimization removes the `n`, not the `k` (we still restrict
  every gene tree each round). Whether all `k` trees are needed — vs a sampled subset, as
  ASTRAL-MP's `addBipartitionsFromSignleIndTreesToX` uses a *single* reference tree — is a
  separate policy question (see consensus-design §8.4 vs the polytomy-design.md §5 finding that
  ASTRAL-MP's `baseTrees` is one UPGMA tree). If accuracy permits a sampled subset, that is a
  multiplicative win orthogonal to the `O(n)→O(d log d)` one.
- **Reuse across rounds/polytomies.** The Euler+RMQ structure is per gene tree and **immutable**,
  so it is built once and reused across all rounds, all polytomies, and all 7 snapshots. Build it
  alongside (or share with) the completion phase's existing tour data.
- **`d > 31`.** Step B currently skips these (int-bitmap limit). The auxiliary-tree method has no
  such limit; if we lift the bitmap to `long[]`/bitset-of-`d`, the same `O(d log d)` applies for
  the size-budget's larger `d` (up to ≈ √(50+25n)).

---

## 4. Recommended sequencing

1. **Unblock first, optimize second.** The emission→X bridge (§2) is what makes the entire
   consensus phase *useful*; the O(d) restriction (§3) only makes an already-working phase
   faster. Do §2 first.
2. **§2 depends on multi-range clusters.** Land the CPU multi-range cluster support
   (multi-range-cluster-design.md, phases 1–3) — the consensus tree is the first real consumer
   (its §4 "first consumer" milestone). Then wire `EmissionBuffer → ClusterTable`: gene-tree
   lookup when possible, synthesized consensus-anchored multi-range exemplar otherwise, plus the
   `ConsensusTree` inverse `positionMap` (§2.3).
3. **§3 is independent and infra-ready.** The O(d log d) restriction reuses `EulerTourBuilder`;
   it can be done any time, with the multiplicity decision (§3.3) made explicitly. Highest value
   at large `n`.
4. **Validate** both against the current `O(n)`/buffer path: the emission *set* (signatures)
   must be unchanged by §3 under the "faithful" multiplicity choice, and the DP result must be
   unchanged by §2 when the emitted sets happen to already be in X.
