# Recovering Tripartition Dedup via Exemplar-Tree Grouping — Design

> Status: **implemented**. Refines the prefix-sum tree-DP weight kernel
> ([weight-prefix-sum-tree-dp-design.md](weight-prefix-sum-tree-dp-design.md)) by
> putting cross-tree tripartition **deduplication back in** — at *no* extra memory cost,
> reversing the "forgo dedup" decision of that doc's §3.1 / §6.2.

---

## 1. The Insight

The prefix-sum kernel processes, per candidate split, **all** `O(nk)` gene-tree internal
nodes — one tripartition per node, grouped by tree so each tree's leaf prefix sums are
built once. We dropped global dedup because the dedup-preserving variant *appeared* to
need `O(nk)` prefix data live per in-flight split (a unique tripartition "could belong to
any tree", so you'd want every tree's prefix available at once).

**That was the wrong variant.** A deduplicated tripartition still has an **exemplar tree**
(the tree it was first extracted from). If we **group the unique tripartitions by their
exemplar tree** and process tree-by-tree — build that tree's prefix once, score its unique
tripartitions, move on — then only **one tree's prefix is ever live**, exactly like the
current no-dedup path. Dedup is recovered at **O(L) memory, not O(nk)**.

So instead of scoring `totalNodes = Σ_g (L_g − 2)` nodes per split, we score
`numUnique = |PartitionTable|` unique tripartitions per split, each multiplied by its
frequency.

---

## 2. Why It Is Correct

A unique tripartition `(M1|M2|M3)` is scored using only its exemplar tree `g₀`'s prefix.
Two conditions must hold; both do:

1. **Intersections are set-based.** `|Mᵢ ∩ A|` depends only on the taxon sets, not on which
   tree realizes them. The prefix difference `prefixA[mid]−prefixA[lo]` over `g₀` gives the
   right count.

2. **The row sum `lgA = |A ∩ Lg|` is consistent across all trees sharing the tripartition.**
   Here `Lg = M1∪M2∪M3`. Because `M3 = Lg \ sub(u)`, the tripartition's hash **encodes Lg**
   — two tripartitions from different trees dedup-match only if they have the *same* `Lg`.
   The exemplar gives `prefixA[L_{g₀}] = |A ∩ L_{g₀}| = |A ∩ Lg|`, valid for every tree that
   shares it (they all have that same `Lg`). ✓

Hence the full 3×3 matrix and `QI` are identical for all `f = frequency` occurrences, and
the contribution is exactly `f · QI`. The result is **bit-identical** to summing over every
node individually:

```
Σ_{all nodes}  QI(node)   ≡   Σ_{unique parts}  frequency · QI(part)
```

This is the same identity the GPU/CPU paths already rely on for the deduped CPU scorer.

---

## 3. What Shrinks, What Doesn't (Honest Accounting)

Per-split work = **prefix-building** + **node-scoring**.

| Term | Before (no dedup) | After (exemplar dedup) |
|---|---|---|
| Node-scoring | `totalNodes` = `O(nk)` | **`numUnique`** (≤ totalNodes) |
| Prefix-building | `Σ_g O(L_g)` = `O(nk)` | `Σ_{g : u_g>0} O(L_g)` — skips exemplar-empty trees |
| Device memory (`nodeData`) | `totalNodes·3` ints | **`numUnique·(3+1)`** ints (smaller) |

- **Node-scoring** drops by the dedup ratio `numUnique / totalNodes`. This is the main win —
  and node-scoring carries the heavy per-node arithmetic (the 6-permutation `QI`), so it is
  a large fraction of kernel time.
- **Prefix-building** is the `O(nk)` floor and is *not* reduced in general (you still scan
  each exemplar tree's full leaf array). It only shrinks for trees that end up
  **exemplar-empty** (`u_g = 0`) — those skip the prefix build entirely. With first-seen
  exemplar assignment most trees keep ≥1 unique tripartition, so the floor largely remains.
- **Never worse:** if dedup ratio ≈ 1 (no sharing), `numUnique ≈ totalNodes`; we've added
  only one `frequency` multiply per node and a smaller-or-equal `nodeData`. No regression.

### When it helps

- **Small-n / large-k** (e.g. 50–200 taxa × thousands of loci): high tripartition sharing →
  large node-scoring reduction → real speedup. Common phylogenomics regime.
- **Large-n** (tens of thousands of taxa): tripartition coincidences are rare → low dedup →
  little gain, and that regime is prefix-dominated anyway (and runs the global-memory path).
  No harm, just little benefit there.

**Measure first:** `PartitionTable` already logs `"<candidates> -> <unique> tripartitions"`.
That ratio is the achievable node-scoring reduction; check it on the target dataset.

---

## 4. Implementation

The deduplicated tripartitions are **already computed** every run — `PartitionTable`
(Phase 4) builds them and the CPU scorer uses them. The GPU path currently *ignores* that
table and re-derives all nodes by raw tree traversal. This change makes the GPU path reuse
`PartitionTable`, so it is a cleanup as much as an optimization.

### 4.1 Host: build the CSR from `PartitionTable`, grouped by exemplar tree

Replace `buildNodeCSR(partTrees)` (raw traversal, all nodes) with a bucket-by-exemplar pass
over `partTable.entries()`:

```
nodeOffset[g+1] = count of unique parts whose exemplar.treeIndex == g   (then prefix-sum)
for each unique entry e (exemplar p = e.exemplar):
    g   = p.treeIndex
    pos = cursor[g]++                       // bucketed write position
    nodeData[3·pos + 0] = p.leftStart       // lo
    nodeData[3·pos + 1] = p.leftEnd         // mid (= p.rightStart)
    nodeData[3·pos + 2] = p.rightEnd        // hi
    nodeFreq[pos]       = e.frequency
partLeafCount[g] = partTrees.get(g).leafCount
maxLeafCount     = max L_g over trees with u_g > 0   // only non-empty exemplars need a prefix
```

`NodeCSR` gains an `int[] nodeFreq` (length `numUnique`). `totalNodes` becomes `numUnique`.

Sizing `maxLeafCount` over only **non-empty** exemplar trees can lower the shared-memory
requirement (and may keep larger inputs on the fast shared path).

### 4.2 Kernel: frequency multiply + skip empty trees

`scoreSplit()` changes minimally:

```
for each gene tree g:
    nbeg = nodeOffset[g];  nend = nodeOffset[g+1]
    if (nbeg == nend) continue;             // exemplar-empty → no prefix needed (uniform skip)
    build prefixA, prefixB over g  (unchanged)
    for ni in [nbeg, nend) striped over threads:
        ... derive 3×3, twoQI (unchanged) ...
        threadAccum += (long long) nodeFreq[ni] * twoQI     // NEW: weight by frequency
```

The `continue` is uniform across the block (all threads see the same `nodeOffset`), so it
skips `buildPrefix`'s `__syncthreads` collectively — no divergence, no deadlock.

Both kernel variants (`<false>` shared, `<true>` global pool) take the new `nodeFreq`
pointer; nothing else in the shared/global adaptive machinery changes.

### 4.3 JNI

Add one `int[] nodeFreq` parameter to `computeWeightsGPU` (host uploads it alongside
`nodeData` as static, per-batch-independent data).

---

## 5. Correctness & Regression Plan

- Scores must be **bit-identical** to the pre-change kernel (the dedup identity is exact).
  Validate against the independent Python verifier on TC1–TC13 (GPU/CPU × local/full) and
  against the prior GPU/CPU output on the 200-tree/37-taxon dataset.
- Exercise both prefix backends: shared path (small inputs) and forced global path
  (`ASTRALX_WEIGHT_FORCE_GLOBAL=1`) — both must still match.
- Log the dedup ratio (`totalNodes` vs `numUnique`) so the achieved reduction is visible.

---

## 6. Future Refinement (Optional, Not Implemented)

The prefix-building `O(nk)` floor could be attacked by choosing exemplar assignments that
**minimize the number of distinct exemplar trees** (so more trees become skippable). Each
unique tripartition can be served by any tree that contains it, so this is a **minimum set
cover** (NP-hard; greedy gives a log-factor approximation). Potential upside: fewer full
prefix builds. Likely marginal and added complexity — left as a note, not a plan.
