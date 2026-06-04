# Weight Calculation via Per-Tree Prefix Sums (Tree-DP) — Design

> Status: **design only**, not yet implemented. This document supersedes the inner
> loop of `astralx_weight.cu` / `WeightTable.computeScore`. The batching machinery,
> the 3×3 matrix derivation (`DESIGN/intersection-optimization.md`), and the QI formula
> are all retained unchanged. Only *how the 4 core intersection counts are obtained*
> changes.

---

## 1. The Problem with the Current Inner Loop

The weight calculation is the dominant cost of ASTRAL-X. The current kernel
(`computeWeightsKernel`) is:

```
for each candidate split  x = (A | B | C):        // 1 GPU thread per split
    for each unique gene-tree tripartition P = (M1 | M2 | M3):   // loop, deduped
        a0 = |M1 ∩ A|   a1 = |M2 ∩ A|             // 4 core intersections,
        b0 = |M1 ∩ B|   b1 = |M2 ∩ B|             //   each via coreIntersect()
        derive remaining 5 entries of the 3×3 matrix
        2QI += freq(P) * twoQI(matrix)
```

Each `coreIntersect(M_range, cluster_range)` (`astralx_weight.cu:47`) **iterates element
by element over the smaller of the two ranges**, looking each taxon up in the other
tree's `invIndex`. This is the bottleneck inside the bottleneck.

Two problems with this:

1. **The smaller-side cost is not O(1).** We proved that, *assuming balanced binary gene
   trees*, always walking the smaller side keeps the total at roughly `O(n²k²)`. But:
   - There is a real constant factor (a scattered `invIndex` global load + branch per
     element).
   - The balance assumption is not something we control. A caterpillar / ladder gene
     tree makes the smaller-side sum degrade toward `O(n)` per node, i.e. `O(n²)` per
     (split, tree), pushing the whole computation toward `O(n³k²)` in the worst case.

2. **Tripartitions of one tree are treated as independent.** The internal nodes of a
   single gene tree are *deeply nested* intervals. We currently throw that structure
   away (we dedup tripartitions globally and re-derive every intersection from scratch),
   re-walking overlapping ranges over and over.

The idea below removes the element-by-element walk entirely: every one of the 4 core
intersections becomes a genuine **O(1)** lookup, and the per-(split, tree) cost becomes
**exactly O(L) regardless of tree shape** (`L` = number of leaves in that tree).

---

## 2. Core Insight: A Gene-Tree Tripartition Is a Range, So Intersection Is a Prefix-Sum Difference

### 2.1 The key structural fact

A gene-tree tripartition stored by ASTRAL-X is *not* an arbitrary set partition. Because
each gene tree is stored as a **postorder leaf array** with contiguous subtree ranges
(`Tree.postorderArray`, and `PartitionTable` ranges), an internal node `u` of gene tree
`g` corresponds to a contiguous leaf interval

```
[lo, hi)   split by its two children at  mid:
    M1 = sub(left)  = leaves [lo,  mid)
    M2 = sub(right) = leaves [mid, hi )
    M3 = Lg \ sub(u) = leaves [0, lo) ∪ [hi, L)      (everything else in g)
```

(Here `lo = leftStart`, `mid = leftEnd = rightStart`, `hi = rightEnd`,
`L = g.leafCount`; see `PartitionTable.extractNode`.)

### 2.2 Intersection as a counting query over `g`'s leaves

Fix a candidate cluster `A` (one side of the split). For gene tree `g`, define the
**leaf membership indicator** over `g`'s postorder positions:

```
indA[p] = 1  if  the taxon at g.postorderArray[p] belongs to A
        = 0  otherwise          (p = 0 .. L-1)
```

Membership is an **O(1)** test, because cluster `A` is itself a range `[loA, hiA)` (or its
complement) in its *completed* exemplar tree `tA`:

```
taxon t ∈ A   ⇔   ( invIndex[tA][t] ∈ [loA, hiA) )  XOR  complementA
```

`tA` is a completed gene tree (full taxon set), so `invIndex[tA][t]` is always valid — no
"missing taxon" special case in the membership test. This is exactly the lookup the
current kernel already does, just one element at a time.

Now build the **prefix sum** of the indicator:

```
prefixA[0] = 0
prefixA[p+1] = prefixA[p] + indA[p]        // O(L) to build, once per (split, tree)
```

Then **every** intersection count for **every** internal node of `g` is an O(1)
difference:

```
|M1 ∩ A| = prefixA[mid] − prefixA[lo]
|M2 ∩ A| = prefixA[hi]  − prefixA[mid]
|Lg ∩ A| = prefixA[L]                    (= lgA, the row sum — see §2.3)
```

and identically for cluster `B` with a second prefix array `prefixB`. Cluster `C`'s
counts are never computed directly; they fall out of the 3×3 derivation in §2.3.

### 2.3 The 3×3 matrix derivation is unchanged — and incompleteness becomes free

With `a0,a1` (from `prefixA`), `b0,b1` (from `prefixB`), and the part sizes
`sz1,sz2,sz3` (stored in the node), the existing reconstruction
(`DESIGN/intersection-optimization.md`, §4) applies verbatim:

```
lgA = prefixA[L]          // |A ∩ Lg|  — the corrected row sum for incomplete g
lgB = prefixB[L]          // |B ∩ Lg|
a2  = lgA − a0 − a1       // row constraint on A
b2  = lgB − b0 − b1       // row constraint on B
c0  = sz1 − a0 − b0       // column constraint on M1
c1  = sz2 − a1 − b1       // column constraint on M2
c2  = sz3 − a2 − b2       // column constraint on M3
```

**Incomplete gene trees handled for free.** The current code needs a special
`intersectWithFullTree` call to get `lgA` when `g` is incomplete
(`WeightTable.java:355`). Here `lgA` is simply `prefixA[L]` — the last prefix entry — so
the "3×3 from 4 values" machinery needs *no extra work* for missing taxa. The prefix
already counts only the leaves that are present in `g`.

The QI computation (`twoQI`, 6 permutations) and the negative-value skip are unchanged.

### 2.4 Equivalence to the "DP on the tree" idea

The user's framing — "at node X with children Y, Z, `|X∩P| = |Y∩P| + |Z∩P|`, leaves are
O(1)" — is *exactly* this prefix sum. `|sub(u) ∩ A| = prefixA[hi] − prefixA[lo]` is the
subtree sum, and `prefixA[hi]−prefixA[lo] = (prefixA[mid]−prefixA[lo]) +
(prefixA[hi]−prefixA[mid])` is the child-addition recurrence. The prefix-sum form is just
the **flattened, random-access** version of the same DP — and it is the more GPU-friendly
of the two (see §4). Both compute the identical numbers.

---

## 3. Complexity: What We Gain

Let `n` = taxa, `k` = gene trees, `L_g` = leaves of tree `g` (`Σ_g L_g = O(nk)`),
`S` = number of candidate splits (`O(nk)` tree-local, more under full cross-tree search).

| | Current | Prefix-sum tree-DP |
|---|---|---|
| Core intersection cost | `O(min range)` per part, element walk | **`O(1)`** per part |
| Per (split, tree) | `Σ_nodes O(min range)` ≈ `O(L log L)` balanced, `O(L²)` worst | **`O(L)` always** |
| Per split | `O(nk · min-factor)` | **`O(nk)`** |
| Total | `O(n²k² · min-factor)`, balance-dependent | **`O(n²k²)`, guaranteed** |
| Per-leaf inner op | scattered `invIndex` load + branch | one add (after a single membership load during the scan) |
| Incomplete-tree `lgA` | extra full-tree intersection | free (`prefixA[L]`) |

The asymptotic class is the same `O(n²k²)` the user expected — but it is now
**guaranteed independent of tree balance** and carries a much smaller constant: the inner
operation is an integer add over a contiguous, coalescible array instead of a scattered
membership probe.

### 3.1 The dedup trade-off (honest accounting)

The current path deduplicates tripartitions *globally* across all trees (frequency
counts), so it iterates `numUniqueParts ≤ Σ_g (L_g − 1)` parts. The prefix-sum path is
naturally **per-tree** (the prefix is tied to one tree's leaf ordering), so it processes
every internal node of every tree — it forgoes cross-tree tripartition dedup.

- We recover the cheap part of dedup at the **tree level**: collapse identical gene trees
  with an integer multiplicity and multiply that tree's contribution once. (Whole-tree
  duplicates are common when gene trees are bootstrapped/limited.)
- We lose cross-tree *tripartition* dedup (shared clades across otherwise-different
  trees). But each such part now costs O(1) arithmetic instead of an O(min-range) walk,
  so reprocessing duplicates is far cheaper than before. Net expectation: win, especially
  on unbalanced trees; possible mild loss only when dedup ratio is very high *and* trees
  are near-perfectly balanced.
- A dedup-preserving variant exists (§6.2) but costs `O(nk)` prefix storage per in-flight
  split, which breaks the clean memory bound — not recommended as default.

---

## 4. GPU Design (the hard part)

The user's concern is precise: the current "1 thread per split, O(1) state, fully
parallel, clean O(nk) device memory" pattern does not survive a naive port, because a DP
needs O(L) working memory. The resolution is to make that O(L) working memory **transient
shared memory per thread-block**, never a global `O(S × n)` array.

### 4.1 Recommended scheme — one block per split, loop trees inside, prefix in shared memory

```
GRID  : one thread-block per candidate split (batched over splits, §4.4)
BLOCK : e.g. 128 threads, cooperating on one split at a time

per block (split x = (A|B)):
    twoScoreLocal = 0
    for each gene tree g (looped on-device):           // reuse shared buffer per tree
        L = g.leafCount
        # Phase 1 — indicators (coalesced over leaves, O(L/threads))
        for p in [0, L) striped over threads:
            t = orderings[g][p]
            indA = membership(t, A);  indB = membership(t, B)
            sA[p] = indA;  sB[p] = indB            // shared scratch
        # Phase 2 — block inclusive scan → prefixA, prefixB in shared mem (O(L))
        prefixA = scan(sA);  prefixB = scan(sB)
        lgA = prefixA[L];   lgB = prefixB[L]
        # Phase 3 — internal nodes, O(1) each, striped over threads
        for node (lo,mid,hi, sz1,sz2,sz3) of g striped over threads:
            a0 = prefixA[mid]-prefixA[lo];  a1 = prefixA[hi]-prefixA[mid]
            b0 = prefixB[mid]-prefixB[lo];  b1 = prefixB[hi]-prefixB[mid]
            ... derive a2,b2,c0,c1,c2 ; skip if any < 0 ...
            partial += twoQI(...)                  // per-thread accumulator
        # Phase 4 — block-reduce partials, multiply tree multiplicity
        twoScoreLocal += mult(g) * blockReduce(partial)
    twoScores[x] = twoScoreLocal                    // single write, no atomics
```

**Why this is the right granularity:**

- **Parallelism stays massive.** There are `S = O(nk)` splits → `O(nk)` blocks, each
  fully occupied (threads cooperate within a tree across its `L` leaves and internal
  nodes). This is at least as parallel as the current 1-thread-per-split design, with far
  better memory-access behaviour (coalesced leaf scans vs. scattered probes).
- **The O(L) DP state is transient shared memory**, reused tree-by-tree within a block.
  It never becomes a global `O(S·n)` allocation. The user's "we'd need O(n) memory"
  worry is real but is confined to a few dozen KB of shared memory per *active* block,
  bounded by the number of resident blocks, **not** by the number of splits.
- **No atomics**: each block owns one split's output and writes it once.
- **Drop-in with existing batching.** Static device data (orderings, invIndex, plus a new
  per-tree internal-node CSR) is uploaded once; splits stream in adaptive batches exactly
  as today (`astralx_weight.cu` batch loop). We change *what a block does*, not the
  host-side streaming.

**Device-resident memory (per block, shared):** `2·(L+1)` ints ≈ `8L` bytes. For
`L = 4096`, that is 32 KB — within the 48–96 KB shared-memory budget. The shared buffer is
sized to the *maximum* `L` in the dataset (or per-batch max).

### 4.2 Static device memory — still O(nk)

```
orderings  : numTrees × n ints          (unchanged)
invIndex   : numTrees × n ints          (unchanged — used for cluster membership)
nodeCSR    : Σ_g (L_g − 1) × 3 ints      (NEW: per-tree internal nodes as (lo,mid,hi))
nodeOffset : numTrees + 1 ints           (NEW: CSR row pointers into nodeCSR)
treeMult   : numTrees ints               (NEW, optional: whole-tree multiplicity)
splits     : batchSize × 10 ints         (streamed, as today)
twoScores  : batchSize longs             (streamed out, as today)
```

`nodeCSR` is just the current `parts` array reorganized **per tree** (grouped, not
globally deduped) and carrying `(lo, mid, hi)` + the three sizes. It is still `O(nk)`. So
the **peak device footprint is unchanged at O(nk) static + O(batch) transient** — the clean
property the user wanted to preserve is preserved.

### 4.3 Handling very large `L` (when prefix won't fit in shared memory) — IMPLEMENTED

If `L` exceeds the shared-memory budget (`2·(L+1)·4 B + red[]` over the device opt-in
limit, ≈ `L ≳ 12.5k` at the 99 KB sm_86 cap), the kernel switches to a **global-memory
prefix pool**:

1. **Per-block global scratch (shipped).** `prefixA/prefixB` live in global memory sized
   `(number of resident blocks) × 2(L+1)`, **not** `S × 2L`. The grid is capped to the
   resident-block count (`numSM × cudaOccupancyMaxActiveBlocksPerMultiprocessor`), and each
   block **grid-strides** over splits, reusing its own slot. Bounded VRAM: e.g. for
   `n = 25000`, ~100 resident blocks → ≈ 20 MB pool. Correct, slower per-access than shared.
2. *(Not needed in practice)* Tiled scan / additive wavefront (§4.5) would avoid storing
   the full prefix entirely; unnecessary given option 1's small footprint.

**This is now implemented adaptively** in `astralx_weight.cu`: `scoreSplit()` is a shared
device function; `computeWeightsKernel<false>` keeps prefixes in dynamic shared memory (one
block per split), `computeWeightsKernel<true>` keeps them in the global pool (resident-capped
grid-stride). The host picks the mode from the shared-memory feasibility check and only the
prefix storage differs — the algorithm and results are identical. Verified: forcing the
global path (`ASTRALX_WEIGHT_FORCE_GLOBAL=1`) yields byte-identical species trees and scores
vs. the shared path and CPU on TC1–TC13 and the 200-tree/37-taxon dataset.
Typical phylogenomic `n` (hundreds to low thousands) stays on the fast shared-mem path; the
global path only engages for `n` in the tens of thousands.

### 4.4 Batching

Identical strategy to the current kernel: upload static data once; stream splits in
adaptive batches sized from free VRAM after the static upload. The only change to the
budget formula is per-split output is still 8 B and per-split input still 40 B, so the
existing `batchSizeHint` / `vramFraction` logic carries over unchanged. (Static now
includes `nodeCSR`, accounted once.)

### 4.5 Alternative — additive wavefront DP (documented, not recommended as default)

The user explicitly raised wavefront parallelism. It is the direct realization of the
"child addition" recurrence without materializing a prefix array:

- Lay each gene tree out by **levels** (or by postorder waves). All nodes at one level are
  independent and computed simultaneously: `cntA[u] = cntA[left] + cntA[right]`.
- Only the current frontier of `cntA/cntB` values must be live, not the whole prefix —
  scratch is `O(width of level)` rather than `O(L)`, which can be smaller for some shapes
  but is `O(L)` for balanced trees anyway.

Trade-offs vs. the prefix-sum scheme:
- **+** Never stores a full prefix; can stream very large trees.
- **−** Needs explicit child-pointer structure (the postorder *interval* trick is lost);
  more irregular memory access; harder to get coalesced loads; level extraction is extra
  preprocessing.
- **−** Equivalent asymptotics and, in practice, worse constants than the contiguous
  prefix scan on the common (moderate `n`) case.

Recommendation: ship the **prefix-sum / shared-memory** scheme (§4.1) as the default; keep
the wavefront form as the conceptual fallback for the very-large-`L` regime (§4.3 option
2).

---

## 5. CPU Path

The same reformulation accelerates the CPU path with no special hardware. Replace
`computeScore`'s per-part `coreIntersect` calls (`WeightTable.java:349-358`) with:

```
per (split, tree):
    build prefixA[0..L], prefixB[0..L]      // one O(L) pass each
    for each internal node: 4 O(1) prefix differences → 3×3 → twoQI
```

This alone removes the inner element walk and the `intersectWithFullTree` call on the
existing multi-threaded CPU path; useful for correctness cross-checking the GPU kernel
(Phase verifiers) and for GPU-less runs.

---

## 6. Variants and Extensions

### 6.1 Sharing one prefix across both passes of `C`
`prefixC` is never built: `c0,c1,c2` come from column constraints. Only `A` and `B` need
a scan — two scans per (split, tree), not three.

### 6.2 Dedup-preserving variant (cluster-major), and why it is not the default
For a fixed split, building `prefixA/prefixB` for **all** ≤k exemplar trees up front would
let us keep global tripartition dedup (each unique part queries its own exemplar tree's
prefix in O(1)). But that holds `O(nk)` prefix data live per in-flight split — incompatible
with the bounded shared-memory design. Only worthwhile if dedup ratio is extreme and
memory is abundant. Left as a future option behind a flag.

### 6.3 Polytomy / d-partition support
The prefix-sum query generalizes directly: a polytomy node covering `[lo,hi)` with `d`
contiguous child ranges yields `|M_i ∩ A| = prefixA[end_i] − prefixA[start_i]` for each
child in O(1), i.e. the 3×d matrix of `intersection-optimization.md` §8 at O(d) per node.
No new mechanism needed.

---

## 7. Implementation Plan (when we proceed)

1. **Per-tree node CSR.** Add an extractor that emits, per tree, the list of internal-node
   `(lo, mid, hi, sz1, sz2, sz3)` in postorder, plus CSR offsets and optional whole-tree
   multiplicity. (Reuse `PartitionTable.extractNode`'s traversal; drop global dedup.)
2. **CPU reference kernel** (§5) behind a config flag, validated against the existing
   `computeScore` on TC1–TC13 — must match scores exactly.
3. **GPU kernel** (§4.1): shared-memory prefix scan + striped node evaluation +
   block-reduce. Validate against the CPU reference and the current kernel on all test
   cases.
4. **Large-`L` fallback** (§4.3) gated by a shared-memory-capacity check.
5. **Benchmarks** vs. the current kernel on balanced *and* deliberately unbalanced
   (caterpillar) gene-tree inputs to demonstrate the balance-independence win.

## 8. Open Questions

- **Whole-tree multiplicity dedup** — is it worth the bookkeeping, or do biological inputs
  rarely have identical trees? Measure dedup ratio at both granularities first.
- **Shared-memory `L` threshold** — tune against real GPUs (48 KB vs 96 KB opt-in).
- **Block size** (128 vs 256) and scan algorithm (Blelloch vs. warp-scan + carry) — pick
  by profiling Phase 2, which dominates when trees are large.
- **Grid size vs. launch count** — `O(nk)` blocks per batch; confirm scheduler overhead is
  negligible vs. kernel work (it should be, each block does `O(nk)` arithmetic).
