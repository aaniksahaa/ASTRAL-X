# Polytomy Support in ASTRAL-X — Design Document

> **Source-of-truth**: ASTRAL-MP implementation at
> `astral-my/ASTRAL/main/phylonet/coalescent/` — specifically
> `WQDataCollection.java` (cluster / search-space construction),
> `Polytomy.java` (d-partition object),
> `Polytree.java` (weight computation with polytomous nodes).

---

## 0. Overview

This document describes every change required to add gene-tree polytomy support to
ASTRAL-X.  All existing binary test cases must continue to pass with bit-identical
results.

---

## 1. What a "Cluster" Means for Polytomous Gene Trees (Confirmed)

A **cluster** is exactly the taxon set of a connected subtree.  For a polytomous node u
with children c₀, c₁, …, c_{k-1}:

| Set | Is a cluster in X? |
|-----|--------------------|
| sub(u) = c₀∪c₁∪…∪c_{k-1} | **YES** (the whole polytomous subtree) |
| S \ sub(u) | **YES** (complement, same as binary case) |
| sub(cᵢ) for each child i | **YES** (from recursing into children) |
| sub(cᵢ) ∪ sub(cⱼ) for i≠j (e.g. AB, BC, AC, …) | **NO** |

> **Key quote from ASTRAL-MP** (`WQDataCollection.java`, line 163):
> *"For polytomies, if we don't do anything extra, the cluster associated with the
> polytomy may not have any resolutions in X.  We don't want that.  We use the
> greedy consensus trees and random sampling to add extra bipartitions to the input
> set when we have polytomies."*

ASTRAL-MP handles this by **random sampling** (see §5), NOT by adding intermediate
contiguous combo clusters.  Our implementation follows the same decision.

---

## 2. Terminology

| Term | Meaning |
|------|---------|
| **polytomous node** | internal gene-tree node with k ≥ 3 children in the rooted representation |
| **d-partition** | partition M = M₁\|M₂\|…\|Mₐ of the gene-tree's leaf set; d=3 for binary non-root, d=k+1 for a non-root polytomous node with k children (k children subtrees + 1 complement) |
| **`Polytomy`** | ASTRAL-MP's class for d>3 partitions (analogous to `Tripartition`) |
| **QI** | ASTRAL quartet incompatibility score; always a non-negative integer |

---

## 3. Layer-by-Layer Change Analysis

### 3.1 TreeNode (tree/TreeNode.java)

**Current**: strictly binary — `left`, `right`, `parent`.

**Change**: add `children` field for polytomous nodes.

```java
// NEW field
public TreeNode[] children;  // null for binary nodes; length k ≥ 3 for polytomous
```

**Invariants after the change**:

| Node type | `left` | `right` | `children` |
|-----------|--------|---------|------------|
| Leaf | null | null | null |
| Binary internal | child[0] | child[1] | null |
| Polytomous internal (k children) | `children[0]` | `children[k-1]` | non-null array of length k |

- `isLeaf()`: unchanged — `left == null`.
- `isPolytomous()`: new helper — `children != null` (implies length ≥ 3 by construction).
- `getSibling()`: unchanged for children of binary nodes only; **must not be called**
  for children of polytomous nodes (see DPTable §3.7).
- `rangeStart` / `rangeEnd` still span the full subtree in both cases:
  `node.rangeStart = node.left.rangeStart` (leftmost child),
  `node.rangeEnd   = node.right.rangeEnd`  (rightmost child).

---

### 3.2 TreeParser (tree/TreeParser.java)

**Current**: throws for any non-root node with ≠ 2 children.

**Change in `validateAndConvert`**:

```
nc == 2             → unchanged binary path
nc == 3 && isRoot   → unchanged arbitrary-rooting path
nc ≥ 4 && isRoot    → new: treat as unrooted polytomy at root;
                       isolate children[0] as left, collect children[1..nc-1]
                       into a polytomous right child, set left/right/parent
nc ≥ 3 && !isRoot   → NEW: create polytomous TreeNode
                       node.children = new TreeNode[nc]   (all children in order)
                       node.left     = node.children[0]   (leftmost)
                       node.right    = node.children[nc-1] (rightmost)
                       set parent for all children
```

**Change in `assignRangesAndFillArray`**:

```java
// binary path: unchanged (no children array → use left, right)
// polytomous path:
for (TreeNode child : node.children) {
    assignRangesAndFillArray(child, arr, counter);
}
node.rangeStart = node.left.rangeStart;
node.rangeEnd   = node.right.rangeEnd;
```

**Why the `nc ≥ 4 && isRoot` rooting is correct** (same principle as the existing `nc==3`
case): ASTRAL treats gene trees as unrooted, and a degree-`nc` root is the unrooted
representation of an `nc`-way polytomy spanning all taxa. Isolating `child[0]` as the
root's left and making `children[1..nc-1]` a polytomous right child means the root itself
becomes a degree-2 node (skipped in extraction, empty complement), while the **inner
polytomous node recovers the full `nc`-partition**: its `nc-1` explicit children plus its
complement (= `child[0]`) reconstruct exactly `M₁|…|M_{nc}` of the original unrooted
polytomy. The arbitrarily-chosen `child[0]` arm just lands in the complement slot — and
since each arm of an unrooted polytomy is separated by a single edge, `sub(inner)` and
each `sub(armᵢ)` are all genuine edge-induced clusters (no spurious cluster is created).

**Side-effect analysis**:
- `postorderArray` is built identically — children in postorder left-to-right order.
- Range [rangeStart, rangeEnd) remains a contiguous span of the postorder array.
- Binary trees are completely unaffected.

---

### 3.3 ClusterTable (cluster/ClusterTable.java)

**Change in `walkNodes`**: recurse into all children for polytomous nodes.

```java
private void walkNodes(TreeNode node, ...) {
    if (!node.isLeaf()) {
        if (node.isPolytomous()) {
            for (TreeNode child : node.children) walkNodes(child, ...);
        } else {
            walkNodes(node.left, ...);
            walkNodes(node.right, ...);
        }
    }
    if (node.isRoot()) return;

    // Existing: register sub(u) and its super-complement — UNCHANGED
    int lo = node.rangeStart, hi = node.rangeEnd;
    registerCluster(ti, lo, hi, false, hi - lo, L, pref, numTaxa);
    int superCompSize = numTaxa - (hi - lo);
    if (superCompSize > 0)
        registerCluster(ti, lo, hi, true, superCompSize, numTaxa, pref, numTaxa);
    count[0]++;
    // NO combo clusters added — confirmed ASTRAL-MP behaviour
}
```

**Consequence**: a polytomous node contributes the same two clusters as a binary node of
the same subtree size.  No new intermediate clusters.

---

### 3.4 Partition (partition/Partition.java)

**Current**: fixed 3-part layout with named fields.

**New**: variable-length d-part layout.

```java
public final class Partition {
    public final int d;                // number of parts: 3 for binary, k+1 for polytomous
    public final ClusterHash[] hashes; // [0..d-1]; hashes[d-1] is always the complement
    public final int[] sizes;          // sizes[0..d-1]
    public final int treeIndex;
    // Ranges for parts 0..d-2 (the non-complement parts = children's subtree ranges)
    public final int[] partStarts;     // length d-1
    public final int[] partEnds;       // length d-1
}
```

**Backward compatibility for d=3 (binary)**:
- `hashes[0]` = left subtree hash  (was `hash1`)
- `hashes[1]` = right subtree hash (was `hash2`)
- `hashes[2]` = complement hash    (was `hash3`)
- `partStarts[0]` = leftStart, `partEnds[0]` = leftEnd
- `partStarts[1]` = rightStart, `partEnds[1]` = rightEnd

All callers that used named fields (`hash1`, `leftStart`, etc.) must update to array
indexing.  The full set of direct accessors (verified by grep, **do not under-count**):
- `PartitionTable.java` — extraction (writes the fields).
- `WeightTable.java` — CPU scoring and the GPU CSR/parts packers.
- `Phase4Verifier.java` — reads `size1/2/3`, `leftStart/End`, `rightStart/End`,
  `exemplar.size1/size3`; **and asserts binary-only invariants** (see §3.4.1).
- `Phase3Verifier.java` — reads `exemplar.treeIndex` (cluster path; low impact).

`Phase4Verifier` is the one that needs real thought, not a mechanical rename — see next.

#### 3.4.1 Phase4Verifier (astralx/Phase4Verifier.java) — MUST be made polytomy-aware

The current verifier hard-codes binary assumptions that are **false** for polytomies and
will fire spurious failures the moment a polytomous tree is parsed:

| Current assertion | Why it breaks on polytomy | Fix |
|-------------------|---------------------------|-----|
| `size1 + size2 + size3 == leafCount` | a d-partition has `d` parts | `Σ sizes[0..d-1] == leafCount` |
| `size3 > 0` for all | the complement is `sizes[d-1]` | `sizes[d-1] > 0` |
| `hash1 + hash2 + hash3 == totalHash` | d hashes, not 3 | `Σ hashes[0..d-1] == totalHash` |
| "binary tree: n−2 non-root internal nodes" candidate count | polytomous trees have fewer internal nodes | count actual non-root internal nodes per tree |
| per-part taxa dump uses left/right/comp ranges | d−1 child ranges + complement | loop `partStarts/partEnds[0..d-2]` then complement |

This must be in the change set and validated **before** running polytomous test cases,
otherwise the verifier — our own correctness oracle — produces false alarms.

---

### 3.5 PartitionHash (partition/PartitionHash.java)

**Current**: order-invariant over (h1, h2) pair; h3 appended separately.

**New**: order-invariant over ALL d parts — confirmed by `Polytomy.java` in ASTRAL-MP,
which sorts all d clusters by `hash1` before storing.

Algorithm: build one `long[]` fingerprint per ClusterHash, sort all d fingerprints
lexicographically, concatenate, hash.

```java
public PartitionHash(ClusterHash[] parts) {
    int d = parts.length, m = parts[0].sums.length;
    long[][] fps = new long[d][2 * m];
    for (int i = 0; i < d; i++) {
        System.arraycopy(parts[i].sums, 0, fps[i], 0, m);
        System.arraycopy(parts[i].xors, 0, fps[i], m, m);
    }
    // Sort fingerprints lexicographically (unsigned long comparison)
    Arrays.sort(fps, (a, b) -> { for(int s=0;s<a.length;s++){int c=Long.compareUnsigned(a[s],b[s]);if(c!=0)return c;} return 0; });
    // Flatten and hash
    int h = 1;
    for (long[] fp : fps) for (long v : fp) h = 31 * h + Long.hashCode(v);
    this.cachedHashCode = h;
    this.data = flatten(fps);  // stored for equals()
}
```

**Effect on binary trees (d=3)**: any permutation of {M1,M2,M3} produces the same hash.
For binary gene trees where M3 is always the complement (a unique set), the old and new
schemes produce identical deduplication results on all existing test cases.

---

### 3.6 PartitionTable (partition/PartitionTable.java)

**Change in `extractNode`**:

Binary path (nc==2, non-root): unchanged — build `Partition` with d=3 using arrays.

Polytomous path (nc≥3, non-root):
```
k   = node.children.length   (number of children)
d   = k + 1                  (k child subtrees + 1 complement)

For i = 0 .. k-1:
    sizes[i]     = children[i].rangeEnd - children[i].rangeStart
    hashes[i]    = buildHash(ti, children[i].rangeStart, children[i].rangeEnd,
                             false, sizes[i], pref)
    partStarts[i] = children[i].rangeStart
    partEnds[i]   = children[i].rangeEnd

sizes[k]  = L - (node.rangeEnd - node.rangeStart)   // complement
hashes[k] = buildHash(ti, node.rangeStart, node.rangeEnd, true, sizes[k], pref)

Skip if sizes[k] == 0 (u is root → empty complement)

PartitionHash ph = new PartitionHash(hashes)
Deduplicate as before; increment frequency on collision
```

`extractNode` still recurses into children before processing the node itself:
```java
private void extractNode(TreeNode node, ...) {
    if (node.isLeaf()) return;
    if (node.isPolytomous()) {
        for (TreeNode child : node.children) extractNode(child, ...);
    } else {
        extractNode(node.left, ...);
        extractNode(node.right, ...);
    }
    if (node.isRoot()) return;
    // ... extraction logic
}
```

---

### 3.7 DPTable (dp/DPTable.java)

**Key design decision (confirmed by ASTRAL-MP)**:
Polytomous gene-tree nodes do NOT directly add DP transitions.  They only contribute
through the QI/weight formula.  The search space for clusters that only appear as
polytomous subtrees relies on:
1. Binary nodes in other gene trees that happen to contain the same cluster.
2. Mode 2 (cross-tree transitions), which adds all valid binary splits of X-clusters.
3. Future work: random sampling around polytomies (see §5).

**Changes in `emit`**:

```java
private void emit(TreeNode u, int ti, PrefixHashArrays pref) {
    if (u.isLeaf()) return;

    // Recurse into all children
    if (u.isPolytomous()) {
        for (TreeNode child : u.children) emit(child, ti, pref);
        // Polytomous node: NO direct Type 1 or Type 2 transitions from u itself
        return;
    }

    // Binary path: unchanged
    emit(u.left,  ti, pref);
    emit(u.right, ti, pref);

    // Type 1: sub(u) → sub(left) | sub(right)
    ClusterHash hU     = hashRange(ti, u.rangeStart,       u.rangeEnd,       false, pref);
    ClusterHash hLeft  = hashRange(ti, u.left.rangeStart,  u.left.rangeEnd,  false, pref);
    ClusterHash hRight = hashRange(ti, u.right.rangeStart, u.right.rangeEnd, false, pref);
    addTransition(hU, hLeft, hRight);

    // Type 2: S\sub(u) → sub(sibling) | S\sub(parent)
    // GUARD: skip when the parent is polytomous.  A binary node u whose PARENT is a
    // polytomous node still reaches this code (it went through the binary path above),
    // but u.getSibling() — which does `parent.left==this ? parent.right : parent.left`
    // — is meaningless for a child of a polytomous parent (no well-defined sibling).
    // The `!u.parent.isPolytomous()` clause is therefore REQUIRED for correctness, not
    // an optimization.  (The all-taxa/root path is handled separately via Mode 2.)
    if (!u.isRoot() && !u.parent.isPolytomous()) {
        TreeNode sib    = u.getSibling();   // safe: parent is binary
        TreeNode parent = u.parent;
        ClusterHash hCompU      = hashRange(ti, u.rangeStart,      u.rangeEnd,      true,  pref);
        ClusterHash hSib        = hashRange(ti, sib.rangeStart,    sib.rangeEnd,    false, pref);
        ClusterHash hCompParent = hashRange(ti, parent.rangeStart, parent.rangeEnd, true,  pref);
        if (hCompParent.size > 0) {
            addTransition(hCompU, hSib, hCompParent);
        }
    }
}
```

The single added clause `&& !u.parent.isPolytomous()` is the entire DPTable change — it
prevents calling `getSibling()` on a child of a polytomous node while keeping the binary
path bit-identical (for binary trees no node has a polytomous parent, so the clause is
always true and behavior is unchanged).

---

### 3.8 WeightTable — QI Formula (weight/WeightTable.java)

#### 3.8.1 Current binary formula (for reference)

For a 3-partition (M₁|M₂|M₃) — the current O(6) inner loop:
```
2·QI = Σ_{(i,j,k) distinct perms of {0,1,2}}  a[i]·b[j]·c[k]·(a[i]+b[j]+c[k]-3)
```

#### 3.8.2 Generalized O(d) ASTRAL-III formula

For species-tree split A|B (C = S \ (A∪B)) and gene-tree d-partition M₀|…|M_{d-1}:

**Intersection table** (d entries each):
```
aᵢ = |A ∩ Mᵢ|,   bᵢ = |B ∩ Mᵢ|,   cᵢ = |Mᵢ| - aᵢ - bᵢ
```

**Global sums and cross-products**:
```
Sₐ = Σaᵢ,   S_b = Σbᵢ,   S_c = Σcᵢ
S_{ab} = Σaᵢbᵢ,   S_{ac} = Σaᵢcᵢ,   S_{bc} = Σbᵢcᵢ
```

**O(d) formula for 2·QI** (same convention as current code — divide by 2 at end):
```
2·QI = Σᵢ aᵢ(aᵢ-1)·[(S_b-bᵢ)·(S_c-cᵢ) - S_{bc} + bᵢcᵢ]
     + Σᵢ bᵢ(bᵢ-1)·[(Sₐ-aᵢ)·(S_c-cᵢ) - S_{ac} + aᵢcᵢ]
     + Σᵢ cᵢ(cᵢ-1)·[(Sₐ-aᵢ)·(S_b-bᵢ) - S_{ab} + aᵢbᵢ]
```

This is provably equivalent to the O(d³) triple-sum formula (see §4 for the derivation).
For d=3 it produces exactly the same value as the current 6-permutation loop — this
will be verified by a unit test before committing.

#### 3.8.3 Intersection computation for d-partitions

For binary partitions: 4 core intersections computed, rest derived.
For d-partitions: compute 2(d-1) core intersections (aᵢ and bᵢ for i = 0..d-2)
then derive:

```
cᵢ = sizes[i] - aᵢ - bᵢ    for i = 0..d-2
a_{d-1} = lgA - Σᵢ₌₀^{d-2} aᵢ   (row constraint; lgA = |A ∩ gene-tree taxa|)
b_{d-1} = lgB - Σᵢ₌₀^{d-2} bᵢ   (row constraint)
c_{d-1} = sizes[d-1] - a_{d-1} - b_{d-1}   (complement part)
```

For each non-complement part i (0..d-2):
```java
aᵢ = IntersectionCounter.intersect(tGT, p.partStarts[i], p.partEnds[i],
                                    tA, cA.left, cA.right, cA.complement, p.sizes[i]);
bᵢ = IntersectionCounter.intersect(tGT, p.partStarts[i], p.partEnds[i],
                                    tB, cB.left, cB.right, cB.complement, p.sizes[i]);
```

All three variants (LONG, DOUBLE, INT128) follow the same structure; only the numeric
type differs.

#### 3.8.4 GPU path with polytomy (full design) — ✅ IMPLEMENTED & VALIDATED

> **Status (2026-06):** both GPU kernels implement polytomy natively.  Prefix-sum:
> `scorePolyNodes<ACC>` + `scorePolyNodesI128`, a two-pass O(1)-memory loop over the
> per-tree poly CSR, reusing the same `pA/pB/lgA/lgB` as the binary loop (any degree d).
> Smaller-side: `ssScorePoly<ACC>` + `ssScorePolyI128`, the two-pass-rewalk over a poly
> CSR.  Both binary loops are untouched; empty poly CSR ⇒ no-op ⇒ byte-identical on
> binary inputs.  Validated: TC1–13 bit-identical on BOTH kernels (LONG/INT128); on
> polytomous inputs **oracle == CPU == prefix-sum GPU == smaller-side GPU == INT128**
> (5-way agreement, complete + incomplete, up to d≈n/2), combined with mechanism-B
> multi-range clusters, and on the large-L global prefix path.  No CPU fallback is
> forced by polytomy.



This section gives a **real GPU design**, not a CPU fallback.  The key observation that
makes it tractable:

> **The prefix-sum kernel's per-tree prefix arrays `pA[]`, `pB[]` are independent of node
> structure.**  They are built once per (split, tree) over the tree's leaf postorder
> array.  A polytomous node's children are *still contiguous postorder ranges*, so each
> of its d-partition parts is recovered by the **same O(1) prefix difference** the binary
> path already uses.  Polytomy support therefore needs:
> - **zero changes** to prefix-sum construction (`buildPrefix`),
> - **zero extra shared/global memory** (no new per-thread working set),
> - only a new *per-node inner computation* (the O(d) formula) fed by a variable-length
>   boundary list instead of the fixed `[lo, mid, hi]` triple.

Background: the relevant kernel is in `src/native/astralx_weight.cu`, `scoreSplit<ACC>`
(the per-thread node loop at lines ~255–291) and its INT128 twin `scoreSplitI128`.
The per-tree node CSR is `nodeData[3·ni + {0,1,2}] = (lo, mid, hi)` built by
`buildDedupNodeCSR` in `WeightTable.java`.

##### (a) Two-tier node CSR — keeps the binary path bit-identical

To guarantee **no regression** on existing binary inputs (in all three numeric modes,
including DOUBLE where FP summation order matters), we do **not** rewrite the binary
node loop.  Instead we split the deduplicated partitions into two CSRs, both bucketed by
exemplar tree:

1. **Binary CSR** — exactly the current `nodeData[3·ni]=(lo,mid,hi)` + `nodeFreq[ni]`,
   carrying all `d == 3` partitions.  Processed by the **unchanged** binary node loop and
   6-permutation formula → bit-identical to today.

2. **Polytomy CSR** (new) — carries only `d > 3` partitions:
   ```
   polyTreeOffset[g]..polyTreeOffset[g+1]   CSR row pointers: poly nodes of tree g
   polyBoundOffset[pn]..polyBoundOffset[pn+1]  range into polyBounds for node pn;
                                                length = d (the partition degree)
   polyBounds[...]   flat int array: concatenated boundary lists.
                     For a degree-d node: b[0]=lo, b[1], …, b[d-1]=hi.
                     The d-1 child parts are [b[i], b[i+1]) for i=0..d-2;
                     the d-th part is the complement  Lg \ [b[0], b[d-1]).
   polyFreq[pn]      occurrence count (frequency) of unique poly partition pn
   ```
   `partLeafCount[g]` (already uploaded) supplies `L` for the complement part.

Since the overwhelming majority of nodes are binary, the fast bit-identical path
dominates; the new path touches only the rare polytomous nodes.

##### (b) New device routine `scorePolyNodes<ACC>` (and INT128 twin)

Called once per (split, tree) right after the existing binary node loop, reusing the same
`pA`, `pB`, `scan` buffers (already populated for this tree).  Each thread grid-strides
over the tree's polytomous nodes (`for pn = polyTreeOffset[g]+tid; pn < end; pn += nthreads`),
computing the **full O(d)** QI for its node — no inter-thread cooperation needed:

```c
int base = polyBoundOffset[pn];
int d    = polyBoundOffset[pn + 1] - base;   // partition degree
int b0   = polyBounds[base];                 // = lo
int bD   = polyBounds[base + d - 1];         // = hi
int lgA  = pA[L], lgB = pB[L];

// ---- Pass 1: marginals over all d parts (child parts via O(1) prefix diffs) ----
ACC Sa=0,Sb=0,Sc=0, Sab=0,Sac=0,Sbc=0;
int sumA=0, sumB=0;                          // child-part column sums (for complement)
for (int i = 0; i < d - 1; i++) {            // the d-1 child intervals
    int lo = polyBounds[base + i], hi = polyBounds[base + i + 1];
    int ai = pA[hi] - pA[lo];
    int bi = pB[hi] - pB[lo];
    int ci = (hi - lo) - ai - bi;            // ≥ 0 by construction
    Sa+=ai; Sb+=bi; Sc+=ci;
    Sab+=(ACC)ai*bi; Sac+=(ACC)ai*ci; Sbc+=(ACC)bi*ci;
    sumA+=ai; sumB+=bi;
}
// complement part (index d-1)
int aC = lgA - sumA;
int bC = lgB - sumB;
int szC = L - (bD - b0);
int cC = szC - aC - bC;
if (aC < 0 || bC < 0 || cC < 0) continue;    // incomplete-tree row mismatch → skip node
Sa+=aC; Sb+=bC; Sc+=cC;
Sab+=(ACC)aC*bC; Sac+=(ACC)aC*cC; Sbc+=(ACC)bC*cC;

// ---- Pass 2: O(d) QI accumulation (recompute child parts; complement reused) ----
ACC twoQI = 0;
for (int i = 0; i < d; i++) {
    int ai, bi, ci;
    if (i < d - 1) {
        int lo = polyBounds[base + i], hi = polyBounds[base + i + 1];
        ai = pA[hi]-pA[lo]; bi = pB[hi]-pB[lo]; ci = (hi-lo)-ai-bi;
    } else { ai = aC; bi = bC; ci = cC; }
    twoQI += (ACC)ai*(ai-1) * ((Sb-bi)*(Sc-ci) - Sbc + (ACC)bi*ci);
    twoQI += (ACC)bi*(bi-1) * ((Sa-ai)*(Sc-ci) - Sac + (ACC)ai*ci);
    twoQI += (ACC)ci*(ci-1) * ((Sa-ai)*(Sb-bi) - Sab + (ACC)ai*bi);
}
threadAccum += (ACC) polyFreq[pn] * twoQI;
```

Notes:
- **No per-thread storage of the d parts** — both passes recompute child parts via O(1)
  prefix differences, so working memory stays O(1) regardless of d (≤ n).
- The routine accumulates into the **same `threadAccum`** as the binary loop, so the
  block reduction and transport are unchanged.
- Three instantiations (`long long`, `double`, INT128 via the `i128_*` helpers) mirror
  the existing `scoreSplit` / `scoreSplitI128` split.  Magnitude budget is the same as
  binary (per-term ≤ n⁴; the LONG/DOUBLE/INT128 mode decision in `WeightTable` already
  covers it — reuse it verbatim).
- Validity check parallels the binary `a2/b2/c0/c1/c2 < 0` guard: only the
  complement-derived parts can be negative (child parts are non-negative by construction).

**Load balancing (deferred optimization).** In the current kernel each thread handles
one whole node. A thread that lands on a degree-`d` polytomy does O(d) work while
neighbours on binary nodes do O(1), causing warp divergence. For typical inputs (few,
low-degree polytomies) this is negligible, so the first implementation keeps the
one-thread-per-node model. If a dataset has *many high-degree* polytomies, a later
optimization could (a) sort/bucket poly nodes by degree to reduce intra-warp variance,
or (b) assign a warp/sub-warp per high-degree node with a cooperative reduction. This is
explicitly out of scope for the initial polytomy support and should be revisited only if
profiling shows poly nodes dominating a real workload.

##### (c) Host-side packing (`WeightTable.java`)

Extend `buildDedupNodeCSR` to emit **both** CSRs in one pass over `partTable.entries()`:
route `d == 3` entries to the existing `nodeData/nodeFreq/nodeOffset` arrays, and `d > 3`
entries to the new `polyBounds/polyBoundOffset/polyFreq/polyTreeOffset` arrays.  The poly
boundary list for a degree-d entry is `[partStarts[0], partStarts[1], …, partStarts[d-2], partEnds[d-2]]`
(the children are contiguous, so `partEnds[i] == partStarts[i+1]`; the last boundary is
the end of the final child = `hi`).

Add four `int[]` JNI parameters to `computeWeightsGPU` (and the INT128 path) for the poly
CSR; pass empty arrays (and `polyTreeOffset` all-zero) when there are no polytomies, so the
kernel's poly loop is a no-op and behavior is byte-for-byte identical to today on binary
inputs.

##### (d) Smaller-side path (the kernel we usually run) — single-pass design

The smaller-side kernel (`computeWeightsSmallerSideKernel`) is one-thread-per-split, no
prefix sums: it walks the smaller of two ranges per intersection. Its binary parts are a
fixed 9-int row `[treeIdx, lo1,hi1, lo2,hi2, sz1,sz2,sz3, freq]`. Because this is the path
we work in most often, we give it **first-class polytomy support** rather than rerouting
to prefix-sum.

**Why the prefix-sum two-pass trick does NOT transfer here.** In the prefix-sum kernel,
recomputing `a_i` in pass 2 is an O(1) prefix difference. In the smaller-side kernel,
computing `a_i = |M_i ∩ A|` is a **range walk** (O(range)). A naive two-pass O(d) would
walk every child range *twice*. We avoid that with a **single-pass moment formulation**:
walk each child range exactly once, feed `(a_i, b_i, c_i)` into a fixed set of scalar
moment accumulators, then combine them in O(1) at the end using the global sums.

**The moment identity.** Expand the O(d) formula's a-term bracket:
```
(S_b - b_i)(S_c - c_i) - S_bc + b_i c_i
  = S_b·S_c - S_b·c_i - S_c·b_i + 2·b_i·c_i - S_bc
```
so, with weight wᵢ = aᵢ(aᵢ-1),
```
T_a = Σᵢ aᵢ(aᵢ-1)·[…]
    = S_b·S_c·Wa  −  S_b·Wa_c  −  S_c·Wa_b  +  2·Wa_bc  −  S_bc·Wa
```
where the four **a-moments** are
```
Wa    = Σ aᵢ(aᵢ-1)
Wa_b  = Σ aᵢ(aᵢ-1)·bᵢ
Wa_c  = Σ aᵢ(aᵢ-1)·cᵢ
Wa_bc = Σ aᵢ(aᵢ-1)·bᵢ·cᵢ
```
T_b and T_c are symmetric (swap roles of a/b/c):
```
T_b = S_a·S_c·Wb − S_a·Wb_c − S_c·Wb_a + 2·Wb_ac − S_ac·Wb
T_c = S_a·S_b·Wc − S_a·Wc_b − S_b·Wc_a + 2·Wc_ab − S_ab·Wc
```
`2·QI = T_a + T_b + T_c`.

**Single-pass accumulator set (18 scalars), all updated once per part:**
```
globals : S_a S_b S_c   S_ab S_ac S_bc
a-moments: Wa  Wa_b  Wa_c  Wa_bc
b-moments: Wb  Wb_a  Wb_c  Wb_ac
c-moments: Wc  Wc_a  Wc_b  Wc_ab
```
The final combine uses only globals × moments, so it is valid *after* the pass completes
(all globals are known). For d=3 this reproduces the binary score exactly (verified by the
unit test in §8.2), so it is a strict generalization.

**Per-node procedure (one thread, one poly node):**
```c
// header: ssPolyMeta[pn] = {treeIdx, L_GT, freq};  bounds via ssPolyBoundOffset CSR
walk each child range i=0..d-2 ONCE:
    a_i = ssIntersect(tGT, [b_i,b_{i+1}), A);   // smaller-side walk
    b_i = ssIntersect(tGT, [b_i,b_{i+1}), B);
    c_i = (b_{i+1}-b_i) - a_i - b_i;
    update all 18 accumulators with (a_i,b_i,c_i);  sumA+=a_i; sumB+=b_i;
// complement part (no walk): row constraint
lgA = (L_GT==totalN) ? sizeA : ssCoreIntersect(tGT,[0,L_GT),A)-adjusted;   // as today
lgB = similarly;
a_C = lgA - sumA;  b_C = lgB - sumB;
c_C = (L_GT - (b_{d-1}-b_0)) - a_C - b_C;
if (a_C<0||b_C<0||c_C<0) skip node;
update all 18 accumulators with (a_C,b_C,c_C);
twoScore += freq · combine18();    // O(1)
```
Walk cost per poly node = the d-1 child ranges (×2 for A,B) plus the existing per-node
`[0,L_GT)` row walks for incomplete trees — i.e. the **same character** as a binary part,
just summed over the children. No range is walked twice.

**Implementation-complexity note (INT128).** The moment combine has triple products
(`S_b·S_c·Wa`, up to ≈ n⁴) and the moment `Wa_bc = Σ aᵢ(aᵢ-1)bᵢcᵢ` can itself exceed
64 bits. The existing `i128_*` helpers only do `I128 × 64-bit`, so the INT128 moment path
needs careful widening: keep `Wa_bc / Wb_ac / Wc_ab` as `I128` accumulators, and form each
triple product as `i128_mul_u64(S_b*S_c, Wa)` etc. (each pairwise factor ≤ n² fits 64-bit;
two `i128_mul_u64`s compose the n⁴ term). This is materially more code than the binary
INT128 path — it does **not** just "mirror the existing split."

**Simpler correctness-first alternative (recommended for the INT128 variant, optional for
all).** Because polytomous nodes are *rare*, a **two-pass-with-rewalk** is an acceptable
fallback: pass 1 walks the child ranges to accumulate only the marginals
(`S_a,S_b,S_c,S_ab,S_ac,S_bc`); pass 2 re-walks them and applies the §3.8.2 O(d) formula
per part exactly as the prefix-sum/CPU paths do. This doubles the range-walk cost on poly
nodes only, but reuses the *identical* per-part arithmetic as binary (trivially correct in
LONG/DOUBLE/INT128, no 18-moment bookkeeping). **Recommendation:** ship two-pass-rewalk
first to de-risk INT128 correctness; switch the LONG/DOUBLE variants to single-pass moments
later if profiling shows poly re-walks matter. Both must agree exactly (cross-check, §(e)).

**Smaller-side poly CSR (host-side, analogous to the prefix-sum poly CSR):**
```
ssPolyMeta[3·pn]   = {treeIdx, L_GT, freq}              // L_GT = gene-tree leaf count
ssPolyBoundOffset[pn]..[pn+1]                           // range into ssPolyBounds, length d
ssPolyBounds[...]  = boundary positions b[0..d-1]       // child i = [b[i], b[i+1])
```
Binary parts keep the existing 9-int layout and the **unchanged** binary loop (bit-identical).
The poly loop runs after it; empty poly arrays → no-op → byte-identical on binary inputs.
Same load-balancing caveat as the prefix-sum path applies (deferred optimization).

Add `boolean hasPolytomousPartitions()` to `PartitionTable` (true if any entry has `d > 3`).
Both GPU paths (prefix-sum and smaller-side) now support polytomy natively; **no CPU
fallback is forced by polytomy alone** — CPU is used only if the chosen GPU path is
otherwise infeasible (e.g. allocation failure), exactly as today.

> **Note (combining with multi-range clusters):** polytomy makes the *partition (M)* side
> variable; multi-range clusters (see [multi-range-cluster-design.md](multi-range-cluster-design.md))
> make the *split (A/B)* side variable. These are orthogonal axes. In the prefix-sum kernel
> they are consumed in separate phases (node loop vs `buildPrefix`) and never interact — 2+2
> branches, not 2×2. They meet only in the smaller-side kernel's inner intersection. Polytomy
> scoring ships with single-range splits, so this interaction is a later concern; full
> analysis in multi-range-cluster-design.md §5.5.

##### (e) GPU validation (both kernels)

- With **no** polytomies: confirm the poly CSR is empty / `poly*Offset` all-zero, the poly
  loop never executes, and GPU scores are **bit-identical** to the pre-change build on
  TC1–TC13 — for **both** the prefix-sum and smaller-side kernels, in LONG, DOUBLE, INT128.
- With polytomies: cross-check GPU scores against the CPU O(d) path on the same input
  (LONG/INT128 must match exactly; DOUBLE within FP tolerance). Also cross-check the two
  GPU kernels against each other (prefix-sum two-pass vs smaller-side single-pass moments —
  they must agree exactly in LONG/INT128).
- Stress: a single high-degree polytomy (d = n−1) to exercise the O(d) path and, for
  prefix-sum, the large-L global-prefix path simultaneously.

---

### 3.9 Inference / DP Solver (dp/Inference.java)

**No changes needed.**  The solver operates on binary bipartition splits; polytomy
support is entirely absorbed by generalized QI scoring.

---

## 4. The O(d) QI Formula — Derivation

The O(d³) formula sums over ordered distinct triples (i,j,k):
```
2·QI = Σ_{i≠j,i≠k,j≠k}  aᵢ·bⱼ·cₖ·(aᵢ+bⱼ+cₖ-3)
```

Split `(aᵢ+bⱼ+cₖ-3) = (aᵢ-1) + (bⱼ-1) + (cₖ-1)` and separate into three sums.
For the first sub-sum (factor is `aᵢ(aᵢ-1)`), fix i and sum over j≠i, k≠i,k≠j:

```
Σ_{j≠i} bⱼ · Σ_{k≠i,k≠j} cₖ
= Σ_{j≠i} bⱼ · (S_c - cᵢ - cⱼ)
= (S_b - bᵢ)(S_c - cᵢ) - Σ_{j≠i} bⱼcⱼ
= (S_b - bᵢ)(S_c - cᵢ) - (S_{bc} - bᵢcᵢ)
```

So the first sub-sum becomes `Σᵢ aᵢ(aᵢ-1)·[(S_b-bᵢ)(S_c-cᵢ) - S_{bc} + bᵢcᵢ]`.
The other two sub-sums follow by symmetry.  This is the O(d) formula stated in §3.8.2.

**Numerical correctness**: the O(d) formula is an exact algebraic identity — it gives
the same integer as the O(d³) formula for every input.  A unit test verifies this.

---

## 5. Search Space Enrichment for Polytomies (ASTRAL-MP approach — Future Work)

**First, separate two concerns that are easy to conflate:**

1. **Scoring an input gene-tree polytomy** — handled by the d-partition QI formula (§3.8).
   This is the actual quartet *signal*, summed over **all** gene trees in the weight step.
   This is what our main design implements.

2. **Enriching the candidate set X** — a *search-space* heuristic that only proposes
   candidate bipartitions; it computes no signal. This §5 is about (2), and it is largely
   **orthogonal** to input-gene-tree polytomy support.

> **CORRECTION (verified 2026-06).** An earlier draft of this section claimed the
> sampled polytomies live *only* in internally-built greedy-consensus trees, "not in the
> raw input gene trees." **That is wrong for single-individual data.** At
> `WQDataCollection.java:652–661`, `allGreedies[gt] = [the input gene tree, relabelled]`
> — i.e. for single-individual datasets `allGreedies` IS the set of input gene trees.
> `FormSetXLoop` (:761/828) then calls `addBipartitionsFromSignleIndTreesToX` on **each
> input gene tree**, whose polytomy block (:172–227) resolves every input-gene-tree
> polytomy against the base tree `ST` (3 samples, `getBitsets` + `addbackAfterSampling`)
> and adds the resulting **arm-union (multi-range) clusters** to X.  So ASTRAL-MP DOES
> enrich X from input gene-tree polytomies — a distinct mechanism from the d-partition
> *scoring* in §3.8.  ASTRAL-X now has multi-range cluster support, so this enrichment is
> implementable as a follow-up "gene-tree polytomy sampler" (resolve each input polytomy
> against the UPGMA guide tree → arm-union emissions).  It is **orthogonal to and not
> required for** correct d-partition scoring (which this design implements).

**What ASTRAL-MP actually does** (verified in `WQDataCollection.java`):

- It resolves polytomies in **two** places: (1) each **input gene tree** directly (for
  single-individual data `allGreedies` = input trees; `addBipartitionsFromSignleIndTreesToX`
  resolves their polytomies against `ST`), and (2) internally-built **greedy-consensus
  trees** + the reference tree `ST` (`addExtraBipartitionByHeuristics`).  Call sites:
  line 749 (traverse `ST`), line 761/828 (traverse `allGreedies` = input trees), and the
  greedy-consensus path at :799.
- For each polytomous node of such a tree (`childbslist.size() > 2`, lines 172–227):
  1. Build the d arms: `children[0..k-1]` + complement (`remaining`).
  2. Pick one random taxon per arm → `randomSample` (d taxa).
  3. Restrict the **reference set `baseTrees`** to those d taxa and take its bipartitions
     (`Utils.getBitsets`).
  4. Expand each restricted bipartition back: OR in the full arm bitset for each sampled
     taxon (`addbackAfterSampling`), then add to X (`addSpeciesBitSetToX`).
  5. Repeat 3× (the `for ii` loop).

**The cost is bounded — it is NOT per input gene tree.** `baseTrees` is a **single** tree:
the reference tree `ST` from `speciesMatrix.inferTreeBitsets()` (lines 725–748; the
`allGenesGreedy` alternative is commented out). `inferTreeBitsets()` is literally
`return UPGMA()` — so **`ST` is a UPGMA tree over the quartet-based species similarity
matrix** (`geneMatrix` populated by `populateByQuartetDistance`, then collapsed to species
level), a single fully-resolved binary tree (hence `ST` itself has no polytomies; it is the
*resolved reference* used to propose resolutions). This is the same UPGMA-on-similarity
machinery ASTRAL-X already has in `completion/` (`SimilarityMatrixBuilder` + `UPGMAClusterer`).
So per polytomous node the work is `3 samples × 1 reference tree × O(d) bipartitions ≈ O(3d)`
candidates added to X — not O(genes). The "average signal across all gene trees" is never
computed here; it is computed later in the weight step, over whatever candidates this step
proposed.

**Why porting this is non-trivial in our range-based system**: the expanded bipartitions
in step 4 are unions of *non-adjacent* arms (e.g. arm₀ ∪ arm₂), which are **non-contiguous**
taxon sets and cannot be represented as a single `[lo, hi)` postorder range. Implementing
it faithfully requires multi-range cluster support — a separate refactor — and it would
operate on our own consensus/inferred trees (we already build greedy consensus in
`greedy/`), not the input gene trees.

**Consequence for this implementation**: a cluster that appears *exclusively* as a
polytomous subtree gets no Mode 1 transition of its own. Crucially, **it does not need
one** — the DP only has to resolve clusters that lie on the path it actually chooses, and
it is free to build the species tree from the singletons and the (many) binary clusters
contributed by other gene-tree nodes and by Mode 2. A polytomy is by definition an
*unresolved* node, so it correctly does **not** force any particular resolution. Its
quartet signal is still fully captured: the d-partition contributes QI to every candidate
split it is compatible with (§3.8). So the only thing we lose by skipping the random
sampling is some extra candidate bipartitions in X — never correctness of the scoring.
(Mode 2 may still supply a split for the polytomous cluster if other trees contributed the
needed sub-clusters, but that is incidental, not required.)

**This is correct and acceptable for a first implementation.**  The random-sampling
enrichment can be added later as a separate improvement once multi-range cluster support
is in place (or by a dedicated PolytomySampler class that works within the existing
range-based X via restriction to the individual-children ranges).

---

## 6. Summary: What Changes, What Does Not

| Component | Changes? | Nature of change |
|-----------|----------|-----------------|
| `TreeNode.java` | YES | Add `children` field; `isPolytomous()` helper |
| `TreeParser.java` | YES | Remove polytomy rejection; handle k≥3 at non-root; update range assignment |
| `ClusterTable.java` | YES | Recurse into all children; **no combo clusters** (confirmed ASTRAL-MP) |
| `Partition.java` | YES | Generalize to d-part with arrays |
| `PartitionHash.java` | YES | Sort all d parts (not just 2) |
| `PartitionTable.java` | YES | Extract d-partition for polytomous nodes; recurse into all children |
| `DPTable.java` | YES | Skip Type 2 for children of polytomous parents (one-line guard); polytomous nodes themselves add no transitions |
| `WeightTable.java` | YES | O(d) QI formula for all 3 numeric modes (CPU); emit two-tier (binary + polytomy) CSR/parts for both GPU kernels |
| `Phase4Verifier.java` | YES | Generalize binary-only assertions to d parts (§3.4.1) — required so the oracle doesn't false-alarm |
| `Phase3Verifier.java` | NO* | Reads only `exemplar.treeIndex` (cluster path); unaffected by the Partition refactor |
| `IntersectionCounter.java` | NO | Existing `intersect()` handles any range pair |
| `Inference.java` | NO | DP solver operates on binary splits only — unchanged |
| `BipartitionSplit.java` | NO | Unchanged |
| `Cluster.java` | NO | Range cluster representation unchanged |
| `ClusterHash.java` | NO | Hash arithmetic unchanged |
| `PrefixHashArrays.java` | NO | Prefix sum arrays unchanged |
| `src/native/astralx_weight.cu` | YES | Prefix-sum: new `scorePolyNodes<ACC>` (+INT128) two-pass, reuses existing prefix arrays. Smaller-side: new single-pass moment poly loop. Both binary loops unchanged (bit-identical) |
| `GPUWeightCalculator.java` | YES | Add poly-CSR int[] params to **both** JNI signatures (empty when no polytomy) |
| `test/verify_weights.py` | YES | Generalize tripartition extraction + QI reference to d-partitions so it can validate polytomous runs |
| Greedy consensus (`greedy/`) | NO | Already uses `SNode` with arbitrary children |
| Completion (`completion/`) | NO | Operates on completed binary trees |

---

## 7. Precise Answer to "What Cluster Means for Polytomous Nodes"

> "If a node had children A, B, C, D as a polytomy, what does ASTRAL-MP call a cluster?
>  Only ABCD? Or also AB, BC, CD?"

**Answer (confirmed from ASTRAL-MP source)**:
- sub(ABCD) — YES (whole polytomous subtree)
- S \ sub(ABCD) — YES (complement)
- sub(A), sub(B), sub(C), sub(D) — YES (individual children, from recursing)
- sub(AB), sub(BC), sub(CD), sub(AC), sub(BD), sub(ABC), sub(BCD), etc. — **NO**

The search-space enrichment for polytomies comes from random sampling (future work),
not from pre-computed intermediate clusters.

---

## 8. Test Plan

### 8.1 Non-regression (binary trees)

All existing TC1–TC13 must produce **bit-identical output** after the changes.
Run the full test suite and verify all pass.

### 8.2 QI formula unit test

Write a test (Java JUnit or Python script) that:
- For d=3: asserts `twoQI_od(a,b,c) == twoQI_brute_6perms(a,b,c)` for 10,000 random inputs.
- For d=4 and d=5: asserts `twoQI_od == twoQI_brute_d3sum` for 1,000 random inputs.
- Edge cases: some aᵢ/bᵢ/cᵢ = 0, singletons.

### 8.3 Verifier generalization (prerequisite for polytomous test cases)

The in-process and Python verifiers are our correctness oracles; both are binary-only
today and must be generalized **before** they can validate polytomous runs:
- **`Phase4Verifier.java`** — generalize the five binary assertions per §3.4.1.
- **`verify_weights.py`** — `extract_tripartitions()` and the QI reference currently
  assume `M1=left, M2=right, M3=complement` (binary). Generalize to extract a d-partition
  per internal node (d−1 child ranges + complement) and compute the reference QI by the
  O(d³) brute force (the independent, obviously-correct formula) so it can cross-check the
  Java O(d) result on polytomous inputs.

### 8.4 Polytomous gene-tree test cases

Craft small inputs with polytomous nodes and verify:
- TreeParser accepts them without throwing.
- Correct d-partitions are extracted (correct d value, correct sizes, complement = `sizes[d-1]`).
- The (generalized) Phase4Verifier and verify_weights.py pass.
- DP produces a binary species tree (a polytomous cluster need not be a DP intermediate —
  see §5; its quartet signal still enters the weights).

### 8.5 Polytomy simulator

New `simulate_polytomous.py`:
1. Take an existing binary gene tree file.
2. For each tree, randomly "collapse" k internal nodes by removing the separator node
   and making its parent directly adopt both grandchildren (creating a polytomy).
3. Verify the resulting Newick is parseable by ASTRAL-X.
4. Run ASTRAL-X and compare species tree with the binary-input run.

---

## 9. Implementation Order (safe, no-regression sequence)

1. `TreeNode` + `TreeParser`: add `children` field and parsing support.
   Validate: binary trees parse identically (same `postorderArray`).

2. `ClusterTable`: update recursion to visit all children.
   Validate: binary trees produce same cluster counts.

3. `Partition` + `PartitionHash`: generalize to d-parts.
   Validate: for d=3 inputs, frequency counts match previous run.

4. `PartitionTable`: generalize extraction.
   Validate: binary inputs produce same unique partition count.

5. `WeightTable` (CPU): implement O(d) QI formula for LONG/DOUBLE/INT128; add
   `hasPolytomousPartitions()`.
   Validate: binary inputs give bit-identical scores (d=3 path is numerically equivalent).

6. `DPTable`: add the one-clause parent-polytomy guard for Type 2.
   Validate: binary trees produce same transition counts.

7. **Generalize the oracles**: update `Phase4Verifier` (§3.4.1) and `verify_weights.py`
   (§8.3) to d-partitions. On binary inputs they must still pass identically.

8. Run full test suite on the **CPU path**: TC1–TC13 must all pass (with the generalized
   verifiers).

9. GPU: build the two-tier (binary + polytomy) CSR/parts in `WeightTable` for both kernels;
   add the prefix-sum `scorePolyNodes<ACC>` (+INT128) and the smaller-side poly loop
   (two-pass-rewalk first for INT128 safety, single-pass moments as the LONG/DOUBLE
   optimization) in `astralx_weight.cu`; extend both JNI signatures.
   Validate: with empty poly CSR, both kernels are bit-identical to pre-change on TC1–TC13;
   with polytomies, both GPU kernels match the CPU O(d) path and each other.

10. Add polytomous test cases (TC14+) and run on **both** CPU and GPU.

11. Polytomy simulator (`simulate_polytomous.py`).
