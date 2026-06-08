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

All callers that used named fields (`hash1`, `leftStart`, etc.) update to array indexing.
Only `PartitionTable`, `WeightTable`, and the GPU serialization paths access `Partition` directly.

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

    // Type 2: only for children of binary parents (getSibling() is defined)
    // Children of polytomous nodes reach here via emit(child,...) above and
    // return early if child.parent.isPolytomous() — so Type 2 is naturally skipped.
    if (!u.isRoot()) {
        TreeNode sib    = u.getSibling();   // safe: parent is binary (ensured below)
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

**Guarding `getSibling()` for children of polytomous parents**:
`u.getSibling()` calls `parent.left == this ? parent.right : parent.left` — this is
**wrong** for a child of a polytomous node because the parent has `children` array and
the "sibling" concept is undefined.

Safe guard: in the binary `emit` path above, Type 2 only runs when we are processing a
**binary** node u.  When `emit` is called on a child `c` of a polytomous node p, `c`
may itself be a binary node, and it will reach the Type 2 check with `c.parent = p`
(polytomous).  We must skip Type 2 in that case:

```java
// At the Type 2 block, replace the existing isRoot check with:
if (!u.isRoot() && !u.parent.isPolytomous()) {
    TreeNode sib = u.getSibling();   // safe: parent is binary
    ...
}
```

This one-line guard prevents calling `getSibling()` on a child of a polytomous node,
while keeping the binary path entirely intact.

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

#### 3.8.4 GPU path with polytomy

The GPU prefix-sum kernel assumes `[lo, mid, hi]` (3 ints, binary-only).
The GPU smaller-side kernel packs `[treeIdx, lo1, hi1, lo2, hi2, sz1, sz2, sz3, freq]`
(9 ints, binary-only).  Neither layout supports d > 3.

**Decision**: when any gene tree contains a polytomous node, fall through to the CPU
path for all weight computation.

```java
// In WeightTable constructor, before GPU path selection:
if (useGPU && partTable.hasPolytomousPartitions()) {
    Logging.info("Input contains polytomous gene-tree nodes — GPU weight path disabled, using CPU");
    useGPU = false;
}
```

Add `boolean hasPolytomousPartitions()` to `PartitionTable` (returns true if any entry
has `d > 3`).

GPU polytomy support is left as a future enhancement.

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

ASTRAL-MP enriches X around each polytomous node by **random sampling**
(`WQDataCollection.java`, lines 172–227):

1. Build the polytomy's d parts: `children[0..k-1]` bitsets + complement bitset.
2. Pick one random taxon from each part (d taxa total).
3. For each gene tree, compute the bipartitions of that gene tree restricted to the d
   sampled taxa.
4. Expand each restricted bipartition back: replace each sampled taxon with the entire
   bitset of its originating arm.
5. Add the result as a new bipartition to X.
6. Repeat 3 times.

**Why this is non-trivial in our range-based system**: the expanded bipartitions in
step 4 are unions of non-adjacent arms (e.g. arm₀ ∪ arm₂), which are **non-contiguous**
sets and cannot be represented as a single `[lo, hi)` range in our postorder array.
Implementing this faithfully requires extending the cluster representation to support
multi-range unions — a significant refactor.

**Consequence for this implementation**: clusters that appear *exclusively* as polytomous
nodes in all gene trees will have no Mode 1 DP transitions.  They will still be resolved
by Mode 2 cross-tree transitions if any binary sub-cluster is available in X.  In the
worst case (cluster appears only as a complete polytomy across all trees with no helpful
binary gene trees), the DP will return score 0 for all possible resolutions of that
cluster, meaning any resolution is equally valid.

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
| `WeightTable.java` | YES | O(d) QI formula for all 3 numeric modes; GPU disable flag when any d>3 partition present |
| `IntersectionCounter.java` | NO | Existing `intersect()` handles any range pair |
| `Inference.java` | NO | DP solver operates on binary splits only — unchanged |
| `BipartitionSplit.java` | NO | Unchanged |
| `Cluster.java` | NO | Range cluster representation unchanged |
| `ClusterHash.java` | NO | Hash arithmetic unchanged |
| `PrefixHashArrays.java` | NO | Prefix sum arrays unchanged |
| GPU code (`gpu/`) | NO | GPU path disabled when polytomy detected; kernel unchanged |
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

### 8.3 Polytomous gene-tree test cases

Craft small inputs with polytomous nodes and verify:
- TreeParser accepts them without throwing.
- Correct d-partitions are extracted (correct d value, correct sizes).
- DP produces a binary species tree (even if scores are all 0 for unresolved polytomous clusters).

### 8.4 Polytomy simulator

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

5. `WeightTable`: implement O(d) QI formula; add `hasPolytomousPartitions()` check
   and GPU disable flag.
   Validate: binary inputs give bit-identical scores (d=3 path is numerically equivalent).

6. `DPTable`: add the one-line parent polytomy guard for Type 2.
   Validate: binary trees produce same transition counts.

7. Run full test suite: TC1–TC13 must all pass.

8. Add polytomous test cases (TC14+).

9. Polytomy simulator (`simulate_polytomous.py`).
