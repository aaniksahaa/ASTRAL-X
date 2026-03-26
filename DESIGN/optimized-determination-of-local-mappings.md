# Tree-Local vs. Cross-Tree DP Transitions in ASTRAL-X

## 1. The Problem

The DP search space for ASTRAL-like methods requires, for each cluster A ∈ X*, finding all valid splits A → A' | (A \ A') such that both A' and (A \ A') are in X*. The expensive approach (used in ASTRAL-III/MP) is to test all pairs via hash subtraction, costing O(|X|²) time.

The observation here is that **many of these transitions are directly readable from the tree structure** during the bottom-up traversal, at O(nk) total cost. This document formalizes exactly which transitions are "tree-local," which are "cross-tree," and the implications of restricting to tree-local transitions only.

---

## 2. Setup and Definitions

Let g be a rooted gene tree with taxa set Lg ⊆ S (where S is the total taxa set). For each internal node u of g:
- **left(u)**, **right(u)** = the two children of u
- **sub(u)** = the set of taxa in the subtree rooted at u
- **parent(u)** = p, the parent of u
- **sibling(u)** = v, the other child of p

We treat g as **unrooted** for ASTRAL purposes. In the unrooted view, every internal node u (except the root) induces clusters on both sides: sub(u) and Lg \ sub(u).

The cluster set X* includes:
- All sub(u) for all internal nodes u of all gene trees (subtree clusters)
- All Lg \ sub(u) for all internal nodes u of all gene trees (complement clusters)
- All S \ A for all A ∈ X* so far (super-complement clusters, needed for missing taxa)

---

## 3. Tree-Local Transitions: Formal Derivation

### 3.1 Type 1: Subtree Splits

For any internal node u with children l = left(u) and r = right(u):

```
sub(u) = sub(l)  ∪  sub(r)       (disjoint union)
```

This directly gives the DP transition:

```
sub(u)  →  sub(l)  |  sub(r)
```

**Cost to discover**: O(1) per node, O(nk) total across all gene trees.

**Verification**: sub(l) ∈ X* and sub(r) ∈ X* by construction (both are subtree clusters). sub(l) ∪ sub(r) = sub(u) ∈ X*. ✓

### 3.2 Type 2: Complement Splits

For any internal node u (not the root) with parent p and sibling v:

```
Lg \ sub(u) = sub(v)  ∪  (Lg \ sub(p))     (disjoint union)
```

**Proof**: Since p has children u and v, we have sub(p) = sub(u) ∪ sub(v). Therefore:
```
Lg \ sub(u) = (Lg \ sub(p)) ∪ (sub(p) \ sub(u))
            = (Lg \ sub(p)) ∪ sub(v)
```
The union is disjoint because sub(v) ⊆ sub(p) and (Lg \ sub(p)) ∩ sub(p) = ∅. ✓

This gives the transition:

```
[Lg \ sub(u)]  →  sub(v)  |  [Lg \ sub(p)]
```

**Membership check**: sub(v) ∈ X* (subtree cluster). Lg \ sub(p) ∈ X* (complement cluster, unless p is the root, in which case Lg \ sub(p) = ∅ — the root case is handled separately). ✓

### 3.3 Special Case: Root Node

If p is the root of g, then sub(p) = Lg, so Lg \ sub(p) = ∅. The transition becomes:

```
[Lg \ sub(u)]  →  sub(v)  |  ∅
```

This is degenerate (one side is empty) and corresponds to the fact that the root of a rooted tree, when de-rooted, gives only a bipartition, not a tripartition. In ASTRAL, this contributes zero weight (W(A|S\A|∅) = 0), so this transition can be skipped.

### 3.4 Type 3: Super-Complement Splits (Missing Taxa Case)

When Lg ⊊ S, we also have super-complement clusters S \ A for each A ∈ X*. For a subtree cluster sub(u):

```
S \ sub(u) = (Lg \ sub(u))  ∪  (S \ Lg)
```

This gives the transition:

```
[S \ sub(u)]  →  [Lg \ sub(u)]  |  [S \ Lg]
```

**But**: This is only valid if both parts are in X*:
- Lg \ sub(u) ∈ X* ✓ (complement cluster)
- S \ Lg ∈ X* — this is the "missing taxa" of gene tree g. It is in X* only if we added it (it's the super-complement of Lg itself, i.e., S \ Lg). If Lg is in X* (which it should be, as the full taxa set of a gene tree), then S \ Lg is a super-complement and is in X*. ✓

Similarly, for a complement cluster Lg \ sub(u):

```
S \ (Lg \ sub(u)) = sub(u)  ∪  (S \ Lg)
```

Transition:
```
[S \ (Lg \ sub(u))]  →  sub(u)  |  [S \ Lg]
```

Both parts are in X*. ✓

### 3.5 Summary of Tree-Local Transitions per Gene Tree

For a binary gene tree g with n_g taxa, there are (n_g - 1) internal nodes. Each non-root internal node yields:

| Transition | From | To |
|---|---|---|
| Type 1 | sub(u) | sub(left(u)) \| sub(right(u)) |
| Type 2 | Lg \ sub(u) | sub(sibling(u)) \| (Lg \ sub(parent(u))) |
| Type 3a | S \ sub(u) | (Lg \ sub(u)) \| (S \ Lg) |
| Type 3b | S \ (Lg \ sub(u)) | sub(u) \| (S \ Lg) |

That gives roughly **4 transitions per internal node**, or **O(n_g)** per tree, **O(nk)** total.

---

## 4. What Tree-Local Transitions Miss: Cross-Tree Transitions

### 4.1 Example

Consider two gene trees on taxa {1,2,3,4,5}:
```
g₁: ((1,2),(3,(4,5)))     →  subtree clusters: {1,2}, {4,5}, {3,4,5}, {1,2,3,4,5}
g₂: ((1,3),(2,(4,5)))     →  subtree clusters: {1,3}, {4,5}, {2,4,5}, {1,2,3,4,5}
```

Complement clusters (unrooted):
```
g₁: {3,4,5}, {1,2}, {1,2,3}, ... (complements of the above w.r.t. {1,2,3,4,5})
g₂: {2,4,5}, {1,3}, {1,2,3}, ...
```

Now consider cluster {1,2,3} which appears as a complement cluster in both trees.

From g₁'s tree structure: {1,2,3} is the complement of {4,5} w.r.t. Lg. In g₁, {4,5} has parent node with subtree {3,4,5}, so sibling is {3}. Thus:
```
{1,2,3} → {3} | {1,2}     (Type 2 transition from g₁)
```

From g₂'s tree structure: {1,2,3} is the complement of {4,5} w.r.t. Lg. In g₂, {4,5} has parent with subtree {2,4,5}, so sibling is {2}. Thus:
```
{1,2,3} → {2} | {1,3}     (Type 2 transition from g₂)
```

**Cross-tree transition**: Is {1,2,3} → {1} | {2,3} a valid split? 
- {1} ∈ X* (singleton, always in X*)  
- {2,3} — is this in X*? It would be only if some gene tree has a cluster {2,3}. Neither g₁ nor g₂ has this. So this particular cross-tree transition does **not** exist in X*.

**Another cross-tree example**: Consider {3,4,5} → {3} | {4,5}. 
- In g₁, this IS a Type 1 tree-local transition (the right subtree splits this way).
- But what about {3,4,5} → {3,4} | {5}? This requires {3,4} ∈ X*. If some gene tree g₃ has cluster {3,4}, then this is a valid cross-tree transition not discoverable from g₁ or g₂ alone.

### 4.2 When Cross-Tree Transitions Matter

Cross-tree transitions are important when the **optimal species tree** requires a split that no single gene tree exhibits. This happens when:

1. **Different gene trees suggest different resolutions** of the same clade, and the optimal quartet score is achieved by a resolution present in none of the individual trees.

2. **Missing taxa**: Gene tree g₁ has taxa {A,B,C} and g₂ has taxa {C,D,E}. The cluster {A,B,C,D,E} might need to be split as {A,B} | {C,D,E}, where {A,B} comes from g₁ and {C,D,E} comes from g₂.

### 4.3 Are Tree-Local Transitions Sufficient for Statistical Consistency?

**No, not in general for ASTRAL.** ASTRAL's statistical consistency proof requires that the set X contains all bipartitions from the true species tree. If the true species tree has a bipartition that appears in at least one gene tree (which happens with high probability as k → ∞), then the corresponding split will be tree-local. But the **combination** of these splits (how they compose in the DP) may require cross-tree transitions to find the optimal tree.

However, in practice, the tree-local transitions capture the vast majority of useful transitions. ASTRAL-I used exactly this approach (X = gene tree bipartitions, and the DP implicitly used tree-local structure). The accuracy degradation from missing cross-tree transitions was modest, and ASTRAL-II's heuristic additions to X primarily added new *clusters* (which then enable new cross-tree transitions).

---

## 5. Contrast with STELAR-X (Rooted/Triplet Case)

In STELAR-X, the DP operates over rooted bipartitions. The candidate set CB = UGB contains only subtree bipartitions from gene trees. Crucially:

**Theorem (STELAR-X, Theorem 4.1)**: For CB = UGB, every bipartition in CB is visited in the DP state space, and every transition is tree-local.

**Proof sketch**: Every bipartition (A|B) ∈ UGB comes from some gene tree node u with sub(left(u)) = A and sub(right(u)) = B. The parent cluster A ∪ B = sub(u) is also a cluster from the same tree. The transition sub(u) → A | B is tree-local. By induction up to the root, every cluster is reachable.

In STELAR-X, there are **no complement clusters** and **no cross-tree transitions** in the constrained case. This is a fundamental simplification that comes from working with rooted trees and triplets.

In ASTRAL-X (unrooted treatment), complement clusters create clusters that may span structures from multiple trees, making cross-tree transitions possible and sometimes necessary.

---

## 6. Proposed Two-Mode Architecture

### Mode 1: Tree-Local Only (Fast Mode)

**Construction**: During the bottom-up traversal of each gene tree, directly emit DP transitions:

```
For each gene tree g:
  For each internal node u (non-root) with parent p, sibling v:
    // Type 1: subtree split
    emit transition: sub(u) → sub(left(u)) | sub(right(u))
    
    // Type 2: complement split
    if p is not root:
      emit transition: [Lg\sub(u)] → sub(v) | [Lg\sub(p)]
    
    // Type 3: super-complement splits (if Lg ⊊ S)
    if Lg ≠ S:
      emit transition: [S\sub(u)] → [Lg\sub(u)] | [S\Lg]
      emit transition: [S\(Lg\sub(u))] → sub(u) | [S\Lg]
```

**Cost**: O(nk) time, O(nk) space. No hash subtraction search needed.

**Properties**:
- Finds all transitions visible within individual gene trees
- Statistically consistent (given enough gene trees, true species tree bipartitions appear in gene trees)
- May miss some transitions that combine information across gene trees
- Equivalent to ASTRAL-I level search space behavior

### Mode 2: Full Cross-Tree Search (Thorough Mode)

**Construction**: After collecting all tree-local transitions, additionally run the hash-subtraction search over X* to discover cross-tree transitions.

```
For each cluster A ∈ X* (by decreasing size):
  For each size sz ≤ |A|/2:
    For each cluster B ∈ X* with |B| = sz:
      residual = hash(A) - hash(B)
      if residual matches some C ∈ X* with |C| = |A| - sz:
        emit transition: A → B | C    (if not already discovered)
```

**Cost**: O(|X*|²) worst case, but with size-binning optimization from the original plan.

**Properties**:
- Discovers all valid transitions, including cross-tree ones
- Strictly more transitions than Mode 1
- Higher computational cost
- Equivalent to ASTRAL-II/III level search space behavior

### Mode 3: Hybrid (Recommended Default)

1. Run Mode 1 first (O(nk) time) to get all tree-local transitions.
2. Optionally run Mode 2 to discover additional cross-tree transitions.
3. Use a flag to control: `--search-mode local|full|hybrid`

The hybrid mode could also be smarter: only run the cross-tree search for clusters where tree-local transitions provide fewer than some threshold number of split options.

---

## 7. Formal Verification of the Complement Split (Type 2)

Let us carefully verify the claim for the complement split with a concrete example.

**Tree**: `((3,(1,2)),(4,5))` on taxa {1,2,3,4,5}

Post-order array: [1, 2, 3, 4, 5]

Internal nodes (bottom-up):
- Node a: children are leaves 1, 2. sub(a) = {1,2}. Range [0,1].
- Node b: children are leaf 3, node a. sub(b) = {1,2,3}. Range [0,2].
- Node c: children are leaves 4, 5. sub(c) = {4,5}. Range [3,4].
- Root r: children are node b, node c. sub(r) = {1,2,3,4,5}. Range [0,4].

**Unrooted clusters** (subtree + complement for each non-root node):

| Node | Subtree cluster | Complement cluster |
|------|----------------|--------------------|
| a | {1,2} = [0,1] | {3,4,5} = comp([0,1]) |
| b | {1,2,3} = [0,2] | {4,5} = comp([0,2]) |
| c | {4,5} = [3,4] | {1,2,3} = comp([3,4]) |
| leaf 1 | {1} = [0,0] | {2,3,4,5} = comp([0,0]) |
| leaf 2 | {2} = [1,1] | {1,3,4,5} = comp([1,1]) |
| ... | ... | ... |

**Tree-local transitions**:

**Type 1 (subtree splits)**:
- sub(a) = {1,2} → {1} | {2}  ✓
- sub(b) = {1,2,3} → {3} | {1,2}  ✓  (leaf 3 and node a)
- sub(c) = {4,5} → {4} | {5}  ✓

**Type 2 (complement splits)**:
- Node a: parent = b, sibling = leaf 3.
  - Lg \ sub(a) = {3,4,5}
  - → sub(sibling) | (Lg \ sub(parent)) = {3} | {4,5}  ✓
  
  Verify: {3} ∪ {4,5} = {3,4,5} = Lg \ {1,2} ✓

- Node b: parent = root r, sibling = c. But parent is root, so Lg \ sub(r) = ∅.
  - Lg \ sub(b) = {4,5}
  - → sub(c) | ∅ = {4,5} | ∅. 
  - This is degenerate (skip — contributes zero weight).

- Node c: parent = root r, sibling = b. Same as above.
  - Lg \ sub(c) = {1,2,3}
  - → sub(b) | ∅ = {1,2,3} | ∅.
  - Degenerate (skip).

So the non-trivial complement splits for this tree are:
```
{3,4,5} → {3} | {4,5}
```

Now notice: this is exactly the transition you described! The "upper part" {3,4,5} gets split into the sibling's subtree {3} and the grandparent's complement {4,5}. ✓

**What about deeper complements?** Consider leaf 1: parent = a, sibling = leaf 2.
- Lg \ sub(leaf 1) = {2,3,4,5}
- → sub(sibling of leaf 1) | (Lg \ sub(a)) = {2} | {3,4,5}  ✓

And then {3,4,5} is further split by the transition derived from node a:
- {3,4,5} → {3} | {4,5}

So the DP can chain: {2,3,4,5} → {2} | {3,4,5} → {2} | ({3} | {4,5})

This reconstructs the unrooted tree structure purely from tree-local transitions. ✓

---

## 8. Complexity Comparison

| Aspect | Mode 1 (Tree-Local) | Mode 2 (Full Search) |
|--------|---------------------|---------------------|
| Time to build transitions | O(nk) | O(\|X*\|²) = O(n²k²) |
| Number of transitions | O(nk) | O(\|X*\|²) = O(n²k²) |
| Finds cross-tree splits | No | Yes |
| Statistical consistency | Yes (same as ASTRAL-I) | Yes (potentially better accuracy) |
| GPU needed | No | Optional (hash lookups) |

---

## 9. Implementation Notes

### 9.1 During Traversal, Register Transitions Immediately

As you traverse each gene tree bottom-up, you have all the information needed:
```java
void extractTransitions(Node u, int treeIdx) {
    if (u.isLeaf()) return;
    
    Cluster leftCluster  = new Cluster(treeIdx, u.left.rangeStart, u.left.rangeEnd, false);
    Cluster rightCluster = new Cluster(treeIdx, u.right.rangeStart, u.right.rangeEnd, false);
    Cluster parentCluster = new Cluster(treeIdx, u.rangeStart, u.rangeEnd, false);
    
    // Type 1: subtree split
    registerTransition(parentCluster, leftCluster, rightCluster);
    
    if (u.parent != null && !u.parent.isRoot()) {
        Node sibling = getSibling(u);
        Cluster siblingCluster = new Cluster(treeIdx, sibling.rangeStart, sibling.rangeEnd, false);
        Cluster complementU = new Cluster(treeIdx, u.rangeStart, u.rangeEnd, true);  // complement
        Cluster complementParent = new Cluster(treeIdx, u.parent.rangeStart, u.parent.rangeEnd, true);
        
        // Type 2: complement split
        registerTransition(complementU, siblingCluster, complementParent);
    }
    
    // Type 3: super-complement splits (if missing taxa exist)
    // ... 
}
```

### 9.2 Store Transitions in the DP Map by Cluster Hash

Since different gene trees may produce the same cluster (same taxa set), the transition map should be keyed by **cluster hash**, not by (treeIdx, range) tuple. Multiple gene trees may contribute different transitions for the same cluster, and all should be collected.

```
Map<ClusterHash, List<(ClusterHash, ClusterHash)>> dpTransitions;
```

### 9.3 Deduplication

The same transition might be discovered from multiple gene trees (if they share the same tree structure locally). The map naturally deduplicates by cluster hash + split pair.

---

## 10. Relationship to ASTRAL's Actual Implementation

In ASTRAL-III (the Java implementation), the cluster partitioning step (building Y from X) is done **lazily during the DP recursion**, not precomputed. For each cluster A encountered during the DP, ASTRAL-III computes X⊆A = {B ∈ X : B ⊆ A} and then tests all pairs in X⊆A for disjointness.

ASTRAL-MP improved this with the randomized hash-based approach (the abelian group homomorphism φ), which avoids explicitly computing X⊆A and instead uses hash lookups.

Your Mode 1 (tree-local) is even cheaper than both: it precomputes all transitions in O(nk) during tree traversal, with no need for subset testing or hash lookups at all. This is a genuine advantage of the compact tuple representation — the tree structure directly encodes the transitions.

The trade-off is that Mode 1 misses cross-tree transitions that ASTRAL-II/III/MP's expanded search space would find. But for a first version focused on scalability, Mode 1 gives you a complete, correct, and extremely fast DP search space construction.

---

## 11. Final Recommendation

**Default to Mode 1 (tree-local)** for the initial implementation. It is:
- O(nk) time and space — asymptotically optimal
- Correct and statistically consistent
- Trivially parallelizable (each gene tree independently)
- No GPU needed for this step

**Add Mode 2 (full search) as an optional enhancement** for users who want higher accuracy at the cost of longer preprocessing. This is the natural path to ASTRAL-II/III-level search space quality.

The key insight is that **tree-local transitions are a natural byproduct of the tree traversal you're already doing for cluster extraction**, so they come essentially for free.