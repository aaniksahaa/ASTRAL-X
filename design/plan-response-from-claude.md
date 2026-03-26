# Rigorous Review of the ASTRAL-X Design Plan

## Overall Verdict

The plan is fundamentally sound and represents a genuine contribution: bringing STELAR-X's compact integer-tuple representation and GPU-accelerated weight precomputation into ASTRAL's quartet-based framework. The pipeline stages are logically ordered and the key data structures (Cluster with tuple representation, prefix-hash arrays, partition hashtable) are well-motivated.

However, there are issues spanning from a critical mathematical subtlety to several architectural concerns and missed optimizations. I organize these by severity.

---

## 1. CRITICAL: Rooted vs. Unrooted Confusion in the DP and Scoring

### The Problem

The plan says: *"since we will build a rooted tree, we consider candidates as bipartitions"* and then later: *"the internal node in rooted tree that induces X|Y, also actually induces the tripartition X|Y|L-X-Y"*.

This is correct in the sense that ASTRAL's DP *transitions* over bipartitions (splitting cluster A into A' and A\A'), but the *score* attached to each transition is the weight of the **tripartition** (A'|A\A'|S\A). The plan does state this correctly in the weight calculation section, but the language throughout conflates "candidate bipartition" (the DP object) with what gets scored (a tripartition). This risks a subtle implementation bug.

More importantly: **ASTRAL works on unrooted gene trees**. The plan says input trees are rooted, but then says there should be a flag to treat them as unrooted. For quartet scoring to be correct, each internal node of a gene tree (treated as unrooted) defines a tripartition (or d-partition for polytomies). If you treat gene trees as rooted and extract *bipartitions* (left-child | right-child), you are extracting the wrong structures for QI scoring. A rooted binary gene tree node induces a bipartition of its descendant taxa, not a tripartition of all taxa in that gene tree.

### The Fix

For gene tree partitions used in weight calculation, you must treat the gene trees as **unrooted**. Each internal node of the unrooted gene tree induces a tripartition (or d-partition). Concretely, for a rooted binary gene tree, each internal node u (except the root) induces:
- Left subtree leaves = M₁
- Right subtree leaves = M₂  
- Everything else in Lg = M₃ = Lg \ M₁ \ M₂

This third part M₃ is the **complement** w.r.t. Lg. The root's two children just give a bipartition (2-partition), which contributes zero quartets (since one side is empty in the tripartition sense — this is the W(A'|S-A'|∅) = 0 boundary in the ASTRAL paper).

So: the cluster extraction for gene tree partitions **must** produce complements (unrooted treatment), and the partition hashtable should store **tripartitions** (or d-partitions), not bipartitions. Your plan does mention this possibility but seems to waver. Be firm: gene tree partitions for scoring = unrooted tripartitions.

The **species tree** DP, on the other hand, searches over bipartitions of clusters (A' | A\A'), which then induce a tripartition (A' | A\A' | S\A) for scoring. This is the standard ASTRAL formulation and is correct.

### Recommendation

Make the distinction explicit in the architecture:
- **Gene tree partitions** (for scoring): always unrooted tripartitions/d-partitions. Each non-root internal node of a rooted gene tree produces a tripartition (left | right | complement-wrt-Lg).
- **Candidate splits** (for the DP): bipartitions (A' | A\A'), which *induce* the tripartition (A' | A\A' | S\A) for weight lookup.
- The Partition class should clearly mark whether the "whole set" is Lg (gene tree partition) or S (species tree tripartition).

---

## 2. IMPORTANT: The 3×3 Intersection Optimization — Your Count of 6 Is Off

### The Problem

The plan says: *"we kinda need to compute 6 intersections, and may infer the other three with subtraction"*.

Let us verify carefully. The 3×3 matrix for QI((X|Y|Z), (A|B|C)) where Z = S\X\Y and C = Lg\A\B is:

```
         A           B           C=Lg\A\B
X     |X∩A|       |X∩B|       |X∩C|
Y     |Y∩A|       |Y∩B|       |Y∩C|
Z     |Z∩A|       |Z∩B|       |Z∩C|
```

Row constraints (always valid since A,B,C partition Lg and X⊆S⊇Lg... wait, X is NOT necessarily a subset of Lg):

Actually, the row constraint says: |X∩A| + |X∩B| + |X∩C| = |X∩Lg| (not |X|), because A∪B∪C = Lg, not S. Elements of X outside Lg do not appear in any of A, B, C.

Similarly |Y∩A| + |Y∩B| + |Y∩C| = |Y∩Lg|.

And the column constraint says: |X∩A| + |Y∩A| + |Z∩A| = |A| (always valid since X∪Y∪Z = S ⊇ A).

So from the 2×2 top-left block {|X∩A|, |X∩B|, |Y∩A|, |Y∩B|} you can derive:

- |Z∩A| = |A| - |X∩A| - |Y∩A|  (column constraint, valid)
- |Z∩B| = |B| - |X∩B| - |Y∩B|  (column constraint, valid)
- |X∩C| = |X∩Lg| - |X∩A| - |X∩B|  (row constraint, needs |X∩Lg|)
- |Y∩C| = |Y∩Lg| - |Y∩A| - |Y∩B|  (row constraint, needs |Y∩Lg|)
- |Z∩C| from either row or column

So you need: **4 intersections** (the 2×2 block) + **2 restricted sizes** (|X∩Lg| and |Y∩Lg|), giving you all 9 values. That is potentially **6 computations**, but |X∩Lg| and |Y∩Lg| are simpler to compute than a general intersection — they are just "how many taxa of X are present in gene tree g", which can be precomputed once per (candidate cluster, gene tree) pair.

However, the plan says *"we cannot guarantee that X is a subset of Lg"* — this is correct but the plan then jumps to saying 6 full intersections are needed. In reality, you need only the 4 true intersections (|X∩A|, |X∩B|, |Y∩A|, |Y∩B|) plus the precomputable restricted sizes. This is a meaningful optimization.

### Recommendation

Precompute |X∩Lg| for each candidate cluster X and each gene tree g. This can be done efficiently using the position maps: iterate over taxa in the smaller of X or Lg and check membership in the other. Store these. Then during QI computation, you only need 4 actual cross-tree intersections per (candidate split, gene tree partition) pair.

---

## 3. IMPORTANT: What Gene Tree Partitions Go into the Partition Hashtable?

### The Problem

The plan says to hash gene tree tripartitions and store unique ones with frequencies. But it is not fully clear which partitions are extracted.

For ASTRAL with unrooted binary gene trees, each internal node (except root, when rooted arbitrarily) defines a tripartition. For a rooted binary gene tree with n_g leaves, there are (n_g - 1) internal nodes. When treated as unrooted (de-rooting), the root's two children merge into one node, giving (n_g - 2) tripartition-inducing nodes. Each such node defines a tripartition (left | right | complement).

The plan mentions registering "for each internal node... 2 clusters, the subtree range and its complement." But for the tripartition at an unrooted node, you need **three** parts: left subtree, right subtree, and everything else. The plan's cluster extraction (subtree + complement) gives you a bipartition, not a tripartition.

### The Fix

For each non-root internal node u of the rooted gene tree:
- Part 1: left child subtree range → Cluster(i, l_left, r_left, false)
- Part 2: right child subtree range → Cluster(i, l_right, r_right, false)
- Part 3 (implicit): complement of the union = Lg \ (Part1 ∪ Part2)

The tripartition/partition for scoring is then (Part1 | Part2 | Part3). Part3 is the complement of the range [l_left, r_right] (i.e., the parent's subtree complement w.r.t. Lg). This can be represented as Cluster(i, l_left, r_right, complement=true).

For the root node of a rooted gene tree: this corresponds to the de-rooting point. The root's two children give a bipartition of Lg, which corresponds to a tripartition with one empty side. ASTRAL sets W(A|S-A|∅) = 0, so this contributes zero weight and can be skipped.

### Recommendation

When building the partition hashtable, extract tripartitions as (leftRange | rightRange | complementOfParentRange) for each non-root internal node. Store these as Partition objects with 2 explicit clusters (left, right) and the implicit third (complement of their union w.r.t. Lg).

---

## 4. IMPORTANT: The DP Search Space Construction — Correctness of Hash Subtraction for Subset Testing

### The Problem

The plan proposes: for a fixed cluster A of size szA, check all clusters B of size sz < szA/2, compute hash(A) - hash(B), and look up whether this "residual hash" exists in the bin of size (szA - sz).

This is mathematically sound under the hash associativity assumption: if A = B ∪ C (disjoint), then hash_sum(A) = hash_sum(B) + hash_sum(C). So hash(A) - hash(B) should equal hash(C) if and only if C = A\B with high probability.

**But there is a subtle issue**: this only works if B ⊆ A. Just because hash(A) - hash(B) matches hash(C) in the table does not guarantee B ⊆ A — it could be a hash collision. Moreover, the plan does not discuss verifying that B ⊆ A.

With m independent hash pairs (sum + XOR), the collision probability per false match is ~1/2^(64m) for sum and independently ~1/2^(64m) for XOR. With m=2 (4 independent hashes), this is ~1/2^(256), which is negligible. So this is likely fine in practice, but the plan should acknowledge this.

### A More Serious Issue: Missing the Case sz = szA/2

The plan says "checking less or equal half suffices, since the other one will lie on the upper half." But when sz = szA/2 exactly, both parts have the same size, and you need to check pairs *within* the same bin. The plan should handle this edge case — you need to check all pairs (B₁, B₂) within bin(szA/2) such that hash(B₁) + hash(B₂) = hash(A).

### Recommendation

- Handle sz = szA/2 as a special case (pairs within the same bin).
- Since you have 2m independent hashes, the false positive rate is negligible, so no explicit subset verification is needed. But document this.
- Consider also the **reverse lookup**: instead of iterating over all B in bin(sz) and looking up the residual, build a hashmap for each bin and do O(1) lookups. This is what the plan seems to suggest, but make it explicit.

---

## 5. IMPORTANT: Wavelet Matrix Strategy — Memory and Complexity Analysis

### The Concern

The plan proposes: for each gene tree gᵢ, build wavelet matrices between gᵢ and all other gene trees, compute all needed intersections, then free the memory and move to the next gene tree.

Memory per round: O(nk log n) — storing k wavelet matrices, each of size O(n log n). This is fine.

But the **total work** needs careful analysis. For each gene tree gᵢ (contributing tripartitions), you need to score every candidate bipartition X|Y against every tripartition from gᵢ. Let T = number of unique tripartitions and C = number of candidate bipartitions. For each (candidate bip, tripartition from gᵢ) pair, you compute 4 intersections, each costing O(log n) with the wavelet matrix.

Total work per round: O(C × |tripartitions from gᵢ| × 4 × log n)
Total over all rounds: O(C × T × 4 × log n)

Compare with STELAR-X's approach (iterating over the smaller range): O(C × T × min(s,t)) per pair, which for balanced trees gives O(n²k²).

With wavelet matrices: O(C × T × log n). Since C = O(nk) and T = O(nk), this is O(n²k² log n) — actually *worse* by a log factor than STELAR-X's balanced-tree case!

### Where Wavelet Matrices Win

They win when the clusters are **large** — for balanced trees with cluster sizes ~n/2, STELAR-X's min(s,t) approach costs O(n/2) per intersection, while wavelet costs O(log n). But for small clusters (which are much more numerous), the iteration approach is faster.

### Recommendation

Consider a **hybrid approach**: 
- For small clusters (size < threshold), use the direct iteration method (STELAR-X style).
- For large clusters (size ≥ threshold), use wavelet matrices.
- The threshold should be around O(log n).

Alternatively, since the wavelet matrix approach requires O(k) rounds of GPU memory allocation/deallocation, the overhead might dominate for large k. Profile carefully.

---

## 6. MODERATE: Cluster Set X Construction — Missing ASTRAL-II/III Search Space Expansion

### The Problem

The plan constructs X from:
1. Subtree ranges and their complements from each gene tree (rooted or unrooted treatment)
2. "Super-complements" w.r.t. the total taxa set S

This gives you the basic X₀ (bipartitions from input gene trees) plus complements, which is sufficient for **statistical consistency** (ASTRAL-I level).

But ASTRAL-II and ASTRAL-III **expand** X using heuristics:
- Similarity matrix → UPGMA tree → add its bipartitions
- Greedy consensus at multiple thresholds → resolve polytomies → add bipartitions
- Pectinate trees based on similarity

These additions significantly improve accuracy, especially with few genes, high ILS, or many taxa (see ASTRAL-II Table 1: improvements up to 40% on high-ILS conditions).

### Recommendation

The plan should at minimum acknowledge this and design the Cluster/X infrastructure to support future additions. For a first version (proving scalability), the basic X₀ is fine. But for competitive accuracy, you will need to implement at least the similarity matrix + UPGMA expansion from ASTRAL-II. The similarity matrix computation itself requires O(n²k) and needs to be thought through in the compact-representation framework (you cannot use bitsets).

---

## 7. MODERATE: Partition Hashing — Order-Invariance Needs Care

### The Problem

The plan says: *"hash a partition, as simply the set of hashes of the constituent clusters, but note that, the order does not matter."*

For order-invariant hashing of a set of cluster hashes {h₁, h₂, ..., hₚ}, common approaches are:
- XOR of all hᵢ: fast but high collision rate (e.g., {A,B} and {C,D} collide if h_A ⊕ h_B = h_C ⊕ h_D)
- Sum of all hᵢ: same issue
- Sorted tuple then hash: good but requires sorting
- Symmetric polynomial hash: h₁·h₂ + h₁·h₃ + h₂·h₃ (for 3 elements) — better but expensive

Since you already have m independent hash functions, you can use multiple independent XOR/sum hashes over different seeds to reduce collision probability. But the plan should specify this explicitly.

### Recommendation

For tripartitions (2 explicit clusters + 1 implicit), hash as: sorted pair of the two explicit cluster hashes, then combine. Since the implicit third is determined by the two explicit ones plus the whole set, it does not need to be included in the hash. Sorting two hashes is trivial (just compare and swap).

For d-partitions with polytomies, sort all d-1 explicit cluster hashes and combine sequentially.

---

## 8. MODERATE: The "Candidate Bipartition" Scores vs. What ASTRAL Actually Precomputes

### The Problem

ASTRAL precomputes w(T) for each **tripartition** T = (A'|A\A'|S\A), not for each "candidate bipartition" (A'|A\A'). Different DP states can produce the same tripartition — for example, if the DP is at cluster A and splits into (A'|A\A'), and also at cluster B and splits into (B'|B\B'), these produce different tripartitions if S\A ≠ S\B.

This means you **cannot** precompute a single score per candidate bipartition (A'|A\A') — the score depends on which parent cluster A it came from, because the third part Z = S\A changes.

Wait — actually, looking at ASTRAL's formulation more carefully: the weight w(A'|A\A'|S\A) depends on the tripartition, which depends on A (the DP state). So the weight is a function of three arguments, not two.

However, in ASTRAL, they precompute w for each tripartition in the search space Y = {(A', A\A', S\A) : A'⊂A, A'∈X*, A∈X*, A\A'∈X*}. The number of such tripartitions is O(|X|²).

### The Implication for Your Plan

Your plan talks about precomputing scores for "candidate bipartitions," but actually you need to precompute for **tripartitions** (which include the context of what S\A is). This is important — the same pair (X, Y) might appear in different DP states (with different parent clusters), and the score changes each time because Z = S \ parentCluster changes.

In practice, many of these tripartitions will share the same third part, but the plan should be architecturally aware of this.

### Recommendation

The weight precomputation should be indexed by tripartitions (all three parts), not just bipartitions. Alternatively, during the DP traversal, compute the weight on-the-fly for each (A', A\A', S\A) encountered, rather than trying to precompute all of them. This is what ASTRAL-II/III actually does — it computes weights lazily as the DP encounters new tripartitions.

This is actually a significant design decision. STELAR-X can precompute all bipartition weights because the score depends only on the bipartition (A|B), not on context. In ASTRAL, the score depends on the tripartition (A|B|C), which is context-dependent. You may need to either:
(a) Precompute all O(|X|²) tripartition weights (expensive in memory), or
(b) Compute weights on-the-fly during the DP (ASTRAL's approach), or
(c) Cache computed weights in a hash table keyed by tripartition hash (lazy evaluation with memoization).

Option (c) is probably best for your architecture — it allows GPU-batched computation of weights as they are needed.

---

## 9. MINOR: Missing Taxa Handling in Prefix Hash Arrays

### The Concern

The plan says to use -1 for missing taxa in the post-order arrays. For prefix sum/XOR hashes, you need to ensure that -1 entries contribute nothing. Two approaches:

1. Map -1 to hash value 0 (for both sum and XOR, 0 is the identity element). This works but means hash(0) for a real taxon must never be 0, which is almost certainly true with SplitMix64 (probability 1/2^64).

2. Skip -1 entries in the prefix computation (only increment the prefix when the entry is valid).

Approach 1 is simpler and GPU-friendlier (no branching). Just ensure that the prefix arrays have the same length n for all gene trees, with 0-hash padding for missing taxa.

### Recommendation

Use approach 1. Set hash(-1) = 0 for all seeds. Document the (negligible) assumption that no real taxon hash is 0.

---

## 10. MINOR: The 1/2 Factor in Quartet Scoring

### The Concern

ASTRAL's formulation has w(T) = (1/2) Σ QI(T, M). The DP score C(S) = 2 × (actual quartet score). The plan mentions the QI formula but does not discuss where the 1/2 factor is applied.

In ASTRAL's DP, the 1/2 is folded into the weight definition. The DP accumulates C(S) = Σ w(T_node) for all nodes in the optimal tree, and the final quartet score is C(S)/2. But during the DP maximization, the 1/2 does not affect which tree is optimal (it is a global constant), so you can either:
- Include the 1/2 in w and report C(S)/2
- Omit the 1/2 in w and report C(S)/4
- Work with 2×QI throughout (no division at all) and just divide at the very end

### Recommendation

For simplicity and to avoid floating point, work with integer scores throughout: compute Σ QI(T, M) without the 1/2, and only divide by 2 when reporting the final score. This is what ASTRAL implementations typically do.

---

## 11. ARCHITECTURAL SUGGESTION: Pipeline Summary

Based on the above, here is a cleaned-up pipeline:

```
Phase 0: Parse & Preprocess
  - Parse Newick trees (multithreaded, as in STELAR-X)
  - Build post-order arrays + position maps
  - Compute single-taxon hashes (m seeds × n taxa)
  - Build prefix-hash arrays (2m arrays per gene tree)

Phase 1: Extract Clusters → Build X
  - For each gene tree, bottom-up traversal:
    - Register range clusters (subtree ranges)
    - Register complement clusters (for unrooted treatment)
  - Deduplicate via cluster hash table
  - Add super-complements w.r.t. S
  - [Future: similarity matrix → UPGMA → expand X]

Phase 2: Extract Gene Tree Partitions
  - For each non-root internal node of each gene tree:
    - Build tripartition (left | right | complement)
    - Hash it (order-invariant)
    - Deduplicate in partition hash table with frequency

Phase 3: Build DP Search Space
  - For each cluster A ∈ X, find all valid splits A' | (A\A')
    where both A' and A\A' are in X
  - Use size-binning + hash subtraction (CPU or GPU)
  - Store as: cluster A → list of (A', A\A') pairs

Phase 4: Weight Computation (GPU-accelerated)
  - For each tripartition encountered during DP:
    - Score against all unique gene tree partitions (with frequencies)
    - Use 4 intersections + row/column constraints for 3×3 matrix
    - Intersection method: hybrid (direct iteration for small 
      clusters, wavelet matrix for large clusters)
  - Cache computed weights (memoization by tripartition hash)

Phase 5: Inference DP
  - Top-down from S, recursively split clusters
  - At each cluster A, try all valid splits, pick max-weight
  - Backtrack to build the species tree
  - Output in Newick format

Phase 6: [Future] Polytomy handling, branch support (localPP)
```

---

## Summary of Key Action Items

| # | Issue | Severity | Action |
|---|-------|----------|--------|
| 1 | Rooted vs unrooted confusion | Critical | Gene tree partitions must be unrooted tripartitions |
| 2 | Intersection count (6 vs 4+2) | Important | Precompute |X∩Lg| per (cluster, gene tree); only 4 cross-tree intersections needed |
| 3 | Gene tree partition extraction | Important | Extract tripartitions (left\|right\|complement), not just bipartitions |
| 4 | Hash subtraction correctness | Important | Handle sz=szA/2 case; document collision bounds |
| 5 | Wavelet matrix complexity | Important | Hybrid approach; wavelet only wins for large clusters |
| 6 | Search space expansion (X) | Moderate | Design X infrastructure for future ASTRAL-II/III heuristics |
| 7 | Partition hash order-invariance | Moderate | Sort pair of cluster hashes before combining |
| 8 | Tripartition vs bipartition scores | Moderate | Scores depend on tripartitions (context-dependent); use lazy+memoize |
| 9 | Missing taxa in prefix arrays | Minor | Map -1 → hash 0 |
| 10 | The 1/2 factor | Minor | Work with integers, divide at end |