# ASTRAL-X — Remaining Major Tasks

Current status: end-to-end pipeline is working correctly on complete binary gene trees
(37 taxa / 200 trees and 200 taxa / 1000 trees verified, CPU multi-threaded + GPU CUDA).

The items below are not yet implemented. They are grouped by what they fix or improve.

---

## 1. Incomplete Gene Tree Support (Missing Taxa)

**What**: Gene trees where some taxa are absent (`Lg ⊊ S`).
**Why now**: All three test datasets happen to be complete, but real-world inputs routinely have missing taxa.

### 1a. Super-Complement Clusters in ClusterTable

- When `Lg ≠ S`, for each cluster `A ∈ X` from gene tree `g`, the set `S \ A` is a new cluster not equal to `Lg \ A`.
- Currently skipped with comment "no new clusters added for complete trees".
- Must compute hash via `allTaxaHash - clusterHash` (sum) and `allTaxaHash ^ clusterHash` (XOR) and insert into `ClusterTable` if not already present.

**File**: `src/astralx/cluster/ClusterTable.java`

### 1b. Type 3 DP Transitions in DPTable

Two new transition types that only matter when `Lg ≠ S`:

| Transition | From | To |
|---|---|---|
| Type 3a | `S \ sub(u)` | `(Lg \ sub(u))` \| `(S \ Lg)` |
| Type 3b | `S \ (Lg \ sub(u))` | `sub(u)` \| `(S \ Lg)` |

- Currently neither is emitted during tree traversal.
- `S \ Lg` (the "missing taxa" cluster of gene tree g) must be in `X` for these to be valid — it is, because it's the super-complement of `Lg` itself.

**File**: `src/astralx/dp/DPTable.java`

### 1c. Corrected Column Constraint in Weight Calculation

For complete trees: `c0 = sz1 - a0 - b0` works because `|A| = |A ∩ Lg|`.
For incomplete trees the correct formula is:

```
c0 = |A ∩ Lg| - a0 - b0
c1 = |B ∩ Lg| - a1 - b1
```

`|A ∩ Lg|` and `|B ∩ Lg|` must be precomputed once per (partition, candidate-split) pair via `intersect(Lg_range, A_range)` — but since `Lg` is a contiguous range `[0, leafCount)` in tree `tGT`, this reduces to just `coreIntersect(tGT, 0, leafCount, tA, loA, hiA, loComp)`.

Must be fixed in both:
- **`src/astralx/weight/WeightTable.java`** (CPU path, `computeScore` method)
- **`src/native/astralx_weight.cu`** (GPU kernel `computeWeightsKernel`)

---

## 2. Polytomy Handling in Gene Trees

**What**: Gene trees with internal nodes having more than 2 children (multifurcating / non-binary nodes).
**Why**: Real gene tree inference tools (e.g. IQ-TREE, RAxML with collapsed branches) can produce polytomies.

### 2a. TreeParser — Non-Binary Node Support

- Currently throws `RuntimeException` when a node has `children.size() != 2`.
- Must be relaxed: allow any number of children ≥ 2 for internal nodes.
- A polytomous node with `d` children induces a d-partition of the gene tree's taxa.

**File**: `src/astralx/tree/TreeParser.java`

### 2b. PartitionTable — d-Partition Extraction

- Currently only binary internal nodes produce a tripartition (3-part partition).
- For a node with `d` children, must extract a d-partition: the `d` subtree ranges plus the complement (everything else in `Lg`).
- The `Partition` class stores only `leftStart/leftEnd`, `rightStart/rightEnd`; needs generalization to a variable-length list of parts.
- Alternatively: resolve polytomies into all possible binary refinements (common ASTRAL approach) — simpler but less faithful.

**File**: `src/astralx/partition/Partition.java`, `src/astralx/partition/PartitionTable.java`

### 2c. Weight Calculation — d-Part QI Formula

The current QI formula sums over 6 permutations of `{0,1,2}` (fixed for d=3).
For a d-partition `(M1 | M2 | ... | Md)` the formula generalizes to:

```
QI(T, M) = sum over all ordered triples (i,j,k) with i,j,k distinct, each in [1..d]:
             a[i] * b[j] * c[k] * (a[i] + b[j] + c[k] - 3) / 2
```

This is `d*(d-1)*(d-2)` terms instead of 6. Must be handled in both CPU and GPU paths.

**Files**: `src/astralx/weight/WeightTable.java`, `src/native/astralx_weight.cu`

---

## 3. Cross-Tree DP Transitions (Mode 2 — Full Search)

**What**: Currently only tree-local transitions are discovered (`O(nk)` total). Mode 2 additionally finds all valid splits `A → B | (A\B)` by hash subtraction across the entire cluster set `X`, even when `A`, `B`, `A\B` come from different gene trees.
**Why**: Equivalent to ASTRAL-II/III search space quality — higher accuracy, especially for lower gene tree coverage.

### 3a. Hash-Subtraction Search (CPU)

Algorithm (size-binned):
```
For each cluster A in X (by decreasing size):
  For each size sz in [1 .. |A|/2]:
    For each cluster B in bin(sz):
      residual_hash = hash(A) - hash(B)
      if residual_hash exists in bin(|A| - sz):
        emit transition: A → B | (A \ B)   [if not already discovered]
```

**File**: `src/astralx/dp/DPTable.java` (new method `buildCrossTreeTransitions`)

### 3b. CLI Flag

Add `--search-mode local|full` (default: `local`). `Config.java` already has the pattern for boolean flags.

**Files**: `src/astralx/Config.java`, `src/astralx/Main.java`

### 3c. GPU Hash-Set Acceleration (Optional)

The hash-subtraction lookup in 3a can be accelerated on GPU using a GPU hash set.
Working reference code is available in `ref-cuda/` (GPU hash-set kernel).
Only worth implementing if Mode 2 CPU speed is a bottleneck.

**New file**: `src/native/astralx_dp.cu` (GPU hash-set build + lookup)

---

## 4. Wavelet Matrix Intersection Counting

**What**: Replace current `O(min(|A|, |M|))` direct iteration intersection counting with GPU wavelet matrix giving `O(log n)` per query.
**Why**: For large `n` (hundreds of taxa), intersection counting is the inner loop bottleneck. Wavelet matrix trades build time for faster individual queries.

### Strategy (from design docs)

Process one reference gene tree `g_i` at a time:
1. Build wavelet matrix for `g_i` vs all other trees → `O(nk log n)` memory for this round.
2. Batch all intersection queries involving `g_i` as one side → launch single GPU kernel.
3. Free wavelet matrices for this round.
4. Repeat for each reference tree.

This keeps peak GPU memory at `O(nk log n)` instead of `O(n^2 k log n)`.

Working GPU wavelet matrix code is already available at `ref-cuda/wavelet_matrix.cu`.

### Implementation

- Port `ref-cuda/wavelet_matrix.cu` into `src/native/astralx_wavelet.cu`
- Add batch query kernel: given array of `(tree1, l1, r1, tree2, l2, r2)` queries, return intersection counts
- Integrate into `src/astralx/gpu/GPUWeightCalculator.java` as an alternative to current direct-iteration kernel

**New files**: `src/native/astralx_wavelet.cu`, updated `src/astralx/gpu/GPUWeightCalculator.java`

---

## 5. GPU Memory Batching for Very Large Inputs

**What**: Current kernel allocates all splits and all partitions in one GPU allocation.
**Why**: For very large inputs (1000+ taxa, 10000+ trees), the combined size of `splits[N×10]` + `parts[P×9]` + `orderings[T×n]` may exceed available VRAM (currently 6GB on RTX 3050).

### Approach

Option A — **batch over splits**: divide splits into chunks of size `B`, launch `ceil(numSplits/B)` kernel calls, accumulate results.
Option B — **batch over partitions**: divide partitions into chunks, accumulate partial `twoScores` across calls.

Option B is preferable because `orderings` and `invIndex` (the large `T×n` arrays) only need to be uploaded once. Only the `parts` array needs to be re-uploaded per batch.

**Files**: `src/astralx/weight/WeightTable.java` (GPU path), `src/native/astralx_weight.cu`

---

---

## 6. Similarity Matrix

**What**: Build the `SimilarityMatrix` (quartet co-occurrence score) in addition to / instead of the current `DistanceMatrix`. This is ASTRAL-MP's **default** matrix type.

**Why**: `SimilarityMatrix[i][j]` = average quartet co-occurrence score between taxon i and j, which is directly aligned with what ASTRAL's quartet objective maximizes. The current `DistanceMatrix` (branch-count distance) is ASTRAL-MP's non-default `--ustar-dist` mode. For accuracy, the similarity matrix is the correct signal to use.

**Algorithm** (`SimilarityMatrix.populateByQuartetDistance` in ASTRAL-MP):

For each gene tree, at each internal node v with children C₁, C₂, ... and "other" subtree:
```
for each pair of groups (i, j):
    sim = totalPairs - lcp_i - rcp_j      // fully resolved quartet count
    for a in group_i, b in group_j:
        matrix[a][b] += sim
        denom[a][b]  += (n_tree - 2) * (n_tree - 3) / 2
After all trees: matrix[i][j] /= (denom[i][j] / 2)
```

**Note on GPU difficulty**: DistanceMatrix maps cleanly to a per-pair LCA formula (easy GPU). SimilarityMatrix requires a per-node scatter over all child-pair combinations (harder GPU — non-trivial reformulation needed). A CPU implementation is straightforward first.

**Files**: new `src/astralx/completion/SimilarityMatrix.java`, `src/astralx/completion/SimilarityMatrixBuilder.java`

---

## 7. UPGMA Guide Tree → X Enrichment

**What**: After building the (Similarity or Distance) matrix, run UPGMA on it to produce one species-tree estimate, then inject all its n−2 bipartitions into `ClusterTable` (X). This is ASTRAL-MP's "Track A" and runs unconditionally — even for complete gene trees.

**Why**: The UPGMA tree is a globally-informed consensus signal. For large n with high ILS, many true bipartitions exist only in the UPGMA tree and not in any individual gene tree. Without this, the DP can never find them. This is the single largest accuracy gap for complete-tree inputs at large n.

**Algorithm**:
1. Build n×n SimilarityMatrix (or DistanceMatrix as fallback).
2. Run UPGMA: at each step merge the pair with highest average similarity; record the bipartition at each merge.
3. Produce a list of n−2 bipartitions (BitSet form).
4. For each bipartition, hash it and insert into `ClusterTable` if not already present.
5. For each new cluster added to `ClusterTable`, also add its super-complement `S \ cluster` (to keep X closed under complement).
6. Rebuild size-bins in `ClusterTable` to include the new entries before DPTable is built.

**UPGMA time**: O(n² log n) using a sorted structure; memory reuses the already-built n×n matrix.

**Files**: new `src/astralx/completion/UPGMATreeBuilder.java`; integrate in `src/astralx/Main.java` between Phase 1b and Phase 3.

---

## 8. Greedy Consensus Trees → X Enrichment

**What**: Build 7 greedy consensus trees at frequency thresholds `{0, 1/100, 1/50, 1/20, 1/10, 1/5, 1/3}` and inject their bipartitions (plus polytomy resolutions) into X. This is ASTRAL-MP's `addExtraBipartitionByHeuristics` and is the **largest contributor** to X enrichment.

**Why**: No single gene tree contains every true bipartition under high ILS. A bipartition supported by 15% of gene trees is real signal but may not survive in any individual tree. The 7-threshold approach systematically covers bipartitions at every support level — from strong majority (T1 ≥33%) down to any-tree support (T7 ≥0%). This is what drives ASTRAL-MP's accuracy advantage at large n.

**Algorithm** (4 phases):

**Phase 1** — Count bipartition frequencies across all k gene trees:
```
for each gene tree:
    post-traverse; at each internal node, compute cluster (BitSet of subtree taxa)
    increment count[cluster] in a HashMap (deduplicate A and S\A as same bipartition)
```
Time: O(k × n²/64) using BitSet ops.

**Phase 2** — Sort clusters by frequency descending.

**Phase 3** — Build 7 greedy consensus trees, one per threshold, in parallel:
```
Walk sorted clusters from most to least frequent.
At each threshold crossing, take a snapshot and call buildTreeFromClusters(snapshot):
  start with star tree (all n taxa under one root)
  for each cluster in snapshot (most frequent first):
    find LCA of cluster's leaves in current tree
    check if cluster's leaves are exactly a subset of LCA's children's subtrees
    if compatible: create new internal node, adopt the matching children
```

**Phase 4** — For each of 7 trees, for each polytomy node:
- **Step A**: Run UPGMA on the d×d sub-matrix of the polytomy's d groups → bipartitions → X
- **Step B**: 10–100 adaptive rounds of `sampleAndResolve`: pick one random representative from each group, find gene-tree bipartitions consistent with that sample, map back to full taxon set → X

**Constants** (matching ASTRAL-MP):
- 7 thresholds: `{0, 1/100, 1/50, 1/20, 1/10, 1/5, 1/3}`
- Base rounds per polytomy: 10; max: 100; improvement reward: +2 rounds when ≥5 new clusters found
- Polytomy size limit: `50 + n×25` (sum-of-squares budget to skip massive polytomies)

**Files**: new `src/astralx/consensus/GreedyConsensusBuilder.java`; requires SimilarityMatrix (task 6) for polytomy UPGMA sub-matrix resolution; integrate in `src/astralx/Main.java` after Phase 1b.

---

## Priority Order

| # | Task | Type | Effort |
|---|---|---|---|
| 1 | Incomplete gene trees | Correctness | Medium |
| 2 | Polytomy handling | Correctness | Medium |
| 3 | Cross-tree transitions (Mode 2) | Accuracy | Medium |
| 4 | Wavelet matrix intersections | Performance | High |
| 5 | GPU memory batching | Scalability | Low |
| 6 | Similarity Matrix | Accuracy | Medium |
| 7 | UPGMA guide tree → X enrichment | Accuracy | Medium |
| 8 | Greedy consensus trees → X enrichment | Accuracy | High |
