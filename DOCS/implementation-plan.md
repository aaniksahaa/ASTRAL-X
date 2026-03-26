# ASTRAL-X Implementation Plan

## Overview

ASTRAL-X brings STELAR-X's integer-tuple representation and GPU acceleration to ASTRAL's
quartet-based species tree inference. The key innovations:

1. **Integer tuple clusters**: O(1) memory per cluster instead of O(n) bitsets
2. **Prefix hash arrays**: O(1) range hash computation via prefix sums/XORs
3. **GPU-accelerated weight computation**: Wavelet matrix for O(log n) intersection queries
4. **GPU hash lookups**: For DP search space construction

### Environment
- Java 21, CUDA 12.0 (sm_86), RTX 3050 6GB
- Test data: 37 taxa/200 genes, 100 taxa/1000 genes, 200 taxa/1000 genes
- All gene trees are rooted and complete (no missing taxa initially)

### Simplifying Assumptions (First Version)
- Gene trees are complete (all taxa present) -- enables 4-intersection optimization (3x3 matrix from 2x2)
- No polytomies -- all gene trees are binary
- No heuristic X expansion (no UPGMA, distance matrix, etc.)
- Tree-local DP transitions only (Mode 1 from design doc) -- O(nk) construction
- Rooted input trees

---

## Project Structure

```
astral-x/
  src/
    astralx/
      Main.java                    -- CLI entry point
      Config.java                  -- Global configuration
      Logging.java                 -- Structured logging
      taxon/
        TaxonRegistry.java         -- Taxon name <-> integer ID mapping
      tree/
        TreeNode.java              -- Tree node
        Tree.java                  -- Parsed tree with postorder array
        TreeParser.java            -- Newick parser
      cluster/
        Cluster.java               -- Integer tuple (treeIdx, left, right, complement)
        ClusterHash.java           -- 2m-hash value object (sum+xor per seed)
        ClusterTable.java          -- Hash table of unique clusters (X set)
      partition/
        Partition.java             -- Tripartition (2 explicit clusters + whole set marker)
        PartitionTable.java        -- Hash table of unique gene tree tripartitions
      hash/
        TaxonHasher.java           -- Single taxon hash (SplitMix64, m seeds)
        PrefixHashArrays.java      -- Prefix sum/XOR arrays per gene tree
      dp/
        DPSearchSpace.java         -- Cluster -> list of (child1, child2) transitions
        InferenceDP.java           -- DP solver (top-down memoized)
        TreeReconstructor.java     -- Backtrack DP choices -> Newick tree
      weight/
        WeightCalculator.java      -- Interface
        CPUWeightCalculator.java   -- CPU intersection counting
        IntersectionCounter.java   -- Intersection via inverse index (STELAR-X style)
        TripartitionScorer.java    -- QI formula from 3x3 matrix
      gpu/
        CUDABridge.java            -- JNI/JNA bridge to CUDA kernels
        GPUWeightCalculator.java   -- GPU-accelerated weight computation
      util/
        Threading.java             -- Thread pool management
  native/
    Makefile
    cuda_bridge.cu                 -- CUDA kernels + JNI entry points
    wavelet_matrix.cuh             -- Wavelet matrix device code
    hash_lookup.cuh                -- GPU hash set for DP space construction
  draft/
    cuda/                          -- CUDA test/warmup programs
  docs/
    implementation-plan.md         -- This file
    design-decisions.md            -- Key design choices and rationale
```

---

## Phase 0: Project Scaffold & Build System

### Goal
Compilable Java project with CLI argument parsing, logging framework, and CUDA build.

### Tasks
1. Create directory structure
2. `Main.java`: Parse CLI args (`-i input.tre -o output.tre --threads N --mode cpu|gpu -v`)
3. `Config.java`: Singleton holding all runtime config (input file, output file, thread count,
   computation mode, verbosity level, number of hash seeds m=2)
4. `Logging.java`: Leveled logging (INFO, DEBUG, TRACE) with timestamps. Key design:
   - INFO: Major pipeline stages ("Parsing 200 gene trees...", "Extracted 4521 unique clusters")
   - DEBUG: Per-stage summaries ("Tree 0: 37 leaves, postorder length 37")
   - TRACE: Per-element detail (individual cluster hashes) -- guarded by level check to avoid
     string construction overhead on large inputs
5. `Threading.java`: Fixed thread pool (from STELAR-X pattern)
6. `native/Makefile`: Compile CUDA code to shared library for JNI
7. Shell script `build.sh` to compile Java + native code

### Checkpoint 0
```
$ ./build.sh
$ java -cp build astralx.Main -i all_gt_bs_rooted_37.tre -o /dev/null -v
[INFO] ASTRAL-X v0.1
[INFO] Input: all_gt_bs_rooted_37.tre
[INFO] Threads: 8
[INFO] Mode: CPU
```

---

## Phase 1: Tree Parsing & Preprocessing

### Goal
Parse Newick trees into internal representation with postorder arrays and position maps.

### Data Structures

**TaxonRegistry** (like STELAR-X's Taxon + taxaMap):
- `String[] idToName` -- taxon ID -> name
- `Map<String, Integer> nameToId` -- name -> taxon ID
- `int count` -- total unique taxa (n)
- Two-pass: first pass collects all taxon names across all trees, assigns IDs; second pass parses.

**TreeNode**:
- `int index` -- node index in tree's node list
- `TreeNode left, right, parent`
- `int taxonId` -- -1 for internal nodes
- `int postorderStart, postorderEnd` -- range [start, end) in postorder array

**Tree**:
- `int treeIndex` -- index in gene tree list (0..k-1)
- `int[] postorderArray` -- left-to-right leaf ordering (taxon IDs)
- `int[] positionMap` -- positionMap[taxonId] = index in postorderArray (-1 if absent)
- `TreeNode root`
- `int leafCount` -- number of leaves in this tree
- `boolean isComplete` -- leafCount == n

### Algorithm
1. **First pass** (can be parallel): Read all Newick lines, collect unique taxon names.
   Lock TaxonRegistry.
2. **Second pass** (parallel, chunked like STELAR-X): Parse each Newick string:
   - Stack-based parser (same as STELAR-X's Tree.parseFromNewick)
   - During parsing, assign postorder ranges to each node
   - Build `postorderArray` via in-order (left-to-right) leaf traversal
   - Build `positionMap` as inverse of postorderArray
   - Validate: rooted (root has exactly 2 children), binary (every internal node has 2 children)

### Logging
- INFO: "Parsed {k} gene trees with {n} unique taxa in {time}ms"
- DEBUG: "Tree {i}: {leafCount} leaves, complete={isComplete}"

### Checkpoint 1
```
Test: Parse all_gt_bs_rooted_37.tre
Assert: n == 37, k == 200
Assert: Every tree has leafCount == 37 (complete trees)
Assert: postorderArray has no -1 entries for complete trees
Assert: positionMap is a valid permutation for complete trees
Assert: For tree 0, positionMap[postorderArray[i]] == i for all i
```

---

## Phase 2: Taxon Hashing & Prefix Hash Arrays

### Goal
Compute sparsified taxon hashes and prefix sum/XOR arrays for O(1) range hash queries.

### Data Structures

**TaxonHasher**:
- `int m` -- number of hash seeds (default 2)
- `long[] seeds` -- m random seeds (fixed for reproducibility)
- `long[][] hashes` -- hashes[seed][taxonId], shape (m, n)
- Hash function: SplitMix64 -- `mix64(taxonId + seed)`

**PrefixHashArrays**:
- `long[][][] prefixSums` -- prefixSums[treeIdx][seed][pos], shape (k, m, n+1)
  - prefixSums[t][s][0] = 0
  - prefixSums[t][s][i] = sum of hashes[s][postorderArray[t][j]] for j=0..i-1
- `long[][][] prefixXors` -- same shape, with XOR
- `long[][] totalSums` -- totalSums[treeIdx][seed] = hash of all taxa in tree t
- `long[][] totalXors` -- same with XOR
- `long[] allTaxaSums` -- allTaxaSums[seed] = sum of hashes[s][j] for j=0..n-1
- `long[] allTaxaXors` -- same with XOR

### Range Hash Computation
For a range [l, r) in tree t, seed s:
- `rangeSum(t, s, l, r) = prefixSums[t][s][r] - prefixSums[t][s][l]`
- `rangeXor(t, s, l, r) = prefixXors[t][s][r] ^ prefixXors[t][s][l]`

For a complement range (complement of [l,r) w.r.t. tree t's taxa):
- `compSum(t, s, l, r) = totalSums[t][s] - rangeSum(t, s, l, r)`
- `compXor(t, s, l, r) = totalXors[t][s] ^ rangeXor(t, s, l, r)`

For a super-complement (complement w.r.t. ALL taxa):
- `superCompSum(t, s, l, r) = allTaxaSums[s] - rangeSum(t, s, l, r)`
- `superCompXor(t, s, l, r) = allTaxaXors[s] ^ rangeXor(t, s, l, r)`

### Important Details
- All arithmetic is unsigned 64-bit (Java: use `long`, wrapping is automatic mod 2^64)
- For missing taxa (-1 in postorder array): hash to 0 (identity for both sum and XOR).
  Since SplitMix64 output for real taxa is almost certainly non-zero, this is safe.
- For complete gene trees: totalSums == allTaxaSums, totalXors == allTaxaXors

### Logging
- INFO: "Computed prefix hash arrays: m={m} seeds, k={k} trees in {time}ms"
- DEBUG: "Taxon 0 ({name}): hashes = [{h0}, {h1}]"

### Checkpoint 2
```
Test with tiny example: 3 taxa (A=0, B=1, C=2), tree ((A,B),C)
Postorder: [A, B, C] = [0, 1, 2]
Manual computation:
  h[0][0] = mix64(0 + seed0), h[0][1] = mix64(1 + seed0), h[0][2] = mix64(2 + seed0)
  prefixSums[0][0] = [0, h00, h00+h01, h00+h01+h02]
  rangeSum(0, 0, 0, 2) = h00 + h01  (cluster {A,B})
  compSum(0, 0, 0, 2) = h02          (complement = {C})
Assert: rangeSum + compSum == totalSum
Assert: rangeXor ^ compXor == totalXor

Also test on actual 37-taxa data:
Assert: For each tree, prefixSums[t][s][leafCount] == totalSums[t][s]
Assert: For complete trees, totalSums[t][s] == allTaxaSums[s]
```

---

## Phase 3: Cluster Extraction, Hashing & Deduplication

### Goal
Extract all clusters from gene trees (subtree + complement + super-complement),
hash them, and deduplicate into the cluster set X.

### Data Structures

**Cluster** (immutable value object, ~20 bytes):
- `int treeIndex` -- which gene tree
- `int left` -- start of range (inclusive)
- `int right` -- end of range (exclusive)
- `boolean complement` -- if true, represents complement of [left, right) w.r.t. tree's taxa
- `boolean superComplement` -- if true, complement w.r.t. ALL taxa (not just tree's taxa)

**ClusterHash** (immutable, used as HashMap key):
- `long[] sums` -- m sum hashes (mixed/finalized)
- `long[] xors` -- m xor hashes (mixed/finalized)
- `int size` -- number of taxa in the cluster
- Finalization: apply SplitMix64 mixing to raw sum/xor for better distribution in hash tables.
  The raw values are needed for associative operations; the mixed values are for hash table keys.

**ClusterTable** (the set X):
- `Map<ClusterHash, ClusterEntry> table`
- `ClusterEntry`: { `ClusterHash hash`, `Cluster exemplar`, `int frequency`, `int size` }
- `Map<Integer, List<ClusterHash>> sizeBins` -- clusters grouped by size (for DP space construction)

### Algorithm

For each gene tree g (index t), bottom-up traversal:
1. **Subtree clusters**: For each node u (including leaves, excluding root):
   - Cluster(t, u.postorderStart, u.postorderEnd, false, false)
   - Size = u.postorderEnd - u.postorderStart
   - Compute ClusterHash from prefix arrays (raw sums/xors -> mix for hash key)

2. **Complement clusters** (unrooted treatment): For each node u (excluding root):
   - Cluster(t, u.postorderStart, u.postorderEnd, complement=true, false)
   - Size = tree.leafCount - (u.postorderEnd - u.postorderStart)
   - Hash = totalHash - rangeHash (for sum), totalHash ^ rangeHash (for xor)

3. **Full tree taxa set**: Cluster for [0, leafCount) -- but this equals all-taxa for complete
   trees, so skip adding to X (the all-taxa cluster is the DP root, handled specially).

For each cluster, compute its ClusterHash and insert into ClusterTable:
- If hash already exists: increment frequency (cluster is duplicate)
- If hash is new: insert with exemplar and frequency=1

4. **Super-complements**: For each cluster A already in X:
   - Compute S\A where S = all taxa set
   - Hash: allTaxaHash - A.hash (sum), allTaxaHash ^ A.hash (xor)
   - Size: n - A.size
   - If this hash is not already in X, add it
   - For complete gene trees: S\A is the same as Lg\A (since Lg=S), so no new clusters
     are added in this step! This is an important optimization for our initial case.

### Important: Excluding Trivial Clusters
- Do NOT add the all-taxa cluster to X (it's the DP root, handled separately)
- Do NOT add empty clusters (size 0)
- Single-taxon clusters (leaves) ARE included in X

### Logging
- INFO: "Extracted {total} cluster candidates, {unique} unique clusters in {time}ms"
- INFO: "Size distribution: min={min}, max={max}, median={med}"
- DEBUG: "Cluster hash={hash}, size={size}, freq={freq}, exemplar=tree{t}[{l},{r})"

### Checkpoint 3
```
Test on 37-taxa data:
Assert: All clusters have size in [1, n-1]
Assert: For each cluster A in X, complement (n - A.size) is also in X (for complete trees)
Assert: sizeBins cover all sizes from 1 to n-1
Assert: Frequency of any cluster >= 1
Assert: No duplicate ClusterHash keys in table

Specific test: For a known tree ((A,B),(C,D)), manually verify:
  Subtree clusters: {A}, {B}, {C}, {D}, {A,B}, {C,D}
  Complement clusters: {C,D}, {A,B}, {A,B}, {C,D}, {C,D}, {A,B}
  (many duplicates collapsed)
  Unique: {A}, {B}, {C}, {D}, {A,B}, {C,D}, {A,B,C}, {A,B,D}, {A,C,D}, {B,C,D}
  (the 3-element clusters come from complements of singletons)
```

---

## Phase 4: Gene Tree Tripartition Extraction

### Goal
For each non-root internal node of each gene tree (treated as unrooted), extract the
tripartition (left | right | complement) and deduplicate.

### Data Structures

**Partition** (tripartition):
- `ClusterHash part1Hash` -- hash of first explicit part (e.g., left subtree)
- `ClusterHash part2Hash` -- hash of second explicit part (e.g., right subtree)
- `ClusterHash part3Hash` -- hash of implicit third part (complement of union)
- `int part1Size, part2Size, part3Size`
- The "whole set" is the gene tree's taxa set Lg (for gene tree partitions)

**PartitionHash** (used as HashMap key):
- Computed from sorted pair of (part1Hash, part2Hash) -- order-invariant
- Since part3 is determined by part1 + part2 + wholeSet, it need not be in the hash

**PartitionTable**:
- `Map<PartitionHash, PartitionEntry> table`
- `PartitionEntry`: { `PartitionHash hash`, `Partition exemplar`, `int frequency` }

### Algorithm

For each gene tree g (index t):
  For each internal node u that is NOT the root:
    - part1 = Cluster(t, left.start, left.end, false) -- left subtree
    - part2 = Cluster(t, right.start, right.end, false) -- right subtree
    - part3 = complement of [u.start, u.end) w.r.t. tree t -- everything else in Lg
    - Compute hashes for all three parts
    - Create Partition, compute order-invariant PartitionHash
    - Insert into PartitionTable (increment frequency if duplicate)

Note: The root node's two children define a bipartition of Lg (2-partition), which corresponds
to a tripartition with part3 = empty set. This contributes 0 to the QI score, so we skip it.

### Logging
- INFO: "Extracted {unique} unique gene tree tripartitions (from {total} total) in {time}ms"
- DEBUG: "Tripartition: |part1|={s1}, |part2|={s2}, |part3|={s3}, freq={f}"

### Checkpoint 4
```
Test on 37-taxa data:
Assert: Each tripartition has part1Size + part2Size + part3Size == n (complete trees)
Assert: part3Size > 0 for all non-root tripartitions
Assert: Total tripartitions extracted = sum over trees of (numInternalNodes - 1)
  For binary tree with n leaves: numInternalNodes = n-1, so tripartitions per tree = n-2
  Total before dedup: k * (n-2) = 200 * 35 = 7000 for 37-taxa

Specific test: For tree ((A,B),(C,D)):
  Internal nodes (non-root): node(A,B), node(C,D)
  Tripartitions:
    node(A,B): ({A,B} | {C,D} | {}) -- root child, SKIP (part3 empty)
    Wait -- in a 4-leaf tree, root has 2 children, each has 2 leaf children.
    Non-root internal nodes: node(A,B) and node(C,D).
    node(A,B): part1={A}, part2={B}, part3={C,D}
    node(C,D): part1={C}, part2={D}, part3={A,B}
  These are the correct tripartitions for an unrooted view.
```

---

## Phase 5: DP Search Space Construction (Tree-Local Transitions)

### Goal
For each cluster A in X, find all valid splits A -> A' | (A\A') using tree-local transitions.
This is O(nk) total -- no hash subtraction search needed.

### Data Structures

**DPSearchSpace**:
- `Map<ClusterHash, List<DPTransition>> transitions`
- `DPTransition`: { `ClusterHash parent`, `ClusterHash child1`, `ClusterHash child2` }

### Algorithm (Mode 1: Tree-Local Only)

For each gene tree g (index t), bottom-up traversal:
  For each internal node u (non-root) with parent p, sibling v:

  **Type 1: Subtree split**
    - parent = sub(u) = Cluster(t, u.start, u.end, false)
    - child1 = sub(left(u))
    - child2 = sub(right(u))
    - Emit: hash(parent) -> (hash(child1), hash(child2))

  **Type 2: Complement split** (if p is not root)
    - parent = Lg\sub(u) = Cluster(t, u.start, u.end, complement=true)
    - child1 = sub(sibling(u))
    - child2 = Lg\sub(parent(u)) = Cluster(t, p.start, p.end, complement=true)
    - Emit: hash(parent) -> (hash(child1), hash(child2))

  **Type 3: Super-complement splits** (if Lg != S, i.e., tree is incomplete)
    - Skip for now (all trees are complete)

Additionally, for the all-taxa cluster (DP root), we need initial transitions.
For complete trees, ANY cluster A in X with size < n gives a valid split:
  allTaxa -> A | (S\A)
Since S\A is guaranteed to be in X (we added super-complements).
But we don't need to enumerate all of these upfront -- during DP, when we process
the all-taxa cluster, we can iterate over X and use hash subtraction.

Actually, a cleaner approach: during tree-local traversal, for each root's children:
  - The root gives a bipartition sub(left_root) | sub(right_root)
  - This IS a valid transition for the all-taxa cluster
  - Emit: allTaxaHash -> (hash(sub(left_root)), hash(sub(right_root)))

### Deduplication
Multiple gene trees may emit the same transition (same parent hash, same child hashes).
Store as a Set per parent to avoid duplicates.

### Logging
- INFO: "Built DP search space: {numClusters} clusters, {numTransitions} transitions in {time}ms"
- DEBUG: "Cluster hash={h}, size={s}: {numSplits} candidate splits"

### Checkpoint 5
```
Test on 37-taxa data:
Assert: Every cluster in X (except singletons) has at least 1 transition
Assert: Singletons (size 1) have 0 transitions (base case)
Assert: allTaxaCluster has at least k transitions (one per gene tree root)
Assert: For each transition (parent -> c1, c2):
  - c1.size + c2.size == parent.size
  - c1 and c2 are both in X

Structural test: For tree ((A,B),(C,D)):
  Type 1 transitions:
    {A,B} -> {A} | {B}
    {C,D} -> {C} | {D}
  Type 2 transitions (node(A,B) has parent=root, so skip; same for node(C,D))
  Root transition:
    {A,B,C,D} -> {A,B} | {C,D}
```

---

## Phase 6: Weight Calculation

### Goal
For each candidate tripartition (encountered during DP), compute its total weight
across all unique gene tree tripartitions.

### Key Insight: Complete Gene Trees -> 4 Intersections Suffice

For candidate tripartition (X|Y|Z) where Z = S\X\Y, and gene tree tripartition
(A|B|C) where C = Lg\A\B:

When Lg = S (complete gene trees), the full 3x3 intersection matrix:
```
         A           B           C
X    |X cap A|     |X cap B|     |X| - |X cap A| - |X cap B|
Y    |Y cap A|     |Y cap B|     |Y| - |Y cap A| - |Y cap B|
Z    |A|-|X cap A|-|Y cap A|   |B|-|X cap B|-|Y cap B|   (by difference)
```

Only need 4 actual intersection computations: |X cap A|, |X cap B|, |Y cap A|, |Y cap B|.

### QI Scoring Formula

For tripartition T=(X|Y|Z) and gene tree tripartition M=(A|B|C):
```
QI(T,M) = sum over all 6 permutations (i,j,k) of (A,B,C):
           a_i * b_j * c_k * (a_i + b_j + c_k - 3) / 2
```
where a_i = |X cap M_i|, b_j = |Y cap M_j|, c_k = |Z cap M_k|.

Total weight: w(T) = (1/2) * sum over all gene tree tripartitions M (with frequency):
                      freq(M) * QI(T, M)

### Intersection Counting (CPU, STELAR-X style)

**IntersectionCounter**:
- `int[][] inverseIndex` -- inverseIndex[treeIdx][taxonId] = position in postorder array (-1 if absent)
- `int[][] postorderArrays` -- postorderArrays[treeIdx][pos] = taxonId

`countIntersection(tree1, l1, r1, complement1, tree2, l2, r2, complement2)`:
  - For non-complement ranges: iterate over smaller range, for each taxon check if it falls
    in the other range via inverseIndex. O(min(|range1|, |range2|)).
  - For complement ranges: use the identity |A^c cap B| = |B| - |A cap B| when A, B are
    from the same universe, or iterate and check exclusion.

Actually, simpler for complete trees: any cluster from X can be mapped to actual taxa
via its exemplar. But for weight calculation, we work with the DP's candidate tripartition
(X|Y|Z) which may involve clusters from different trees.

**Approach**: For a candidate tripartition (X|Y|Z) and gene tree tripartition (A|B|C):
- X and Y are clusters from potentially different trees.
- A and B are from the same gene tree.
- Need |X cap A|: for each taxon in the smaller of X and A, check membership in the other.

Membership check: For cluster Cluster(t, l, r, comp):
- Non-complement: taxon `tid` is in cluster iff inverseIndex[t][tid] is in [l, r)
- Complement: taxon `tid` is in cluster iff inverseIndex[t][tid] is NOT in [l, r)
  (but IS in the tree, i.e., inverseIndex[t][tid] != -1)

To enumerate taxa in a cluster:
- Non-complement, small: iterate postorderArray[t][l..r), get taxon IDs
- Complement: more expensive; iterate all taxa, check membership

**Optimization**: Since we need 4 intersections per (candidate, gt-tripartition) pair,
and the candidate's clusters (X, Y) are used across many gt-tripartitions, we can:
1. For each candidate split, enumerate the taxa in X and Y once
2. For each gt-tripartition, do 4 range membership lookups

### Weight Calculation Architecture

**CPUWeightCalculator**:
- For each tripartition T encountered during DP:
  - For each unique gt-tripartition M in PartitionTable (with frequency f):
    - Compute 4 intersections: |X cap A|, |X cap B|, |Y cap A|, |Y cap B|
    - Derive full 3x3 matrix
    - Compute QI(T, M)
    - weight += f * QI(T, M)
  - Return weight / 2

**Lazy + Memoized**: Don't precompute all weights. Compute on-the-fly during DP,
cache by tripartition hash. This avoids computing weights for tripartitions never
visited by the DP.

### Logging
- INFO: "Weight calculation: {computed} tripartitions scored in {time}ms"
- DEBUG: "Tripartition weight={w}, scored against {numGtTrips} gt-tripartitions"

### Checkpoint 6
```
Test with tiny example: 5 taxa, 2 gene trees, manually compute QI scores.

Validation on 37-taxa data:
Assert: All weights are non-negative (QI terms are non-negative)
Assert: For symmetric tripartitions, score is consistent regardless of which
  representation is used (order invariance)
Assert: Weight of root's first-level split > 0 (not degenerate)

Cross-check: Score a known species tree by summing all tripartition weights,
compare with brute-force quartet counting (on small dataset).
```

---

## Phase 7: Inference DP & Tree Reconstruction

### Goal
Solve the DP over the search space, find optimal species tree.

### Algorithm

**InferenceDP**:
- `Map<ClusterHash, Long> memo` -- cluster hash -> best score
- `Map<ClusterHash, DPTransition> choice` -- cluster hash -> best split choice

`solve()`:
  1. Compute allTaxaHash
  2. Call dp(allTaxaHash)
  3. Reconstruct tree from choice map

`dp(ClusterHash clusterHash)`:
  1. If memo contains clusterHash, return memo[clusterHash]
  2. Get cluster size from ClusterTable
  3. Base case: size <= 1 -> return 0 (leaf)
  4. Base case: size == 2 -> return 0 (trivial split, only one option)
  5. bestScore = Long.MIN_VALUE
  6. For each transition (child1, child2) in DPSearchSpace[clusterHash]:
     - Compute the candidate tripartition:
       T = (child1 | child2 | S\clusterHash)
       where S\clusterHash is the complement of the current cluster w.r.t. all taxa
     - weight = weightCalculator.getWeight(T)  // lazy + memoized
     - score = dp(child1) + dp(child2) + weight
     - if score > bestScore: update bestScore, bestChoice
  7. memo[clusterHash] = bestScore; choice[clusterHash] = bestChoice
  8. Return bestScore

**Note on the all-taxa cluster**: The all-taxa cluster is the DP root.
Its tripartition is (child1 | child2 | empty), which contributes 0 weight
(QI with an empty set is 0). So the root split's own weight is 0, but its
children's subtree scores count.

**TreeReconstructor**:
  `reconstructTree(ClusterHash root)`:
  1. If size <= 1: return leaf node with taxon name
  2. Get best choice (child1, child2) from choice map
  3. leftTree = reconstructTree(child1)
  4. rightTree = reconstructTree(child2)
  5. Return internal node with leftTree and rightTree
  6. Output in Newick format

### Logging
- INFO: "DP solved: optimal score = {score}, {numStates} states explored in {time}ms"
- INFO: "Species tree: {newickString}"
- DEBUG: "DP state: cluster size={s}, {numSplits} candidates, bestScore={score}"

### Checkpoint 7 -- MAJOR MILESTONE
```
End-to-end test on 37-taxa dataset:
$ java -cp build astralx.Main -i all_gt_bs_rooted_37.tre -o output_37.tre -v

Assert: Output is a valid Newick tree with exactly 37 leaves
Assert: All 37 taxon names are present in the output
Assert: The tree is binary (every internal node has exactly 2 children)
Assert: The optimal score is non-negative

Comparison: Run ASTRAL on the same input and compare:
  - Topology should be identical or very similar
  - Score should match (if X sets are the same)
```

---

## Phase 8: GPU Acceleration

### 8A: GPU Wavelet Matrix for Intersection Counting

Replace CPU intersection counting with GPU wavelet matrix for large clusters.

**Strategy** (from design doc):
- For each gene tree g_i, build wavelet matrices between g_i and all other gene trees
- This enables O(log n) intersection queries between g_i's ranges and any other tree's ranges
- After processing all candidate tripartitions involving g_i, free the wavelet matrices
- Memory per round: O(nk log n)

**Implementation**:
1. Port wavelet_matrix.cu into the project (already have working code in ref-cuda/)
2. Adapt for batch queries: given candidate tripartitions and gt-tripartitions,
   batch all intersection queries and launch one kernel
3. JNI bridge: Java passes arrays of (tree1, l1, r1, tree2, l2, r2) queries,
   CUDA kernel returns intersection counts

**Hybrid approach**: Use wavelet matrix only for large clusters (size > threshold ~log(n)).
For small clusters, CPU iteration is faster.

### 8B: GPU Hash Lookup for DP Space Construction (Future)

Use GPU hash set for cross-tree transition discovery (Mode 2).
Reference code in ref-cuda/gpu_set_lookup.cu already works.

### Checkpoint 8
```
Test: Run same 37-taxa dataset with --mode gpu
Assert: Output tree topology matches CPU mode exactly
Assert: Score matches CPU mode exactly
Assert: GPU mode is faster for 200-taxa dataset
```

---

## Phase 9: End-to-End Testing & Benchmarking

### Test 1: 37 taxa, 200 gene trees (small)
- Expected: completes in seconds
- Validate correctness against ASTRAL output

### Test 2: 100 taxa, 1000 gene trees (medium)
- Expected: completes in minutes
- Track memory usage (should be << ASTRAL due to tuple representation)

### Test 3: 200 taxa, 1000 gene trees (large)
- Expected: completes in reasonable time
- Compare with ASTRAL (may be infeasible for ASTRAL with bitsets)

### Metrics to Track
- Wall clock time per phase
- Peak memory usage
- Number of unique clusters, tripartitions, DP states
- GPU utilization (if applicable)

---

## Design Decisions Log

### D1: Hash Seeds (m=2)
Using m=2 seeds gives 4 independent 64-bit hashes (2 sum + 2 XOR).
Collision probability per pair: ~1/2^256. Safe for n up to millions.

### D2: Tree-Local Transitions Only (Initial Version)
Mode 1 from the design doc. O(nk) construction, statistically consistent.
Cross-tree transitions (Mode 2) can be added later as optional enhancement.

### D3: Lazy + Memoized Weight Calculation
Don't precompute all O(|X|^2) tripartition weights. Compute on-demand during DP
and cache. This saves enormous memory and compute since the DP only visits a
fraction of all possible tripartitions.

### D4: Integer Scores (No Floating Point)
Work with integer QI scores throughout. The 1/2 factor can be applied at the very
end when reporting the final score. This avoids floating-point precision issues.

### D5: Complete Gene Trees Assumption
Enables the 4-intersection optimization (full 3x3 matrix from 2x2 submatrix).
Also means super-complement step adds no new clusters (since Lg = S for all trees).

### D6: ClusterHash as Immutable Value Object
ClusterHash stores the finalized (mixed) hash values and is used as HashMap key.
Raw (unmixed) values are used only during construction for associative operations.
The mixed values give better hash table distribution.

---

## Execution Order Summary

```
Phase 0  ->  Checkpoint 0 (CLI works)
Phase 1  ->  Checkpoint 1 (trees parse correctly)
Phase 2  ->  Checkpoint 2 (hashing is correct)
Phase 3  ->  Checkpoint 3 (clusters extracted correctly)
Phase 4  ->  Checkpoint 4 (tripartitions extracted correctly)
Phase 5  ->  Checkpoint 5 (DP space is valid)
Phase 6  ->  Checkpoint 6 (weights are correct)
Phase 7  ->  Checkpoint 7 (MAJOR: correct species tree on 37-taxa)
Phase 8  ->  Checkpoint 8 (GPU matches CPU exactly)
Phase 9  ->  Full benchmarks on 37/100/200 taxa
```

Each phase builds on the previous and can be tested independently.
