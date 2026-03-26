# ASTRAL-X Design Decisions

## Key Architectural Choices

### 1. Gene Tree Tripartitions vs Bipartitions

ASTRAL scores **tripartitions**, not bipartitions. Every non-root internal node of a gene
tree (treated as unrooted) induces a tripartition: (left subtree | right subtree | everything else).

The DP *searches* over bipartitions (splitting cluster A into A' and A\A'), but the *score*
attached to each split is the weight of the tripartition (A' | A\A' | S\A). The third part
Z = S\A depends on which parent cluster we are splitting, making the weight context-dependent.

Decision: Use lazy + memoized weight calculation, keyed by the full tripartition hash.

### 2. Raw vs Mixed Hashes

Two levels of hashing:
- **Raw hashes**: The direct prefix sum / prefix XOR values. These are associative:
  raw_sum(A union B) = raw_sum(A) + raw_sum(B). Needed for complement computation
  and cross-tree recombination.
- **Mixed hashes**: Raw values passed through SplitMix64 finalizer. Better distribution
  for use as HashMap keys. NOT associative.

Decision: Store both. Use raw for algebraic operations, mixed for hash table lookups.
ClusterHash stores mixed values (for HashMap key quality). Raw values computed on-the-fly
from prefix arrays when needed.

### 3. Intersection Counting Strategy

For the 3x3 intersection matrix between candidate tripartition (X|Y|Z) and gene tree
tripartition (A|B|C):

With complete gene trees (Lg = S), only 4 intersections needed:
|X cap A|, |X cap B|, |Y cap A|, |Y cap B|. The other 5 derived by subtraction.

CPU method (STELAR-X style): For each intersection |X cap A|, iterate over the smaller
cluster and check membership in the larger via inverse index. O(min(|X|, |A|)).

GPU method (wavelet matrix): Build wavelet matrix per gene tree pair, query in O(log n).
Better for large clusters, worse for small ones. Use hybrid approach.

### 4. DP Search Space: Tree-Local vs Cross-Tree

Tree-local transitions (Mode 1): Directly readable from tree structure during bottom-up
traversal. O(nk) total, no hash lookups needed. Statistically consistent.

Cross-tree transitions (Mode 2): Requires hash subtraction search over all cluster pairs.
O(|X|^2) worst case. Finds more transitions but much more expensive.

Decision: Start with Mode 1 only. It captures the vast majority of useful transitions
and is equivalent to ASTRAL-I's search space. Mode 2 can be added as --search-mode full.

### 5. Memory Layout for Prefix Arrays

Layout: `prefixSums[treeIdx][seedIdx][position]`

This puts all prefix values for one tree+seed contiguous in memory, which is optimal for
range queries (which access consecutive positions in the same tree+seed).

Alternative: `prefixSums[seedIdx][treeIdx][position]` -- worse locality for range queries.

### 6. Partition Hash Order-Invariance

A tripartition (A|B|C) is the same as (B|A|C), (C|A|B), etc. For hashing:
- Sort the two explicit cluster hashes (the third is implicit) before combining
- Since we only store 2 explicit clusters, sorting 2 values is trivial

For future polytomy support (d-partition with d > 3):
- Sort all d-1 explicit cluster hashes before combining

### 7. The 1/2 Factor in QI

ASTRAL defines w(T) = (1/2) * sum QI(T, M). During DP, the 1/2 is a global constant
that doesn't affect which tree is optimal.

Decision: Work with 2*w(T) throughout (just sum QI without the 1/2). Divide by 2 only
when reporting the final score. This keeps everything as integers.

### 8. All-Taxa Cluster Handling

The all-taxa cluster is the DP root. It is NOT stored in X.

For its transitions: during tree-local traversal, each gene tree root contributes
one transition (left_root_subtree | right_root_subtree). These are the initial splits.

The weight of the root split itself is 0 (tripartition with Z = empty set), but the
children's subtree scores accumulate.
