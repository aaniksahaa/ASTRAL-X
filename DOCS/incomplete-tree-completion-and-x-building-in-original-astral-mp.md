# Incomplete Gene Tree Completion and Search Space X Construction

Analysis of the ASTRAL-MP heuristic pipeline for handling incomplete gene trees,
based on reading the legacy codebase at `astral-mp-legacy-codebase/`.

Primary source files:
- `astral-mp-legacy-codebase/WQDataCollection.java`
- `astral-mp-legacy-codebase/AbstractMatrix.java`
- `astral-mp-legacy-codebase/DistanceMatrix.java`
- `astral-mp-legacy-codebase/SimilarityMatrix.java`
- `astral-mp-legacy-codebase/Matrix.java`


---

## 1. The Core Problem: Incomplete Gene Trees

A gene tree is **incomplete** when some taxa are missing from it — the gene was not
sequenced for those taxa, or the sequences were filtered out.  This creates two
problems for ASTRAL:

1. **Missing bipartitions**: a bipartition of the full taxon set `{A,B,C,D,E}` cannot
   be read from a tree that contains only `{A,B,C,D}`.  E must be placed somewhere
   before the bipartition is meaningful over all n taxa.

2. **Incorrect quartet counts**: the quartet score formula assumes all taxa are present.
   A tripartition extracted from an incomplete tree covers a subset of taxa, which
   distorts the weight computation.

ASTRAL-MP's solution: **complete every incomplete gene tree before building X**, using
a distance/similarity matrix built from all input trees.  ASTRAL-X takes a different
approach (handles incompleteness natively in the weight formula), but understanding
ASTRAL-MP's method is essential for implementing a compatible heuristic pipeline.


---

## 2. The Distance / Similarity Matrix

### 2.1 What it is

A dense **n × n symmetric float array** (`AbstractMatrix.matrix`), where n is the total
number of taxa across all gene trees.  Entry `matrix[i][j]` stores the average
pairwise distance (or similarity) between taxon i and taxon j, aggregated over all
k gene trees in which both taxa co-appear.

Two variants exist, selected by a command-line option:

| Variant | Class | Entry meaning | Used for |
|---|---|---|---|
| Branch distance | `DistanceMatrix` | Average branch-length distance between i and j | PhyDstar tree inference |
| Quartet similarity | `SimilarityMatrix` | Average quartet co-occurrence score between i and j | UPGMA clustering |

### 2.2 Construction algorithm

**`DistanceMatrix.matricesByBranchDistance()`** — `DistanceMatrix.java` lines 223–330:

```
for each gene tree T:
  post-order traverse T:
    at each leaf v:
      distanceMap[v] = 0
    at each internal node v with children L, R:
      for each taxon i in L, taxon j in R:
        matrix[i][j]    += distanceMap[i] + distanceMap[j] + 2
        pairCount[i][j] += 1
      increment distanceMap for all taxa passing through v

normalize: matrix[i][j] /= pairCount[i][j]
```

**`SimilarityMatrix.populateByQuartetDistance()`** — `SimilarityMatrix.java` lines 63–240:

Similar traversal but accumulates quartet co-occurrence scores with thread-level
parallelism (memory chunks + synchronized blocks).

### 2.3 Normalization

`pairCount[i][j]` records exactly how many gene trees contained both taxon i and j.
The final value is an **average**, not a cumulative sum:

```java
matrix[i][j] /= pairNumMatrix[i][j];   // DistanceMatrix.java line ~304
```

This means pairs that co-appear in only 1 tree get an entry based on a single
observation (high variance), while pairs present in 100 trees get a stable average.
ASTRAL-MP treats all normalized entries as equally reliable — a known weakness for
heavily incomplete data.

Pairs that never co-appear in any tree are marked `-99` (sentinel for undefined).

### 2.4 Memory and time

| Quantity | Formula | Example (n=25,000, k=1,000) |
|---|---|---|
| Matrix storage | 4n² bytes | 2.5 GB |
| pairCount array | 4n² bytes | 2.5 GB |
| Total peak | ~3 × 4n² bytes | ~7.5 GB |
| Construction time | O(k × n²) | ~625 billion ops |

The O(n²) memory is the fundamental scalability wall.  It is not a GPU problem —
the matrix simply does not fit for n > ~80,000 on any current hardware.

| n | Matrix size | Feasible? |
|---|---|---|
| 5,000 | 100 MB | Yes |
| 25,000 | 2.5 GB | Yes (tight) |
| 100,000 | 40 GB | No |
| 500,000 | 1 TB | Never |


---

## 3. Completing Incomplete Gene Trees

### 3.1 Purpose

**`WQDataCollection.getCompleteTree()`** — `WQDataCollection.java` lines 261–341.

The Javadoc states it directly:

> *"Completes an incomplete tree for the purpose of adding to set X.
>  Otherwise, bipartitions are meaningless."*

A bipartition extracted from an incomplete tree (e.g. `{A,B} | {C,D}` when E is
absent) is not a valid bipartition over the full taxon set.  To add it to X, E must
be inserted first.

### 3.2 The four-point navigation algorithm

For each missing taxon x, the algorithm navigates the gene tree and inserts x at
the most consistent location using the **four-point metric** from the distance matrix.

**Step 1 — Find the closest present taxon**

```java
int closestId = geneMatrix.getClosestPresentTaxonId(gtAllBS, missingId);
// AbstractMatrix.java lines 52–67
```

`orderedTaxonBySimilarity[x]` is a pre-sorted list of all taxa by proximity to x
(computed once at first use).  Iterate this list; stop at the first taxon that is
either present in this gene tree or has a smaller ID (already inserted in a previous
iteration for this tree).  Call this taxon **a** (the anchor).

**Step 2 — Reroot the gene tree at a**

```java
trc.rerootTreeAtNode(closestNode);
Trees.removeBinaryNodes(trc);
```

After rerooting, the tree has a on one side of the root and everything else
(`subtree_rest`) on the other.  Set `start = subtree_rest`.

**Step 3 — Navigate down using six matrix entries per step**

```
while start is an internal node:
    c1, c2 = children of start
    c1rep = leftmost leaf of c1
    c2rep = leftmost leaf of c2
    betterSide = getBetterSideByFourPoint(x, a, c1rep, c2rep)
    if betterSide == a:    stop here (insert x at this node)
    if betterSide == c1rep: descend into c1, reset c2rep
    if betterSide == c2rep: descend into c2, c1rep = c2rep, reset c2rep
```

**The six matrix values used** (`DistanceMatrix.java` lines 40–51):

```
d(x, a)    d(x, c1rep)    d(x, c2rep)
d(a, c1rep)  d(a, c2rep)  d(c1rep, c2rep)

ascore = (d(x,c1rep) + d(a,c2rep)) - (d(x,a) + d(c1rep,c2rep))
bscore = -ascore
cscore = (d(x,c1rep) + d(a,c2rep)) - (d(x,c2rep) + d(a,c1rep))
return argmax(a=ascore, c1rep=bscore, c2rep=cscore)
```

The **four-point condition** from distance theory: among four taxa on any tree, the
sum of the two "crossing" pairwise distances equals each other and exceeds the
"same-clade" sum.  Comparing these sums identifies the topology of the four taxa
without needing the tree — just the 6 pairwise distances.  This tells us which
subtree x should descend into, or whether to stop and insert at the current node.

**Why use the closest taxon as anchor?**

The closest taxon a has the lowest-variance distance estimate to x (co-appeared most
often across gene trees).  Using a as anchor maximizes reliability of the four-point
decisions.  Choosing a random taxon would work mathematically but produce noisier
navigation.

Note: "closest in the distance matrix" ≠ "sister in the topology."  The navigation
may place x far from a in the final tree — a is only an anchor, not the target.

**Step 4 — Insert x at the landing position**

```java
// Stopped at a leaf:
newinternalnode = start.getParent().createChild()
newinternalnode.adoptChild(start)          // leaf becomes child of new node
newinternalnode.adoptChild(new leaf x)     // x becomes sister to that leaf

// Stopped at an internal node (four-point said: insert between a and this subtree):
start.createChild(x)                       // x as new direct child of start
newinternalnode = start.createChild()      // wrap c1, c2 under new internal node
newinternalnode.adoptChild(c1)
newinternalnode.adoptChild(c2)
```

**Step 5 — Repeat for all missing taxa**

The outer loop (`WQDataCollection.java` line 271) inserts missing taxa in order of
taxon ID.  Each insertion modifies the tree; later insertions can use previously
inserted taxa as anchors.

### 3.3 Complexity per gene tree

| Operation | Time | Notes |
|---|---|---|
| Pre-sort all n rows once | O(n² log n) | Done once for all trees |
| Find closest taxon | O(1) after sort | Walk sorted list |
| Navigate to insertion point | O(depth × 6) | 6 matrix reads per level |
| Insert taxon | O(1) | Tree pointer ops |
| **Per missing taxon** | **O(depth)** | depth = O(log n) balanced, O(n) worst |
| **Per gene tree** | **O(missing × depth)** | |
| **All trees** | **O(k × n × depth)** | Cheap relative to matrix construction |

All 6 entries per navigation step come from the **global matrix** (averaged over all k
trees), not from the current gene tree.  The 3 entries not involving x (between present
taxa a, c1rep, c2rep) could alternatively be computed from the gene tree directly, but
ASTRAL-MP uses the global matrix for consistency.

### 3.4 Can we avoid the full n×n matrix?

The navigation needs `d(x, c1rep)` and `d(x, c2rep)` where c1rep and c2rep are
arbitrary taxa (the leftmost leaf of each child subtree).  In the current
implementation, you need x's full matrix row.

A **k-nearest-neighbor (kNN) sparse index** (K entries per taxon) would suffice only
if the representative selection is changed: instead of "leftmost leaf", pick the
representative to be "x's nearest neighbor within that subtree."  Then all three
x-involving entries are guaranteed to be in x's kNN list.

However, a deeper problem exists: if x never co-appears with any taxon in a subtree
across all k gene trees, that entry is `-99` (undefined) regardless of K.  For
complete gene trees (x appears everywhere), the full n×n is unavoidable.  For
incomplete data, **co-occurrence sparsity** is the natural structure: only store
`d(x, y)` for pairs that actually appear together in at least one tree.  This is
sparser than kNN when data is heavily incomplete, and degenerates to full n×n for
complete trees.


---

## 4. Building the Search Space X after Completion

All incomplete trees are completed before X is built
(`WQDataCollection.java` lines 613–623).

### 4.1 The full pipeline

```
calculateDistances()           → build n×n matrix
if any incomplete trees:
    completeGeneTrees()        → complete all trees using 4-point navigation
else:
    completedGeeneTrees = copy of originalIncompleteTrees

// Now build X from completedGeeneTrees:

[Track A] UPGMA/PhyDstar tree → add its bipartitions to X
[Track B] For each completed gene tree → extract bipartitions → add to X
```

### 4.2 Track A — matrix-derived tree

```java
// WQDataCollection.java lines 727–751
for (BitSet b : speciesMatrix.inferTreeBitsets()) { ... }
Tree ST = Utils.buildTreeFromClusters(STls, ...);
addBipartitionsFromSingleIndTreesToX(ST, baseTrees, taxonIdentifier);
```

Run UPGMA (SimilarityMatrix) or PhyDstar (DistanceMatrix) on the species-level
matrix → one species tree → post-traverse and add every internal node's bipartition
to X.

Time: O(n² log n) for UPGMA.  Memory: uses the already-built n×n matrix.

### 4.3 Track B — gene tree bipartitions (single-individual path)

**For single-individual datasets** (`isSingleIndividual() == true`),
lines 652–661:

```java
for (Tree gt : completedGeeneTrees) {
    STITree gtrelabelled = new STITree(gt);
    spm.gtToSt(gtrelabelled);            // relabel: identity for 1:1 datasets
    allGreedies[gtindex++] = [gtrelabelled];   // single-element list, NO sampling
}
```

No random sampling.  No greedy consensus.  Each completed gene tree is used
**directly**.  Then (`WQDataCollection.java` lines 757–776):

```java
// secondRoundSampling = 1 for single-individual
for j in 0..k-1:
    FormSetXLoop(allGreedies[j].get(0), baseTrees, latch)
```

Each `FormSetXLoop` call runs `addBipartitionsFromSingleIndTreesToX` on one
completed gene tree:

```
post-traverse the tree:
  at each internal node:
    add its bipartition (BitSet over all n taxa) to X
  if the node is a polytomy (degree > 2):
    3 rounds of random-sample-based polytomy resolution:
      sample one taxon from each partition of the polytomy
      extract matching bipartitions from baseTrees
      map back to full taxon set → add to X
```

**For multi-individual datasets** (lines 662–719), a much heavier path runs:
- K=100 random individual samples per species × secondRoundSampling rounds
- Contract each gene tree to each sample → build greedy consensus trees
- Extract bipartitions from those consensus trees → add to X

This is O(K × S × k × n²) and is irrelevant for single-individual datasets.


---

## 5. Complete Complexity Analysis (Single-Individual)

For single-individual datasets with n taxa, k gene trees, fraction f_miss of
taxa missing per tree on average:

| Phase | Time | Memory | Bottleneck |
|---|---|---|---|
| Build distance matrix | O(k × n²) | O(n²) = 4n² bytes | **Dominant — scales quadratically** |
| Pre-sort matrix rows | O(n² log n) | O(n²) | One-time cost |
| Complete all trees | O(k × f_miss × n × depth) | O(n) | Cheap; depth ≈ O(log n) |
| UPGMA tree → X | O(n² log n) | (reuses matrix) | Cheap |
| Gene tree bipartitions → X | O(k × n) | O(\|X\| × n/8) | Cheap; \|X\| ≤ O(k×n) |
| Polytomy resolution | O(k × p × d × n) | O(n) | Cheap; p, d are small |
| **Total** | **O(k × n²)** | **O(n²)** | Matrix dominates |

For n=25,000, k=1,000:
- Distance matrix: 2.5 GB, ~625B ops (minutes on CPU, ~60s on GPU)
- Everything else: negligible

The entire multi-individual sampling + greedy consensus block (O(K × S × k × n²))
**does not execute** for single-individual datasets.  The pipeline reduces cleanly
to: build matrix → complete trees → walk each tree once → insert bipartitions into X.


---

## 6. GPU Transferrability Summary

| Step | GPU fit | Reason |
|---|---|---|
| Distance matrix construction | **High** | O(k×n²) scatter-add, embarrassingly parallel over trees; matrix fits GPU for n≤25,000 |
| Matrix row sorting | Low | Global sort per row, irregular comparators |
| 4-point tree navigation | Poor | Pointer-chasing tree traversal, sequential decisions per level |
| Tree insertion (leaf/internal) | Poor | Mutable tree structure, branching logic |
| Gene tree post-traverse → X | Low | Tree traversal; O(k×n) total is already fast on CPU |
| UPGMA from matrix | Low-medium | Iterative best-pair selection, dynamic active set |
| Polytomy resolution | Poor | Tiny random samples, branching control flow |

**Practical recommendation**: GPU-accelerate the distance matrix construction only.
Everything else is either fast enough on CPU or fundamentally irregular.  The matrix
construction kernel is structurally very similar to the weight calculation kernel
already in ASTRAL-X — same `orderings`/`invIndex` arrays, same post-order traversal,
same scatter-add pattern.  The existing static data already uploaded to the GPU
during Phase 6 (weight calculation) could be reused for matrix construction if the
phase ordering is adjusted.


---

## 7. Design Implications for ASTRAL-X

### Philosophical difference

| | ASTRAL-MP | ASTRAL-X (current) |
|---|---|---|
| Incomplete trees | Complete first, extract bipartitions from completed trees | Extract tripartitions directly from incomplete trees |
| Weight DP input | Completed trees | Original incomplete trees with adjusted formula |
| X source | Bipartitions of completed trees + UPGMA tree | Bipartitions of raw incomplete trees |
| Memory cost | O(n²) for matrix | O(n) — no distance matrix |

### What ASTRAL-X gains from adding this pipeline

Completing incomplete trees and adding the resulting bipartitions to X enriches the
search space with full-taxa bipartitions that are not visible in the raw incomplete
trees.  This can improve accuracy on datasets with high rates of missing taxa.

### Scalability boundary

The distance matrix gates everything.  For n ≤ ~25,000 (2.5 GB matrix, fits on
RTX 4090), the full ASTRAL-MP compatible pipeline is feasible with a GPU kernel for
matrix construction.  Beyond that, an approximate alternative is needed:

- **Co-occurrence sparse matrix**: only store `d(x, y)` for pairs that actually
  co-appear in ≥1 gene tree.  Natural sparsity for incomplete data; degenerates to
  full n×n for complete trees.
- **kNN index**: store K nearest neighbors per taxon (K ≈ 50–100), change
  representative selection in navigation to pick from x's kNN.  Breaks down when
  a subtree contains none of x's K neighbors.
- **Subsample-based UPGMA**: run UPGMA on a random sample of n'=2000 taxa,
  extract bipartitions, map back.  Only affects Track A (matrix tree), not completion.

For complete gene trees (f_miss = 0), the completion step is skipped entirely
(the else branch at line 616), and only the matrix-derived UPGMA tree contributes
to X enrichment.  In that case, the matrix is built but never used for navigation —
only for UPGMA.  This is a significant wasteful cost for complete-tree datasets.
