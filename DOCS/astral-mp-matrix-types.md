# Matrix Types in ASTRAL-MP: Complete Reference

Detailed analysis of the two matrix types used in ASTRAL-MP, their construction
algorithms, internal representations, and the exact roles each plays in the
pipeline.

Source files:
- `astral-mp-legacy-codebase/Matrix.java` — interface
- `astral-mp-legacy-codebase/AbstractMatrix.java` — shared implementation
- `astral-mp-legacy-codebase/SimilarityMatrix.java`
- `astral-mp-legacy-codebase/DistanceMatrix.java`
- `astral-mp-legacy-codebase/WQDataCollection.java` — usage site

---

## 1. The Matrix Interface

`Matrix.java` defines the contract that both concrete types implement:

```java
public interface Matrix {
    int    getSize();
    float  get(int i, int j);
    boolean isDistance();

    // Gene tree completion:
    int  getClosestPresentTaxonId(BitSet presentBS, int missingId);
    int  getBetterSideByFourPoint(int x, int a, int b, int c);

    // Guide tree construction:
    List<BitSet> inferTreeBitsets();

    // Matrix construction:
    Matrix populate(List<STITreeCluster> treeAllClusters,
                    List<Tree> geneTrees, SpeciesMapper spm);

    // Polytomy resolution (extra mode):
    List<BitSet> resolvePolytomy(List<BitSet> bsList, boolean original);
    Matrix       getInducedMatrix(HashMap<String,Integer> randomSample, TaxonIdentifier id);
    Iterable<BitSet> getQuadraticBitsets();
}
```

Both concrete types are **fully interchangeable** through this interface. The
variable `geneMatrix` in `WQDataCollection` is of type `Matrix`. Which concrete
class is instantiated depends on a single command-line flag.

---

## 2. Selection: Which Matrix is Used?

`WQDataCollection.calculateDistances()` (line 842):

```java
if (options.isUstarDist()) {
    this.geneMatrix = new DistanceMatrix(n);    // --ustar-dist flag
} else {
    this.geneMatrix = new SimilarityMatrix(n);  // DEFAULT (no flag)
}
this.speciesMatrix = this.geneMatrix.populate(treeAllClusters,
                         originalIncompleteTrees,
                         GlobalMaps.taxonNameMap.getSpeciesIdMapper());
```

Two matrices are produced:
- `geneMatrix`: individual-level n×n matrix (all individuals)
- `speciesMatrix`: species-level n_s×n_s matrix (collapsed to species)

`speciesMatrix` is derived from `geneMatrix` via `SpeciesMapper.convertToSpeciesDistance()`.

---

## 3. SimilarityMatrix (Default)

### 3.1 What it measures

`SimilarityMatrix[i][j]` = average **quartet co-occurrence score** between taxon
i and taxon j across all gene trees where both co-appear.

Concretely: for each internal node v in a gene tree with children C₁, C₂, ..., Cₘ
and "other" subtree (taxa in the tree but not in v's subtree), the similarity
between any taxon in Cᵢ and any taxon in Cⱼ (i ≠ j) is incremented by the
number of "fully resolved quartets" consistent with that split at v.

### 3.2 Construction algorithm

`SimilarityMatrix.populateByQuartetDistance()` (lines 63–240):

For each gene tree, at each internal node v:

```
children = all child subtrees of v
others   = taxa in tree but NOT in v's subtree

for each pair of groups (i, j) in {children ∪ others}:
    lc   = |left group i|
    rc   = |right group j|
    totalPairs = Σ_groups c*(c-1)/2         // all within-group pairs
    sim = totalPairs - lcp_i - rcp_j        // fully resolved quartets
    for a in group_i, b in group_j:
        matrix[a][b] += sim
        denom[a][b]  += (n_tree - 2) * (n_tree - 3) / 2   // normalization
```

The "fully resolved quartets" count `sim` is the number of 4-taxon subsets
that include one from group i and one from group j, and whose topology at this
node is unambiguous.

After all trees:
```java
matrix[i][j] /= (denom[i][j] / 2)
```

### 3.3 Sign convention

Higher value = MORE similar = closer relationship.

`isDistance()` returns `false`.

In `compareTwoValues()`:
```java
int compareTwoValues(float f1, float f2) {
    return -Float.compare(f1, f2);  // reversed: higher similarity comes first
}
```

### 3.4 Four-point formula for tree completion

```java
public int getBetterSideByFourPoint(int x, int a, int b, int c) {
    double xa = matrix[x][a], xb = matrix[x][b], xc = matrix[x][c];
    double ab = matrix[a][b], ac = matrix[a][c], bc = matrix[b][c];
    double ascore = xa + bc  - (xb + ac);   // NOTE: similarity, not distance
    double bscore = xb + ac  - (xa + bc);
    double cscore = xc + ab  - (xb + ac);
    return ascore >= bscore ?
           ascore >= cscore ? a : c :
           bscore >= cscore ? b : c;
}
```

Returns the side (a, b, or c) that x is MOST similar to. The four-point
condition in similarity space: x groups with whichever side maximizes
`sim(x, that_side) + sim(anchor, other_side)`.

### 3.5 Guide tree construction

`inferTreeBitsets()` → **`UPGMA()`**

Runs UPGMA on the n×n similarity matrix. At each step, merges the pair of
clusters with highest average similarity. Produces one tree whose bipartitions
are added to X. (See `DOCS/upgma-efficient-implementation.md` for full UPGMA
analysis.)

### 3.6 Polytomy resolution

`resolvePolytomy()` → **`resolveByUPGMA(bsList, original)`**

For a polytomy with d child groups, computes the d×d induced sub-matrix of
inter-group average similarities, runs UPGMA on it, produces resolution
bipartitions for X.

---

## 4. DistanceMatrix (`--ustar-dist` flag)

### 4.1 What it measures

`DistanceMatrix[i][j]` = average **branch-count distance** (number of edges on
the path) between taxon i and taxon j across all gene trees where both co-appear.

### 4.2 Construction algorithm

`DistanceMatrix.matricesByBranchDistance()` (lines 223–330):

For each gene tree, post-order traversal:

```java
at each leaf v:
    distanceMap[v] = 0

at each internal node v with children L, R (and possibly more):
    for each pair (i in L, j in R):
        d = distanceMap[i] + distanceMap[j] + 2
        matrix[i][j] += d
        pairNumMatrix[i][j] += 1
    for each taxon k in v's subtree:
        distanceMap[k] += 1   // propagate distance up
```

`distanceMap[k]` at node v = number of edges from leaf k to v. When leaves i
and j are split at their LCA v, their path length = `distanceMap[i] + distanceMap[j] + 2`
(the `+2` is for the two edges connecting L and R to v, i.e. the branch from
i's subtree to v plus the branch from j's subtree to v; each step up increments
by 1).

After all trees:
```java
matrix[i][j] /= pairNumMatrix[i][j]   // average over co-appearing trees
// if pairNumMatrix[i][j] == 0: matrix[i][j] = -99 (sentinel)
```

**Note**: this is actually computable using the LCA formula:
`d(i,j) = depth[i] + depth[j] - 2×depth[LCA(i,j)]`, which is the basis for
the GPU design in `DOCS/gpu-distance-matrix-complete-design.md`.

### 4.3 Sign convention

Lower value = closer relationship (it is a distance).

`isDistance()` returns `true`.

In `compareTwoValues()`:
```java
int compareTwoValues(float f1, float f2) {
    return Float.compare(f1, f2);  // natural order: lower distance = better
}
```

### 4.4 Four-point formula for tree completion

```java
public int getBetterSideByFourPoint(int x, int a, int b, int c) {
    double xa = matrix[x][a], xb = matrix[x][b], xc = matrix[x][c];
    double ab = matrix[a][b], ac = matrix[a][c], bc = matrix[b][c];
    double ascore = (xb + ac) - (xa + bc);   // distance: opposite sign to similarity
    double bscore = (xa + bc) - (xb + ac);
    double cscore = (xb + ac) - (xc + ab);
    return ascore >= bscore ? ascore >= cscore ? a : c
                            : bscore >= cscore ? b : c;
}
```

Returns the side that x is LEAST distant from. The sign is the exact inverse of
SimilarityMatrix's formula — same algorithm, opposite direction.

### 4.5 Guide tree construction

`inferTreeBitsets()` → **`PhyDstar()`**

PhyD* is a neighbor-joining variant. It is invoked by:
1. Serializing the distance matrix into PHYLIP format (a text string)
2. Calling the PhyD* Java library
3. Parsing the returned Newick tree
4. Extracting its bipartitions as BitSets

PhyD* is specifically designed for branch-distance matrices and typically gives
better trees than UPGMA when using branch lengths.

### 4.6 Polytomy resolution

`resolvePolytomy()` → **`resolveByPhyDstar(bsList, original)`**

For a polytomy with d groups, builds a d×d sub-matrix of inter-group distances
by random sampling (10 rounds of random representative pairs), then runs PhyD*
on the d×d matrix to produce a resolution tree.

The random sampling (lines 120–156):
```java
for each round (10 rounds):
    pick random pair leaves (i1, i2) from group i
    pick random pair leaves (j1, j2) from group j
    internalMatrix[i][j] += (d(i1,j1) + d(i2,j2) - d(i1,i2) - d(j1,j2)) / 2
// average over 10 rounds
```
This estimates the inter-group distance from a random sample of cross-group pairs.

---

## 5. Side-by-Side Comparison

| Aspect | SimilarityMatrix (default) | DistanceMatrix (`--ustar-dist`) |
|---|---|---|
| Entry meaning | avg quartet co-occurrence score | avg branch-count distance |
| Higher/lower = closer | Higher | Lower |
| `isDistance()` | false | true |
| Sort direction | Descending (highest first) | Ascending (lowest first) |
| Construction method | `populateByQuartetDistance` | `matricesByBranchDistance` |
| Construction cost | O(k × n²) | O(k × n²) |
| Memory | O(n²) float | O(n²) float |
| Sentinel for missing pairs | 0 (never co-appeared → sim=0) | -99 |
| Tree completion algorithm | 4-point similarity | 4-point distance (signs flipped) |
| Guide tree algorithm | UPGMA | PhyDstar (NJ-variant) |
| Polytomy resolution | UPGMA on sub-matrix | PhyDstar on sub-matrix |
| GPU acceleration difficulty | Hard (per-node formula) | Easy (per-pair LCA formula) |

---

## 6. Why Two Separate Approaches?

### The conceptual reason

DistanceMatrix measures **where taxa are in tree space** (branch lengths). It is
sensitive to rate variation, long branches, and saturation — the same artifacts
that affect distance-based tree reconstruction in general.

SimilarityMatrix measures **how taxa co-cluster in quartet topologies** — which is
directly aligned with what ASTRAL's objective function maximizes. A high quartet
co-occurrence score between i and j means they frequently appear as sisters in
resolved quartets across gene trees, which is exactly the signal ASTRAL is
designed to detect.

### The practical reason

SimilarityMatrix is the default because it is more robust for typical phylogenomic
datasets. DistanceMatrix + PhyDstar is the older approach (retained for
compatibility and as an alternative when datasets have very complete gene trees
where branch-length information is reliable).

---

## 7. How `orderedTaxonBySimilarity` Works (Shared)

Both types use the same `AbstractMatrix` infrastructure for finding the closest
taxon during tree completion. `AbstractMatrix.sortByDistance()` builds a list of
n `TreeSet<Integer>` objects, one per taxon, sorted by the matrix value in the
direction relevant to the concrete type (via the abstract `compareTwoValues()`):

- SimilarityMatrix: sorted **descending** (most similar first)
- DistanceMatrix: sorted **ascending** (closest first)

`getClosestPresentTaxonId(presentBS, missingId)` walks `orderedTaxonBySimilarity[missingId]`
and returns the first taxon that is either already present in the gene tree or
has already been inserted (has smaller ID). This is the anchor taxon used for
4-point navigation.

The `orderedTaxonBySimilarity` list is **lazily computed** on first use
(`assureOrderedTaxa()`), then cached for all subsequent tree completions.

**Memory note**: this list has the same O(n²) TreeSet overhead as the UPGMA
`indsBySim` structure (48 bytes × n² entries). For n=25,000 this is ~27 GB —
the same memory pathology described in `DOCS/upgma-efficient-implementation.md`.

---

## 8. The Two Matrices: `geneMatrix` vs `speciesMatrix`

`populate()` in both types produces TWO matrices at once:

**`geneMatrix`** (n×n, individual level):
- One entry per individual across all gene trees
- Used for the 4-point navigation in gene tree completion
- `n` = total number of individuals (all taxa across all gene trees)

**`speciesMatrix`** (n_s×n_s, species level):
- Collapsed to species via `SpeciesMapper.convertToSpeciesDistance()`
- Multiple individuals per species → values averaged to species level
- Used for UPGMA/PhyDstar guide tree construction
- `n_s` = number of species (≤ n)

For single-individual datasets: n = n_s, so `geneMatrix` and `speciesMatrix`
are the same size and nearly identical (just converted via SpeciesMapper which
is identity for 1:1 datasets).

---

## 9. The `convertType` Utility (AbstractMatrix)

`AbstractMatrix.convertType(in)` converts a DistanceMatrix into a SimilarityMatrix-like
format by inverting the values:

```java
SM[i][j] = speciesCount - in.matrix[i][j]
```

This transforms distances into pseudo-similarities (high distance → low "similarity").
Used internally when DistanceMatrix values need to be passed to code expecting
the similarity convention (e.g. certain UPGMA paths). The `-99` sentinel is
preserved.

---

## 10. Implications for GPU Acceleration

| Matrix type | GPU kernel type needed | Difficulty |
|---|---|---|
| DistanceMatrix | Per-pair: `d = depth[a] + depth[b] - 2×depth[LCA(a,b)]` | **Easy** — pure per-pair formula |
| SimilarityMatrix | Per-internal-node: accumulate quartet counts across ALL child subtrees | **Hard** — depends on all siblings, not just the pair |

DistanceMatrix maps cleanly to the tiled GPU design in
`DOCS/gpu-distance-matrix-complete-design.md`. Each pair (a,b) is independent.

SimilarityMatrix requires a different kernel: for each internal node v (not for
each pair), compute the `sim` contribution and scatter it to all relevant pairs
(a ∈ Cᵢ, b ∈ Cⱼ). This is inherently a scatter operation unless reformulated
per-pair via a clever node-decomposition, which is non-trivial.

Since SimilarityMatrix is the default and is the better-performing option,
it is the higher-priority target for GPU acceleration despite its difficulty.

---

## 11. Summary Diagram

```
calculateDistances():
                              ┌─ --ustar-dist ─→  DistanceMatrix
          command-line flag ──┤
                              └─ (default) ────→  SimilarityMatrix
                                                    │
                              geneMatrix ←──────────┤
                              speciesMatrix ←────────┘

geneMatrix used for:
  ├── getClosestPresentTaxonId()  → find anchor taxon for tree completion
  └── getBetterSideByFourPoint()  → navigate to insertion point in tree

speciesMatrix used for:
  ├── inferTreeBitsets()           → UPGMA (Similarity) / PhyDstar (Distance)
  │                                  → guide tree bipartitions → X
  └── resolvePolytomy()           → UPGMA (Similarity) / PhyDstar (Distance)
                                     → polytomy resolution bipartitions → X
```

Both roles (tree completion and guide tree) use whichever matrix type was
selected. The interface is identical; only the metric and direction differ.
```
