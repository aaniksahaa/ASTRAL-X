# Efficient UPGMA Implementation: Design and Analysis

Analysis of the UPGMA algorithm as used by ASTRAL-MP, a diagnosis of why the
current implementation is memory-pathological, and a complete specification of
an efficient replacement that achieves a 21× memory reduction and improves
expected running time simultaneously.

Reference file: `astral-mp-legacy-codebase/SimilarityMatrix.java`

---

## 1. What UPGMA Is and Why It Is Used Here

After building the n×n similarity matrix, ASTRAL-MP runs UPGMA on it to produce
one additional "guide" tree whose bipartitions are donated to the search space X.
This enriches X with bipartitions that may not appear in any individual gene tree
but are supported by the aggregated signal across all k trees.

UPGMA (Unweighted Pair Group Method with Arithmetic Mean) is an agglomerative
hierarchical clustering algorithm:

```
start with n singleton clusters, one per taxon
repeat n-2 times:
    find the pair (I, J) with highest similarity
    merge them into a new cluster I
    update similarity of new I to every other active cluster k:
        sim(I, k) = (w_I * old_sim(I,k) + w_J * old_sim(J,k)) / (w_I + w_J)
    record the merged cluster as a bipartition → add to X
```

Each merge produces one bipartition (the set of taxa now in I). After n-2 merges,
all n taxa are in one cluster and the bipartitions form a binary tree. These
bipartitions are what get added to X.


---

## 2. What ASTRAL-MP Currently Does

### 2.1 Data structures

`SimilarityMatrix.UPGMA()` initialises these structures for n taxa:

```java
List<BitSet>           bsList;        // n BitSets, each of n bits
List<TreeSet<Integer>> indsBySim;     // n sorted sets, each with n-1 entries
List<float[]>          sims;          // n float arrays, each of length n (matrix copy)
List<Integer>          weights;       // n integers
```

`indsBySim.get(i)` is a `TreeSet<Integer>` whose elements are the indices of all
OTHER active clusters, sorted in descending order of `sims.get(i)[j]`. It allows
`indsBySim.get(i).first()` to retrieve the most similar neighbor of cluster i in
O(log n).

### 2.2 The main loop

Each of n-2 iterations:

**Find best pair** (`upgmaLoop` lines 353–365):
```java
for (int i = 0; i < indsBySim.size(); i++) {
    int j = indsBySim.get(i).first();          // best neighbor of i
    if (sims.get(i)[j] > bestHit) { ... }      // track global max
}
```
Scans all n rows, calls `.first()` on each TreeSet. **O(n) per iteration**.

**Update distances** (`upgmaLoop` lines 371–393):
```java
for (int k = 0; k < sims.size(); k++) {
    float newSim = weighted_average(iDist[k], jDist[k], w_i, w_j);
    indsBySim.get(k).remove(closestI);
    sims.get(k)[closestI] = newSim;
    indsBySim.get(k).add(closestI);            // re-insert with new key
    // same for indsBySim.get(closestI)
}
```
For each of n active rows: one `TreeSet.remove` + one `TreeSet.add`.
Each is a red-black tree operation: **O(log n) per row → O(n log n) per iteration**.

### 2.3 Complexity of the current implementation

| Phase | Time |
|---|---|
| Initialisation (n sorted TreeSets) | O(n² log n) |
| Find best pair (total, n iterations × O(n)) | O(n²) |
| Distance updates (total, n iterations × O(n log n)) | **O(n² log n)** |
| **Total** | **O(n² log n)** |

### 2.4 Memory of the current implementation

The `sims` list is a full copy of the n×n matrix: **4n² bytes**.

The `indsBySim` list is the problem. Each `TreeSet<Integer>` stores Java Integer
objects in a red-black tree. A single Java `TreeSet` node (red-black tree entry)
holds:

- 1 Integer object: 16 bytes header + 4 bytes value = 20 bytes
- TreeMap.Entry wrapping it: ~32 bytes (value ref, left, right, parent, color)
- Total per entry: **~40–48 bytes**

At initialisation there are n rows, each with n-1 entries:

```
n × (n-1) entries × 48 bytes ≈ 48n² bytes
```

| Structure | Size | n=1,000 | n=10,000 | n=25,000 |
|---|---|---|---|---|
| `sims` (float copy of matrix) | 4n² bytes | 4 MB | 400 MB | 2.5 GB |
| `indsBySim` (TreeSet nodes) | ~40n² bytes | 40 MB | 4 GB | **25 GB** |
| `bsList` (BitSets) | n²/8 bytes | 125 KB | 12.5 MB | 78 MB |
| `weights` | 4n bytes | 4 KB | 40 KB | 100 KB |
| **Total** | **~44n²** | **~44 MB** | **~4.4 GB** | **~27 GB** |

For n = 25,000, the UPGMA step alone requires ~27 GB of JVM heap. This is the
dominant memory consumer in the entire ASTRAL-MP pipeline, dwarfing even the
distance matrix construction.

### 2.5 Cache behaviour

Every `.first()` call on a TreeSet dereferences a Java heap pointer to the root
of the red-black tree, then follows left-child pointers. The Integer keys are
boxed objects at arbitrary heap locations. Each access is a pointer dereference
with no spatial locality → **cache miss per access**. For n = 25,000 and n
iterations per find-best step, that is ~25,000 cache misses per iteration, every
iteration.


---

## 3. Why TreeSets Are Wasteful Here

The TreeSet per row maintains a sorted order so that `.first()` retrieves the best
neighbor of row i in O(1). The O(log n) cost is paid on every `remove` + `add`
during the update step.

But look at how the global best pair is found:

```java
for (int i = 0; i < indsBySim.size(); i++) {
    int j = indsBySim.get(i).first();    // O(1) per row
    if (sims.get(i)[j] > bestHit) ...
}
```

This still scans **all n rows**. The sorted order within each TreeSet is never
used for the global search — it only provides the per-row maximum. A plain `float`
storing the per-row maximum would serve identically. The entire TreeSet structure
exists only to answer one question per iteration: "what is the best neighbor of
row i?" — a question that can be answered with a single float and a single int.


---

## 4. The Efficient Implementation

### 4.1 Key insight

Replace each n-entry TreeSet with two primitive scalars:

```
float bestSim[i]  =  max over active j≠i of sim(i, j)
int   bestJ[i]    =  index of that best neighbor
```

These are flat primitive arrays: zero object overhead, zero boxing, sequential
memory layout, hardware-prefetcher-friendly.

The global best pair is then found by a single linear scan of `bestSim[]`.

### 4.2 Data structures

```
float[] sim         // upper triangle, size n*(n+1)/2, primitive, no objects
float[] bestSim     // size n, one float per cluster
int[]   bestJ       // size n, one int per cluster
boolean[] active    // size n
int[]   weight      // size n
BitSet[] clusters   // n BitSets of n bits each → for bipartition output
```

Upper triangle indexing (i ≤ j):
```java
int idx(int i, int j) {
    if (i > j) { int t = i; i = j; j = t; }   // ensure i ≤ j
    return i * n - i * (i + 1) / 2 + j;
}
```

### 4.3 Initialisation

One full O(n²) pass over the matrix. For each row i, scan all j≠i to find
`bestJ[i]` and `bestSim[i]`:

```java
for (int i = 0; i < n; i++) {
    active[i]  = true;
    weight[i]  = 1;
    clusters[i] = new BitSet(n);
    clusters[i].set(i);
    bestSim[i] = Float.NEGATIVE_INFINITY;
    bestJ[i]   = -1;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float s = sim[idx(i,j)];
        if (s > bestSim[i]) { bestSim[i] = s; bestJ[i] = j; }
    }
}
```

Time: **O(n²)**. This is a single cache-sequential pass over `sim[]` per row.

### 4.4 Main loop

Runs n-2 times.

**Step 1 — Find global best pair: O(n)**

```java
int I = -1;  float best = Float.NEGATIVE_INFINITY;
for (int i = 0; i < n; i++) {
    if (!active[i]) continue;
    if (bestSim[i] > best) { best = bestSim[i]; I = i; }
}
int J = bestJ[I];
```

`bestSim[]` is 100 KB for n=25,000 — fits entirely in L2 cache.

**Step 2 — Record bipartition: O(n/64)**

```java
BitSet merged = (BitSet) clusters[I].clone();
merged.or(clusters[J]);
output.add(merged);          // this bipartition goes into X
clusters[I] = merged;
```

**Step 3 — Merge: O(1)**

```java
weight[I] += weight[J];
active[J]  = false;
```

**Step 4 — Update distances and caches: O(n) amortised**

```java
for (int k = 0; k < n; k++) {
    if (!active[k] || k == I) continue;

    // --- update the stored similarity ---
    float oldIK = sim[idx(I, k)];
    float oldJK = sim[idx(J, k)];
    float newIK = (oldIK * w_I + oldJK * w_J) / (w_I + w_J);
    sim[idx(I, k)] = newIK;   // write new value unconditionally

    // --- update bestJ[k] cache ---
    if (bestJ[k] == J) {
        // J is gone; forced full recompute of row k
        recomputeBest(k);
    } else if (newIK > bestSim[k]) {
        // found a strictly better neighbor
        bestJ[k]   = I;
        bestSim[k] = newIK;
    } else if (bestJ[k] == I && newIK < bestSim[k]) {
        // cached best got worse; something else may now be better
        recomputeBest(k);
    }
    // else: bestJ[k] is some other active m, sim(k,m) unchanged, cache valid
}

// row I's values all changed; always recompute
recomputeBest(I);
```

`recomputeBest(k)` is a linear scan of row k:

```java
void recomputeBest(int k) {
    bestSim[k] = Float.NEGATIVE_INFINITY;
    bestJ[k]   = -1;
    for (int m = 0; m < n; m++) {
        if (!active[m] || m == k) continue;
        float s = sim[idx(k, m)];
        if (s > bestSim[k]) { bestSim[k] = s; bestJ[k] = m; }
    }
}
```

### 4.5 Correctness argument

After merging (I, J), the only values that change in the similarity matrix are
`sim(k, I)` for all active k. All other pairwise similarities are unchanged.

Therefore `bestJ[k]` can only become stale in two ways:
1. `bestJ[k] == J` — J no longer exists. Handled by full recompute.
2. `bestJ[k] == I` and `newIK < old sim(k,I)` — cached maximum decreased.
   Something else might now be globally better for row k. Handled by full recompute.

In all other situations the cached `bestJ[k]` and `bestSim[k]` remain valid:
- If `bestJ[k]` is some active m ≠ I,J: `sim(k, m)` did not change, and
  `newIK ≤ bestSim[k]`, so m is still the best. Cache valid.
- If `bestJ[k] == I` and `newIK ≥ old sim(k,I)`: best got better or equal.
  Update `bestSim[k] = newIK` in O(1). Still valid.
- If `newIK > bestSim[k]` (regardless of who bestJ[k] was): I is now the best.
  Update in O(1). Valid.

No case is missed. The algorithm is correct.

### 4.6 Time complexity

| Phase | Time | Notes |
|---|---|---|
| Initialisation | O(n²) | Single pass per row |
| Find best pair (all iterations) | O(n²) | O(n) × n iterations |
| Distance update — O(1) cases | O(n²) | Dominant; O(n) rows × n iterations |
| Full recomputes | O(R × n) | R = total recomputes across all iterations |
| **Total** | **O(n² + R×n)** | |

**R analysis**: a full recompute for row k is triggered only when `bestJ[k]`
is invalidated (cases 1 or 2b above). This happens when k had I or J as its
best neighbor and the merge degrades that. For typical similarity matrices derived
from phylogenomic data, only O(1) rows per iteration require recompute, so R = O(n)
total → **O(n²) expected time**.

Worst case: every row has bestJ[k] ∈ {I, J} every iteration → R = O(n²) →
**O(n³) worst case**. This requires a highly degenerate similarity matrix where
one cluster is universally the most similar to all others, every iteration. This
does not occur on real data.

Compare to current:

| | Current (TreeSet) | Proposed (flat array) |
|---|---|---|
| Update step | O(n log n) guaranteed | O(n) expected, O(n²) worst |
| Total time | O(n² log n) guaranteed | O(n²) expected, O(n³) worst |
| Realistic behaviour | Always O(n² log n) | Always O(n²) |

The O(log n) factor in the current implementation is paid unconditionally on every
row, every iteration, because TreeSet.remove + add is always performed. The
proposed implementation pays O(1) for the common case and O(n) only for the rare
cache-invalidation case.


---

## 5. Memory Comparison

| Structure | Current | Proposed |
|---|---|---|
| Similarity data | `List<float[]>` full n×n copy: **4n²** bytes | `float[]` upper triangle: **2n²** bytes |
| Sorted neighbor structure | `List<TreeSet<Integer>>` with ~40 bytes/entry: **40n²** bytes | `float[] bestSim` + `int[] bestJ`: **8n** bytes |
| Cluster BitSets | n²/8 bytes | n²/8 bytes (same) |
| active / weight | negligible | negligible |
| **Total** | **~44n² bytes** | **~2.1n² bytes** |
| **Reduction factor** | — | **~21×** |

Absolute values:

| n | Current peak | Proposed peak | Savings |
|---|---|---|---|
| 1,000 | 44 MB | 2.1 MB | 42 MB |
| 5,000 | 1.1 GB | 52 MB | ~1 GB |
| 10,000 | 4.4 GB | 210 MB | ~4.2 GB |
| 25,000 | **27 GB** | **1.3 GB** | **~26 GB** |
| 50,000 | >100 GB (impossible) | **~5.3 GB** (feasible) | — |

The proposed implementation makes n = 25,000 comfortably feasible within a 4 GB
budget for the UPGMA step alone. The current implementation makes it impossible on
any single consumer machine.


---

## 6. Cache Behaviour

### Current

Finding the global best pair requires n calls to `TreeSet.first()`. Each call:
1. Dereferences the `TreeSet` root pointer (random heap location)
2. Follows left-child pointers to find the minimum-key node (2–3 pointer hops)
3. Dereferences the `Integer` key object (another random heap location)

For n = 25,000, this is ~75,000 pointer dereferences per find-best step, each
likely a cache miss. At ~100 ns per LLC miss, that is ~7.5 ms per iteration just
for find-best, or ~180 seconds total across n iterations.

The update step (n TreeSet remove+add operations) is similarly cache-hostile.

### Proposed

Finding the global best pair: linear scan of `float[] bestSim`, 100 KB for
n = 25,000. Fits entirely in L2 cache (typically 256 KB–1 MB). Hardware
prefetcher handles this pattern perfectly. Effectively zero cache misses.

Updating distances: linear scan of `float[] sim` (upper triangle row), sequential
memory access. The matrix itself (1.25 GB for n = 25,000) does not fit in cache,
but sequential access allows the prefetcher to hide most latency.

The cache advantage alone means the proposed implementation will be faster in wall
time by a substantial margin even for moderate n, independent of the asymptotic
improvement.


---

## 7. Summary: Why the Current Implementation is Simply a Poor Fit

The TreeSet-per-row structure was chosen to provide O(log n) re-insertion after an
update, and O(1) best-neighbor lookup per row. In isolation these are reasonable
properties. The problem is that the global best pair still requires a full O(n)
scan of all rows — making the O(1) per-row lookup irrelevant to the dominant cost.

The log n factor is paid on every single row update (n rows × n iterations = n²
updates) but buys nothing, because the bottleneck is elsewhere.

The resulting implementation has:
- **O(n² log n) time** (vs O(n²) possible)
- **~44n² memory** (vs ~2.1n² possible)
- **Cache-hostile pointer chasing** (vs sequential float array scans)

The efficient implementation replaces TreeSets with two flat primitive arrays,
handles cache invalidation with a simple case analysis (proven correct above),
and achieves a 21× memory reduction and a log n factor time improvement —
simultaneously, with no tradeoff.


---

## 8. Python Brute-Force Reference and Testing

A plain O(n³) Python implementation provides a correctness oracle for the Java
code.  For n ≤ 50 (the typical taxon count for automated testing), O(n³) takes
well under a second.

### 8.1 Python algorithm

```python
def upgma_bipartitions(sim, n):
    d       = {(i,j): sim[i][j] for i in range(n) for j in range(i+1,n)}
    clusters = [frozenset([i]) for i in range(n)]
    weights  = [1] * n
    active   = list(range(n))
    bips     = set()

    for _ in range(n - 1):
        # O(n²) find-best
        best_s, I, J = -inf, -1, -1
        for ii in range(len(active)):
            for jj in range(ii+1, len(active)):
                i, j = active[ii], active[jj]
                key  = (min(i,j), max(i,j))
                if d[key] > best_s:
                    best_s, I, J = d[key], i, j

        wI, wJ = weights[I], weights[J]
        merged = clusters[I] | clusters[J]
        if len(merged) < n:          # skip all-taxa cluster
            bips.add(merged)

        # O(n) weighted-average update
        for k in active:
            if k in (I, J): continue
            ki = (min(k,I), max(k,I));  kj = (min(k,J), max(k,J))
            d[ki] = (d[ki]*wI + d[kj]*wJ) / (wI + wJ)

        clusters[I] = merged;  weights[I] = wI + wJ
        active.remove(J)

    return bips   # set of frozensets of taxon indices
```

### 8.2 Test script

`test/test_upgma.py` runs end-to-end comparison for 20+ random seeds:

1. Generate n (5..30) taxa and k (8..20) random gene trees (some incomplete).
2. Compute similarity matrix in Python (same brute-force as
   `test_similarity_matrix.py`).
3. Run Python UPGMA → set of frozensets.
4. Run `java astralx.Main -i <trees> --verify-upgma -C` → parse
   `bipartition=...` lines → set of frozensets.
5. Assert the two sets are identical.

Run: `python3 test/test_upgma.py [-n NUM] [-v]`

Ties in similarity are vanishingly rare for floating-point inputs generated
from random trees (probability zero for continuous distributions), so
tie-breaking order does not affect correctness of the comparison.


---

## 9. Tree-Pointer Construction (No BitSets)

Rather than maintaining `BitSet[] clusters` to represent which taxa belong to
each active cluster, the implementation builds the UPGMA tree directly using
`TreeNode` pointer objects — the same type used throughout ASTRAL-X.

### 9.1 Design

```
TreeNode[] clusterRoot   // clusterRoot[i] = current subtree root for cluster i
```

At initialisation, each `clusterRoot[i]` is a leaf node with `taxonId = i`.

At merge(I, J):
```java
TreeNode newNode = new TreeNode();
newNode.left          = clusterRoot[I];
newNode.right         = clusterRoot[J];
clusterRoot[I].parent = newNode;
clusterRoot[J].parent = newNode;
clusterRoot[I]        = newNode;
```

After n−1 merges, `clusterRoot[I]` is the root of the full dendrogram.

### 9.2 Postorder walk → Tree object

A single DFS assigns `rangeStart`, `rangeEnd`, and fills `postorderArray`:

```java
void assignRanges(TreeNode node, int[] postArr, int[] cursor) {
    if (node.isLeaf()) {
        node.rangeStart = cursor[0];
        postArr[cursor[0]++] = node.taxonId;
        node.rangeEnd = cursor[0];
    } else {
        node.rangeStart = cursor[0];
        assignRanges(node.left,  postArr, cursor);
        assignRanges(node.right, postArr, cursor);
        node.rangeEnd = cursor[0];
    }
}
```

The result is wrapped in a `Tree(treeIndex, root, postArr, posMap, n, n)` —
exactly the same object type returned by `TreeParser`.

### 9.3 Bipartition extraction

`ClusterTable.addTree(upgmaTree, pref, n)` calls the existing `extractFromTree`
walk that already handles subtree ranges + super-complements, deduplication, and
all-taxa skipping.  Zero new code is needed for the bipartition extraction step;
the UPGMA tree is simply one more source tree.

### 9.4 Why not BitSets?

BitSets require O(n/64) per merge step for the `or` operation (merging two
cluster bitmasks) and O(n²/64) per output step (iterating bits to emit taxon
sets). The pointer tree uses O(1) per merge and O(n) total for the postorder
walk — strictly better, and consistent with the rest of ASTRAL-X's design.


---

## 10. Parallelization

The outer merge sequence is inherently serial (each merge changes state for the
next), but three inner operations per iteration can be parallelized.

### 10.1 Parallel find-best (Step 1)

`bestSim[]` is only 200 KB for n = 25,000 (double[]) — fits in L2 cache.
T threads each scan a disjoint chunk of `activeArr[0..activeCount-1]`, keep
a thread-local (bestI, bestSim), then one serial reduce over T winners.

```
parallel_for t in 0..T-1:
    scan activeArr[lo_t..hi_t), track local (I_t, S_t)
reduce: I = argmax S_t over t
```

### 10.2 Parallel row update (Step 3)

For each active k ≠ I, computing `newIK = (oldIK*wI + oldJK*wJ)/(wI+wJ)` is
independent across k.  T threads partition the active list; each thread owns
disjoint k values:

- Writes to `mat[idx(I,k)]` for different k go to disjoint memory addresses.
- Writes to `bestSim[k]` and `bestJ[k]` are per-k, disjoint across threads.
- Reads of `mat[idx(J,k)]` are read-only (J is deactivated before this phase).

No synchronization is needed except for the `ConcurrentLinkedQueue` used to
collect stale-row indices.

### 10.3 Parallel stale-row recompute (Step 4)

Rows that need a full recompute (`bestJ[k]` invalidated) are collected in a
`ConcurrentLinkedQueue<Integer>` during the parallel update.  A second
`processRangeParallel` dispatches `recomputeBest(k)` for each stale k.
Different stale rows are independent; no synchronization needed.

### 10.4 Serial row-I recompute (Step 5)

Row I's entire set of similarities changed; it always requires a full recompute.
This is done serially after all parallel phases complete.

### 10.5 Active-list maintenance: swap-and-shrink

A naïve `buildActiveList()` scan of all n entries every iteration costs O(n²)
total just for bookkeeping.  The swap-and-shrink trick reduces removal to O(1):

```
posInActive[j]            = position of cluster j in activeArr
removeActive(j):
    pos       = posInActive[j]
    last      = activeArr[activeCount - 1]
    activeArr[pos]    = last
    posInActive[last] = pos
    activeCount--
```

All parallel phases iterate `activeArr[0..activeCount-1]` directly — no stale
entries, no wasted scans.

### 10.6 Expected speedup

| Phase | Serial cost | With T threads |
|---|---|---|
| Find-best | O(n) per iter | O(n/T) per iter |
| Row update | O(n) per iter | O(n/T) per iter |
| Stale recompute | O(R_iter × n) | O(R_iter × n / T) |
| Row-I recompute | O(n) per iter | O(n) (serial) |

The serial row-I recompute limits speedup via Amdahl's law: for T = 16 threads
and typical R_iter ≈ 1, the parallel fraction is roughly (n + R_iter × n) /
(2n + R_iter × n) ≈ 2/3 per iteration, capping theoretical speedup at ~3×.
In practice: find-best and row-update together dominate the iteration cost,
and the parallel fraction is higher, giving a more useful speedup especially
for large n.
