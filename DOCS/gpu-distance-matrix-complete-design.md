# GPU Distance Matrix Construction: Complete Design

Full design for computing the n×n pairwise distance matrix from k gene trees
with maximum GPU parallelism while keeping GPU VRAM at O(B²) or O(B×n) rather
than O(n²).

Reference: `astral-mp-legacy-codebase/DistanceMatrix.java` (CPU baseline).

---

## 1. Problem Statement

Given k gene trees over a universe of n taxa (trees may be incomplete —
some taxa absent from some trees), compute:

```
D[a][b] = (1 / C[a][b]) * Σ_{t: both a,b present in T_t} d_t(a, b)
C[a][b] = number of trees where both a and b co-appear
```

where `d_t(a, b)` is the branch-count distance between taxa a and b in tree T_t.

If C[a][b] = 0 (pair never co-appears): D[a][b] = −99 (sentinel).

**Goals:**
- CPU RAM: O(n²) allowed — the final matrix lives here
- GPU VRAM: O(B² + tree-batch data) — NOT O(n²)
- Time: as close to O(k × n²) arithmetic as possible, maximally parallel


---

## 2. Why the Naive GPU Port Fails

The CPU algorithm (DistanceMatrix.java) does a post-order traversal. At each
internal node v splitting left subtree L and right subtree R:

```
for i in L, for j in R:
    D[i][j] += distanceMap[i] + distanceMap[j] + 2
```

This scatters writes to arbitrary (i,j) cells of a global n×n output buffer.

Problems with direct GPU port:
- n×n output buffer = 2.5 GB for n=25,000 — exceeds VRAM budget
- Multiple threads write to the same D[i][j] → atomic operations required
- Scattered global atomics → serialization, terrible throughput
- Different pairs get different amounts of work depending on tree shape → warp divergence

**The fix**: reformulate so each pair (a,b) is computed independently via a
formula over read-only tree data. No global scatter. No atomics.


---

## 3. Core Mathematical Reformulation

For a tree T rooted arbitrarily at node r, and any two taxa a, b both present:

```
d_T(a, b) = depth[a] + depth[b] - 2 × depth[LCA(a, b)]
```

where `depth[v]` = number of edges from root r to node v.

**Why this is exact**: the path between two leaves passes through their LCA.
`depth[a] - depth[LCA(a,b)]` = distance from a to LCA. Same for b. Sum = total path.

**Why this is GPU-friendly**:
- Each pair (a,b) is computed independently from read-only arrays
- No write conflicts: thread for (a,b) owns exactly one output cell
- All threads do the same amount of work structure: O(log n) for LCA

This single formula converts the problem from "per-node scatter" to
"per-pair independent evaluation." That is the architectural pivot.


---

## 4. Two Independent Batching Dimensions

There are two distinct batching dimensions, each with its own purpose:

### Dimension 1: Output tile size B

Controls how much of the n×n output matrix lives in GPU VRAM at once.

- Row-block variant: process B rows × n cols at a time → VRAM = O(B×n)
- Square tile variant: process B rows × B cols at a time → VRAM = O(B²)

Square tiles are strictly better: for the same VRAM budget V, tiles allow
B = √(V/4) while row-blocks allow B = V/(4n). Since B_tile >> B_rowblock
for large n, tiles achieve better GPU occupancy per launch.

### Dimension 2: Tree batch size Δ

Controls how many trees' data live in GPU VRAM simultaneously.

- Small Δ: less VRAM for tree data, more kernel launches
- Large Δ: more VRAM for tree data, fewer launches (less overhead)

These two dimensions are independent. For each output tile, we loop over
Δ-sized tree batches. Larger Δ means each tile-computation loop body does
more work before the next launch.

### VRAM formula

```
VRAM_total = B²  × sizeof(float) × 2    [sum and count tile buffers]
           + Δ   × per_tree_size         [tree batch data]

per_tree_size ≈ 2 × n_t × LOG_N × 4     [binary lifting table]
              + 2 × n_t × 4              [depth array]
              + n × 1                    [presence mask]
              + n × 4                    [local_id mapping]
```

For n_t = n = 25,000, Δ=1 tree: per_tree_size ≈ 3.3 MB.

Typical configuration (B=256, Δ=32): VRAM ≈ 0.5 MB (tiles) + 106 MB (trees) ≈ 107 MB.


---

## 5. Data Structures

### 5.1 Per-tree flat arrays

For a gene tree T with n_t taxa and 2n_t−1 total nodes (for a binary tree):

```
int   anc[2*n_t - 1][LOG_N]    binary lifting: anc[v][j] = 2^j-th ancestor of v
int   depth[2*n_t - 1]         depth from arbitrary root
int   local_id[n]              global taxon id → leaf node id in this tree (-1 if absent)
bool  present[n]               quick presence check (redundant with local_id < 0)
int   global_id[n_t]           leaf node id → global taxon id (inverse of local_id)
```

`LOG_N = ceil(log2(2*n_t))` ≈ 15 for n_t = 25,000.

Binary lifting table size: 2 × 25,000 × 15 × 4 = 3 MB per tree (n_t = 25,000).
For incomplete trees with n_t < n: proportionally smaller. A tree with 1,000
taxa uses only ~120 KB.

### 5.2 Flattened pool structure for tree batches

DO NOT store per-tree arrays as separate GPU allocations. Instead, flatten all
Δ trees in a batch into contiguous arrays with offset arrays:

```
int   parentPool[]             concatenation of all Δ trees' parent arrays
int   depthPool[]              concatenation of all Δ trees' depth arrays
int   ancPool[][]              concatenation of all Δ trees' binary lifting tables
int   localIdPool[]            concatenation of all Δ trees' local_id[n] arrays
int   treeNodeOffset[Δ+1]      start index of tree t's nodes in parentPool/depthPool/ancPool
int   treeTaxonOffset[Δ+1]     start index of tree t's local_id in localIdPool
int   treeNumNodes[Δ]          number of nodes in each tree
```

Benefits:
- Single GPU memory allocation per batch (no pointer-per-tree overhead)
- Contiguous memory = better prefetching
- Simpler kernel indexing: tree t's node v → ancPool[treeNodeOffset[t] + v][j]

### 5.3 Output tile buffers (persistent across tree batches)

```
float  sumTile[B][B]     accumulates Σ_trees d_t(a,b)
int    countTile[B][B]   accumulates co-occurrence count C[a,b]
```

These persist across all Δ-batches for one output tile. Zero them at the start
of each new tile, populate across all Δ-batches, then download to CPU.


---

## 6. Algorithm: Tiled Pair Evaluation

### 6.1 Outer structure

```
CPU: preprocess all k trees into flat arrays (parallel with GPU execution)

for each upper-triangular tile (I, J) where I ≤ J, I,J ∈ {0, B, 2B, ...}:

    zero sumTile[B][B] and countTile[B][B] on GPU

    for tree batch τ = 0, Δ, 2Δ, ..., k-Δ:
        upload flattened arrays for trees [τ, τ+Δ)
        launch kernel: B×B threads
        [kernel accumulates into sumTile, countTile across trees τ..τ+Δ-1]

    // All k trees processed for this tile
    normalize on GPU: sumTile[u][v] /= countTile[u][v]  (or set -99 if count=0)
    download sumTile[B][B] to CPU
    write CPU matrix D[I:I+B][J:J+B] = sumTile
    if I ≠ J: mirror D[J:J+B][I:I+B] = sumTile^T   (symmetry)

Total tiles: (n/B)×(n/B+1)/2  [upper triangle]
Total kernel launches: tiles × (k/Δ)
```

### 6.2 Kernel: one thread per pair per tree batch

```
kernel compute_tile_contribution(
    int I, int J,              // tile top-left corner in global matrix
    int B,                     // tile size
    int* ancPool,              // flattened binary lifting tables
    int* depthPool,            // flattened depth arrays
    int* localIdPool,          // flattened local_id[n] arrays
    int* treeNodeOffset,       // node start per tree
    int* treeTaxonOffset,      // taxon-map start per tree
    int  numTreesInBatch,
    float* sumTile,            // [B][B] persistent across batches
    int*   countTile           // [B][B] persistent across batches
)

thread (u, v):    // u = row in tile, v = col in tile
    a = I + u     // global taxon id for row
    b = J + v     // global taxon id for col

    float localSum = 0.0f
    int   localCount = 0

    for t = 0 to numTreesInBatch-1:

        int* localId = localIdPool + treeTaxonOffset[t]
        int aNode = localId[a]
        int bNode = localId[b]
        if aNode < 0 or bNode < 0: continue   // taxon absent from tree t

        int* anc   = ancPool   + treeNodeOffset[t] * LOG_N
        int* depth = depthPool + treeNodeOffset[t]

        int lca = binary_lifting_lca(aNode, bNode, anc, depth)
        int dist = depth[aNode] + depth[bNode] - 2 * depth[lca]

        localSum   += dist
        localCount += 1

    sumTile[u * B + v]   += localSum
    countTile[u * B + v] += localCount
```

Key properties:
- Thread (u,v) is the ONLY writer to `sumTile[u][v]` and `countTile[u][v]`
- **No atomic operations needed**
- Each thread accumulates in registers across tree loop → one global write per thread per batch
- Inner loop over trees: all threads do the same iterations (no warp divergence from loop count)
- Divergence only from `if aNode < 0`: for dense data this is rarely taken

### 6.3 Binary lifting LCA subroutine

```
device int binary_lifting_lca(int u, int v, int* anc, int* depth):
    // Bring u to same depth as v (u = deeper node)
    if depth[u] < depth[v]: swap(u, v)
    int diff = depth[u] - depth[v]
    for j = LOG_N-1 downto 0:
        if (diff >> j) & 1: u = anc[u * LOG_N + j]

    if u == v: return depth[u]   // LCA found

    // Lift both until they diverge
    for j = LOG_N-1 downto 0:
        if anc[u * LOG_N + j] != anc[v * LOG_N + j]:
            u = anc[u * LOG_N + j]
            v = anc[v * LOG_N + j]

    return depth[anc[u * LOG_N + 0]]   // depth of LCA
```

This is O(log n) = ≈15 iterations for n=25,000. Each iteration: 2–3 accesses
to `anc[]` (3 MB table for n=25,000). The table fits in GPU L2 cache (72 MB on
RTX 4090), so accesses hit L2 rather than DRAM after the first few passes.

### 6.4 Thread layout for memory coalescing

```
dim3 blockDim(32, 8)           // 256 threads/block: 32 along v-axis, 8 along u-axis
dim3 gridDim(B/32, B/8)        // for tile size B=256: 8×32 = 256 blocks
```

Within a warp: 32 consecutive threads share the same u (same row taxon a),
varying v (varying col taxon b):
- All 32 threads read `depth[a_node]`: same value → broadcast (single transaction)
- All 32 threads write `sumTile[u][v:v+32]`: consecutive floats → coalesced write ✓
- `localId[b]` for 32 consecutive b values: stride-1 access → coalesced read ✓
- `anc[b_node][j]` during LCA: b_node values are scattered → L2 cache handles this


---

## 7. Shared Memory Optimization

For small tile sizes (B ≤ 32), it is possible to preload per-tile taxon metadata
into shared memory.

For a B=32 tile processing taxa rows {a₀..a₃₁} and cols {b₀..b₃₁}:

```
// In thread block initialization (before tree loop):
__shared__ int sDepth_a[32]      // depth[aNode] for each row taxon
__shared__ int sAncRoot_a[32]    // anc[aNode][0] (parent) for each row taxon
__shared__ int sLocalA[32]       // local node ID for each row taxon
__shared__ int sLocalB[32]       // local node ID for each col taxon

// Thread (u, v) where u < 32, v < 32:
// Thread (u, 0) loads row taxon u's metadata
// Thread (0, v) loads col taxon v's metadata
// Synchronize, then all threads use shared memory
```

What CAN be preloaded:
- `local_id[a]` and `local_id[b]` for the 2B taxa in the tile: **O(2B integers)**
- `depth[aNode]` and `depth[bNode]`: **O(2B integers)**

What CANNOT be preloaded:
- Intermediate ancestor nodes during LCA traversal — these depend on the path
  up the tree from aNode and bNode, which are arbitrary node IDs not known before
  the tree loop starts. The `anc` table must remain in global memory (L2 cache).

**Net benefit of shared memory**: saves 2B global memory reads of local_id and
depth per tree per thread in the block. For B=32 and Δ=32 trees per batch:
saves 2×32 × 32 = 2,048 reads per block per batch. Modest but real.

The main performance driver for large B is L2 cache hit rate on the `anc` table,
not shared memory. The `anc` table (3 MB for n=25,000) fits in L2 — that is the
caching mechanism that matters most.


---

## 8. CPU Preprocessing

For each gene tree T_i, before uploading to GPU:

```
1. Root T_i at any node (e.g. node 0 or leftmost leaf)
2. DFS to compute:
   - depth[v] for all 2*n_t - 1 nodes
   - parent[v] for all nodes
3. Build binary lifting table:
   anc[v][0] = parent[v]
   for j = 1..LOG_N-1:
       anc[v][j] = anc[anc[v][j-1]][j-1]
4. Build local_id[n] mapping:
   for each leaf node l with global taxon id g:
       local_id[g] = l   (node index in this tree)
   for all other g: local_id[g] = -1
```

Time per tree: O(n_t log n_t).
For k=1,000 trees, n=25,000: 1,000 × 25,000 × 15 ≈ 375M ops → ~0.4 seconds.

**Pipeline overlap**: CPU preprocesses tree T_{i+1} while GPU processes batch
containing T_i. Use a double-buffer: CPU writes into "staging" CPU memory while
GPU reads from "active" GPU memory.

```
Thread A (CPU):  preprocess T_0 → T_1 → T_2 → ...
Thread B (GPU):  upload+compute T_0 batch → T_1 batch → ...
Overlap:         while GPU runs on T_i, CPU is preprocessing T_{i+1}
```

Zero preprocessing overhead in wall time.


---

## 9. VRAM Budget Analysis

For tile size B and tree batch Δ, with n=25,000 and n_t≈n:

| Component | VRAM | B=256, Δ=32, n=25K |
|---|---|---|
| sumTile[B²] | B² × 4 bytes | 0.26 MB |
| countTile[B²] | B² × 4 bytes | 0.26 MB |
| anc table (Δ trees) | Δ × 2n × LOG_N × 4 | Δ × 3 MB = 96 MB |
| depth arrays (Δ trees) | Δ × 2n × 4 | Δ × 0.2 MB = 6.4 MB |
| localId maps (Δ trees) | Δ × n × 4 | Δ × 0.1 MB = 3.2 MB |
| **Total** | | **~106 MB** |

Effect of tuning B and Δ:

| Configuration | VRAM | Kernel launches (n=25K, k=1K) |
|---|---|---|
| B=128, Δ=8 | 0.07 MB + 24 MB = ~24 MB | 24,600 |
| B=256, Δ=32 | 0.5 MB + 106 MB = ~107 MB | 3,200 |
| B=512, Δ=64 | 1 MB + 205 MB = ~206 MB | 800 |
| B=1024, Δ=128 | 4 MB + 410 MB = ~414 MB | 200 |

For RTX 4090 (24 GB VRAM): B=1024, Δ=128 uses <2% of available VRAM and needs
only 200 kernel launches for n=25,000, k=1,000.

**Comparison to naive O(n²)**:

| | Naive (full matrix on GPU) | Tiled design |
|---|---|---|
| n=25,000 | 2.5 GB (sum) + 2.5 GB (count) = 5 GB | **~100–400 MB** |
| n=50,000 | 20 GB (exceeds 24 GB VRAM) | **~100–400 MB (same)** |
| n=100,000 | 80 GB (impossible) | **~100–400 MB (same)** |

The tiled design's VRAM is **independent of n**. It scales only with B and Δ.


---

## 10. Incomplete Trees

Handling absent taxa is natural and requires no special cases in the kernel:

```
int aNode = localId[a]
int bNode = localId[b]
if aNode < 0 or bNode < 0: continue   // skip: absent from this tree
```

The `localId` array (size n) stores −1 for absent taxa. The check costs one
comparison per (thread, tree). For trees where most taxa are present (n_t ≈ n),
this branch is rarely taken and the GPU branch predictor handles it well.

For very sparse trees (n_t << n), most threads skip the entire tree loop body.
These threads still loop over the tree indices but do very little work — this is
wasted compute. Optimization: process only trees relevant to the current tile
by pre-filtering which trees contain at least one taxon in {I..I+B-1} and one
in {J..J+B-1}. Maintain a per-tree presence bitvector:

```
// Skip tree t if neither row taxon a nor col taxon b is present
if tile_presence[t][I/B] == 0 or tile_presence[t][J/B] == 0: continue
```

This reduces wasted iterations for heavily incomplete data.


---

## 11. Normalization and Output

After all k tree batches complete for a tile:

```
for (u, v) in tile [0..B) × [0..B):
    a = I + u, b = J + v
    if countTile[u][v] == 0:
        D_CPU[a][b] = -99.0f       // sentinel: never co-appeared
    else:
        D_CPU[a][b] = sumTile[u][v] / countTile[u][v]

    if I != J:
        D_CPU[b][a] = D_CPU[a][b]  // mirror symmetric entry
```

This can run as a small GPU kernel before download, or on CPU after download.
Running on CPU is simpler and the normalization cost is negligible (O(B²) ops).

**Diagonal tiles** (I = J): only process upper triangle within the tile, set
diagonal entries D[a][a] = 0. Use `if v >= u` in the kernel condition.


---

## 12. Complexity: Formal Statement

**Theorem**: The tiled algorithm computes D exactly in:

- Arithmetic: O(k × n²) pair evaluations, each O(log n) → total O(k × n² × log n)
- GPU VRAM: O(B² + Δ × n log n) — independent of n for fixed B, Δ
- CPU RAM: O(n²) for final matrix — unavoidable (it is the output)
- Kernel launches: O((n/B)² × k/Δ) — minimized by larger B and Δ
- Data transfer GPU→CPU: O(n²) total (one tile at a time, each O(B²)) — unavoidable

The O(log n) per pair (from binary lifting LCA) vs O(1) amortized per pair in
the CPU post-order algorithm is the only overhead. For n=25,000 this is a factor
of ≈15. GPU parallelism (10,000+ concurrent threads) more than compensates.

If O(1) LCA via Euler tour + sparse table is implemented instead, the log n
factor is eliminated. The sparse table (O(n log n) data, same size as binary
lifting) enables O(1) LCA queries at the cost of a more complex preprocessing step.


---

## 13. Row-block vs Square Tile: Formal Comparison

For the same VRAM budget V bytes:

| | Row-block | Square tile |
|---|---|---|
| Buffer shape | B_r × n | B × B |
| B from budget V | B_r = V/(4n) | B = √(V/4) |
| Threads per launch | B_r × n = V/4 | B² = V/4 |
| Tiles needed | n/B_r = 4n²/V | (n/B)² = 4n²/V |
| **Total launches** | **k × 4n²/V × 1/Δ** | **k × 4n²/V × 1/Δ** |

**Both require the same total kernel launches for the same VRAM budget.**

The real advantages of square tiles over row-blocks:

1. **Symmetric access to col metadata**: with B×n threads, you cannot preload
   n column taxa into shared memory. With B×B threads (B=32), both row and col
   taxa metadata fit in shared memory (2B entries = 256 bytes).

2. **Natural symmetry exploitation**: a tile (I,J) with I≠J computes exactly
   the same pairs as tile (J,I). Download once, mirror → halves total transfers.
   Row-blocks handle this awkwardly.

3. **Better for multi-GPU**: tiles partition the output matrix cleanly into
   independent chunks. Each GPU handles a subset of tiles.

4. **Occupancy flexibility**: B can be tuned independently of n. Row-blocks
   force B×n threads regardless of n.

**Recommendation**: implement square tiles. Use row-blocks only as a simpler
initial prototype (Stage 1 below).


---

## 14. Implementation Roadmap

### Stage 0: Prove correctness on paper

Define exactly:
- Which distance formula (branch count vs branch length)?
- How are missing pairs handled?
- What is `g_t(a,b)` for SimilarityMatrix (harder — see note below)?

### Stage 1: CPU reference with preprocessed arrays

Implement the same computation on CPU using:
- Arbitrary rooted trees
- Binary lifting LCA
- Row-by-row pair evaluation

This is the correctness oracle. Verify against existing DistanceMatrix.java output.

### Stage 2: GPU kernel, single tile, all trees

Single B×B tile, one tree at a time (Δ=1).
Verify correctness per tile.

### Stage 3: Tree batching

Add Δ > 1. Upload Δ trees per batch. Verify identical output to Stage 2.

### Stage 4: Full tile iteration

Iterate over all upper-triangular tiles. Mirror to CPU matrix. Verify final
matrix matches CPU reference.

### Stage 5: Pipeline overlap

CPU preprocessing of next tree batch overlaps with GPU computation of current.
Measure speedup.

### Stage 6: Tune B and Δ

Profile with different values. Target: maximize GPU SM utilization while
staying within VRAM budget. Typical sweet spot: B=256–512, Δ=32–128.

### Stage 7: Shared memory optimization

For small B (≤32): preload 2B taxon depths and local IDs into shared memory.
Measure improvement (likely 5–10% for large n where L2 already handles most).

### Stage 8 (future): Edge-based contribution aggregation

See Section 15.


---

## 15. Multi-GPU Extension

The tiled structure maps directly to multi-GPU:

```
GPU 0: handles tiles (I,J) where I mod num_GPUs == 0
GPU 1: handles tiles (I,J) where I mod num_GPUs == 1
...
```

Each GPU:
- receives all k trees (or tree batches) — same upload for all GPUs
- computes its assigned tiles independently
- streams finished tiles to CPU RAM

CPU assembly: receives tiles from all GPUs, places into final matrix.

No cross-GPU communication needed. Scales linearly with number of GPUs.


---

## 16. Future: Subtree-Contribution Aggregation

In the long run, there is a more efficient kernel structure that avoids
per-pair LCA queries entirely. The idea: for each internal node v in tree T,
every edge on the path from v to a leaf contributes exactly 1 to the distance
for all pairs (a,b) separated by that edge.

Formally, for each edge (v, parent[v]) in tree T:
```
D[a][b] += 1   for all a ∈ subtree(v), b ∉ subtree(v)
```

This is a rank-1 update: a Cartesian product of two sets. If the taxa in
subtree(v) are contiguous in the output tile's row index, this becomes an
outer-product ADD that can exploit SIMD very efficiently.

**The catch**: taxa in a subtree are NOT contiguous in the global taxon ordering.
To make them contiguous, taxa must be ordered by DFS pre-order of the current
tree — but this DFS order changes per tree.

**Practical realization**: for each tile, reorder the B row taxa and B col taxa
by DFS pre-order of the current tree. Then subtree membership becomes a contiguous
range in reordered coordinates. Each edge contributes to a rectangular subblock
of the tile.

This reduces the problem from B² LCA queries (O(B² log n) per tree per tile)
to O(n_t) edge contributions (O(n_t) per tree per tile), each contributing to
a subblock. Since n_t < n << B², this is a dramatic improvement.

However:
- Reordering adds preprocessing cost per (tile, tree): O(B log B)
- Subblock update requires atomics if multiple edges overlap the same cell —
  or a tree-local accumulator before the final += into sumTile
- Incomplete trees complicate the subtree-size arithmetic

This is a Stage 8+ optimization. For B² << n (typical: B=256, n=25,000 →
B²=65,536 << 625M = n²), the per-pair approach is already acceptable. The
edge-based approach matters most for very large tiles or very large n.


---

## 17. SimilarityMatrix: a Different Problem

The GPU design above is for `DistanceMatrix` (branch-count distance). The
`SimilarityMatrix` (quartet co-occurrence score) cannot use the same per-pair
LCA formula. Its contribution per internal node v depends on the sizes of ALL
sibling subtrees at v:

```
sim(a,b) += resolved quartets at v where a,b are on the same side
          = f(|C1|, |C2|, ..., |Cm|) — depends on ALL children of v
```

This is NOT decomposable as a simple per-pair formula. A GPU kernel for
SimilarityMatrix requires a per-internal-node design, not per-pair. The output
writes are scattered (same race-condition problem as the CPU algorithm). This
requires either:
- O(n²) VRAM for the accumulator (same problem as naive), OR
- A tiled reduction approach where each node only contributes to
  pairs within the current output tile

SimilarityMatrix GPU acceleration is a separate harder problem.
DistanceMatrix GPU acceleration (this document) is the priority, as it is
directly needed for gene tree completion and the design is clean.


---

## 18. Summary

| | CPU baseline | This design |
|---|---|---|
| Output buffer | O(n²) in RAM | O(B²) in VRAM + O(n²) in CPU RAM |
| Parallelism | None | B² threads per tile per tree batch |
| Scatter writes | Yes (post-order) | No (per-pair, no conflicts) |
| VRAM scaling | N/A | Independent of n |
| Incomplete trees | Natural | Natural (local_id check) |
| Multi-GPU | Hard | Natural (tile partition) |
| Arithmetic | O(k × n²) | O(k × n² × log n) [binary lifting] |
| n=25K, k=1K estimate | ~5–10 min | **~30–90 seconds** |
| n=50K feasible? | Yes (RAM) | **Yes** (VRAM unchanged) |
| n=100K feasible? | Yes (RAM only) | **Yes** (VRAM unchanged) |

The design keeps GPU VRAM at O(B² + Δ×n log n) regardless of n, streams the
O(n²) output to CPU RAM tile by tile, and achieves full GPU parallelism across
both the n² pairs and k trees simultaneously.
