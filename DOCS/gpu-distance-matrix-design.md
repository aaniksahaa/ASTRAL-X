# GPU-Accelerated Distance Matrix Construction: Design Plan

Goal: compute the n×n pairwise distance matrix from k gene trees with maximum
GPU parallelism, while keeping GPU VRAM at O(B×n) rather than O(n²).

---

## 1. The Problem

The distance matrix `D[i][j]` = average branch distance between taxon i and taxon j
over all gene trees where both co-appear.

Building it requires, for each of the k trees, computing the distance between
every pair of taxa that co-appear in that tree. Total work: O(k×n²). This is
the dominant cost in the ASTRAL-MP pipeline and the primary target for GPU
acceleration.

The naive CPU algorithm (`DistanceMatrix.java`) does a post-order traversal of
each tree. At each internal node v splitting left subtree L and right subtree R:

```
for i in L, j in R:
    D[i][j] += distanceMap[i] + distanceMap[j] + 2
    pairCount[i][j] += 1
increment distanceMap[*] as taxa propagate up
```

The problem with directly porting this to GPU: the writes `D[i][j] += ...` scatter
to arbitrary cells of the n×n output buffer. The output buffer alone is O(n²) =
2.5 GB for n=25,000. This is exactly what we want to eliminate from VRAM.

---

## 2. The Core Reformulation

Instead of computing contributions node-by-node (scatter to arbitrary cells),
reformulate as a **per-pair formula using LCA**.

For any two taxa i and j in the same gene tree T (rooted arbitrarily):

```
d(i, j) = depth[i] + depth[j] - 2 × depth[LCA(i, j)]
```

where depth[v] = number of edges from root to node v.

This is exact: the path between two leaves in a tree always passes through
their LCA. Depth of LCA = distance from root to the split point.

**Why this reformulation is GPU-friendly:**

In the scatter approach, two different pairs (i,j) and (i,k) may write to
overlapping cells if they share an ancestor → atomic operations or race conditions.

With the LCA formula, each pair (i,j) independently computes d(i,j) from
read-only tree data. No writes to shared cells within one tree's computation.
The output is a row buffer (described below) with perfectly partitioned writes.

---

## 3. Row-Batching Strategy

Process the matrix **B rows at a time** (B = batch size, a tunable parameter).

For a batch of sources `{s₀, s₁, ..., s_{B-1}}`:

```
row_buffer[B][n]     ← accumulates Σ_trees d(sᵦ, t)
count_buffer[B][n]   ← accumulates co-occurrence counts
```

For each of the k trees (sequentially on GPU, one per kernel launch):

```
launch B×n threads:
  thread (b, t):
    s = s_start + b
    if present[s] and present[t]:
        lca_depth = LCA_depth(s, t)     // O(log n) with binary lifting
        d = depth[s] + depth[t] - 2×lca_depth
        row_buffer[b][t] += d
        count_buffer[b][t] += 1
    // else: contribute nothing (taxon absent from this tree)
```

After all k trees finish for this batch:

```
for b in 0..B-1, t in 0..n-1:
    if count_buffer[b][t] > 0:
        row_buffer[b][t] /= count_buffer[b][t]   // normalize
    else:
        row_buffer[b][t] = -99                    // never co-appeared (sentinel)
download row_buffer[B][n] to CPU matrix rows [s_start, s_start+B)
```

Repeat for the next batch until all n rows are computed.

**Key property**: thread (b, t) is the ONLY thread writing to `row_buffer[b][t]`
within a single kernel launch. No atomic operations needed. Direct += is safe.

---

## 4. LCA on GPU: Binary Lifting

The LCA computation is the heart of each GPU thread.

**Binary lifting table** (precomputed on CPU per tree):

```
int anc[2*n_t - 1][LOG_N]   // anc[v][j] = 2^j-th ancestor of v
int depth[2*n_t - 1]        // depth from root
bool present[n]              // whether global taxon id is in this tree
int global_id[n_t]          // leaf index in tree → global taxon id
int local_id[n]             // global taxon id → leaf index in tree (-1 if absent)
```

`LOG_N = ceil(log2(2*n_t - 1))` ≈ 15 for n=25,000.

**LCA with binary lifting** (standard algorithm, O(log n) sequential steps per thread):

```
int lca_depth(int u, int v, int* depth, int anc[][LOG_N]):
    // bring both to same depth
    if depth[u] < depth[v]: swap(u, v)
    diff = depth[u] - depth[v]
    for j = LOG_N-1 downto 0:
        if (diff >> j) & 1: u = anc[u][j]
    // now depth[u] == depth[v]
    if u == v: return depth[u]
    // lift both together
    for j = LOG_N-1 downto 0:
        if anc[u][j] != anc[v][j]:
            u = anc[u][j]; v = anc[v][j]
    return depth[anc[u][0]]
```

Each of the LOG_N iterations accesses `anc[v][j]` — a memory access into the
binary lifting table. The table is O(n log n) per tree = ~3 MB for n=25,000.
This fits in GPU L2 cache (RTX 4090: 72 MB L2), so repeated accesses are fast.

**Why binary lifting over Euler tour + sparse table:**

Sparse table gives O(1) LCA queries but requires a 3-array indirection
(Euler tour → first[] → sparse_table[]). Binary lifting gives O(log n)
queries with simpler memory access pattern:
- All accesses into one flat `anc[v][j]` 2D array
- Better cache line utilization (sequential j traversal)
- Simpler to implement correctly in CUDA

For n=25,000: 15 iterations per pair. Total work per thread: 15 array accesses
+ 2 depth lookups = 17 memory operations. With L2 cache hit, ~30–50 ns per thread.

---

## 5. Data Structures and VRAM Budget

### Per-tree GPU upload (one tree at a time, streamed)

| Data | Type | Size | n_t=25,000 |
|---|---|---|---|
| `anc[2n_t-1][LOG_N]` | int | 4 × 2n_t × LOG_N | ~3 MB |
| `depth[2n_t-1]` | int | 4 × 2n_t | 200 KB |
| `present[n]` | bool | n | 25 KB |
| `local_id[n]` | int | 4n | 100 KB |
| **Per-tree total** | | | **~3.3 MB** |

For incomplete trees (n_t < n), the binary lifting table shrinks proportionally.
A tree with 1,000 taxa uses ~120 KB for the anc table.

### Row-batch buffers (persistent across all k trees for one batch)

| Data | Type | Size | B=64, n=25,000 |
|---|---|---|---|
| `row_buffer[B][n]` | float | 4 × B × n | 6.4 MB |
| `count_buffer[B][n]` | int | 4 × B × n | 6.4 MB |
| **Batch buffer total** | | | **~12.8 MB** |

### Total VRAM

```
VRAM = per_tree_data + batch_buffers
     = O(n log n)    + O(B × n)
     ≈ 3.3 MB        + 12.8 MB   (for n=25,000, B=64)
     = ~16 MB
```

For B=1024: ~206 MB total. Still very comfortable.

**VRAM vs B tradeoff:**

| B | VRAM (row+count buffers) | Batches needed | GPU launches per batch |
|---|---|---|---|
| 16 | 3.2 MB | n/16 = 1,563 | k = 1,000 |
| 64 | 12.8 MB | 391 | 1,000 |
| 256 | 51 MB | 98 | 1,000 |
| 1,024 | 205 MB | 25 | 1,000 |
| 4,096 | 820 MB | 7 | 1,000 |

More VRAM → fewer batches → fewer kernel launches (lower overhead).
For RTX 4090 with 24 GB available: B=4096 uses only 820 MB → 7 total batches.

---

## 6. Kernel Structure and Thread Layout

```
kernel: compute_distance_batch(
    int s_start, int B,           // batch range
    int* anc, int* depth,         // current tree binary lifting table
    bool* present, int* local_id, // current tree taxon presence
    int n_tree_nodes, int n,      // tree size, total taxon count
    float* row_buffer,            // [B][n] output, persists across trees
    int* count_buffer             // [B][n] co-occurrence counts
)

thread index = b * n + t    (b = source index within batch, t = target taxon)

b = threadIdx / n  (or computed from 2D launch)
t = threadIdx % n

s = s_start + b
s_local = local_id[s]
t_local = local_id[t]

if s_local < 0 or t_local < 0:
    return    // taxon absent from this tree

d = depth[s_local] + depth[t_local] - 2 * lca_depth(s_local, t_local, depth, anc)
row_buffer[b * n + t] += d
count_buffer[b * n + t] += 1
```

**2D thread launch for better occupancy:**

```
dim3 blockDim(32, 8)    // 32 threads along t-axis, 8 along b-axis = 256 threads/block
dim3 gridDim(n/32, B/8) // ceil divides
```

The 32-wide t-dimension gives coalesced writes to `row_buffer[b][t:t+32]` —
consecutive addresses in a warp → single memory transaction per warp. ✓

The 8-wide b-dimension means 8 source taxa per block. Each source's data
(local_id[s], depth[s_local]) is shared by 32 threads → load into shared memory
once per block.

---

## 7. CPU Preprocessing Per Tree

On the CPU, before uploading each tree to GPU:

```
for each gene tree T_i (in newick / array form):
    1. Root T_i at any node (e.g. leftmost leaf)
    2. DFS to compute:
       - depth[v] for all 2*n_t-1 nodes
       - parent[v] for all nodes
    3. Build binary lifting table:
       anc[v][0] = parent[v]
       for j = 1..LOG_N:
           anc[v][j] = anc[anc[v][j-1]][j-1]
    4. Build present[0..n-1] and local_id[0..n-1]
```

Time per tree: O(n_t log n_t).
Time for all k trees: O(k × n log n).

For k=1000, n=25000: 1000 × 25000 × 15 ≈ 375M ops. ~0.4 seconds on CPU.

This preprocessing can **pipeline with GPU computation**: while the GPU processes
tree T_i, the CPU preprocesses tree T_{i+1}. Zero preprocessing overhead.

---

## 8. Full Algorithm

```
// CPU preprocessing: streaming, overlapped with GPU
preprocess_thread: for each tree T_i: build anc[], depth[], present[], local_id[]

// GPU computation
for s_start = 0 to n-1 step B:          // n/B outer iterations
    clear row_buffer[B][n]
    clear count_buffer[B][n]

    for i = 0 to k-1:                    // k trees, sequential on GPU
        upload: anc[i], depth[i], present[i], local_id[i]  // ~3.3 MB

        launch kernel: B×n threads
            for each (b, t): compute d(s_start+b, t) in tree i
                             accumulate into row_buffer, count_buffer

    normalize: row_buffer[b][t] /= count_buffer[b][t]  (or set -99)
    download row_buffer[B][n] to CPU D[s_start .. s_start+B][0..n-1]
```

---

## 9. Complexity Analysis

### Time

| Component | Operations |
|---|---|
| CPU preprocessing (all k trees) | O(k × n log n) |
| GPU: total threads launched | k × (n/B) batches × B × n = k × n² threads |
| GPU: work per thread | O(log n) for binary lifting LCA |
| **GPU total ops** | **O(k × n² × log n)** |
| Data upload per tree per batch | O(n log n) upload × k × n/B = O(k × n² log n / B) |

The O(log n) factor from binary lifting is the only overhead vs the O(1) CPU
post-order traversal. For n=25,000, log n ≈ 15. This is a constant factor
well compensated by GPU parallelism (10,000+ threads running concurrently).

If O(1) LCA (sparse table) is implemented instead, the GPU total drops to O(k×n²).

### VRAM (peak)

```
VRAM_peak = O(n log n)   [current tree binary lifting table]
          + O(B × n)     [row and count buffers]
          ≈ O(B × n)     [for B >> log n, which is typical]
```

### CPU RAM

The final n×n distance matrix in CPU RAM: O(n²) = 2.5 GB for n=25,000.
This is unavoidable — it is the output. The pairCount matrix is the same size.
Both can be kept in CPU RAM throughout.

---

## 10. Handling Incomplete Gene Trees

A gene tree T_i with n_t < n taxa has:
- Binary lifting table of size O(n_t log n_t) << O(n log n)
- `present[s] = false` for n - n_t absent taxa

Each GPU thread checks `local_id[s] < 0 or local_id[t] < 0` and returns early
for absent pairs. The `count_buffer[b][t]` for such pairs stays 0, and after
normalization those cells get the -99 sentinel.

The tree upload cost scales with n_t, not n. A tree covering 20% of taxa uses
only 20% of the VRAM upload budget. This makes the algorithm naturally efficient
for sparse datasets.

---

## 11. Parallelism Dimensions

| Dimension | CPU approach | GPU approach |
|---|---|---|
| Across k trees | Sequential | Sequential (one tree per kernel) |
| Across n² pairs per tree | Sequential (post-order) | **B×n parallel threads** |
| Across B sources per batch | Sequential | **Parallel (B axis)** |
| Within one (source, target) pair | O(1) CPU | O(log n) GPU threads |

The GPU simultaneously parallelizes the k-independent trees AND the n²-pairs
within each tree. The k-sequential bottleneck exists but is mitigated by:
1. Each tree's kernel is large (B×n threads)
2. CPU preprocessing of the next tree overlaps with GPU execution of the current

---

## 12. Memory Coalescing Analysis

The critical memory access for performance is the write to `row_buffer`:

```
row_buffer[b * n + t] += d
```

Within a warp (32 threads), if threads are laid out with consecutive t values
(same b), then:
- `row_buffer[b*n + t], row_buffer[b*n + t+1], ..., row_buffer[b*n + t+31]`
- These are **32 consecutive floats** = one 128-byte cache line = one transaction ✓

The `anc[v][j]` accesses during binary lifting: different threads access different
nodes. This is non-coalesced but the `anc` table (3 MB for n=25,000) fits in L2
cache, so most accesses hit L2. L2 latency ≈ 30–40 cycles vs DRAM ≈ 500 cycles.

The `depth[]` array (200 KB) fits entirely in L1 cache after the first pass.

---

## 13. Comparison to CPU DistanceMatrix

| Aspect | CPU (DistanceMatrix.java) | GPU (this design) |
|---|---|---|
| Algorithm | Post-order scatter-add | LCA formula, per-pair direct |
| Output buffer | O(n²) flat matrix | O(B×n) row buffer (VRAM) |
| Parallelism | None (sequential per tree) | B×n threads per tree |
| LCA cost | Implicit in post-order (O(1) amortized) | O(log n) per pair |
| Cache behavior | Good (sequential tree traversal) | O(n log n) table fits L2 |
| n=25,000, k=1,000 time (estimate) | ~5–10 minutes | ~30–60 seconds |

The log n overhead from explicit LCA computation is real but small. The GPU
achieves ~10,000× more parallelism than a single CPU thread, more than
compensating for the log n factor.

---

## 14. SimilarityMatrix: a Note

The user's UPGMA uses `SimilarityMatrix` (quartet co-occurrence counts) rather
than `DistanceMatrix` (branch distances). The per-pair formula for the similarity
matrix is more complex:

```
sim(i,j) = (number of fully resolved quartets where i and j are on the same side)
           / (total quartets involving i and j)
```

This depends on the sizes of ALL sibling subtrees at each internal node, not
just the two-partition split. It cannot be expressed as a simple d(i,j) = f(LCA)
formula. GPU acceleration for SimilarityMatrix requires a different kernel
design (per-internal-node, not per-pair). The DistanceMatrix design described
here is the priority — it is directly needed for gene tree completion.

---

## 15. Summary

The design achieves O(B×n) VRAM while computing the O(k×n²) distance matrix by:

1. **Reformulating** the scatter-add (arbitrary cell writes) as a per-pair LCA
   formula (direct writes to a partitioned row buffer)

2. **Row-batching** to process B rows at a time: VRAM scales as O(B×n), not O(n²)

3. **Binary lifting LCA** precomputed per tree on CPU (O(n log n)), uploaded
   once per tree per batch (~3.3 MB per tree), O(log n) query per GPU thread

4. **Streaming pipeline**: CPU preprocesses tree T_{i+1} while GPU processes T_i

5. **No atomics**: row_buffer[b][t] has exactly one writer per kernel launch

6. **Incomplete tree support**: natural — absent taxa return early, n_t << n
   trees use proportionally less VRAM

For n=25,000, k=1,000, B=256: VRAM ≈ 54 MB. Estimated speedup over CPU: 10–30×.
