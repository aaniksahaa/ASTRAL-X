# Similarity Matrix — GPU-Parallel Design

## 1. Goal

Compute a taxon-similarity matrix `M[a,b]` from k gene trees over n global taxa:

```
M[a,b] = Σ_{t: a,b ∈ T_t}  num_t(a,b)
          ─────────────────────────────
          Σ_{t: a,b ∈ T_t}  den_t(a,b)
```

where for gene tree T_t with k_t leaves:

- `den_t(a,b) = C2(k_t − 2)`  (total resolved quartets in T_t containing a and b)
- `num_t(a,b)` = number of those quartets where a and b are on the **same side** (see §2)

After normalization: `M[a][a] = 1`, `M[a][b] = M[b][a]`.  
For tree completion we expose `dist[a][b] = 1 − M[a][b]`.

---

## 2. Exact Pairwise Formula

### 2.1 Node-level S[u]

For an internal node u in tree T_t, define its **incident components**:

- one component per child subtree `c₁, c₂, …, c_d`
- one **parent-side** component of size `k_t − subLeafCount[u]` (0 for the root)

Then:

```
S[u] = C2(k_t − subLeafCount[u])  +  Σ_c  C2(subLeafCount[c])
```

### 2.2 Per-pair contribution

For pair (a, b) with `u = LCA_t(a, b)`:

Let `c_a` = child of u on the path toward a,  
    `c_b` = child of u on the path toward b.

```
num_t(a,b) = S[u] − C2(subLeafCount[c_a]) − C2(subLeafCount[c_b])
```

This counts quartets {a,b,x,y} where x and y come from the **same** incident component
that is neither `c_a` nor `c_b` — exactly the quartets supporting a,b on the same side.

**This formula is correct for binary trees AND polytomies.**

For binary trees it simplifies: `num_t(a,b) = C2(k_t − subLeafCount[u])`,
but the implementation uses the general formula to support polytomies.

---

## 3. GPU Architectural Advantage

The original CPU code **scatters** node contributions into many matrix cells:

- many write conflicts
- requires O(n²) GPU accumulator
- heavy atomic writes

The pairwise reformulation turns this into **pair-owned independent evaluation**:

- thread (da, db) owns pair (a0+da, b0+db) for the current tile
- loops over a batch of Δ trees, reading only **its** pair's LCA data
- writes only to its own tile accumulator — **no atomics**
- GPU VRAM stays **O(B² + Δ·n·log n)**, never O(n²)

---

## 4. Tiling and Batching Strategy

Same architecture as the distance matrix kernel:

- **B×B output tile**: B = min(n, ceil(sqrt(n·k))). Tile VRAM = O(B²).
- **Δ-tree batching**: Δ chosen so that Δ·(per-tree GPU bytes) ≤ remaining VRAM.
- **Upper-triangle tiling**: only tiles with a0 ≤ b0 are processed; results mirrored.

For each tile:
1. Zero `numTile[B×B]` and `denTile[B×B]` on GPU
2. For each tree batch of Δ trees: upload → launch kernel → (accumulate across Δ trees)
3. Download tile → CPU merges into full n×n arrays

---

## 5. Per-Tree Preprocessing (CPU, parallel)

For each gene tree T_t, build:

### 5.1 Standard Euler tour + RMQ (shared with distance matrix)

DFS produces:
- `eulerDepths[tourLen]` — depth at each tour position
- `firstOcc[n]`           — first Euler position of each leaf taxon; −1 if absent
- `leafDepth[n]`          — depth of each leaf; −1 if absent (presence test)
- `eulerLen`              — actual tour length (= 3·k_t − 2 for k_t ≥ 2)
- `sparseMin[LOG][tourLen]` — standard min-depth sparse table (left-biased)

### 5.2 New: child-subLeafCount payload arrays

During the same DFS, at each Euler position pos record:

| Position type | `prevChildSubLC[pos]` | `nextChildSubLC[pos]` | `eulerS[pos]` |
|---|---|---|---|
| First visit to internal u | 0 | subLC[first child] | S[u] |
| Intermediate visit between c_i and c_{i+1} | subLC[c_i] | subLC[c_{i+1}] | S[u] |
| Leaf visit | 0 | 0 | 0 |

**Key property**: for pair (a,b) with l = min(firstOcc[a], firstOcc[b]), r = max:
- The LEFTMOST minimum in [l,r] is always an **intermediate visit** of LCA(a,b)
  between the child containing the left leaf and the next child.
- The RIGHTMOST minimum in [l,r] is the intermediate visit between the
  second-to-last child and the child containing the right leaf.

Therefore:
- `prevChildSubLC` at **leftmost argmin** = subLC of child containing the **left** leaf
- `nextChildSubLC` at **rightmost argmin** = subLC of child containing the **right** leaf

### 5.3 New: payload-tracking sparse tables

Build three additional sparse tables alongside `sparseMin`:

| Table | Build rule | Query gives |
|---|---|---|
| `sparseSubLCLeft[LOG][E]`  | left-biased argmin; carry `prevChildSubLC` | subLC of child containing the left leaf |
| `sparseSubLCRight[LOG][E]` | right-biased argmin; carry `nextChildSubLC` | subLC of child containing the right leaf |
| `sparseSLeft[LOG][E]`      | left-biased argmin; carry `eulerS`         | S[u] of LCA |

Build formula for left-biased (`sparseSubLCLeft`, `sparseSLeft`):
```
level 0:  payload[pos] = source[pos]
level ≥1: if sparse[lvl-1][pos] ≤ sparse[lvl-1][pos+half]:   // left wins (left-biased)
              payload[lvl][pos] = payload[lvl-1][pos]
          else:
              payload[lvl][pos] = payload[lvl-1][pos+half]
```

Build formula for right-biased (`sparseSubLCRight`):
```
level 0:  payload[pos] = nextChildSubLC[pos]
level ≥1: if sparse[lvl-1][pos+half] ≤ sparse[lvl-1][pos]:   // right wins (right-biased)
              payload[lvl][pos] = payload[lvl-1][pos+half]
          else:
              payload[lvl][pos] = payload[lvl-1][pos]
```

---

## 6. GPU Kernel Logic

Thread (da, db) handles pair `a = a0+da`, `b = b0+db` for the current B×B tile.

```
for each tree t in current Δ-batch:
    // Presence check
    if leafDepth[t][a] < 0 || leafDepth[t][b] < 0: continue

    fa = firstOcc[t][a];  fb = firstOcc[t][b]
    l  = min(fa, fb);     r  = max(fa, fb)

    // RMQ: 4 payloads in one overlap query
    k_lvl = 31 - clz(r - l + 1)
    l2    = r - (1 << k_lvl) + 1

    d_l = sparseMin[t][k_lvl][l];   d_r = sparseMin[t][k_lvl][l2]

    // Left-biased payloads
    subLC_leftChild = (d_l <= d_r) ? sparseSubLCLeft[t][k_lvl][l]
                                   : sparseSubLCLeft[t][k_lvl][l2]
    S_u             = (d_l <= d_r) ? sparseSLeft[t][k_lvl][l]
                                   : sparseSLeft[t][k_lvl][l2]

    // Right-biased payload
    subLC_rightChild = (d_r <= d_l) ? sparseSubLCRight[t][k_lvl][l2]
                                    : sparseSubLCRight[t][k_lvl][l]

    // Assign to ca/cb based on which leaf is left vs right in the tour
    subLC_ca = (fa <= fb) ? subLC_leftChild  : subLC_rightChild
    subLC_cb = (fa <= fb) ? subLC_rightChild : subLC_leftChild

    // Accumulate
    num  = S_u - C2(subLC_ca) - C2(subLC_cb)
    den  = C2(kt[t] - 2)
    numTile[da * bB + db] += num
    denTile[da * bB + db] += den
```

No atomics — each (da,db) owns a unique cell.

---

## 7. Flat GPU Arrays (per-tree, flattened for Δ-batching)

| Array | Type | Size per tree | Notes |
|---|---|---|---|
| `eulerDepths`       | `short` | E_max    | depth at each Euler position |
| `eulerPrevSubLC`    | `short` | E_max    | prevChildSubLC at each position |
| `eulerNextSubLC`    | `short` | E_max    | nextChildSubLC at each position |
| `eulerS`            | `int`   | E_max    | S[u] at each position |
| `sparseMin`         | `short` | LOG×E_max | min-depth left-biased sparse table |
| `sparseSubLCLeft`   | `short` | LOG×E_max | left-biased prevChildSubLC payload |
| `sparseSubLCRight`  | `short` | LOG×E_max | right-biased nextChildSubLC payload |
| `sparseSLeft`       | `int`   | LOG×E_max | left-biased S[u] payload |
| `firstOcc`          | `int`   | n         | first Euler position per leaf |
| `leafDepth`         | `short` | n         | leaf depth (−1 = absent) |
| `leafCount`         | `int`   | 1         | k_t per tree |
| `eulerLen`          | `int`   | 1         | actual tour length |

Per-tree bytes ≈ E_max × (2+2+2+4 + LOG×(2+2+2+4)) + n×6
                = E_max × (10 + 10·LOG) + 6n
                ≈ 3n × 10 × (1+LOG) + 6n      [E_max ≈ 3n]

---

## 8. Memory Complexity

### GPU VRAM (for one tile + Δ-tree batch)

```
VRAM = 2 · B² · 8          (numTile + denTile, double)
     + Δ · per_tree_bytes   (tree data on GPU)
```

With B = sqrt(n·k):  `B² = n·k`  →  tile VRAM = **O(n·k)**.  
Tree-batch VRAM = O(Δ·n·log n).  
Total GPU VRAM = **O(n·k + Δ·n·log n)** — no O(n²) term.

### CPU RAM

Full n×n similarity matrix: O(n²) — unavoidable (it is the output).  
All preprocessed trees: O(k·n·log n).

---

## 9. Normalization and Output

After all tiles are processed:

```java
sim[a*n+b] = (denSum[a*n+b] > 0) ? numSum[a*n+b] / denSum[a*n+b] : 0.0;
sim[a*n+a] = 1.0;  // diagonal
sim[b*n+a] = sim[a*n+b];  // symmetry
dist[a*n+b] = 1.0 - sim[a*n+b];  // for TreeCompleter
```

---

## 10. Integration with Tree Completion

`TreeCompleter.completeAll(trees, double[] dist, int n)` takes the `dist[]` array from
`SimilarityMatrix.dist` (or `DistanceMatrix.dist` for the CPU-only fallback).

In `Main.java`, when `--autocomplete-incomplete-gene-trees` is active:
1. Build `SimilarityMatrix` from the **original** (pre-completion) gene trees.
2. Call `TreeCompleter.completeAll(trees, sm.dist, n)`.
3. Continue with completed trees for cluster extraction.

This matches ASTRAL-MP's default behavior (similarity-guided completion).
