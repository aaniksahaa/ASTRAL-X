# GPU Weight Kernel — Adaptive Split Batching Design

## 1. The Problem

The ASTRAL-X weight calculation kernel (`astralx_weight.cu`) currently allocates all device
memory in one shot:

```
d_splits    = numSplits * 10 * sizeof(int)
d_twoScores = numSplits * sizeof(long long)
d_orderings = numTrees  * numTaxa * sizeof(int)
d_invIndex  = numTrees  * numTaxa * sizeof(int)
d_parts     = numParts  * 9  * sizeof(int)
```

The first two arrays grow with `numSplits`.  After Mode 2 (`--search-mode full`), the
cross-tree DP search can find O((nk)^1.73) candidate splits in the worst case (the
Tao–Kane bound for ASTRAL's search-space size, where n = taxa, k = gene trees).  In
practice the exponent is closer to 1.1–1.2 on biological data, but the possibility of
it reaching the theoretical bound means a naive single-allocation can exceed GPU VRAM
on large inputs.

**Concrete example.**  For n=200, k=1000 (nk = 200,000):

| Quantity | Formula | Estimate |
|---|---|---|
| Candidate splits (worst case) | O((nk)^1.73) | ~3 × 10⁹ |
| `d_splits` footprint | splits × 40 B | ~120 GB |
| `d_twoScores` footprint | splits × 8 B | ~24 GB |

Neither fits in a typical 24–80 GB GPU.  The static data (orderings + invIndex +
partitions) is O(n·k) and is generally tractable (a few GB for large inputs).


## 2. Why Batching Is Exact (Not an Approximation)

The weight of each candidate split `x` is an independent sum over all gene-tree
tripartitions `P`:

```
score(x) = Σ_P  frequency(P) × QI(x, P)
```

Because the per-partition contributions are additive and the partitions are independent
of each other, the sum can be split arbitrarily into batches without changing the result:

```
score(x) = Σ_b  Σ_{P ∈ batch_b}  frequency(P) × QI(x, P)
```

Similarly, the computations for different candidates `x` are fully independent of one
another — there is no cross-candidate communication inside the kernel.

This means **both the split dimension and the partition dimension are independently
batchable**.  Adaptive batching is **mathematically exact**, not an approximation.


## 3. Choosing the Batching Axis

Two options exist:

| | Batch splits | Batch partitions |
|---|---|---|
| Resident on device | orderings, invIndex, parts | orderings, invIndex, splits, scores |
| Streamed per batch | slice of splits + scores | slice of parts |
| Grows with (nk)^1.73 | splits (batchable) | — |
| Grows with n·k | — | parts (fits easily) |

**Conclusion: batch over splits, keep partitions resident.**

Rationale:

- `numSplits` is the dimension that can explode.  Batching it reduces peak VRAM to a
  fixed ceiling.
- `numParts` ≤ O(n·k) and fits entirely in VRAM alongside the orderings arrays for all
  practical inputs.
- Keeping partitions resident means they are accessed from device L2/DRAM cache during
  every batch, without re-uploading from host — this is efficient.
- Orderings and invIndex (`O(n·k)` ints each) are likewise uploaded once and stay on
  device throughout all batches.


## 4. Memory Layout After the Fix

```
─── Permanent device allocations (uploaded once) ────────────────────────────

  d_orderings   numTrees × numTaxa × 4 B       ← static, stays on device
  d_invIndex    numTrees × numTaxa × 4 B        ← static
  d_parts       numParts × 9 × 4 B              ← static

─── Batch-local device allocations (re-used each batch) ─────────────────────

  d_splits      batchSize × 10 × 4 B            ← streamed in
  d_twoScores   batchSize × 8 B                 ← streamed out

─── Host accumulation ────────────────────────────────────────────────────────

  hTwoScores    numSplits × 8 B                 ← full result, on CPU heap
```

Peak VRAM =  static footprint  +  batchSize × 48 B


## 5. Adaptive Batch-Size Calculation

After uploading static data, query free VRAM with `cudaMemGetInfo`:

```c
size_t freeVRAM, totalVRAM;
cudaMemGetInfo(&freeVRAM, &totalVRAM);

// Reserve 25% headroom for driver, kernel stack, page tables
size_t usable = (size_t)(freeVRAM * 0.75);

// 48 bytes per split: 10 ints (split data) + 1 long long (score)
size_t perSplitBytes = 10 * sizeof(int) + sizeof(long long);  // = 48

int batchSize = (int)min((long long)(usable / perSplitBytes),
                         (long long)numSplits);
batchSize = max(batchSize, 1);  // always process at least 1
```

If `cudaMalloc` for the batch buffers still fails (edge case where VRAM is fragmented),
halve `batchSize` and retry.  This makes the routine robust to fragmentation.

**Number of batches** = ceil(numSplits / batchSize).  Each batch is one kernel launch.
For small inputs (numSplits fits in VRAM), batchSize = numSplits and there is a single
launch — identical to the current behavior.


## 6. Kernel Loop Design

```
Upload orderings, invIndex, parts  (once)
Query free VRAM → compute batchSize
Allocate d_splits[batchSize], d_twoScores[batchSize]

for offset = 0 .. numSplits, step batchSize:
    curBatch = min(batchSize, numSplits - offset)

    cudaMemcpy  d_splits  ← hSplits + offset * 10,   curBatch * 10 * sizeof(int)
    launch  computeWeightsKernel<<<ceil(curBatch/256), 256>>>(
                d_splits, d_parts, d_orderings, d_invIndex,
                curBatch, numParts, numTaxa, totalN, d_twoScores)
    cudaDeviceSynchronize()
    cudaMemcpy  hTwoScores + offset  ← d_twoScores,  curBatch * sizeof(long long)

Free d_splits, d_twoScores, d_parts, d_orderings, d_invIndex
Return hTwoScores
```

No change is needed to the kernel itself — only the JNI wrapper changes.


## 7. Performance Notes

### PCIe traffic per batch

Each batch transfers `curBatch * 40 B` (splits in) + `curBatch * 8 B` (scores out) = 48 B/split over PCIe.  Total host→device traffic is `numSplits * 48 B`.  This is unavoidable because every split must reach the GPU at some point.

The partitions and orderings are transferred only once regardless of the number of
batches, which is the key win of keeping them resident.

### Kernel efficiency

Within each batch, each GPU thread processes one split and iterates over all `numParts`
partitions.  This inner loop is the arithmetic-heavy part and benefits from:
- Coalesced reads of `d_parts` (sequential access pattern in the inner loop).
- L2 cache reuse: with large batches, each warp hits the same `d_parts` rows repeatedly.

The number of batches does not change algorithmic complexity — it only introduces PCIe
round-trip latency per batch.  For a 16 GB/s PCIe link and 48 B/split, transferring 1M
splits costs ~3 ms per batch.  With a typical GPU kernel runtime of tens to hundreds of
milliseconds per batch, PCIe is not the bottleneck.

### Async overlap (future)

The batch loop can be pipelined with CUDA streams:
- Stream A: cudaMemcpy (H→D) for batch i+1 while stream B runs the kernel for batch i.
- Stream B: cudaMemcpy (D→H) for batch i while stream A uploads batch i+1.

This reduces total time by overlapping transfers and compute.  Deferred for now.


## 8. Two-Level Batching (Future: Wavelet Round Design)

When wavelet-matrix intersections are added, a second batching level becomes natural:

```
for each reference gene tree g_i:         ← Level 1: tree round
    build wavelet structures for g_i
    upload wavelet data to device

    for each split batch b:               ← Level 2: split micro-batch
        compute partial weights from (g_i, batch_b)
        accumulate into hTwoScores

    free wavelet structures for g_i
```

Level 1 bounds peak wavelet VRAM to O(n log n) per round (one tree at a time).
Level 2 bounds split buffer VRAM to `batchSize * 48 B`.

This two-level design keeps peak VRAM at roughly:
```
O(n·k) [orderings + invIndex] + O(n log n) [wavelet round] + O(batchSize * 48 B)
```

which is independent of the (nk)^1.73 search-space size.


## 9. When Does Batching Activate?

| Input scale | numSplits fits in VRAM? | Batches |
|---|---|---|
| Small (n ≤ 50, k ≤ 500) | Yes (typically < 1 GB) | 1 (no overhead) |
| Medium (n ≤ 100, k ≤ 1000) | Maybe (1–8 GB) | 1–few |
| Large (n = 200, k = 1000+) | Unlikely for worst-case | Many |

For small inputs the adaptive path degenerates to a single batch, so there is no
performance regression compared to the current implementation.


## 10. Implementation Status

| Task | Status |
|---|---|
| Identified the problem (VRAM blow-up for large Mode 2) | Done |
| Designed split-batching over JNI wrapper | Designed |
| Static-data upload once, batch-local buffers | Designed |
| `cudaMemGetInfo`-based adaptive batch-size | Designed |
| Fallback halving on cudaMalloc failure | Designed |
| Code implementation in `astralx_weight.cu` | **TODO** |
| Async pipeline with CUDA streams | Future |
| Level 1 tree-round batching for wavelet design | Future |


## 11. Summary

The key insight is that the (nk)^1.73 bound governs **total work**, not **peak memory**.
Peak memory is a function of how much of that work we materialize simultaneously on the
GPU.  By uploading static data once and streaming only the growing split dimension in
adaptive batches, peak VRAM stays bounded at:

```
O(n·k)   [static: orderings, invIndex, partitions]
+  batchSize × 48 B   [dynamic: current split batch]
```

where `batchSize` is chosen at runtime to fill available VRAM.  The result is correct,
adaptive, and robust — the computation is exactly equivalent to the monolithic case,
just spread across multiple kernel launches.
