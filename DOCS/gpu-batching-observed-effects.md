# GPU Split Batching — Observed Runtime Effects

Empirical observations from running ASTRAL-X on n=25,000 taxa, k=1,000 gene trees
(RTX 4090, 24 GB VRAM; 16 CPU cores; `--search-mode local`).

| Mode | Time | Peak CPU RAM | Peak GPU VRAM |
|---|---|---|---|
| `--no-gpu-batch` (single launch) | 366 s | 23,017 MB | 961 MB |
| `--gpu-vram-control-factor 0.1` (35 batches) | 1,161 s | 15,472 MB | 859 MB |

Two effects are worth understanding carefully: the unexpected CPU RAM reduction,
and the time increase.


---

## Effect 1 — CPU RAM Drops with GPU Batching

### Observation

Enabling GPU split-batching reduced peak CPU RAM by ~7.5 GB (23 GB → 15.5 GB),
even though batching is a GPU-side change and touches no Java data structures.

### Explanation: JVM GC is blocked during long JNI calls

Java's garbage collector requires the heap to be in a consistent state before it
can run a collection cycle.  When a JNI (native) call is in progress, the JVM
cannot safely compact or collect heap objects that may be referenced from native
code (the native call holds a pointer into the Java heap, or the GC would have to
track every native pointer).

In the **no-batch** case, the single `computeWeightsGPU(...)` JNI call runs for
roughly the entire Phase 6 duration — several hundred seconds.  During this entire
window, the JVM GC is either blocked or severely restricted.  All heap objects
allocated in earlier phases remain live and uncollected:

- **DPTable** (Phase 5): ~830K cluster entries, each with an `ArrayList<BipartitionSplit>`;
  ~2.2M `BipartitionSplit` objects total.  No longer needed once `splitList` is built,
  but GC cannot reclaim it.
- **ClusterTable** (Phase 3): ~880K cluster entries with internal hash/size data.
- **PartitionTable** (Phase 4): ~876K entries with `Partition` exemplars.
- **Tree objects** (Phase 1): 1,000 trees, each holding `postorderArray[25000]` +
  `positionMap[25000]` = 200 KB/tree = 200 MB total.
- **`splitsData` / `orderings` / `invIndex`** host arrays: ~320 MB combined (allocated
  before the call; must remain pinned).

The JVM heap expands to hold all of this simultaneously, and peak RSS is recorded
at this maximum.

In the **batched** case, each JNI call returns after processing `batchSize` splits
(a few seconds).  Between batches, the JVM GC runs freely and reclaims stale
objects from earlier phases.  By the time the 3rd or 4th batch begins, the DPTable,
ClusterTable, and PartitionTable have been collected and their physical pages
returned.  Peak RSS is measured after this partial cleanup, hence the lower reading.

### Quantification

For n=25,000, k=1,000:

| Data structure | Estimated heap | Collected with batching? |
|---|---|---|
| DPTable + 2.2M BipartitionSplit objects | ~1–2 GB | Yes (between batches) |
| ClusterTable (880K clusters) | ~2–4 GB | Yes |
| PartitionTable (876K entries) | ~0.5–1 GB | Yes |
| Tree objects (1K trees × 200 KB) | ~200 MB | No (needed by kernel) |
| JVM heap fragmentation / GC regions | ~1–2 GB | Partially |
| **Total reclaimed** | **~4–9 GB** | matches observed ~7.5 GB |

### Key takeaway

The CPU RAM reduction from batching is a **side effect of giving the JVM GC
windows to run**, not an intentional design goal.  It is nonetheless a real and
significant benefit for large inputs where host RAM is also constrained.


---

## Effect 2 — Time Increases with Many Small Batches

### Observation

With 35 batches (F=0.1), total runtime was 3.2× higher than the single-launch case
(1,161 s vs 366 s), despite identical total arithmetic work.

### Why PCIe transfer is NOT the bottleneck

Each batch transfers `batchSize × 40 B` (splits in) + `batchSize × 8 B` (scores out)
over PCIe.  The total host↔device traffic is `numSplits × 48 B` regardless of the
number of batches — the same data is transferred either way, just in chunks.

For n=25,000, k=1,000 with 2,286,880 splits:
```
total PCIe traffic = 2,286,880 × 48 B ≈ 110 MB
at 16 GB/s PCIe bandwidth → ~7 ms total transfer time
```

35 kernel-launch overheads (each ~5–10 µs) add another ~0.35 ms.  These are
completely negligible compared to the ~800 s of extra runtime.

### Root cause: GPU occupancy collapse

The weight kernel assigns **one GPU thread per split**.  Each thread iterates
over all `numParts` partitions in a loop — the kernel is **memory-latency bound**,
repeatedly accessing large arrays (`parts` = 31.6 MB, `orderings` + `invIndex` =
200 MB) with irregular access patterns that cause frequent L2/DRAM cache misses.

The GPU hides memory latency through **warp-level pipelining**: while one warp
is stalled waiting for a memory fetch, the SM switches to another ready warp.
This requires many active warps per SM simultaneously.

Number of GPU thread blocks = `ceil(batchSize / 256)`.

| Mode | Splits | Blocks | Warps/SM (RTX 4090 = 128 SMs) | Latency hiding |
|---|---|---|---|---|
| no-batch | 2,286,880 | 8,933 | ~70 | excellent — ~69 warps cover each stall |
| F=0.1, 35 batches | 65,749 | 257 | ~2 | near zero — both warps stall together |

With ~2 warps/SM, every cache miss stalls the entire SM with nothing else to run.
With ~70 warps/SM, 69 others keep the SM occupied during each stall.  This is the
entire source of the 3.2× slowdown — the GPU is performing the same work, just
much less efficiently due to under-utilization.

### Multi-threading for data transfer would not help

The bottleneck is GPU SM occupancy, not CPU-side data preparation or PCIe bandwidth.
Adding CPU worker threads to prepare or transfer batch data would not change the
GPU execution time.

### The occupancy threshold

For RTX 4090 (128 SMs), good occupancy requires at least ~16,000–32,000 thread
blocks to keep all SMs saturated across multiple waves.  At blockSize=256:
```
minimum splits for good occupancy ≈ 16,000 × 256 = 4,096,000
```

For datasets where `numSplits < ~4M`, a single launch already under-fills the GPU.
For larger datasets, batching down to ~4M splits/batch preserves good occupancy.


---

## Design Implication: When to Use Batching

| Situation | Recommended setting |
|---|---|
| VRAM is not constrained (e.g. 24 GB GPU, typical dataset) | Default `F=1.0` — single or few batches, full occupancy |
| VRAM is tight (e.g. 6 GB GPU, large dataset) | Lower F (e.g. 0.3–0.5) — trade some speed for VRAM |
| Absolute VRAM minimum needed | `--gpu-batches N` with large N — explicit control |
| Correctness testing / VRAM stress test | `--gpu-batches 100` — many tiny batches, slow but exact |

The default `--gpu-vram-control-factor 1.0` (resident-relative) is designed to
produce a batch large enough to maintain good occupancy on most hardware, while
still being principled about not exceeding available VRAM.


---

## Future: Async Pipelining with CUDA Streams

The current batch loop is fully synchronous:

```
for each batch:
    cudaMemcpy  H→D  (blocking)
    kernel launch
    cudaDeviceSynchronize  (blocking)
    cudaMemcpy  D→H  (blocking)
```

With CUDA streams, transfers and compute can overlap:

```
Stream A: upload batch i+1 splits (H→D)
Stream B: run kernel for batch i
Stream C: download batch i-1 scores (D→H)
```

This would hide most of the per-batch transfer overhead (already small) and
slightly reduce synchronization stalls.  The occupancy issue cannot be solved
by pipelining — only by keeping batch size large enough.
