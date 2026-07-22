# ASTRAL-X multi-GPU execution design

Status: design proposal only. No runtime code is changed by this document.

## 1. Goal and non-negotiable requirements

When CUDA exposes more than one usable GPU, ASTRAL-X should use all selected
devices for phases with enough independent work. A user must also be able to
restrict execution to a subset or to one GPU. The implementation must preserve:

- exactly the same search space, transition set, raw split weights, optimum
  quartet score, and inferred tree as the corresponding single-device run;
- deterministic output regardless of device speed or task completion order;
- the current CPU fallback on systems without CUDA;
- bounded memory, with both per-device and aggregate VRAM reported honestly;
- no silently dropped task, transition, matrix tile, or weight after a CUDA
  error or output-buffer overflow;
- good single-GPU performance and no multi-GPU overhead on small jobs.

The first implementation should require neither peer-to-peer GPU access nor
NCCL. Each selected device may be a different model and may sit behind a
different PCIe root. `CUDA_VISIBLE_DEVICES` remains the outer visibility and
isolation mechanism.

## 2. GPU work that exists today

There are four independent GPU call sites, not one global GPU pipeline.

| Phase | Java entry | Native implementation | Natural independent work |
|---|---|---|---|
| Similarity matrix in Phase 1b | `GPUSimilarityMatrix.computeSimilarityGPU` | `src/native/astralx_similarity.cu` | upper-triangle taxon-pair tiles; optionally tree ranges |
| Distance matrix in Phase 1b | `GPUDistanceMatrix.computeDistancesGPU` | `src/native/astralx_dist.cu` | upper-triangle taxon-pair tiles; optionally tree ranges |
| Full-mode cross-tree construction in Phase 5b | `GPUDPBuilder.findCrossTreeTransitionsGPU` | `src/native/astralx_dp.cu` | `(size bin, A range, B range)` rectangles |
| Weight precomputation in Phase 6 | four methods in `GPUWeightCalculator` and `astralx_weight.cu` | prefix-sum, smaller-side, bitset, simple-tree-walk kernels | disjoint ranges of candidate splits |

All other major phases are currently CPU phases. Multi-GPU support cannot
accelerate parsing, consensus/search-space construction, local-transition
construction, final inference, or tree reconstruction. Whole-program scaling
will therefore be bounded by Amdahl's law even if Phase 6 scales almost linearly.

Today the CUDA runtime is never given a device with `cudaSetDevice`, so every
native call implicitly runs on logical CUDA device 0. `queryGPUStatus` reports
the count but describes only the current device, `queryVRAMMiB` samples only
that device, and the external monitor records the maximum used memory among
all visible GPUs rather than the sum or the selected set.

## 3. The central architecture

### 3.1 One inventory and one selection policy

Add a `CUDADeviceManager` layer that enumerates every visible logical device
and records at least:

- CUDA logical index, name, UUID or PCI bus identifier;
- compute capability and whether it satisfies the packaged artifact minimum;
- total and currently free VRAM;
- SM count, maximum clock, shared-memory limits, and launch limits needed by
  the four weight kernels;
- a conservative throughput weight, initially derived from hardware
  properties and later replaceable by a short calibration.

CUDA logical indices are the indices after `CUDA_VISIBLE_DEVICES` is applied.
ASTRAL-X must never silently reinterpret them as physical `nvidia-smi` indices.

Proposed command-line interface:

```text
--gpu                         use CUDA when at least one selected device is usable
--gpu-devices auto            use all usable visible devices (default under --gpu/auto)
--gpu-devices 0,2,3           use these CUDA logical devices
--max-gpus N                  cap an automatic selection to N devices
--gpu-strict                  never turn a failed GPU phase into a CPU phase
```

`--gpu-devices 0` is the explicit single-GPU control; a separate
`--multi-gpu` flag is unnecessary. In automatic selection, unusable visible
devices are reported and skipped. With an explicit list, an invalid or
incompatible requested device is an error under `--gpu-strict` and a warning
otherwise. Duplicate indices are rejected or de-duplicated with an explicit
warning. Device selection is resolved before input parsing, as compute-mode
selection is today.

For small phases, the scheduler may activate fewer GPUs than were selected.
The log must distinguish "selected for the run" from "active in this phase"
and state why a selected GPU was idle (for example, one matrix tile or launch
overhead larger than estimated useful work).

### 3.2 A native coordinator, not N full Java JNI calls

The tempting implementation is to start N Java threads and call the existing
JNI method once per device. That is unsafe for memory: every call may cause
`Get*ArrayElements` to pin or copy the same very large Java arrays, multiplying
host RAM and copy traffic by N. It also makes partial failure and progress
reporting difficult.

Each phase should instead make one JNI call. That call obtains each Java array
once, then starts one native host worker per active device. Each worker must:

1. call `cudaSetDevice(logicalDeviceId)` before any other CUDA operation;
2. create and own its streams, events, device pointers, pinned staging buffers,
   and CUDA error state;
3. upload the required read-only resident data to its device;
4. consume work assigned by the phase coordinator;
5. write only disjoint host output slots, or return a private task result for a
   deterministic merge;
6. release all per-device resources through RAII on success or failure.

Worker threads must not use the caller's `JNIEnv`. They may read ordinary host
pointers obtained by the still-active JNI call; the coordinator joins all
workers before releasing those pointers or returning to Java.

Shared scheduling/error/progress utilities should live in a small native
common component used by all CUDA libraries rather than being copied four
times. A later session-oriented JNI API is possible, but it is not needed for
the first implementation because one coordinated call already uploads each
phase's resident data only once per GPU.

### 3.3 Transactional task completion

A task has three states: unclaimed, running, and committed. A worker commits a
task only after the kernel has completed successfully and its device-to-host
copy has succeeded. If a device fails, its running task is returned to the
queue and the device is quarantined. Remaining devices may finish it.

No Java-visible result is considered valid until the whole native phase has
succeeded. If no GPU survives:

- normal GPU mode discards or clears partial output and recomputes the entire
  phase on CPU;
- `--gpu-strict` aborts with the device id, phase, task range, CUDA operation,
  CUDA error name/code, and memory inventory.

This is especially important for the matrix JNI functions, which currently
return `void` and can leave partially modified output arrays after an error.
Before multi-GPU work, they must return an explicit success/status object (or
throw a specific exception), and the fallback path must reset all output
arrays before CPU recomputation. A CUDA error must never merely be printed
while stale or zero values are copied into the result.

## 4. Phase 6: all four weight methods

This is the highest-value and lowest-risk first target. Every candidate split
is scored independently, and every current native implementation already
iterates through split batches.

### 4.1 Partition rule

Use a global queue of disjoint split ranges. Each device uploads a read-only
copy of the resident representation required by the selected method, then
claims a range that fits its own scratch capacity:

```text
global split indices: [0 ........................................ numSplits)
GPU 0 claims:         [0, a)
GPU 1 claims:                 [a, b)
GPU 2 claims:                         [b, c)
... faster workers claim additional unprocessed ranges ...
```

Results are written by global split index. There is no cross-GPU arithmetic
reduction. Therefore LONG and INT128 results are exactly identical, and DOUBLE
uses exactly the same operation order inside each split as the single-GPU
kernel. Device completion order cannot affect a score.

A dynamic queue is preferable to equal static slices because heterogeneous
GPUs can differ greatly in speed. Chunks should be large enough for efficient
launches and preferably target tens to a few hundreds of milliseconds, not
one split at a time. The initial version may use each device's current adaptive
batch capacity as its chunk size; later runs can adjust from observed timing.

### 4.2 Method-specific constraints

- **Prefix-sum:** determine shared-memory feasibility separately on every
  device. Allocate the global prefix pool only on devices that require that
  path. A device that cannot launch this kernel is excluded from this method;
  it must not force all other devices to CPU.
- **Smaller-side:** resident parts/orderings/inverse indices are replicated;
  only split inputs and score outputs are sharded.
- **Bitset:** resident cluster and part bitsets are replicated; candidate split
  bitsets and outputs are sharded.
- **Simple-tree-walk:** the measured frontier is global, but launch feasibility
  and scratch capacity are checked per device. A device that cannot execute the
  selected 32/64/128/256/512 frontier specialization is excluded while capable
  devices continue. Token streams and gene bitsets remain read-only replicas.

This deliberately uses the same split-sharding principle for all methods. A
method-specific gene-tree partition could reduce aggregate resident VRAM, but
it would require partial-score reductions and significantly different logic:
deduplicated part frequencies, exemplar trees, polytomies, and autocomplete's
separate cluster-tree/part-tree layouts make that easy to get wrong.

### 4.3 Existing batching controls

The multi-GPU meaning should be explicit:

- automatic batching computes a separate safe maximum chunk for every active
  device from its free VRAM;
- `--gpu-batch-size B` is a per-device maximum task size;
- `--gpu-batches K` denotes K global logical split tasks. With fewer tasks than
  devices, only K devices can work, and the program warns about that;
- `--no-gpu-batch` means one contiguous shard per active device, not one copy of
  all splits on every device;
- the occupancy fraction remains per-device because devices have different
  capacities and background usage.

All meanings must be added to `--help`. Silent reinterpretation is not
acceptable.

## 5. Phase 5b: full-mode cross-tree transitions

The resident cluster hash data and lookup table must initially be replicated
on every active GPU; any worker may need to resolve any residual cluster.

### 5.1 Safe two-dimensional tasks

The natural work item is a rectangle

```text
(size sz, A indices [a0,a1), B indices [b0,b1))
```

within the current size-binned subtraction search. Rectangles are disjoint,
and one `(A,B)` pair can emit at most one triple. Choose both dimensions so

```text
(a1 - a0) * (b1 - b0) <= deviceOutputCapacity
```

This is stronger than the current A-only subdivision. Today, if one B bin is
larger than the configured output capacity, even `aCount = 1` can launch more
pairs than the output buffer can hold. The existing code then clamps the count
and explicitly drops transitions. That behavior must be removed before, not
carried into, multi-GPU execution. Two-dimensional rectangles give a proof
that overflow is impossible. Any violated bound is a hard internal error, not
a warning followed by an incomplete search space.

Workers claim rectangles dynamically. Each result is kept private until its
kernel and copy complete. After every rectangle succeeds, concatenate and sort
triples by `(idxA, idxB, idxResidual)` before constructing Java
`LinkedHashSet`s. Sorting removes device-timing-dependent insertion order and
protects deterministic downstream iteration and tie behavior.

Root transitions remain on CPU exactly as today.

### 5.2 Expected scaling and memory

This phase can scale well when there are many large bin rectangles, but every
device initially needs the complete cluster hashes and hash table. Static
aggregate VRAM therefore grows approximately with the number of devices even
though each device's peak remains bounded. Distributed lookup tables would
avoid replication but would put a remote lookup or communication step in the
inner search and are not justified for the first implementation.

## 6. Phase 1b matrices

The matrix kernels have two independent dimensions: tree batches and
upper-triangle pair tiles. Their scheduler should use two modes rather than a
single policy for every `(n,k)` shape.

### 6.1 Default: pair-tile ownership

For medium and large taxon counts, assign every upper-triangle B-by-B tile to
exactly one GPU. Each worker streams all tree batches and computes only its
owned tiles. It writes disjoint cells of the host matrix, so there is no
cross-device reduction and no race.

Tile assignment should be static but weighted by estimated device throughput
and actual tile cell count (edge and diagonal tiles contain less work). Static
ownership lets a worker upload each tree batch once and process all of its
tiles, matching the efficient loop order used now. A naive dynamic tile queue
that uploads every tree batch again for every claimed tile would destroy
performance.

All devices use common tile geometry so there can be no gap or overlap. A
common tree-batch size is the simplest deterministic first implementation; it
must fit the least-capable active device. Later, different batch sizes are safe
only after the exact accumulator representation and ordering rules below are
satisfied.

The tradeoff is that every GPU reads the full tree stream. On independent PCIe
links those transfers overlap; on a shared root or host-memory-limited system,
upload bandwidth may cap scaling.

### 6.2 Small n / huge k: optional tree-range ownership

For the 363-taxon, roughly 63,000-tree shape, automatic B can equal n, leaving
only one pair tile. Tile ownership can then use only one GPU even though the
tree dimension is enormous.

When a full private partial matrix is small under a configured host-RAM
threshold, shard contiguous tree ranges across devices. Each worker runs the
current loop on its tree range and produces a private n-by-n partial matrix.
The coordinator reduces those matrices in ascending tree-range order. For
363 taxa, two similarity partial matrices are only about 2.1 MB per GPU, so
this mode is inexpensive. It must not be selected for 9,524 taxa, where N
private n-by-n matrices would consume many gigabytes of additional RAM.

The automatic choice is therefore:

1. use tile ownership when there are enough weighted tiles;
2. otherwise use tree ownership only if all private partial matrices fit a
   conservative aggregate host-RAM cap;
3. otherwise use only as many GPUs as there are useful tiles and report why.

Trying to force every selected GPU active when the problem exposes too few
safe tasks can make the run slower or exceed RAM; truthful idling is preferable.

### 6.3 Exactness of matrix reductions

Distance contributions are integral; similarity numerator contributions are
integral or half-integral combinatorial counts. The current storage type is
DOUBLE. Before enabling tree-range sharding, one of these must be established:

- prove and runtime-check that every intermediate remains in DOUBLE's exact
  integer range (scale similarity numerators by two), then reduce exact integer
  representations and convert once; or
- change native partial accumulators to exact 64/128-bit integer units with a
  checked overflow path.

An unconstrained floating-point reduction whose grouping depends on which GPU
finishes first is forbidden. Pair-tile ownership does not need a cross-GPU
reduction and is therefore the safer first matrix implementation. Tests should
require bitwise matrix equality where the current values are exactly
representable, not merely a loose tolerance.

## 7. Memory model and honest VRAM expectations

Multi-GPU has two memory classes:

```text
per-device peak = replicated resident data + that device's bounded scratch
aggregate peak  = sum(per-device peaks across active devices)
```

Split and tile sharding keep scratch bounded and avoid storing the entire
output on each GPU. They do not eliminate resident replication. If one GPU
currently holds R bytes of read-only data, G GPUs initially use approximately
G*R aggregate VRAM. It is impossible to promise both unchanged aggregate VRAM
and full G-device execution with this replication-based design.

The practical first target is:

- each device remains at or below the current single-device policy;
- absolute phase caps, such as similarity tree-data and DP output caps, gain a
  documented aggregate-budget mode that divides scratch among active devices;
- allocation uses each device's own live `cudaMemGetInfo`, with fixed headroom;
- OOM halves only that device's chunk and retries before quarantining it;
- all allocation plans are logged before kernels start.

An advanced second-generation mode may partition resident gene-tree data and
sum partial scores, especially for simple-tree-walk. That could lower aggregate
VRAM, but it introduces exact-reduction and method-specific correctness work
and must not block the safer split-sharded implementation.

Host RAM also needs protection. There is one JNI pin/copy of each input, not
one per worker; pinned staging buffers are bounded per device; matrix
tree-sharding is permitted only below its aggregate partial-matrix cap.

## 8. Progress, diagnostics, and monitoring

Only the coordinator prints the normal progress bar. Worker threads update
atomic completed-work counters; they do not write competing carriage-return
lines. `-vv` may show a stable per-device table with current task, completed
work, throughput, allocated VRAM, and any retry.

For every GPU phase, log:

```text
selected GPUs: 0,1,2,3
active GPUs:   0,1,2,3 (or a reason for the smaller set)
partition:     split ranges / tile owners / DP rectangles
per device:    resident, scratch cap, batch size, tasks, elapsed, throughput
```

Phase VRAM reporting must enumerate the selected logical devices and show:

- peak for each device in this step and so far;
- maximum single-device peak;
- simultaneous aggregate peak (sum of sampled used memory);
- device count and logical ids.

`cudaMemGetInfo` reports total device usage, including other processes. Record
the phase-start baseline so the log can distinguish absolute device usage from
the approximate ASTRAL-X increase. Sampling all devices sequentially is an
approximation and should be labelled as such unless a process-specific NVML
implementation is deliberately added.

The shell monitor currently keeps only the largest GPU value. It should retain
that backward-compatible column and add selected-device count, maximum-single
VRAM, and aggregate VRAM columns. It must honor the same logical device
selection; otherwise unrelated GPUs or jobs contaminate the report.

Fatal reports must include the inventory and, for each failed device, the
phase, task range, last successful CUDA call, requested/free/total bytes, and
whether work was requeued, CPU-fallbacked, or aborted under strict mode.

## 9. Small-work and heterogeneous-device policy

Creating contexts and replicating resident arrays can cost more than a kernel
on small datasets. Each phase estimates useful work after its natural tasks
are known and activates multiple devices only if the expected saving exceeds
context/upload overhead. This decision must be deterministic for a fixed
inventory and configuration.

Heterogeneous devices are supported without forcing every device to use the
weakest launch parameters:

- weights use per-device batch and launch feasibility;
- DP uses per-device output/scratch capacity and dynamic rectangles;
- matrices require common tile geometry but use weighted tile ownership;
- a device slower than its transfer overhead may be left idle after a logged
  deterministic threshold.

No P2P capability is assumed. P2P can be an optional later optimization, never
a correctness dependency.

## 10. Correctness and performance test gates

Multi-GPU should not be considered complete when it merely keeps all devices
busy. The following gates are required.

### 10.1 Exactness matrix

Compare CPU, one GPU through the new coordinator, and 2+ GPUs for:

- all four weight intersection methods;
- LONG, INT128, and DOUBLE score modes;
- local and full search, complete and incomplete gene trees;
- binary and polytomous trees;
- autocomplete on/off, consensus on/off, anchor on/off;
- ordinary inference, score-only mode, and every GPU verification dump.

Compare raw doubled weight arrays by split id, not only the final score. Compare
the complete sorted cross-tree transition triples, both Phase 1b matrices,
the final X/DP reachability counts, quartet score, and output Newick. Integer
outputs must be byte-identical. DOUBLE outputs must be bit-identical whenever
the exact-representability condition holds; any accepted non-bitwise case needs
a written numerical proof and downstream stability test.

### 10.2 Determinism and failure tests

- Repeat the same multi-GPU run at least 20 times with randomized artificial
  worker delays and compare every artifact.
- Test homogeneous and heterogeneous simulated capacities, reordered device
  lists, `CUDA_VISIBLE_DEVICES`, explicit subsets, invalid ids, and one GPU.
- Inject failures during allocation, upload, launch, synchronization, and
  download on each device. Verify task requeue and verify that CPU fallback
  starts from clean output.
- Force tiny DP output caps and B bins larger than a cap; verify two-dimensional
  tiling and zero lost transitions.
- Run compute-sanitizer memory and race checking on reduced test cases.

### 10.3 Performance and memory gates

Benchmark 1, 2, and 4 GPUs separately for each phase on small, medium, and
large shapes, including 37, 200, 363, and 9,524 taxa where data are available.
Record kernel time, transfers, setup, CPU merge, CPU RAM, per-device VRAM,
aggregate VRAM, GPU utilization, and whole-run time.

Acceptance requires:

- no material single-GPU regression;
- no multi-GPU activation for jobs where it measurably slows the phase;
- useful Phase 6 scaling when enough splits exist;
- no unexplained score, tree, transition-count, or matrix change;
- per-device memory within the planned cap and aggregate memory matching the
  logged analytical budget.

### 10.4 Realistic scaling expectations

| Phase | Likely scaling | Main limit |
|---|---|---|
| Four weight methods | best candidate; near-linear while there are many split chunks | replicated upload, memory bandwidth, a few unusually long tail chunks |
| Cross-tree DP | good for many large bin rectangles | replicated hash table, output/download and final sort |
| Matrix, many pair tiles | good if host-to-device links and host RAM bandwidth sustain replicated tree streams | every device reads the full tree stream |
| Matrix, one/few pair tiles | no tile-mode scaling; tree mode can help only when private n-by-n matrices are cheap | host RAM and exact partial reduction |
| Whole run | less than the accelerated phase speedup | all CPU-only phases and serial setup |

No fixed "N GPUs gives N times faster" claim should appear in user-facing
documentation. The program should print phase and whole-run timings so the
actual scaling is visible.

## 11. Concrete code impact

The expected implementation surface is:

| File/component | Required change |
|---|---|
| `Config.java`, `Main.java` | parse, validate, and document device selection and maximum count; preserve requested versus resolved compute mode |
| `Banner.java`, `RuntimeDiagnostics.java`, `FatalReporter.java` | print all visible/selected devices, capabilities, selection decisions, and per-device failures |
| new `astralx.gpu.CUDADeviceManager` JNI bridge | enumerate devices and query per-device memory/capabilities without making the weight library the accidental owner of all CUDA policy |
| new shared native coordinator utilities | device RAII, worker/task state, status/error objects, atomic aggregate progress, and deterministic merge helpers |
| `GPUWeightCalculator.java`, `WeightTable.java`, `astralx_weight.cu` | pass selected ids once; create one resident context per device; distribute global split ranges for all four methods |
| `GPUDPBuilder.java`, `DPTable.java`, `astralx_dp.cu` | use safe A-by-B rectangles, remove transition dropping, distribute tasks, and sort the complete result |
| `GPUSimilarityMatrix.java`, `SimilarityMatrixBuilder.java`, `astralx_similarity.cu` | return explicit status; add common tile ownership and gated exact tree-range mode |
| `GPUDistanceMatrix.java`, `DistanceMatrixBuilder.java`, `astralx_dist.cu` | return explicit status; add the same two scheduling modes with the distance-specific accumulator |
| `PhaseLogger.java` | sample selected devices and report per-device, max-single, and aggregate peaks with baselines |
| `run-astralx-with-monitor.sh` and CSV schema | honor selection and record GPU count, max-single peak, and aggregate peak while retaining the old column |
| portable build scripts | compile/link the shared coordinator into every CUDA artifact and add multi-GPU smoke tests without creating separate single/multi-GPU packages |

Native entry points should take an immutable array of logical device ids and a
small policy/config structure. They should not read `CUDA_VISIBLE_DEVICES`
themselves beyond normal CUDA enumeration, and they should not rely on a
process-global current device.

## 12. Implementation order

The feature should land in reviewable stages, with the code usable after each
stage:

1. **Correctness prerequisites:** structured native status/error propagation;
   eliminate DP transition dropping with two-dimensional rectangles; make
   matrix fallback transactional.
2. **Device layer:** full inventory, selection CLI, help/banner/diagnostics,
   per-device `cudaSetDevice`, and multi-device VRAM sampling.
3. **Weight coordinator:** shared split scheduler for all four methods. This is
   the main expected runtime win and avoids arithmetic reductions.
4. **Monitoring:** aggregate progress, per-device timing/memory, shell monitor
   and CSV extensions.
5. **DP coordinator:** replicated lookup state, dynamic rectangles,
   deterministic sorted merge.
6. **Matrix tile ownership:** common tile geometry and weighted static owners.
7. **Optional matrix tree ownership:** only after exact accumulators/bounds and
   host-RAM gating are implemented.
8. **Advanced memory work:** optional resident tree partitioning if measured
   aggregate VRAM, not compute, is the limiting resource.

The weight stage is high difficulty but structurally clean. DP and matrix
distribution are high difficulty because of output completeness and numerical
ordering. Robust recovery, deterministic merging, monitoring, and the full
test matrix make the complete feature very high difficulty; it should not be
attempted as one monolithic patch.

## 13. What to reuse from legacy ASTRAL-MP

Legacy ASTRAL-MP already exposes a comma-separated GPU selection and creates a
context, queue, resident buffers, and compute worker per selected device. Its
`TurnTaskToScores` also demonstrates the essential scheduling idea: when a
device becomes available, send it another independent tripartition batch, then
restore global result order.

Reuse the concept, not the implementation. ASTRAL-X precomputes a known split
set and already has global output indices, so it does not need ASTRAL-MP's
priority queue or mixed CPU/GPU streaming. CUDA-native coordination gives
cleaner memory ownership, avoids repeated JNI copies, supports transactional
failure, and naturally extends to the three non-weight GPU phases.

## 14. Recommended first milestone

The first production milestone should include Stages 1-4 and multi-GPU support
for all four weight methods only. It provides the largest likely runtime gain
with the strongest exactness argument: each split stays an indivisible task,
and no score is reduced across devices. Full-mode DP and Phase 1b matrices can
then be added behind the same device manager after their specific correctness
gates pass.

This staged boundary is deliberate. A run may initially show, for example,
one GPU in similarity and all GPUs in weight calculation. That is still valid
multi-GPU support and is safer than forcing premature cross-device reductions.
