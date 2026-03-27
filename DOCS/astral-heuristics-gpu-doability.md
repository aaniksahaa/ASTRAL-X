# ASTRAL Heuristic Expansion: GPU Doability and Time Implications

## Context

This note summarizes the legacy ASTRAL heuristic bipartition-expansion pipeline in
[`astral-mp-legacy-codebase`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase),
with emphasis on:

- what the code is doing,
- which parts are realistically GPU-doable,
- which parts are awkward to GPU,
- and whether the GPU-doable part is asymptotically the dominant one.

The short answer is:

- Yes, a meaningful portion of the heuristic pipeline is GPU-doable.
- The most GPU-friendly part is also the asymptotically heavier part.
- So accelerating only that part is still likely worthwhile.
- But once accelerated, the irregular CPU-only stages may become the next visible bottleneck.


## High-Level Pipeline

The relevant logic lives mainly in:

- [`astral-mp-legacy-codebase/WQDataCollection.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java)
- [`astral-mp-legacy-codebase/SimilarityMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/SimilarityMatrix.java)
- [`astral-mp-legacy-codebase/DistanceMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/DistanceMatrix.java)
- [`astral-mp-legacy-codebase/AbstractMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/AbstractMatrix.java)

At a high level, the heuristic expansion stage does four things:

1. Build a gene/species similarity or distance matrix from the input gene trees.
2. Use that matrix to complete incomplete gene trees by inserting missing taxa heuristically.
3. Use the species-level matrix to generate extra bipartitions for the search space.
4. Resolve difficult polytomies using repeated local sampling, greedy consensus, and matrix-based resolution.


## What Exactly Happens in the Code

### 1. Matrix construction

In [`WQDataCollection.calculateDistances()`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java), ASTRAL chooses one of:

- [`SimilarityMatrix.populate(...)`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/SimilarityMatrix.java)
- [`DistanceMatrix.matricesByBranchDistance(...)`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/DistanceMatrix.java)

This stage traverses gene trees and accumulates pairwise taxon statistics into a dense matrix.

This is the most arithmetic-heavy part of the heuristic pipeline.


### 2. Completing incomplete gene trees

In [`WQDataCollection.getCompleteTree(...)`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java), for each missing taxon:

1. Find a closest present taxon using the precomputed matrix through
   [`AbstractMatrix.getClosestPresentTaxonId(...)`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/AbstractMatrix.java).
2. Reroot the current tree at that closest taxon.
3. Walk downward using four-point comparisons via `getBetterSideByFourPoint(...)`.
4. Insert the missing taxon at the selected location.

This is tree-editing and control-heavy logic, not just matrix arithmetic.


### 3. Adding extra bipartitions by distance/similarity

In [`WQDataCollection.addExtraBipartitionByDistance()`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java), the code uses the species matrix to infer additional bipartitions.

Depending on matrix type:

- similarity path: [`SimilarityMatrix.inferTreeBitsets()`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/SimilarityMatrix.java) which uses UPGMA-like clustering
- distance path: [`DistanceMatrix.inferTreeBitsets()`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/DistanceMatrix.java) which calls PhyDstar

This stage turns matrix information into clusters/bipartitions to enlarge the search space.


### 4. Polytomy heuristics

In [`WQDataCollection.sampleAndResolve(...)`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java) and related methods:

- randomly sample one taxon around polytomy branches,
- induce a smaller matrix,
- build greedy resolutions from sampled trees,
- optionally resolve using matrix-based methods again,
- add the resulting bipartitions back into the search space.

This part is repeated and somewhat irregular.


## GPU Doability by Step

### Strongly GPU-doable

#### Matrix construction

Files:

- [`astral-mp-legacy-codebase/SimilarityMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/SimilarityMatrix.java)
- [`astral-mp-legacy-codebase/DistanceMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/DistanceMatrix.java)

Why:

- large bulk accumulation,
- many repeated independent updates,
- dense numeric state,
- natural data-parallel structure over gene trees, clusters, and taxon pairs.

This is the best candidate for GPU acceleration.


#### Batched four-point evaluations

The primitive used in completion,
`getBetterSideByFourPoint(...)`, is numerically simple and could be evaluated in batches on GPU if the surrounding logic were reformulated into arrays.

By itself, this is GPU-doable.

However, the surrounding control flow is not.


#### Induced submatrix extraction

For polytomy heuristics, building many induced submatrices from a larger matrix is also GPU-friendly in principle.

This is mostly gather/copy work and can be batched.


### GPU-doable but awkward

#### UPGMA-like clustering

File:

- [`astral-mp-legacy-codebase/SimilarityMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/SimilarityMatrix.java)

Why awkward:

- iterative best-pair selection,
- repeated global updates,
- dynamic active set,
- control-heavy rather than throughput-heavy.

Possible to accelerate partially, but not a clean GPU fit.


#### PhyDstar-style distance resolution

File:

- [`astral-mp-legacy-codebase/DistanceMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/DistanceMatrix.java)

Why awkward:

- agglomerative / tree-building flavor,
- repeated matrix reduction,
- branching control flow.

Again, possible in theory, but not the cleanest use of GPU.


### Poor GPU fit

#### Gene tree completion and insertion logic

File:

- [`astral-mp-legacy-codebase/WQDataCollection.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java)

The routine:

- reroots trees,
- navigates mutable tree nodes,
- makes branch-dependent decisions,
- inserts leaves dynamically.

This is fundamentally irregular and pointer-structure heavy.

That makes it a poor direct GPU target.


#### Greedy consensus tree updates around polytomies

Also in [`WQDataCollection.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java), the code:

- counts sampled bipartitions,
- sorts them,
- builds a greedy tree incrementally,
- mutates node structures.

This is also CPU-oriented logic.


## Time Implications

## Main question

Is the GPU-doable part asymptotically much more demanding than the non-GPU-doable part?

The judgment here is: yes, generally.


### The likely dominant asymptotic part

The matrix construction stage is the bulk numeric workload.

It repeatedly aggregates information across:

- many gene trees,
- many taxa,
- and many taxon-pair or cluster-pair interactions.

This has the flavor of `k * n^2` or worse depending on tree shape and the exact update pattern.

That is the stage most likely to grow fastest with dataset size.


### The less GPU-friendly parts

The completion and clustering stages are still important, but they are usually:

- more local,
- more branchy,
- more sequential,
- and less dominated by massive pairwise arithmetic.

Examples:

- completing one missing taxon is closer to a guided walk down a tree than a full dense matrix computation,
- UPGMA / PhyDstar style clustering is iterative and global, but its work is not the same kind of huge embarrassingly parallel bulk accumulation,
- polytomy heuristics are repeated but operate on local samples and smaller induced problems.


### Practical consequence

This means:

- accelerating the matrix-heavy part is still meaningful,
- the CPU-only remainder probably does not cancel out that gain at large scale,
- but after GPU acceleration, the remaining heuristic control flow may become more noticeable.

In other words, this is an Amdahl's-law situation:

- before acceleration, matrix building likely dominates,
- after acceleration, completion / clustering / polytomy handling may become the next bottleneck,
- but that does not mean the GPUization was pointless.


## Recommended Engineering View

### Worth GPU-accelerating

- matrix construction
- batched induced-submatrix extraction
- possibly batched four-point score evaluation


### Likely better left on CPU

- mutable tree rerooting and insertion
- greedy consensus tree mutation
- UPGMA / PhyDstar control flow unless profiling proves it dominates


### Best architecture

The best design is likely hybrid:

1. GPU for large regular numeric kernels,
2. CPU for irregular tree/data-structure logic,
3. careful batching so CPU-GPU transfers do not dominate.


## Final Judgment

If ASTRAL-X already GPU-parallelizes the main quartet-weight / DP-heavy path, then the remaining heuristic bipartition expansion is not an all-or-nothing GPU target.

The correct conclusion is:

- the heaviest subpart of that heuristic stage is GPU-doable,
- the most awkward leftover parts are probably not the asymptotic driver,
- so GPU acceleration of the matrix-heavy subpart would still likely help,
- but it will not make the whole heuristic stage vanish,
- and after that, the serial/irregular pieces will become comparatively more visible.


## Short Summary Table

| Step | Main file(s) | GPU suitability | Likely time role |
|---|---|---:|---|
| Build similarity/distance matrix | [`SimilarityMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/SimilarityMatrix.java), [`DistanceMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/DistanceMatrix.java) | High | Likely dominant asymptotic bulk work |
| Complete missing taxa into gene trees | [`WQDataCollection.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java) | Low | Irregular, important, but not likely the main growth term |
| Add extra bipartitions from matrix | [`WQDataCollection.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java) | Medium | Moderate |
| UPGMA / PhyDstar tree inference from matrix | [`SimilarityMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/SimilarityMatrix.java), [`DistanceMatrix.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/DistanceMatrix.java) | Medium-low | Can matter, but control-heavy |
| Polytomy sampling and greedy resolution | [`WQDataCollection.java`](/home/aaniksahaa/research/ASTRAL-X/astral-mp-legacy-codebase/WQDataCollection.java) | Low-medium | Repeated and messy, but typically not the main arithmetic wall |
