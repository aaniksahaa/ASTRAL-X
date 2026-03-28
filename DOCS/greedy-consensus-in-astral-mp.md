# Greedy Consensus Trees in ASTRAL-MP: Design, Algorithm, and Complexity

Analysis of `addExtraBipartitionByHeuristics` and supporting methods in
`astral-mp-legacy-codebase/WQDataCollection.java` and `Utils.java`.

This runs only in the optional extra mode (`addExtra != 0`), after the UPGMA
tree bipartitions and gene tree bipartitions have already been added to X.
Its purpose: enrich X further by producing 7 consensus trees at varying
support thresholds and resolving their polytomies.

---

## 1. Why Greedy Consensus?

The UPGMA tree and direct gene tree bipartitions may still leave gaps in X.
In particular, under high ILS:

- No single gene tree contains every true bipartition
- The UPGMA tree contains the average signal but can be wrong in details
- A bipartition supported by 15% of gene trees is real signal but may not
  survive in any individual gene tree

The greedy consensus approach builds consensus trees at 7 different frequency
thresholds, ranging from "only very frequent bipartitions" (strict, polytomy-heavy)
to "every bipartition that appeared in any gene tree" (relaxed, fully resolved).
Polytomy nodes in these consensus trees represent unresolved regions — exactly
where additional bipartitions need to be injected into X.

---

## 2. Overview of the Full Pipeline

```
addExtraBipartitionByHeuristics(contractedTrees):

  Phase 1: Count bipartitions across all k gene trees
           → HashMap<cluster, count>

  Phase 2: Sort by frequency
           → sorted list, most frequent first

  Phase 3: Build 7 greedy consensus trees, one per threshold
           → allGreedies[7]

  Phase 4: For each of 7 trees, for each polytomy node:
     Step A: Resolve using UPGMA sub-matrix → bipartitions → X
     Step B: sampleAndResolve rounds (10–100 adaptive) → bipartitions → X
```

Constants hardcoded in `WQDataCollection.java` lines 61–70:

```java
double[] GREEDY_ADDITION_THRESHOLDS    = {0, 1/100., 1/50., 1/20., 1/10., 1/5., 1/3.}
int GREEDY_ADDITION_DEFAULT_RUNS       = 10     // base rounds per polytomy
int GREEDY_ADDITION_MAX                = 100    // max adaptive rounds
int GREEDY_ADDITION_IMPROVEMENT_REWARD = 2      // extra rounds when new bips found
int GREEDY_ADDITION_MIN_FREQ           = 5      // min new clusters to count as "improvement"
int GREEDY_ADDITION_MAX_POLYTOMY_MIN   = 50
int GREEDY_ADDITION_MAX_POLYTOMY_MULT  = 25     // polytomy size limit = 50 + n*25
```

---

## 3. Phase 1 — Count Bipartitions Across All Gene Trees

**`Utils.greedyConsensus` lines 239–254. Inner call: `Utils.getGeneClusters`.**

For each of the k gene trees:

```
getGeneClusters(tree):
  post-traverse the tree
  at each internal node v:
    compute BitSet = union of children's BitSets  (via bs.or)
    if 1 < |BitSet| < n:
        add STITreeCluster(BitSet) to list
  return list of all internal-node clusters
```

Each cluster is one side of an unrooted bipartition. The other side is the
complement. Deduplication is handled in the count HashMap:

```java
STITreeCluster comp = cluster.complementaryCluster();
if (count.containsKey(comp)) {
    count.put(comp, count.get(comp) + 1);  // same bipartition seen from other side
    continue;
}
count.put(cluster, 1);
```

This ensures each bipartition is counted only once regardless of which side was
seen in which tree. Result: `HashMap<STITreeCluster, Integer>` mapping each
unique bipartition to how many gene trees support it.

**`getGeneClusters` per tree**:
- Post-traverse: O(n) nodes
- At each node: `BitSet.or` of child BitSets — O(n/64) word operations
- HashMap lookup: `hashCode()` = O(n/64), `equals()` = O(n/64)
- Per tree total: O(n × n/64) = **O(n²/64)**

**Phase 1 total across k trees: O(k × n²/64)**

Note: the `/64` factor from BitSet word operations is real and large in practice.
For n=25,000 that is 390 words per BitSet operation. Phase 1 is equivalent to
~k×n²/64 integer operations — fast with good cache locality since it's sequential.

**Memory — the HashMap**:

Let B = number of unique bipartitions across all k trees. Each entry is a
BitSet key (n/8 bytes) + boxed Integer value (~20 bytes JVM overhead):

```
HashMap memory ≈ B × (n/8 + 20) bytes ≈ B × n/8 bytes (for large n)
```

B is bounded by k×(n−2) but in practice much smaller since trees share bipartitions:

| ILS level | B (typical) | n=25,000 HashMap |
|---|---|---|
| Low ILS, similar trees | ~1–3 × n | 75–225 MB |
| Moderate ILS | ~5–15 × n | 375 MB – 1.1 GB |
| Extreme ILS, all unique | k × n = e.g. 25M | ~78 GB (pathological) |

The pathological case never occurs on real data. For typical phylogenomics
B = O(5–15 × n), so the HashMap is the dominant memory cost at ~500 MB–1 GB
for n=25,000.

---

## 4. Phase 2 — Sort Bipartitions by Frequency

```java
TreeSet<Entry<STITreeCluster,Integer>> countSorted =
    new TreeSet<>(new ClusterComparator(randomize, n));
countSorted.addAll(count.entrySet());
```

All B entries are inserted into a `TreeSet` sorted by count descending (with
BitSet comparison as tiebreaker via `ClusterComparator`).

- B insertions, each O(log B) comparisons
- Each comparison: integer compare O(1) first; if tied, BitSet compare O(n/64)
- **Time: O(B × log(B))** in the common case (most bipartitions have distinct counts)
- **Worst case: O(B × n/64 × log(B))** if all counts are equal
- **Memory: O(B × 48 bytes)** — standard Java TreeSet node overhead (same issue
  as UPGMA's TreeSet, but B entries not n² entries, so much more manageable)

For B = 250,000: TreeSet memory ≈ 250K × 48 = 12 MB. Negligible.

---

## 5. Phase 3 — Build 7 Greedy Consensus Trees

### 5.1 The threshold mechanism

The sorted bipartitions are walked from highest to lowest frequency. A snapshot
of the collected clusters is taken each time the running frequency drops below
a threshold. Each snapshot is submitted as a parallel `greedyConsensusLoop` job.

```
thresholds (high to low): 1/3 → 1/5 → 1/10 → 1/20 → 1/50 → 1/100 → 0

Walk sorted clusters (most frequent first):
  freq=60%                  clusters = [c1]
  freq=55%                  clusters = [c1, c2]
  freq=30%  (drops < 1/3)   → snapshot → build Tree1 from [c1,c2]
  freq=28%                  clusters grow
  freq=18%  (drops < 1/5)   → snapshot → build Tree2
  ...
  freq=0.3% (drops < 1/100) → snapshot → build Tree6
  end of list               → build Tree7 (all clusters)
```

Result: **7 trees**, each built from a different prefix of the sorted list:

| Tree | Threshold | Clusters included | Character |
|---|---|---|---|
| T1 | ≥ 33% | Very frequent only | Highly polytomous, very conservative |
| T2 | ≥ 20% | More | Partially resolved |
| T3 | ≥ 10% | More | Moderate resolution |
| T4 | ≥ 5% | More | Getting binary |
| T5 | ≥ 2% | More | Nearly binary |
| T6 | ≥ 1% | More | Very resolved |
| T7 | ≥ 0 | Every bipartition | Maximally resolved, may include noise |

T1 is essentially the strict majority-rule consensus (binary threshold at 33%).
T7 is the extended majority-rule consensus (every compatible bipartition).

### 5.2 `buildTreeFromClusters` — the tree assembly algorithm

**`Utils.java` lines 63–143.**

Starts with a star tree: all n taxa as direct children of a virtual root.
For each cluster in the snapshot list (most frequent first):

**Step 1** — Rebuild `SchieberVishkinLCA` on current tree:
```java
SchieberVishkinLCA lcaFinder = new SchieberVishkinLCA(tree);
```
Builds an Euler-tour-based range-minimum-query structure for O(1) LCA queries.
**Cost: O(current tree size) ≈ O(n) per cluster.**
This is done INSIDE the loop — the tree changes each iteration, so the LCA
structure cannot be cached.

**Step 2** — Find LCA of the cluster's leaves:
```java
TNode lca = lcaFinder.getLCA(clusterLeaves);
```
O(|cluster|) queries, each O(1) after preprocessing.

**Step 3** — Scan children of LCA, find subsets:
```java
for (TNode child : lca.getChildren()) {
    if (tc.containsCluster(childCluster)) {
        movedChildren.add(child);
        remainingleaves -= childCluster.getClusterSize();
    }
}
if (movedChildren.size() == 0 || remainingleaves != 0) continue; // incompatible
```
O(degree of LCA). If the cluster is incompatible with what's already been
inserted (remainingleaves ≠ 0 after scanning), it is skipped.

**Step 4** — Create new internal node, adopt moved children:
```java
STINode newChild = lca.createChild();
while (!movedChildren.isEmpty()) newChild.adoptChild(movedChildren.remove(0));
```
O(degree). The tree is now one node more resolved.

**Concrete example** (n=5, taxa A–E):

```
Initial star:
         root
      /  / | \ \
     A  B  C  D  E

Insert {A,B,C}:  LCA=root, subset children: A,B,C  →  create [ABC]
         root
        /    \ \
    [ABC]    D  E
    / | \
   A  B  C

Insert {D,E}:  LCA=root, subset children: D,E  →  create [DE]
         root
        /    \
    [ABC]   [DE]
    / | \   / \
   A  B  C  D  E

Insert {B,C,D}: LCA=root, children=[ABC],[DE].
  [ABC]={A,B,C} ⊆ {B,C,D}? No (A ∉ {B,C,D})
  movedChildren empty → SKIP (incompatible with existing structure)
```

**Time for `buildTreeFromClusters`**:
- B_i clusters for tree i × O(n) LCA rebuild each = **O(B_i × n)**
- LCA preprocessing dominates; all other per-cluster work is O(n) worst case
- 7 trees built in parallel: wall time = O(B_7 × n) = O(B × n)
- Total sequential equivalent: O((B_1+...+B_7) × n) ≤ O(7 × B × n)

**Memory for 7 trees**:
- Each tree: at most 2n−1 nodes, each O(1) → O(n) per tree
- 7 trees: O(7n) total → **negligible**
- Cluster list copies: 7 ArrayList<STITreeCluster> sharing the same BitSet
  objects → O(7 × B) pointers, ~negligible

---

## 6. Phase 4 — Polytomy Resolution: Injecting Bipartitions into X

**`addExtraBipartitionByHeuristicsLoop.call()`, lines 1206–1259.**

For each of the 7 consensus trees, for each polytomy node (degree > 2):

### 6.1 Polytomy size limit

Before processing, all polytomy degrees across all 7 trees are collected and
sorted. A size limit is computed:

```java
int N = 50 + n * 25;   // e.g. n=1000: N=25050
// walk sorted degrees, accumulate sum-of-squares until > N
// polytomySizeLimit = last degree processed
```

For n=1000: N=25,050. A polytomy of degree 100 contributes 10,000 to the sum,
so roughly sqrt(N) ≈ 158 is the limit. Polytomies larger than this are **skipped
entirely** — they are too costly to resolve and their signal is too diffuse.

### 6.2 Unrooting the polytomy

The consensus tree is built as rooted (virtual root). To treat the polytomy as
unrooted, the **complement cluster** is added to the children list:

```java
BitSet comp = greedyBS.clone();
comp.flip(0, n);       // all taxa NOT in this subtree
childbs[lastIndex] = comp;
```

So if the polytomy node covers taxa {A,B,C} and has children {A}, {B}, {C},
the array becomes: `[{A}, {B}, {C}, {D,E,F,...}]` — this is the unrooted view
of the polytomy. Now `resolvePolytomy` and `sampleAndResolve` see all d+1
groups that need to be split.

### 6.3 Step A — UPGMA sub-matrix resolution

```java
addSubSampledBitSetToX(
    speciesMatrix.resolvePolytomy(Arrays.asList(childbs), true), tid);
```

**`resolvePolytomy` calls `resolveByUPGMA(bsList, original=true)`** on `SimilarityMatrix`.

This runs UPGMA but **not on the full n×n matrix** — only on the `d` groups
(where d = polytomy degree + 1 for the complement). The similarity between two
groups i and j is the average `matrix[x][y]` over all x ∈ group_i, y ∈ group_j.

```java
for (int k = bsI.nextSetBit(0); k >= 0; k = bsI.nextSetBit(k+1))
    for (int l = bsJ.nextSetBit(0); l >= 0; l = bsJ.nextSetBit(l+1))
        is[j] += this.matrix[k][l];
is[j] /= c;   // average
```

Then `upgmaLoop` runs on d×d entries (not n×n). Time: O(d² log d) per polytomy.
Since d << n (polytomies are small), this is cheap.

Result: bipartitions produced by hierarchically merging the d groups in similarity
order → added to X.

### 6.4 Step B — Adaptive `sampleAndResolve` rounds

```java
int k = 0;
for (int j = 0; j < GREEDY_ADDITION_DEFAULT_RUNS + k; j++) {
    if (sampleAndResolve(childbs, contractedTrees, ...) && k < GREEDY_ADDITION_MAX)
        k += GREEDY_ADDITION_IMPROVEMENT_REWARD;  // +2 rounds as reward
}
```

**Base rounds: 10.** If a round finds ≥ `GREEDY_ADDITION_MIN_FREQ` (5) new clusters
in X, `k` increases by 2 → extra rounds are added. Maximum total rounds: 100.

Each `sampleAndResolve` round:
1. For each of the d groups in the polytomy: pick one random representative taxon
2. Look up bipartitions in `contractedTrees` (all gene trees) that are consistent
   with this random sample
3. Map each found bipartition back to the full taxon set (add back unsampled taxa)
4. Add the mapped bipartitions to X

**Cost per round**: O(d) random choices + O(k × n/64) for scanning gene tree
bipartitions (k trees, each bipartition is a BitSet of n/64 words).

**Cost per polytomy**: O(R × k × n/64) where R ∈ [10, 100] rounds.

**Cost per tree** (P polytomies): O(P × R × k × n/64)

For P=10 polytomies, R=20, k=1000, n=25000: 10×20×1000×390 ≈ 78M operations.
Fast. But polytomies in T1 can have degree n/2 (entire half of the tree unresolved),
in which case R could hit 100 and P could be large.

---

## 7. Complete Complexity Summary

| Phase | Time | Memory |
|---|---|---|
| Phase 1: count bipartitions | O(k × n²/64) | O(B × n/8) bytes (HashMap) |
| Phase 2: sort | O(B × log B) typical | O(B × 48) bytes (TreeSet) |
| Phase 3: 7 trees (parallel) | O(B × n) wall time | O(7n) for trees |
| Phase 4: UPGMA per polytomy | O(P × d² log d) per tree | O(d × n/8) transient |
| Phase 4: sampleAndResolve | O(P × R × k × n/64) per tree | O(k × n/8) transient |
| **Dominant** | **O(k×n²/64 + B×n)** | **O(B × n/8)** |

Absolute numbers for k=1000, n=25,000, B=250,000 (typical moderate ILS):

| Phase | Operations | Wall time (estimate) |
|---|---|---|
| Count bipartitions | 1000 × 625M/64 ≈ 9.8B | ~10 seconds |
| Sort | 250K × 18 ≈ 4.5M | < 1 second |
| Build 7 trees (parallel) | 250K × 25K = 6.25B | ~10–30 seconds |
| Polytomy resolution | O(P×R×k×n/64) — small | seconds |
| **Total** | | **~20–40 seconds** |

---

## 8. Memory Peak: Which Phase Dominates?

| Structure | Size | n=25,000, B=250K |
|---|---|---|
| HashMap (Phase 1) | B × n/8 bytes | **780 MB** |
| TreeSet (Phase 2) | B × 48 bytes | 12 MB |
| 7 output trees | 7 × 2n nodes | negligible |
| Gene trees (input, already in RAM) | k × n/8 bytes | 3 GB |
| **Greedy consensus additional peak** | | **~800 MB** |

The HashMap at ~780 MB is the dominant additional allocation from the greedy
consensus itself. The gene trees (k × n/8 = 3 GB for k=1000, n=25,000) were
already in memory before this phase, so they don't count as new cost.

---

## 9. What X Looks Like After All Sources

After the full `addExtra` pipeline:

| Source | Bipartitions in X |
|---|---|
| Gene tree direct extraction | up to k×(n−2), deduplicated |
| UPGMA full tree (Track A) | n−2 bipartitions |
| Polytomy resolution in gene trees (3 rounds) | variable |
| UPGMA bipartitions re-added via distance path | same n−2 again |
| 7 greedy consensus trees + polytomy resolution | variable, potentially large |

The greedy consensus trees are the **largest contributor** to X enrichment in
the extra mode. They systematically cover bipartitions at 7 frequency levels
and resolve polytomies using both UPGMA signal (high accuracy, data-driven)
and random sampling (high coverage, breadth-first exploration).

---

## 10. Key Observations

**The LCA rebuild bottleneck**: rebuilding `SchieberVishkinLCA` inside
`buildTreeFromClusters` for every cluster is the dominant time cost of Phase 3.
If the LCA structure supported incremental updates (adding one internal node),
the cost would drop from O(B × n) to O(B × log n). This is a known
implementation inefficiency.

**7 trees vs 1**: the threshold-based approach is cleverly designed. Each tree
serves a different purpose: T1 polytomies are large and contain high-ILS signal
(many competing topologies); T7 polytomies are small residual ambiguities.
Resolving all 7 levels ensures X is rich across the full support spectrum.

**Adaptive rounds**: the `GREEDY_ADDITION_IMPROVEMENT_REWARD` mechanism means
productive polytomies (those where random sampling keeps finding new bipartitions)
get more rounds automatically. Up to 100 rounds per polytomy. This is
computationally bounded by `GREEDY_ADDITION_MAX = 100` to prevent runaway cost.

**Polytomy size limit**: the `50 + n×25` budget prevents the algorithm from
spending all its time on one massive polytomy (e.g. a star tree at T1). Only
polytomies up to the computed degree limit are processed. Large polytomies are
skipped — their signal is too diffuse for random sampling to be productive.
