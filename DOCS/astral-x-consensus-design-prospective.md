# ASTRAL-X: Greedy Consensus Construction and Polytomy Resolution

**A tuple-based, bitset-free redesign of ASTRAL-MP's `addExtraBipartitionByHeuristics`**

This document specifies the finalized design for the "extra mode" bipartition
enrichment of ASTRAL-X — the port of ASTRAL-MP's greedy consensus machinery
onto STELAR-X's compact integer-tuple + double-hashing world. It replaces the
two dominant costs of the legacy implementation:

- the **O(B·n) per-cluster LCA rebuild** inside `buildTreeFromClusters`, and
- the **O(n)-per-bipartition bitset encoding** used throughout,

with an incremental union-find laminar builder (**O(B̄·α(n))**) and a
prefix-scan hashing scheme that keeps every structure in **O(nk)** space.

It is organized so that each design decision is justified and traceable. The
two phases are: **(I) Consensus construction** and **(II) Polytomy resolution
→ emission into X**.

---

## 0. Notation and Standing Assumptions

| Symbol | Meaning |
|---|---|
| `n` | number of taxa |
| `k` | number of gene trees |
| `id(t) ∈ {0..n-1}` | global, tree-independent taxon id |
| `A_i` | postorder leaf array of gene tree `i` |
| `π_i(v)` | position of taxon `v` in `A_i` (the index map) |
| `B` | number of unique bipartitions across gene trees |
| `B̄` | summed size of inserted clusters, `Σ |C_i|` |
| `X` | the candidate-bipartition set the DP will consume |
| `σ(S)` | double-hash signature `(ϕ1, ϕ2)` of taxon set `S` |
| `α(n)` | inverse-Ackermann (effectively constant) |

**Hash scheme (inherited from STELAR-X §2.5).** Two permutation-invariant,
associative hash functions over `Z_{2^64}`: `ϕ1` = addition, `ϕ2` = XOR. A
per-element scrambler `H` maps each id into `Z_{2^64}`. For a set `S`,
`σ(S) = (ϕ1({H(id) : id∈S}), ϕ2({H(id) : id∈S}))`. Equivalent bipartitions
(same taxa, any order, either side) share a signature; collision probability is
bounded by `B²/M²` with `M = 2^64`, negligible for `B = Θ(nk)`.

**Cancellation operators.** `ϕ1` admits `⊖₁` = subtraction mod 2⁶⁴; `ϕ2`
admits `⊖₂` = XOR (self-inverse). These make prefix-scan range queries O(1).

**Core discipline.** No length-`n` bitset is ever materialized. Cluster sides
are named by ranges into a tree's postorder array, enumerated by walking that
range, and hashed via prefix scans. **Bipartitions are always emitted by their
smaller side.**

---

# PART I — Greedy Consensus Construction

## 1. Problem Restatement

The legacy pipeline builds **7 consensus trees**, one per frequency threshold
`{1/3, 1/5, 1/10, 1/20, 1/50, 1/100, 0}`, each from the prefix of
frequency-sorted bipartitions exceeding that threshold. Their polytomies mark
unresolved regions to be enriched.

Stripped of tree-surgery language, `buildTreeFromClusters` performs exactly one
operation:

> **Incremental greedy refinement of a laminar family by compatible-cluster
> insertion.** Walk clusters most-frequent-first; maintain a partial tree;
> for each cluster `C`, accept it iff it is laminar (nested-or-disjoint, with
> the unrooted complement allowance) with every accepted cluster, refining one
> polytomy into two; else skip.

The output of accepting a prefix is a laminar family — i.e. a tree.

**Key consequence (collapses the 7-way work).** Because thresholds take nested
*prefixes* of the same sorted list, `T1 ⊂ T2 ⊂ … ⊂ T7` as cluster sets. The
greedy tree of a prefix is exactly the *state of one incremental process* after
consuming that prefix. Therefore we **do not build 7 trees**. We run **one**
incremental insertion pass over the whole sorted list and **snapshot** the
partial tree each time the running frequency crosses a threshold boundary.

This simultaneously (a) removes the O(n) LCA rebuild, (b) eliminates the 7×
redundant rebuilding, and (c) discards the false "build 7 trees in parallel"
parallelism — which is incoherent for an inherently incremental process.

## 2. Data Structures

### 2.1 The laminar forest (the partial tree)

A set of **nodes**, each with:

- `parent[v]`, `children[v]` (list), `size[v]` (taxa in subtree).
- Leaves are nodes (one per taxon, `size = 1`).
- A virtual root `R` initially holds all `n` leaves as children (the star tree).

`size[v]` is maintained additively at each insert (`size[w] = Σ size[adopted]`),
never by scanning — correct by construction since we only ever adopt whole nodes.

### 2.2 The locator (union-find with node payload)

This is the structure that replaces the Schieber–Vishkin LCA. It answers, in
amortized `α(n)`: *"what is the deepest accepted node currently containing
taxon `t`?"*

- `up[id(t)]` : a node containing `t`, possibly stale (not necessarily topmost).
  Initially the leaf `t`.
- `parent_acc[v]` : the accepted internal node that adopted `v`, or `NONE`.
  Set **once**, when `v` is re-parented during an insert.

```
loc(t):
    v = up[id(t)]
    while parent_acc[v] != NONE:      # climb the "absorbed-into" forest
        v = parent_acc[v]
    up[id(t)] = v                     # path compression
    return v
```

`loc` is a path-compressed union-find `find` over the `parent_acc` forest, with
nodes carrying `size`/`children` payload. Critically:

- **Incremental:** accepting a cluster that adopts `k` children costs `k` pointer
  writes (`parent_acc[c] = w`). The structure is *extended*, never rebuilt.
- **Query scales with `|C|`, not `n`:** the LCA of `C` is found by climbing from
  `C`'s taxa and checking convergence — `O(|C|·α(n))`, no `n` term.

> **Why this is the same LCA the legacy code computed, but cheaper.** Finding
> the LCA was never the cost — *maintaining a queryable LCA structure on a
> mutating tree by full rebuild* was. Schieber–Vishkin is static: adding an
> internal node invalidates its Euler tour, forcing the O(n) rebuild. Union-find
> is incremental (O(k) update) and its query/update are the *same* primitive,
> beating a general dynamic-LCA structure's O(log n) with O(α(n)).

## 3. The INSERT Procedure

`INSERT(C)` must correctly handle **all three** behaviors:

1. **Sideways merge** — `C`'s taxa span several top-level accepted nodes that
   share a parent → group them under a new node.
2. **Refinement (descend)** — all of `C` lies inside one accepted node `v`;
   `C` aligns with a sub-collection of `v`'s children → create a new node
   *inside* `v`.
3. **Reject (cross-cut)** — `C` takes some-but-not-all of a child's taxa →
   incompatible.

The discriminator is **whether `C` localizes to one node (descend) or several
(operate at their common parent)**, and rejection is **"any touched child is
only partially covered."**

```
INSERT(C):
    # --- Localize: find the level at which C operates ---
    nodes = { loc(t) : t ∈ C }            # topmost accepted nodes hit, O(|C|·α)
    if |nodes| == 1:
        u = the single node
        if hits-fill u completely (Σ over C == size[u]):
            # C ⊇ u entirely but C is bigger only via taxa outside u → impossible
            # if localized correctly; treat as: C == u → redundant, SKIP
            SKIP
        else:
            v = u                          # DESCEND: operate among u's children
    else:
        v = common parent of all nodes in `nodes`
        if no common parent: REJECT        # C straddles two groups incompatibly

    # --- Resolve C against v's CHILDREN ---
    clear hits
    for t ∈ C:
        c = child_of(v, t)                 # the child of v on the path to t
        hits[c] += 1
    T = { c : hits[c] > 0 }

    # --- Laminarity test (pure integer compares, no bitset) ---
    for c ∈ T:
        if hits[c] != size[c]:
            REJECT                         # c is split by C → cross-cut

    if T == children[v]:  SKIP             # C == v, redundant
    if |T| < 2:           SKIP             # C equals one existing child, nothing new

    # --- Insert ---
    create node w; parent[w] = v
    for c ∈ T:
        remove c from children[v]; add c to children[w]
        parent[c] = w
        parent_acc[c] = w                  # locator update: O(|T|), NOT O(taxa)
    size[w] = Σ_{c∈T} size[c]
    add w to children[v]
```

### 3.1 `child_of(v, t)` — descend-aware locator

We need not the topmost absorbed node but the ancestor of `t` that is a **direct
child of `v`**. Implement `loc` parameterized with a stop level: climb
`parent_acc` from `t` until the *next* hop would pass `v`, return the node just
below `v`. Path-compress carefully so a cached topmost `up[t]` does not hide the
intermediate child needed on a descend (recompute on descent rather than trust a
too-high cached pointer). **This is the single most bug-prone line in Part I.**

### 3.2 Why the tests are correct

- **Whole-node test** (`hits[c] == size[c]`): the set-comparison
  `containsCluster` of the bitset world becomes counting how many of `C`'s taxa
  fell on child `c` versus `c`'s known subtree size. Equal ⇒ `c ⊆ C` fully;
  less ⇒ cross-cut.
- **Common-parent test never spuriously rejects.** Nodes in `nodes` are maximal
  (no one is an ancestor of another, since `loc` returns topmost-absorbed). For
  a compatible `C`, maximal accepted nodes wholly inside `C` are necessarily
  siblings; differing parents imply a real incompatibility already flagged by
  the size test. So unequal parents ⇒ genuine reject.
- **Untouched siblings are fine.** When `C` grabs some of `v`'s children, the
  untouched ones simply remain siblings of the new node `w`. Only *touched*
  children are size-checked.

## 4. The Single-Pass + Snapshot Driver

```
sort all unique bipartitions by frequency, descending      # Phase 2 (unchanged)
init laminar forest = star tree; init locator
ti = 0   # threshold index into {1/3, 1/5, 1/10, 1/20, 1/50, 1/100, 0}
for (C, freq) in sorted list:               # most-frequent first
    while ti < 7 and freq < threshold[ti]:
        snapshot the current forest as tree T[ti]           # O(n)
        ti += 1
    INSERT(C)
while ti < 7:                                # flush remaining thresholds (incl. 0)
    snapshot as T[ti]; ti += 1
```

A snapshot copies the current node structure, `O(n)`. Seven snapshots ⇒ `O(7n)`,
negligible. Each `T[ti]` is exactly the legacy greedy tree at that threshold —
**bit-identical output**.

## 5. Bitset-Free Cluster Handling

A cluster `C` from a gene tree is the tuple `(i, l, r)` = subarray `A_i[l..r]`
(STELAR-X §2.4). Inside `INSERT`:

- **Enumerate `C`'s taxa:** `for j in l..r: t = A_i[j]` — `O(|C|)`, touches no
  bitset. (A bitset would scan all `n/64` words regardless of `|C|`.)
- **Map a taxon to its node:** `loc` / `child_of` over integer-id-keyed arrays —
  no bitset.
- **Laminarity:** integer `hits` vs `size` — no bitset.

The legacy `containsCluster` (O(n/64) bitset subset test **per child**) is
entirely dissolved.

## 6. Part I Complexity

| Quantity | Legacy | ASTRAL-X |
|---|---|---|
| Per cluster | O(n) LCA rebuild + O(n/64)·deg bitset | O(\|C\|·α(n)) |
| Build (all 7) | O(7·B·n) | O(B̄·α(n)) |
| Snapshots | (rebuilt 7×) | O(7n) |
| Memory | O(B·n/8) bitset HashMap | O(nk) tuple/hash |

**Honest floor.** The win is the `α(n)` factor and the eliminated bitset, *not*
escaping `B̄ = Σ|C_i|`. A cluster of size `n` (near-universal clade, common at
**low ILS**) costs O(n) to insert — you must read what you insert. So the bound
degrades toward O(B·n) exactly when clusters are large. This is the
information-theoretic floor, not a defect; the method is optimal up to `α(n)`
against the input description size. The win is largest in the **high-ILS**
regime where clusters are small — which is precisely where greedy consensus
enrichment matters most.

---

# PART II — Polytomy Resolution and Emission into X

## 7. The Key Representation: Hash the Consensus Tree by Its Own Postorder

A polytomy child group `cᵢ` is a node we built; its leaves are scattered in
every *gene tree's* postorder — but **contiguous in the consensus tree's own
postorder**, because postorder visits a subtree's leaves consecutively. The
consensus tree is just another tree, so it receives the **same prefix-scan
hashing treatment as a gene tree**.

### 7.1 Construction (once per snapshot tree, O(n))

1. Postorder pass → `A_cons[0..n-1]` (leaf ids in postorder); stamp `(l,r)` on
   every node (its contiguous range).
2. Prefix-scan arrays, applying the *same* `H` and `ϕ1, ϕ2` as gene trees:
   - `P1[0] = 0` (ϕ1 identity); `P1[m] = P1[m-1] +  H(A_cons[m-1])`  (mod 2⁶⁴)
   - `P2[0] = 0` (ϕ2 identity); `P2[m] = P2[m-1] XOR H(A_cons[m-1])`

### 7.2 O(1) signature of any subtree

For an inclusive range `[l, r]` (0-indexed leaves):

```
σ1[l..r] = P1[r+1] − P1[l]          (mod 2^64)
σ2[l..r] = P2[r+1] XOR P2[l]
σ[l..r]  = (σ1, σ2)
```

Because `ϕ1, ϕ2` are **permutation-invariant**, a consensus-derived signature
and a gene-tree-derived signature for the *same taxon set* are **identical** —
so X deduplicates across all sources automatically. This is the entire reason
the cross-tree matching works.

### 7.3 Disjoint multi-range union (the one landmine)

A resolution uniting **non-adjacent** groups yields a side that is several
disjoint ranges `[l₁,r₁]…[l_m,r_m]`. Combine under the group operators:

```
σ1(union) = Σ_j (P1[r_j+1] − P1[l_j])      (mod 2^64)
σ2(union) = XOR_j (P2[r_j+1] XOR P2[l_j])
```

`O(m)` ops, `m ≤ d`, no taxon re-read.

> **PRECONDITION — ranges MUST be disjoint.** Additive `ϕ1` double-counts
> overlaps; XOR `ϕ2` *cancels* duplicated elements (silent corruption). In the
> polytomy setting children are disjoint by construction, so this is safe — but
> wrap it as `combineDisjointRanges(...)` and assert disjointness, so the
> precondition cannot be silently violated by later reuse.

### 7.4 O(1) complement

The whole-tree signature is `(P1[n], P2[n])`. The complement of any range is
`whole ⊖ range` (valid because a range and its complement are disjoint and
cover everything):

```
σ1(complement of [l,r]) = P1[n] − σ1[l..r]
σ2(complement of [l,r]) = P2[n] XOR σ2[l..r]
```

This gives the unrooting "rest" side in O(1) without any `flip(0,n)`.

## 8. Per-Polytomy Resolution

For a polytomy node `v` with range `[L,R]` and `d` children
`c₁…c_d` (each a sub-range tiling `[L,R]` left-to-right), plus the implicit
complement "rest" = `[0,L-1] ∪ [R+1,n-1]`:

### 8.1 Step 0 — setup (O(d), groups already known)

Children ranges are read directly from the stamped `(l,r)` — `O(d)`, no
per-group leaf collection. One representative per group: `A_cons[l_cᵢ]`.

### 8.2 Polytomy size limit (unchanged policy)

Compute the legacy budget `N = 50 + n·25`; accumulate sorted polytomy degrees'
squares until exceeding `N`; skip polytomies above the resulting degree limit
(≈ √N). Large polytomies have signal too diffuse for sampling to resolve
productively.

### 8.3 Step A — UPGMA on the d groups (data-driven, one resolution)

- Build the `d×d` group-similarity matrix: entry `(i,j)` = average
  `speciesMatrix[x][y]` over `x∈cᵢ, y∈cⱼ` (id-indexed lookups, no bitset). The
  `d×d` **fill dominates** (`O(|v|²)` worst case); UPGMA itself is `O(d² log d)`.
- Each UPGMA merge unites a set of groups → one bipartition side. **Adjacent**
  groups → single range → O(1) hash (§7.2). **Non-adjacent** → disjoint
  multi-range → O(m) hash (§7.3). Emit by smaller side (§7.4 for complement).

### 8.4 Step B — `sampleAndResolve` (random, breadth)

Adaptive rounds, `R ∈ [10, 100]`:

```
k = 0
for j in 0 .. (GREEDY_ADDITION_DEFAULT_RUNS=10) + k - 1:
    reps = one representative id per group         # A_cons[l_cᵢ], O(d)
    for each gene tree i:                          # restrict to the d reps
        positions = { π_i(rep) : rep ∈ reps }      # O(d)
        induced = read induced topology on reps    # O(d) via sorted positions + stack
        for each induced split (subset S of reps | rest):
            map S back to the union of its groups' ranges
            σ = combineDisjointRanges(...)         # O(1) or O(m≤d)
            add σ to thread-local buffer
    if (this round added ≥ GREEDY_ADDITION_MIN_FREQ=5 new to LOCAL buffer)
       and k < GREEDY_ADDITION_MAX=100:
        k += GREEDY_ADDITION_IMPROVEMENT_REWARD=2
```

Per round: `O(k·d)` (was `O(k·n/64)` with bitsets). `d ≪ n` (size-limited), so
this is a real constant-factor win *and* bitset-free.

**The "restrict gene tree to d reps in O(d)" primitive** (Step B hot loop):
look up each rep's position `π_i(rep)`; sort the `d` positions (`O(d log d)`);
sweep them with a stack to read the induced bipartition structure (the topology
of gene tree `i` restricted to the `d` reps) in `O(d)`. No traversal of the
full tree, no `n/64` scan.

> **Correction / status (see
> [consensus-emission-and-restriction-optimization.md](consensus-emission-and-restriction-optimization.md) §3):**
> the sketch above is under-specified — postorder positions *alone* do not
> determine the induced topology; the correct `O(d log d)` primitive is the
> *auxiliary/virtual tree* (sort by Euler order, adjacent-pair LCA depths,
> depth-stack build), which needs an LCA structure — already available in
> `completion/EulerTourBuilder`. The current implementation (`PolytomyResolver.stepB`)
> still does the `O(n)` full walk; the auxiliary-tree variant (incl. the
> frequency-*multiplicity* subtlety) is specified in that doc.

## 9. Emission into X — Always the Smaller Side, Always a Signature

Every resolution yields "side L | rest". To add to X:

1. Compute `σ(L)` via prefix scan (O(1) contiguous, O(m) multi-range).
2. If `|L| > n − |L|`, compute `σ(rest)` instead (O(1), §7.4) and emit that.
3. Insert `σ` into X's double-hash table; dedup in O(1).

X stays entirely in the **O(nk) tuple/hash world** — no bitset ever
materialized for the consensus or polytomy phases. **This is the single most
important invariant for ASTRAL-X**: emitting in any other encoding reintroduces
the O(n²k) bitset memory the whole redesign exists to remove.

## 10. Parallelism — Where It Actually Belongs

The legacy design parallelized the **build** (7-way) and treated polytomy
resolution as an afterthought. **This is inverted.** The build is inherently
incremental (Part I, collapsed to one pass); the *embarrassing* parallelism
lives in **polytomy resolution**.

### 10.1 Task pool: one polytomy = one task

Collect **all** polytomies from **all 7** snapshot trees into a single task
pool — potentially thousands of independent tasks, not 7. Each task's Step A and
Step B read only its own groups plus read-only shared structures (species
matrix, gene trees, `π_i` maps). Dispatch across `TC` CPU threads.

### 10.2 Thread-local buffers, merge once (no lock on hot path)

The shared sink X is the contention point. Use the same map-reduce shape as
STELAR-X's CPU frequency mapping (§2.8): **each thread accumulates emitted
signatures into a thread-local hash table; merge into global X once at the end.**
Merge is `O(total emitted)`, dedup by signature. No lock on the per-emission
path.

### 10.3 Load balancing — work-stealing, longest-first

Polytomy costs are **highly skewed** (Step A is `O(|v|²)`; T1 has huge
polytomies, T7 tiny ones). A static 1/`TC` partition strands one thread on a
giant polytomy. Use a **work-stealing queue**, and dispatch tasks **sorted by
estimated cost descending** (longest-processing-time-first). This matters more
than any micro-optimization in this phase.

### 10.4 The adaptive-rounds correctness decision (state explicitly)

Adaptivity ("≥5 new clusters ⇒ +2 rounds") keys off novelty. Two choices:

- **Local novelty (recommended for scalability):** measure against the
  task-local buffer. Removes the race entirely and makes the number of rounds
  reproducible. The resulting X need not equal a racy shared-buffer run because
  the adaptive stopping point can differ; dedup-on-merge only guarantees that
  duplicate emissions collapse, not that unexecuted rounds are recovered.
- **Global novelty (bit-identical to sequential):** measure against global X →
  forces serialization on the shared structure across polytomies that share
  bipartitions. Kills the parallelism.

This is a genuine **speed ↔ exact-reproducibility tradeoff** to decide
deliberately, not paper over. ASTRAL-X uses task-local novelty and canonical
task ordering/seeding so a fixed input and option set produces a stable X.

## 11. End-to-End Trace (one polytomy)

`A_cons = [B, C, D, E, A, F, G]`, polytomy `v = [0,3]` (B,C,D,E), children
`c₁=[0,1]={B,C}`, `c₂=[2,3]={D,E}`; rest `= [4,6]={A,F,G}`.

- `σ(c₁) = (P1[2]−P1[0], P2[2]⊕P2[0])` — contiguous, O(1).
- **Step A:** UPGMA merges adjacent `c₁,c₂` → union `[0,3]` →
  `(P1[4]−P1[0], P2[4]⊕P2[0])`, single range O(1). Side `{B,C,D,E}` (4) > rest
  (3) ⇒ emit by rest: `σ(rest) = (P1[7]−σ1[0..3], P2[7]⊕σ2[0..3])`, O(1).
- **Step B:** reps `(B,C,E,...)`; restrict each gene tree to reps via `π_i`,
  O(d); read induced splits; a split uniting non-adjacent groups
  `[0,1]∪[5,6]` → `combineDisjointRanges` O(2), disjoint ✓ → hash → local buffer.
- **Merge:** fold local buffer into global X, dedup by signature.

Every operation was a prefix-array lookup, an id-indexed matrix read, a `π_i`
lookup, or a group-combine. No length-`n` object materialized; X stays O(nk).

## 12. Complete Complexity Summary

| Phase | Time | Memory |
|---|---|---|
| Count bipartitions (Phase 1) | O(k·n²/64) → parallel map-reduce over k | O(nk) tuple/hash |
| Sort (Phase 2) | O(B log B) | O(B) |
| **Consensus build (one pass + snapshot)** | **O(B̄·α(n) + 7n)** | O(nk) |
| Polytomy Step A (per polytomy) | O(\|v\|² + d² log d) | O(d²) transient |
| Polytomy Step B (per polytomy) | O(P·R·k·d) | O(d) transient |
| Emission | O(1)–O(d) per bipartition (prefix scan) | O(nk) for X |
| **Dominant** | **O(k·n²/64 + B̄·α(n))** | **O(nk)** |

Versus legacy: build **O(7·B·n) → O(B̄·α(n))**; memory **O(B·n/8) bitset →
O(nk) tuple**.

## 13. Verification Plan (do this before trusting the fast path)

1. **Differential oracle.** Implement a brute-force `O(B·n)` laminarity checker
   (for each incoming `C`, naively compute "deepest cluster containing `t`" for
   all `t`, decide accept/reject/descend). Run the fast builder and the oracle on
   thousands of random cluster streams; assert identical accept/reject decisions
   and identical output trees.
2. **Invariants to assert continuously:**
   - `size[w] == Σ size[children[w]]` at every node.
   - `loc(t)` agrees with the naive "deepest accepted cluster containing `t`".
   - `parent_acc` is a forest (no cycles); every `find` terminates.
   - **Union direction:** `parent_acc` always points child → new internal node
     (toward the *newer* node). The single most dangerous bug is unioning the
     wrong way, which silently breaks the "deepest node" invariant and accepts
     clusters that should be rejected.
3. **Touch-count instrumentation.** Count nodes touched per insert. Legacy
   (Schieber–Vishkin) shows ≈ `n` per insert (the rebuild); the union-find path
   shows ≈ `|C|`. On the A–G traces: `7,7,7,7` vs `2,2,2,4`. That gap *is* the
   asymptotic improvement, made observable.
4. **Disjointness assertion** at every `combineDisjointRanges` call site
   (§7.3) — guards the silent XOR-cancellation / additive-double-count bug.
5. **Cross-source signature check.** Emit the same taxon set from a gene tree
   and from the consensus tree; assert identical `σ`. Validates
   permutation-invariance end to end.

---

## Appendix A — Why Each Legacy Cost Disappeared

| Legacy cost | Mechanism | ASTRAL-X replacement |
|---|---|---|
| O(n) LCA rebuild per cluster | Schieber–Vishkin static, rebuilt on mutation | Incremental union-find, O(k) update, O(α) query |
| 7× redundant tree builds | 7 independent `buildTreeFromClusters` | One incremental pass + 7 O(n) snapshots |
| O(n/64) bitset subset test per child | `containsCluster` BitSet AND | Integer `hits[c] == size[c]` |
| O(n/64) bitset scan to enumerate a side | iterate set bits of length-n BitSet | walk tuple/consensus range `A[l..r]`, O(\|side\|) |
| `comp.flip(0,n)` unrooting bitset | flip a length-n BitSet | symbolic complement; `whole ⊖ range`, O(1) |
| O(n/64) per gene-tree scan in sampling | BitSet AND across k trees | restrict to d reps via `π_i`, O(d) per tree |
| O(B·n/8) bitset HashMap | bitset-keyed map | double-hash signatures, O(nk) |

## Appendix B — Tunable Constants (inherited)

```
GREEDY_ADDITION_THRESHOLDS       = {0, 1/100, 1/50, 1/20, 1/10, 1/5, 1/3}
GREEDY_ADDITION_DEFAULT_RUNS     = 10      # base rounds per polytomy
GREEDY_ADDITION_MAX              = 100     # max adaptive rounds
GREEDY_ADDITION_IMPROVEMENT_REWARD = 2     # extra rounds on a productive round
GREEDY_ADDITION_MIN_FREQ         = 5       # new clusters to count as "improvement"
GREEDY_ADDITION_MAX_POLYTOMY_MIN = 50
GREEDY_ADDITION_MAX_POLYTOMY_MULT= 25      # polytomy size budget = 50 + n*25
```
