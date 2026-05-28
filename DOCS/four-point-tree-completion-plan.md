# Four-Point Tree Completion — Full Implementation Plan

Implementation plan for replacing the current 2-point greedy descent in
`TreeCompleter.java` with the ASTRAL-MP compatible four-point navigation,
including physical rerooting, sorted distance matrix rows, and full
verification strategy.

Primary reference: `astral-mp-legacy-codebase/WQDataCollection.java` lines 261–341
and `astral-mp-legacy-codebase/AbstractMatrix.java` lines 52–67.

---

## 1. What we are changing and why

Current `TreeCompleter.insertTaxon()`:
```
navigate from root, at each node: compare dist[x, leftRep] vs dist[x, rightRep]
always land at a leaf, graft x there
```

ASTRAL-MP `getCompleteTree()`:
```
for each missing taxon x:
  find anchor a = closest present taxon in THIS tree
  physically reroot tree at the edge (anchor, anchor.parent)
  navigate start subtree using four-point condition:
    at each node compare betterSideByFourPoint(x, a, c1rep, c2rep)
    can stop at an internal node (not just a leaf)
  insert x
```

The four-point condition uses 6 pairwise entries and an anchor taxon as a reference,
making navigation decisions richer than a simple 2-point distance comparison.
The rerooting ensures navigation always begins on the correct side of the anchor.

---

## 2. New data structures needed

### 2.1 Sorted distance matrix rows

After building the n×n distance matrix (`double[] dist`, flat row-major), we need
for each taxon x a list of all other taxa sorted by **ascending distance** to x.

Store as a flat int array:
```java
int[] sortedRows;   // [n × n]
// sortedRows[x * n + rank] = taxon ID of x's rank-th nearest neighbor
// rank 0 = closest (dist = 0.0, i.e. x itself; skip x in lookup)
// rank 1 = nearest other taxon
```

Memory: `4 × n²` bytes. For n=25000: 2.5 GB (same order as the distance matrix itself).
This is a one-time cost per run; computed after the distance matrix is finalized.

### 2.2 Per-tree `inTree` membership array

During completion of a single gene tree, we need fast O(1) membership queries:
"Is taxon z currently in this tree (originally present OR already inserted)?"

```java
boolean[] inTree = new boolean[n];
// Initialize: inTree[x] = true  iff  tree.positionMap[x] != -1
// Update:     inTree[x] = true  after inserting taxon x
```

This array is per-tree and reused across all missing taxa for the same tree.
Reset and reinitialize for each new tree.

### 2.3 Taxon-to-node map (per deep copy)

To find the TreeNode corresponding to a taxon ID in the mutable copy:
```java
TreeNode[] taxonNode = new TreeNode[n];
// taxonNode[id] = leaf node for that taxon in the current mutable tree
// Built during deep copy; updated after each insertion (new leaf's entry added)
```

---

## 3. Phase A — Computing sorted rows

### 3.1 CPU path (always available)

After `SimilarityMatrix` or `DistanceMatrix` is built and `dist[]` is finalized,
sort every row:

```
for each taxon x in 0..n-1:
    create index array indices[0..n-1] = {0, 1, ..., n-1}
    sort indices by dist[x * n + indices[j]] ascending
    write result into sortedRows[x * n + 0..n-1]
```

Time: O(n² log n).  For n=25000: ~25000 × 25000 × 15 ≈ 9.4 billion comparisons —
slow but a one-time cost.  Can be parallelised across rows trivially.

### 3.2 GPU path (batched Thrust argsort)

Because the full n×n distance matrix does not fit on GPU (n=25000 → 5 GB),
we sort in batches of B rows per GPU launch.

Algorithm:
```
choose batch size B such that (B × n × (8 + 4)) bytes fit comfortably on GPU
  (8 bytes per float64 dist value + 4 bytes per int index)
  Example: GPU has 20 GB free, n=25000 → B = 20*10^9 / (25000*12) ≈ 66 rows/batch

for each batch [rowStart, rowStart+B):
    // Upload B rows of dist values to GPU
    d_keys  = dist[rowStart*n .. (rowStart+B)*n]   (B*n doubles cast to float)
    // Initialize column indices (0..n-1 repeated B times)
    d_vals  = {0,1,...,n-1, 0,1,...,n-1, ...} (B repetitions, B*n ints)
    // Sort each row: use Thrust segmented sort
    thrust::sort_by_key(d_keys, d_keys + B*n, d_vals, row_comparator)
    // Download sorted indices
    copy d_vals → sortedRows[rowStart*n .. (rowStart+B)*n]
```

The `row_comparator` for Thrust segmented sort can be implemented with
`thrust::stable_sort` + a custom comparator that treats row boundaries as
segment dividers, OR by sorting each row individually in a CUDA kernel
using `cub::DeviceRadixSort::SortPairs` per row.

Reference code: `ref-cuda/gpu_sort.cu` already has `thrust::sort_by_key` for
(key, value) pairs. The per-row batching wrapper is new work.

**HINT for n-ary extension**: sortedRows is purely a property of the distance
matrix and has nothing to do with tree topology — no changes needed here when
polytomies are added.

---

## 4. Phase B — Anchor finding

For missing taxon x, find the closest taxon currently in the tree:

```java
int findAnchor(int x, boolean[] inTree, int[] sortedRows, int n) {
    int base = x * n;
    for (int rank = 0; rank < n; rank++) {
        int candidate = sortedRows[base + rank];
        if (candidate != x && inTree[candidate]) {
            return candidate;
        }
    }
    throw new RuntimeException("No anchor found for taxon " + x);
}
```

This mirrors `AbstractMatrix.getClosestPresentTaxonId()` exactly.

The walk terminates as soon as the first in-tree taxon is found. For typical
datasets (related taxa co-appearing in most trees), this is O(1) to O(small constant).
Worst case O(n) when x co-appears with very few others.

---

## 5. Phase C — Physical rerooting (binary tree)

### 5.1 Overview

We want to reroot the mutable tree at the **edge** between `anchor` (a leaf)
and `anchor.parent` (call it p1). The result should be a new root with two children:
- left = anchor leaf
- right = p1 (the "start" subtree = everything except anchor)

This matches ASTRAL-MP's: after `rerootTreeAtNode(closestNode)` + `removeBinaryNodes`,
the root has exactly 2 children: closestNode and start.

### 5.2 Algorithm

Collect the path from anchor to the old root:
```
path = [anchor, p1, p2, ..., pk]   where pk = old root
```
Length of path = depth of anchor + 1. This is O(log n) average, O(n) worst case.

Then perform the reversal:

```
Step 1: allocate newRoot internal node
        newRoot.left = anchor
        newRoot.right = p1
        anchor.parent = newRoot

Step 2: for i = 1 to k-2  (i.e. nodes p1 through pk-1):
        node         = path[i]
        childOnPath  = path[i-1]      ← child of node toward anchor (keep as child)
        parentOnPath = path[i+1]      ← old parent of node (becomes new child)

        // Replace the child-slot pointing to childOnPath with parentOnPath
        if node.left == childOnPath:
            node.left = parentOnPath
        else:
            node.right = parentOnPath

        // Fix parent pointers
        node.parent = (i == 1) ? newRoot : path[i-1]
        parentOnPath.parent = node

Step 3: collapse old root pk
        // pk's two original children were: path[k-1] (on path) and remainingChild
        // path[k-1] is now pk's PARENT (set in step 2), so pk has only remainingChild
        // Splice out pk: make remainingChild a direct child of pk's new parent (path[k-2])
        // (At this point pk-1 = path[k-1]'s child slot pointing to pk — set in step 2)

        remainingChild = (pk.left == path[k-1]) ? pk.right : pk.left
        // Replace pk in path[k-1]'s children with remainingChild
        if path[k-1].left == pk:
            path[k-1].left = remainingChild
        else:
            path[k-1].right = remainingChild
        remainingChild.parent = path[k-1]
        // pk is discarded (GC'd)
```

After this, `newRoot.right` is the `start` node (= p1).

**Complexity**: O(depth) time, O(depth) space for the path array.
No O(n) passes — no range recomputation, no full traversal.

**rangeStart / rangeEnd fields**: These become stale after rerooting (they were
computed for the original root). That is **intentional and OK** because:
- `leftmostTaxon()` uses `node.left` pointer chains, not range fields → still works
- `rebuildTree()` at the end recomputes all range fields from scratch → corrects them

**HINT for n-ary extension**: In step 2, `node.left == childOnPath / node.right`
becomes a scan of `node.children` list. Replace the matching entry.
Step 3 collapse: scan node.children, remove the null/old-root entry.
No structural change to the algorithm — just list indexing instead of left/right.

### 5.3 After rerooting: extracting start

```java
TreeNode newRoot = reroot(anchor);
// newRoot.left == anchor, newRoot.right == start (by construction)
TreeNode start = newRoot.right;
```

---

## 6. Phase D — Four-point navigation

### 6.1 The four-point score (similarity matrix)

Given taxa x, a (anchor), b (c1rep), c (c2rep):
```
xa = sim[x][a],  xb = sim[x][b],  xc = sim[x][c]
ab = sim[a][b],  ac = sim[a][c],  bc = sim[b][c]

ascore = xa + bc - xb - ac    → votes: x groups with a
bscore = xb + ac - xa - bc    → votes: x groups with b (= c1rep)
cscore = xc + ab - xb - ac    → votes: x groups with c (= c2rep)

return argmax: if ascore >= bscore and ascore >= cscore → return a
               if bscore >= cscore                      → return b
               else                                     → return c
```

(For distance matrix: flip sign of scores — use dist directly.)

This is a direct port of `SimilarityMatrix.getBetterSideByFourPoint()`.

### 6.2 Navigation loop with caching

```java
TreeNode start = newRoot.right;   // "rest of tree" after rerooting
int c1rep = -1;                   // cached leftmost of left child
int c2rep = -1;                   // cached leftmost of right child

while (!start.isLeaf()) {
    TreeNode c1 = start.left;
    TreeNode c2 = start.right;

    // HINT for n-ary: c1 = children.get(0), c2 = children.get(1)
    // For >2 children, apply four-point iteratively across all child pairs,
    // or pick the best child via multi-way tournament. See ASTRAL-MP's
    // addExtraBipartitionByHeuristics for the polytomy resolution pattern.

    if (c1rep == -1) c1rep = leftmostTaxon(c1);
    if (c2rep == -1) c2rep = leftmostTaxon(c2);

    int better = fourPointBetterSide(x, anchor, c1rep, c2rep, sim, n);

    if (better == anchor) {
        break;                    // insert at this internal node
    } else if (better == c1rep) {
        start = c1;
        c2rep = -1;               // right side changes at next level
    } else {                      // better == c2rep
        start = c2;
        c1rep = c2rep;            // c2's leftmost becomes new c1rep
        c2rep = -1;
    }
}
```

**The caching trick** (mirrors original exactly):
When descending into c1: the leftmost of c1's subtree is still `c1rep` (the leftmost
leaf of c1 is always in c1's left-most descendant regardless of depth). So c1rep is
reused as the representative for the new left child. Only c2rep needs refreshing.
Symmetric for descending into c2 (c2rep becomes c1rep for the next step).

### 6.3 Why anchor as stopping condition works

After rerooting, anchor is on the OTHER side of the root from start. So in the
abstract 4-taxa picture `{x, anchor, c1rep, c2rep}`:
- If `ascore` wins (x groups with anchor): x belongs BETWEEN the root and
  the current node — i.e., at this internal node level. Stop here.
- If `bscore`/`cscore` wins: x belongs deeper in c1/c2 subtree. Descend.

---

## 7. Phase E — Insertion

Two cases, matching WQDataCollection.java lines 325–337 exactly:

### 7.1 Stopped at a leaf

```
start.parent
   |
  start (leaf)

After:
start.parent
      |
  newInternal
     /      \
  start     newLeaf(x)
```

```java
TreeNode newLeaf     = new TreeNode();  newLeaf.taxonId = x;
TreeNode newInternal = new TreeNode();
TreeNode p = start.parent;

newInternal.left   = start;
newInternal.right  = newLeaf;
newInternal.parent = p;
start.parent       = newInternal;
newLeaf.parent     = newInternal;

if (p.left == start) p.left = newInternal;
else                 p.right = newInternal;
```

Special case: if `start == newRoot` (tree had only 1 leaf after rerooting), which
cannot happen since we guard `cardinality < 3`.

### 7.2 Stopped at an internal node (betterSide == anchor)

```
start
 /  \
c1   c2

After:
start
 /   \
x   newInternal
       /  \
      c1   c2
```

```java
TreeNode newLeaf     = new TreeNode();  newLeaf.taxonId = x;
TreeNode newInternal = new TreeNode();

newInternal.left   = c1;
newInternal.right  = c2;
newInternal.parent = start;
c1.parent          = newInternal;
c2.parent          = newInternal;

start.left         = newLeaf;
start.right        = newInternal;
newLeaf.parent     = start;
```

Note: `c1` and `c2` are `start.left` and `start.right` as of when the loop broke —
they are already set by the loop body. Do NOT re-fetch from start.

**HINT for n-ary**: In the internal-node case, x becomes a new child of `start`,
and the original children {c1, c2} are wrapped under `newInternal` (also a child of
`start`). For >2 children, wrap ALL existing children under `newInternal`:
`newInternal.children = start.children`, `start.children = [x, newInternal]`.

---

## 8. Membership update and taxon-to-node map update

After inserting taxon x into the tree:
```java
inTree[x] = true;
taxonNode[x] = newLeaf;   // so future anchors for later taxa can be found in O(1)
```

The outer loop then proceeds to the next missing taxon. Each subsequent taxon
can now use x as a potential anchor (inTree[x] == true).

---

## 9. Full per-tree completion procedure

```java
Tree completeTreeFourPoint(Tree tree, double[] sim, int[] sortedRows, int n) {

    // --- Setup ---
    boolean[]   inTree    = new boolean[n];
    TreeNode[]  taxonNode = new TreeNode[n];
    TreeNode    root      = deepCopyNodes(tree.root, null, taxonNode);

    for (int i = 0; i < n; i++)
        if (tree.positionMap[i] != -1) inTree[i] = true;

    // --- Insert each missing taxon in ascending ID order ---
    for (int x = 0; x < n; x++) {
        if (inTree[x]) continue;   // already in tree

        // Phase B: find anchor
        int anchor = findAnchor(x, inTree, sortedRows, n);

        // Phase C: reroot at edge (anchorLeaf, anchorLeaf.parent)
        TreeNode anchorLeaf = taxonNode[anchor];
        TreeNode newRoot    = rerootAtLeafEdge(anchorLeaf, root);  // returns new root
        root = newRoot;

        // Phase D: navigate
        TreeNode start = newRoot.right;   // non-anchor child
        int c1rep = -1, c2rep = -1;
        TreeNode c1 = null, c2 = null;

        while (!start.isLeaf()) {
            c1 = start.left;
            c2 = start.right;
            if (c1rep == -1) c1rep = leftmostTaxon(c1);
            if (c2rep == -1) c2rep = leftmostTaxon(c2);

            int better = fourPointBetterSide(x, anchor, c1rep, c2rep, sim, n);
            if (better == anchor)      break;
            else if (better == c1rep) { start = c1; c2rep = -1; }
            else                      { start = c2; c1rep = c2rep; c2rep = -1; }
        }

        // Phase E: insert
        TreeNode newLeaf = insertTaxon(x, start, c1, c2, start.isLeaf());

        // Phase F: update membership
        inTree[x]    = true;
        taxonNode[x] = newLeaf;
    }

    return rebuildTree(tree.treeIndex, root, n);
}
```

---

## 10. Complexity analysis

| Sub-step | Per missing taxon | Per tree | All trees (k trees, f_miss fraction) |
|---|---|---|---|
| Find anchor | O(rank_of_first_hit) ≈ O(1) avg | O(f_miss × n) | O(k × f_miss × n) |
| Collect reroot path | O(depth) | O(f_miss × n × depth) | O(k × f_miss × n × depth) |
| Physical reroot | O(depth) | same | same |
| Navigate (4-point) | O(depth × 6) = O(depth) | same | same |
| Insert | O(1) | O(f_miss × n) | O(k × f_miss × n) |
| Rebuild tree | O(n) | O(n) | O(k × n) |

`depth` = O(log n) for balanced trees, O(n) worst case (caterpillar trees).
In practice gene trees are close to balanced → O(log n).

**Sort rows (one-time)**:
- CPU: O(n² log n) time, O(n²) memory
- GPU batched: O(n² log n / GPU_speedup) — typically 20–50× faster than CPU

**Dominant cost remains matrix construction** (O(k × n²)), not tree completion.

---

## 11. Implementation steps (ordered)

1. **`SortedRows` builder** — new class `SortedRowsBuilder.java` in `completion/`:
   - `buildCPU(double[] dist, int n)` → `int[] sortedRows`
   - `buildGPU(double[] dist, int n)` → `int[] sortedRows` (batched Thrust argsort)
   - GPU native method: `native void computeSortedRowsGPU(double[] dist, int n, int batchSize, int[] sortedRows)`

2. **Update `deepCopyNodes`** in `TreeCompleter` to also populate `TreeNode[] taxonNode`:
   ```java
   static TreeNode deepCopyNodes(TreeNode src, TreeNode parent, TreeNode[] taxonNode)
   ```

3. **`rerootAtLeafEdge`** static method in `TreeCompleter` (or a new `TreeRerooting.java`):
   - Signature: `static TreeNode rerootAtLeafEdge(TreeNode anchorLeaf)`
   - Implements Section 5.2 path reversal + collapse
   - Returns the new root node

4. **`fourPointBetterSide`** static method:
   - Signature: `static int fourPointBetterSide(int x, int a, int b, int c, double[] sim, int n)`
   - Implements Section 6.1 formula
   - Returns the taxon ID of the winning side

5. **`findAnchor`** static method (Section 4)

6. **Update `completeTree`** to use the new sub-methods (Section 9)

7. **Update `Main.java`** to build `sortedRows` and pass it to `TreeCompleter.completeAll`

---

## 12. Verification strategy

### 12.1 Unit tests: tiny handcrafted trees

Write `TestFourPointCompletion.java` with trees small enough to verify by hand.

Example: 5 taxa {A=0, B=1, C=2, D=3, E=4}, tree `((A,B),(C,D))` (E missing).
Manually set the distance matrix to clearly favour E near A. Check that E ends up
as sister to A in the completed tree.

Test cases to cover:
- E inserts at leaf (stops at leaf during navigation)
- E inserts at internal node (betterSide == anchor)
- Multiple insertions into the same tree (x=3 uses x=2 as anchor)
- Anchor is a previously-inserted taxon (not originally present)
- One-taxa tree edge case (should throw, cardinality < 3)
- Star tree (all taxa sisters under root) — verify no crash

### 12.2 Rerooting unit tests

Test `rerootAtLeafEdge` independently:
- Reroot at each leaf in a small known tree
- Verify the result tree has the same set of bipartitions as the original
  (rerooting changes the root but not the topology)
- Verify `newRoot.left.isLeaf()` == true and `newRoot.left.taxonId` == anchor
- Verify parent pointers are consistent (walk every node)
- Verify all original leaves are still reachable from the new root

### 12.3 Cross-check with astral-my

`astral-my/` contains a dev-friendly ASTRAL-MP build that supports code modifications.

**Step 1**: Add a dump to `WQDataCollection.getCompleteTree()` to print the
completed Newick for each gene tree:

```java
// Add after the for loop in getCompleteTree():
System.err.println("COMPLETED_GT\t" + trc.toStringWD());
```

Recompile astral-my:
```bash
cd astral-my && mvn package -q
```

Run astral-my on a test input and capture output:
```bash
java -jar astral-my/target/astral.jar -i test/incomplete_5taxa.tre -o /tmp/astral_out.tre \
  2>&1 | grep "^COMPLETED_GT" > /tmp/astral_completed.txt
```

**Step 2**: Add equivalent dump to ASTRAL-X's `TreeCompleter.completeTreeFourPoint()`:
```java
Logging.info("COMPLETED_GT\t" + treeToNewick(result));
```

Run ASTRAL-X on the same input and collect the completed trees.

**Step 3**: Compare the two sets of completed trees using `rf.py` or a simple
Newick comparison (the topology must be identical bipartition-for-bipartition).

**Similarity vs distance matrix flag**:
ASTRAL-MP uses `SimilarityMatrix` by default.  To cross-check with distance matrix:
pass `-ustar` to astral-my (which enables `DistanceMatrix`).
In ASTRAL-X: set `--completion-method distance`.

**What to verify**:
- Same set of inserted taxa for each gene tree ✓ (by construction)
- Same anchor chosen for each (x, tree) pair — check by also dumping anchor
- Same completed Newick topology (up to rerooting / leaf label order)

### 12.4 Regression: existing test suite

After the new completer is wired in, run the full ASTRAL-X test suite:
```bash
./build.sh && cd test && python verify_weights.py
```
All 13 test cases (TC1–TC13) should still pass. Since weight scoring uses
`originalTrees` (pre-completion), changes to the completer do not affect
quartet scores for complete-tree tests. Tests that involve incomplete trees
(if any) will now produce different (and better) scores.

---

## 13. Notes on binary vs n-ary (for future polytomy support)

All hints for n-ary are scattered throughout sections above. Summary:

| Component | Binary (now) | N-ary extension |
|---|---|---|
| TreeNode | left, right | add `List<TreeNode> children`; keep left/right as aliases to children.get(0/1) for binary case |
| deepCopyNodes | copy left, right | copy children list |
| rerootAtLeafEdge | replace left/right slots | replace entry in children list; in collapse step, remove entry from children list |
| leftmostTaxon | walk node.left chain | walk children.get(0) chain |
| fourPointNavigation | c1=start.left, c2=start.right | for >2 children, apply four-point tournament across all children pairs; or process as chain of binary splits (see ASTRAL-MP's resolvePolytomy) |
| insertTaxon leaf case | exactly as now | exactly as now |
| insertTaxon internal case | wrap 2 children under newInternal | wrap ALL children under newInternal |
| sortedRows | no change | no change |
| inTree / taxonNode | no change | no change |

The rerooting is the only subtly non-trivial extension: after rerooting at a leaf,
the resulting root can have 2 children (in binary) or exactly 2 children (one=anchor,
one=start) regardless of polytomies in the rest of the tree, because the rerooting
only touches nodes ON the anchor-to-root path and all those nodes remain binary in
the reversal step. Only the collapse of the old root potentially reduces its degree.
So the ASTRAL-MP post-condition (root has 2 children) holds for both binary and n-ary
inputs.
