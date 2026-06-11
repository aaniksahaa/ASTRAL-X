package astralx.dp;

import astralx.Config;
import astralx.Logging;
import astralx.cluster.ClusterHash;
import astralx.cluster.ClusterTable;
import astralx.gpu.GPUDPBuilder;
import astralx.hash.PrefixHashArrays;
import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.ProgressBar;
import astralx.util.Threading;

import java.util.*;

/**
 * DP search space: maps each cluster hash to its set of candidate bipartition splits.
 *
 * Built via Mode 1 (tree-local transitions only), O(nk):
 *
 *   Type 1  -- for every internal node u (incl. root):
 *              sub(u) → sub(left(u)) | sub(right(u))
 *
 *   Type 2  -- for every non-root internal node u whose parent is also non-root:
 *              [Lg \ sub(u)] → sub(sibling(u)) | [Lg \ sub(parent(u))]
 *
 * For complete trees (Lg == S), Types 3a/3b add nothing new and are skipped.
 *
 * The root of the DP is the all-taxa cluster (from ClusterTable.getAllTaxaHash()).
 * Its Type 1 transition(s) are stored at that hash key.
 */
public class DPTable {

    // transitions[parentHash] = set of distinct BipartitionSplits for that cluster
    private final Map<ClusterHash, Set<BipartitionSplit>> transitions = new HashMap<>();

    private final ClusterHash rootHash; // allTaxaHash -- starting point of the DP
    private final int m;
    private final int n; // total taxa count

    // stats
    private int totalEmitted = 0;   // total emitted (with duplicates across trees)
    private int uniqueSplits = 0;   // sum of set sizes after dedup

    // -------------------------------------------------------------------------

    public DPTable(List<Tree> trees, PrefixHashArrays pref, ClusterTable clusterTable) {
        long t0 = System.nanoTime();
        this.m        = pref.numSeeds();
        this.rootHash = clusterTable.getAllTaxaHash();
        this.n        = rootHash.size;

        int treesDone = 0;
        ProgressBar localBar = new ProgressBar("Local DP transitions", trees.size());
        for (Tree tree : trees) {
            extractFromTree(tree, pref);
            localBar.update(++treesDone);
        }
        localBar.done();

        // Count total unique splits
        for (Set<BipartitionSplit> s : transitions.values()) uniqueSplits += s.size();

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("DP table: %d clusters with splits, %d unique splits (%d emitted) in %d ms",
            transitions.size(), uniqueSplits, totalEmitted, ms);
    }

    // -------------------------------------------------------------------------
    // Tree traversal
    // -------------------------------------------------------------------------

    private void extractFromTree(Tree tree, PrefixHashArrays pref) {
        int ti = tree.treeIndex;
        int L  = tree.leafCount;
        emit(tree.root, ti, pref);

        // ── Type 3: for incomplete trees add S → Lg | (S\Lg) ─────────────────
        // Connects the DP root (S) to this gene tree's taxa boundary.
        if (!tree.isComplete) {
            ClusterHash hLg  = hashRange(ti, 0, L, false, pref); // hash(Lg)
            ClusterHash hSLg = hashRange(ti, 0, L, true,  pref); // hash(S\Lg)
            if (hSLg.size > 0) {
                addTransition(rootHash, hLg, hSLg);
            }
        }
    }

    /** Post-order recursion: emit transitions for this node, then children. */
    private void emit(TreeNode u, int ti, PrefixHashArrays pref) {
        if (u.isLeaf()) return;

        // Polytomous node: recurse into all children, but add NO direct transitions of
        // its own.  A polytomy is an unresolved node — it must not force any binary
        // resolution into the DP search space; its quartet signal still enters via the
        // d-partition QI weight.  (polytomy-design.md §3.7.)
        if (u.isPolytomous()) {
            for (TreeNode child : u.children) emit(child, ti, pref);
            return;
        }

        emit(u.left,  ti, pref);
        emit(u.right, ti, pref);

        // ── Type 1: sub(u) → sub(left) | sub(right) ─────────────────────────
        ClusterHash hU     = hashRange(ti, u.rangeStart,       u.rangeEnd,       false, pref);
        ClusterHash hLeft  = hashRange(ti, u.left.rangeStart,  u.left.rangeEnd,  false, pref);
        ClusterHash hRight = hashRange(ti, u.right.rangeStart, u.right.rangeEnd, false, pref);
        addTransition(hU, hLeft, hRight);

        // ── Type 2: S\sub(u) → sub(sibling) | S\sub(parent) ─────────────────
        // For non-root u: if parent is root and tree is complete, S\sub(root)=empty (size 0)
        // so hCompParent.size==0 and we skip.  For incomplete trees, S\sub(root) = S\Lg != empty.
        // GUARD: skip when the parent is polytomous — u.getSibling() has no well-defined
        // value for a child of a polytomous node (polytomy-design.md §3.7).  For binary
        // trees no node has a polytomous parent, so this clause is always true (unchanged).
        if (!u.isRoot() && !u.parent.isPolytomous()) {
            TreeNode sib    = u.getSibling();
            TreeNode parent = u.parent;

            ClusterHash hCompU      = hashRange(ti, u.rangeStart,      u.rangeEnd,      true,  pref);
            ClusterHash hSib        = hashRange(ti, sib.rangeStart,    sib.rangeEnd,    false, pref);
            ClusterHash hCompParent = hashRange(ti, parent.rangeStart, parent.rangeEnd, true,  pref);
            if (hCompParent.size > 0) {
                addTransition(hCompU, hSib, hCompParent);
            }
        }
    }

    // -------------------------------------------------------------------------

    private void addTransition(ClusterHash parent, ClusterHash a, ClusterHash b) {
        totalEmitted++;
        BipartitionSplit split = new BipartitionSplit(a, b);
        transitions.computeIfAbsent(parent, k -> new LinkedHashSet<>()).add(split);
    }

    /**
     * Compute a finalized ClusterHash for the range [lo,hi) in tree ti.
     * complement=true gives the super-complement S\[lo,hi) (w.r.t. ALL n taxa).
     */
    private ClusterHash hashRange(int ti, int lo, int hi, boolean complement,
                                   PrefixHashArrays pref) {
        long[] rawSums = new long[m], rawXors = new long[m];
        for (int s = 0; s < m; s++) {
            rawSums[s] = complement ? pref.superCompSum(ti, s, lo, hi) : pref.rangeSum(ti, s, lo, hi);
            rawXors[s] = complement ? pref.superCompXor(ti, s, lo, hi) : pref.rangeXor(ti, s, lo, hi);
        }
        int sz = complement ? (n - (hi - lo)) : (hi - lo);
        return new ClusterHash(rawSums, rawXors, sz, m);
    }

    // -------------------------------------------------------------------------
    // Mode 2: Cross-tree DP transitions
    // -------------------------------------------------------------------------

    /**
     * Expand the search space with cross-tree splits (ASTRAL Mode 2 / "full" search).
     *
     * For every cluster A ∈ X and every cluster B ∈ X with |B| ≤ |A|/2:
     *   if hash(A) − hash(B) matches another cluster R ∈ X  →  add A → B | R.
     *
     * Also handles the all-taxa root cluster (not in X, but its transitions matter).
     *
     * @param clusterTable  the cluster set X
     * @param useGPU        true to use CUDA acceleration; false for parallel CPU
     */
    public void addCrossTreeTransitions(ClusterTable clusterTable, boolean useGPU) {
        long t0 = System.nanoTime();
        int beforeSplits = 0;
        for (Set<BipartitionSplit> s : transitions.values()) beforeSplits += s.size();

        if (useGPU) {
            addCrossTreeGPU(clusterTable);
        } else {
            addCrossTreeCPU(clusterTable);
        }

        // Recount
        uniqueSplits = 0;
        for (Set<BipartitionSplit> s : transitions.values()) uniqueSplits += s.size();

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Cross-tree transitions (Mode 2): +%d splits (%d total) in %d ms",
            uniqueSplits - beforeSplits, uniqueSplits, ms);
    }

    // ── CPU path ─────────────────────────────────────────────────────────────

    private void addCrossTreeCPU(ClusterTable clusterTable) {
        List<ClusterHash> allHashes = new ArrayList<>();
        for (var e : clusterTable.entries()) allHashes.add(e.hash);
        int N = allHashes.size();

        // Parallel over all clusters A: each thread writes to its own perCluster[idx] slot
        @SuppressWarnings("unchecked")
        Set<BipartitionSplit>[] perCluster = new Set[N];

        java.util.concurrent.atomic.AtomicInteger cpuDone = new java.util.concurrent.atomic.AtomicInteger(0);
        ProgressBar cpuBar = new ProgressBar("Cross-tree DP (CPU)", N);
        Threading.processRangeParallel(N, idx -> {
            ClusterHash hashA = allHashes.get(idx);
            int szA = hashA.size;
            Set<BipartitionSplit> localSet = null; // lazy-init to avoid object churn

            for (int sz = 1; sz <= szA / 2; sz++) {
                for (ClusterHash hashB : clusterTable.getBySize(sz)) {
                    ClusterHash residual = ClusterHash.residual(hashA, hashB);
                    if (clusterTable.contains(residual)) {
                        if (localSet == null) localSet = new LinkedHashSet<>();
                        localSet.add(new BipartitionSplit(hashB, residual));
                    }
                }
            }

            if (localSet != null) perCluster[idx] = localSet;
            cpuBar.update(cpuDone.incrementAndGet());
        });
        cpuBar.done();

        // Serial merge into transitions (different A → different keys, no map contention)
        for (int idx = 0; idx < N; idx++) {
            if (perCluster[idx] != null) {
                ClusterHash hashA = allHashes.get(idx);
                transitions.computeIfAbsent(hashA, k -> new LinkedHashSet<>())
                           .addAll(perCluster[idx]);
            }
        }

        // Also handle the root (all-taxa) cluster — not in clusterTable but is the DP root
        searchRootTransitions(clusterTable);
    }

    // ── GPU path ─────────────────────────────────────────────────────────────

    private void addCrossTreeGPU(ClusterTable clusterTable) {
        List<ClusterTable.Entry> entries = new ArrayList<>(clusterTable.entries());
        int N = entries.size();
        if (N == 0) { searchRootTransitions(clusterTable); return; }

        int maxSize = clusterTable.sizes().stream().mapToInt(Integer::intValue).max().orElse(1);

        // ── Flatten cluster data ──────────────────────────────────────────────
        long[] clusterSums  = new long[N * m];
        long[] clusterXors  = new long[N * m];
        int[]  clusterSizes = new int[N];

        for (int c = 0; c < N; c++) {
            ClusterHash h = entries.get(c).hash;
            clusterSizes[c] = h.size;
            for (int s = 0; s < m; s++) {
                clusterSums[c * m + s] = h.sums[s];
                clusterXors[c * m + s] = h.xors[s];
            }
        }

        // ── Build sortedBySize and binStart ───────────────────────────────────
        // sortedBySize[i] = cluster index (into above arrays) ordered by size asc
        // binStart[sz]    = first index in sortedBySize with size >= sz
        Integer[] order = new Integer[N];
        for (int i = 0; i < N; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingInt(i -> clusterSizes[i]));
        int[] sortedBySize = new int[N];
        for (int i = 0; i < N; i++) sortedBySize[i] = order[i];

        int[] binStart = new int[maxSize + 2];
        int ptr = 0;
        for (int sz = 0; sz <= maxSize + 1; sz++) {
            while (ptr < N && clusterSizes[sortedBySize[ptr]] < sz) ptr++;
            binStart[sz] = ptr;
        }

        // ── Compute maxPerRound to bound GPU output buffer ────────────────────
        // Default 120 MB = 10M triples.  Configurable via --gpu-dp-state-space-construction-output-cap.
        // Sub-batching within each round normally ensures this is never exceeded.
        int    maxPerRound    = Config.getInstance().getGpuDpOutputCapTriples();
        double progressInterval  = Config.getInstance().getGpuDpProgressInterval();
        int    progressMaxSteps  = Config.getInstance().getGpuDpProgressMaxSteps();

        // ── Call GPU ──────────────────────────────────────────────────────────
        Logging.debug("  GPU cross-tree search: N=%d clusters, maxSize=%d", N, maxSize);
        int[] raw = GPUDPBuilder.findCrossTreeTransitionsGPU(
            clusterSums, clusterXors, clusterSizes,
            N, m,
            sortedBySize, binStart, maxSize,
            maxPerRound, progressInterval, progressMaxSteps);

        if (raw == null) {
            Logging.info("  GPU cross-tree search returned null, falling back to CPU");
            addCrossTreeCPU(clusterTable);
            return;
        }

        // ── Process GPU results ───────────────────────────────────────────────
        int count = raw[0];
        Logging.debug("  GPU cross-tree: %d raw pairs found", count);
        for (int i = 0; i < count; i++) {
            int idxA   = raw[1 + i * 3];
            int idxB   = raw[1 + i * 3 + 1];
            int idxRes = raw[1 + i * 3 + 2];
            ClusterHash hashA   = entries.get(idxA).hash;
            ClusterHash hashB   = entries.get(idxB).hash;
            ClusterHash hashRes = entries.get(idxRes).hash;
            transitions.computeIfAbsent(hashA, k -> new LinkedHashSet<>())
                       .add(new BipartitionSplit(hashB, hashRes));
        }

        // Root cluster transitions (handled on CPU, fast)
        searchRootTransitions(clusterTable);
    }

    /** Search transitions for the all-taxa root cluster (not in X itself). */
    private void searchRootTransitions(ClusterTable clusterTable) {
        int szRoot = rootHash.size;
        Set<BipartitionSplit> rootSet =
            transitions.computeIfAbsent(rootHash, k -> new LinkedHashSet<>());
        for (int sz = 1; sz <= szRoot / 2; sz++) {
            for (ClusterHash hashB : clusterTable.getBySize(sz)) {
                ClusterHash residual = ClusterHash.residual(rootHash, hashB);
                if (clusterTable.contains(residual)) {
                    rootSet.add(new BipartitionSplit(hashB, residual));
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    public ClusterHash getRootHash()                       { return rootHash; }
    public Set<BipartitionSplit> getSplits(ClusterHash h)  { return transitions.getOrDefault(h, Collections.emptySet()); }
    public boolean hasSplits(ClusterHash h)                { return transitions.containsKey(h); }
    public int numClusters()                               { return transitions.size(); }
    public int numUniqueSplits()                           { return uniqueSplits; }
    public int numEmitted()                                { return totalEmitted; }
    public Set<Map.Entry<ClusterHash, Set<BipartitionSplit>>> entries() { return transitions.entrySet(); }
}
