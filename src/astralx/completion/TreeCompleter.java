package astralx.completion;

import astralx.Logging;
import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.ProgressBar;
import astralx.util.Threading;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Inserts missing taxa into incomplete gene trees using the ASTRAL-MP compatible
 * four-point tree completion algorithm.
 *
 * Algorithm for inserting taxon x into tree T:
 *   1. Find anchor a = closest taxon currently present in T (via sortedRows).
 *   2. Physically reroot T at the edge (anchor, anchor.parent) so that anchor
 *      is newRoot.left and the rest of the tree (start) is newRoot.right.
 *   3. Navigate start using the four-point condition with x, a, c1rep, c2rep:
 *      - ascore >= both others → insert at this internal node (x groups with anchor)
 *      - bscore wins  → descend into c1 subtree
 *      - cscore wins  → descend into c2 subtree
 *      - start.isLeaf() → stop, graft x as sibling of that leaf
 *   4. Insert x and update inTree / taxonNode maps.
 *
 * Missing taxa are processed in ascending ID order for each tree.
 * Different trees are completed in parallel using the thread pool.
 *
 * Primary reference: WQDataCollection.getCompleteTree() in
 *   astral-mp-legacy-codebase/WQDataCollection.java lines 261–341.
 * Four-point formula: SimilarityMatrix.getBetterSideByFourPoint() in
 *   astral-mp-legacy-codebase/SimilarityMatrix.java lines 45–58.
 * Anchor finding: AbstractMatrix.getClosestPresentTaxonId() in
 *   astral-mp-legacy-codebase/AbstractMatrix.java lines 52–67.
 */
public class TreeCompleter {

    /**
     * Complete all incomplete trees in the list using the four-point algorithm.
     *
     * @param trees      gene trees (may mix complete and incomplete)
     * @param sim        flat n×n similarity matrix (sim[a*n+b] ∈ [0,1]; used for four-point)
     * @param dist       flat n×n distance matrix (dist = 1-sim; used for sortedRows)
     * @param n          total taxon count
     * @return new list where every tree is complete; already-complete trees pass through
     */
    public static List<Tree> completeAll(List<Tree> trees, double[] sim, double[] dist, int n) {
        List<Integer> incomplete = new ArrayList<>();
        for (int i = 0; i < trees.size(); i++) {
            if (!trees.get(i).isComplete) incomplete.add(i);
        }

        if (incomplete.isEmpty()) return trees;

        Logging.info("Tree completion: %d/%d trees incomplete", incomplete.size(), trees.size());

        // Build sortedRows once — shared read-only across all tree completions.
        // sortedRows[x*n + rank] = taxon ID of x's rank-th nearest neighbor (ascending dist).
        int[] sortedRows = SortedRowsBuilder.buildCPU(dist, n);

        // Mutable result array; complete trees pass through unchanged.
        Tree[] result = trees.toArray(new Tree[0]);

        ProgressBar bar   = new ProgressBar("Completing incomplete gene trees", incomplete.size());
        AtomicInteger cnt = new AtomicInteger(0);

        // Each tree's completion is fully independent → safe to parallelise.
        Threading.processParallel(incomplete, idx -> {
            result[idx] = completeTreeFourPoint(trees.get(idx), sim, sortedRows, n);
            bar.update(cnt.incrementAndGet());
        });
        bar.done();

        return Arrays.asList(result);
    }

    // ── Per-tree completion ───────────────────────────────────────────────────

    /**
     * Insert every taxon missing from this tree using the four-point algorithm
     * and return a rebuilt Tree.
     *
     * Mirrors WQDataCollection.getCompleteTree() lines 261–341.
     */
    private static Tree completeTreeFourPoint(Tree tree, double[] sim,
                                              int[] sortedRows, int n) {
        // --- Setup ---
        boolean[]  inTree    = new boolean[n];
        TreeNode[] taxonNode = new TreeNode[n];

        // Deep-copy tree nodes before any mutation; also populates taxonNode.
        TreeNode root = deepCopyNodes(tree.root, null, taxonNode);

        // Initialise inTree from the original positionMap.
        for (int i = 0; i < n; i++) {
            if (tree.positionMap[i] != -1) inTree[i] = true;
        }

        // --- Insert each missing taxon in ascending ID order ---
        // (mirrors WQDataCollection loop: nextClearBit ascending)
        for (int x = 0; x < n; x++) {
            if (inTree[x]) continue;   // already present

            // Phase B: find anchor (closest in-tree taxon)
            // Mirrors AbstractMatrix.getClosestPresentTaxonId() lines 52–67.
            int anchor = findAnchor(x, inTree, sortedRows, n);

            // Phase C: physically reroot at edge (anchorLeaf, anchorLeaf.parent)
            // After rerooting: newRoot.left = anchorLeaf, newRoot.right = start.
            // Mirrors trc.rerootTreeAtNode(closestNode) + removeBinaryNodes in WQDataCollection.java:280.
            TreeNode anchorLeaf = taxonNode[anchor];
            root = rerootAtLeafEdge(anchorLeaf, root);

            // Phase D: four-point navigation
            // Mirrors the while(true) loop in WQDataCollection.java lines 290–323.
            TreeNode start = root.right;   // non-anchor child = "rest of tree"
            int c1rep = -1, c2rep = -1;
            TreeNode c1 = null, c2 = null;

            while (!start.isLeaf()) {
                // HINT for n-ary: c1 = children.get(0), c2 = children.get(1)
                c1 = start.left;
                c2 = start.right;

                if (c1rep == -1) c1rep = leftmostTaxon(c1);
                if (c2rep == -1) c2rep = leftmostTaxon(c2);

                int better = fourPointBetterSide(x, anchor, c1rep, c2rep, sim, n);

                if (better == anchor) {
                    // x groups with anchor → insert at this internal node.
                    break;
                } else if (better == c1rep) {
                    // Descend into left child.
                    // c1rep is still valid for the new left child (leftmost is preserved).
                    start = c1;
                    c2rep = -1;   // right side changes at next level
                } else {
                    // better == c2rep: descend into right child.
                    // c2's leftmost becomes the new c1rep.
                    start = c2;
                    c1rep = c2rep;
                    c2rep = -1;
                }
            }

            // Phase E: insert taxon x
            // c1/c2 hold the last-seen children (used for internal-node case).
            // Mirrors WQDataCollection.java lines 325–337.
            TreeNode newLeaf = insertTaxon(x, start, c1, c2);

            // Phase F: update membership so future iterations can use x as anchor.
            inTree[x]    = true;
            taxonNode[x] = newLeaf;
        }

        return rebuildTree(tree.treeIndex, root, n);
    }

    // ── Anchor finding ────────────────────────────────────────────────────────

    /**
     * Find the closest taxon to x that is currently in the tree.
     *
     * Scans sortedRows[x] in ascending distance order and returns the first
     * in-tree taxon found.
     *
     * Mirrors AbstractMatrix.getClosestPresentTaxonId() lines 52–67.
     * The original checks (missingId > other || presentBS.get(other)); here we
     * unify both conditions into inTree[candidate] which is true for both
     * originally-present and already-inserted taxa.
     */
    private static int findAnchor(int x, boolean[] inTree, int[] sortedRows, int n) {
        int base = x * n;
        for (int rank = 0; rank < n; rank++) {
            int candidate = sortedRows[base + rank];
            if (candidate != x && inTree[candidate]) {
                return candidate;
            }
        }
        throw new RuntimeException("No anchor found for taxon " + x
                + " — inTree array may be empty");
    }

    // ── Four-point score ──────────────────────────────────────────────────────

    /**
     * Determine which of {a (anchor), b (c1rep), c (c2rep)} taxon x groups with,
     * using the four-point condition on the similarity matrix.
     *
     * Scores (higher = better grouping):
     *   ascore = sim[x][a] + sim[b][c] − sim[x][b] − sim[a][c]   (x with anchor)
     *   bscore = sim[x][b] + sim[a][c] − sim[x][a] − sim[b][c]   (x with c1rep)
     *   cscore = sim[x][c] + sim[a][b] − sim[x][b] − sim[a][c]   (x with c2rep)
     *
     * Returns the taxon ID (a, b, or c) of the winning side.
     *
     * Mirrors SimilarityMatrix.getBetterSideByFourPoint() lines 45–58.
     */
    private static int fourPointBetterSide(int x, int a, int b, int c,
                                           double[] sim, int n) {
        double xa = sim[x * n + a];
        double xb = sim[x * n + b];
        double xc = sim[x * n + c];
        double ab = sim[a * n + b];
        double ac = sim[a * n + c];
        double bc = sim[b * n + c];

        double ascore = xa + bc - (xb + ac);
        double bscore = xb + ac - (xa + bc);
        double cscore = xc + ab - (xb + ac);

        // Mirrors the ternary in SimilarityMatrix.getBetterSideByFourPoint exactly.
        return ascore >= bscore
                ? (ascore >= cscore ? a : c)
                : (bscore >= cscore ? b : c);
    }

    // ── Physical rerooting ────────────────────────────────────────────────────

    /**
     * Physically reroot the tree at the edge between anchorLeaf and its parent.
     *
     * After this call:
     *   newRoot.left  = anchorLeaf
     *   newRoot.right = p1 (the former parent of anchorLeaf, now head of the
     *                   "start" subtree containing everything except anchorLeaf)
     *
     * Algorithm (path reversal):
     *   Collect path from anchorLeaf to old root: [anchor, p1, p2, ..., pk].
     *   Allocate newRoot with left=anchor, right=p1.
     *   Reverse parent→child edges along the path.
     *   Collapse the old root (it becomes a unary node after reversal and is
     *   spliced out so the tree stays binary).
     *
     * Mirrors WQDataCollection.java:280 (trc.rerootTreeAtNode + removeBinaryNodes).
     *
     * Complexity: O(depth) time and space.
     *
     * HINT for n-ary extension: in step 2, replace left/right slot with an entry
     * in node.children list. In the collapse step, remove the old-root entry from
     * path[k-1].children.
     *
     * @param anchorLeaf the leaf node that will become newRoot.left
     * @param oldRoot    the current root of the tree (needed only to detect path end)
     * @return the new root node
     */
    static TreeNode rerootAtLeafEdge(TreeNode anchorLeaf, TreeNode oldRoot) {
        // Build path from anchorLeaf up to (and including) old root.
        // path[0] = anchorLeaf, path[k] = oldRoot.
        List<TreeNode> path = new ArrayList<>();
        TreeNode cur = anchorLeaf;
        while (cur != null) {
            path.add(cur);
            if (cur == oldRoot) break;
            cur = cur.parent;
        }
        int k = path.size() - 1;   // index of old root

        // Allocate new root: left = anchor, right = p1.
        TreeNode newRoot = new TreeNode();
        TreeNode p1      = path.get(1);   // former parent of anchorLeaf
        newRoot.left     = anchorLeaf;
        newRoot.right    = p1;
        anchorLeaf.parent = newRoot;

        // Special case: anchorLeaf is a direct child of old root (path length = 2).
        // path = [anchor, root].  After newRoot.right = p1 = root, we need to
        // collapse the old root (now unary after anchor is stolen) into its
        // remaining child.
        if (k == 1) {
            // p1 == oldRoot.  Find the sibling of anchor under old root.
            TreeNode sib = (oldRoot.left == anchorLeaf) ? oldRoot.right : oldRoot.left;
            // Replace old root with sib directly.
            newRoot.right  = sib;
            sib.parent     = newRoot;
            // Old root is discarded.
            return newRoot;
        }

        // General case: path length >= 3 (anchor, p1, ..., pk=oldRoot).
        // Step 2: reverse edges for nodes p1 .. pk-1 (indices 1 .. k-1).
        for (int i = 1; i <= k - 1; i++) {
            TreeNode node         = path.get(i);
            TreeNode childOnPath  = path.get(i - 1);   // toward anchor (keep as child)
            TreeNode parentOnPath = path.get(i + 1);   // old parent (becomes new child)

            // Replace the child slot pointing toward anchor with the old parent.
            // HINT for n-ary: replace matching entry in node.children list.
            if (node.left == childOnPath) {
                node.left = parentOnPath;
            } else {
                node.right = parentOnPath;
            }

            // Fix parent pointers.
            node.parent        = (i == 1) ? newRoot : path.get(i - 1);
            parentOnPath.parent = node;
        }

        // Step 3: collapse old root pk.
        // pk had two children: path[k-1] (now pk's parent after step 2) and remainingChild.
        // Splice pk out: make remainingChild a direct child of path[k-1].
        TreeNode pk             = path.get(k);   // old root
        TreeNode pathKm1        = path.get(k - 1);
        // HINT for n-ary: scan pk.children for the entry that is NOT path[k-1].
        TreeNode remainingChild = (pk.left == pathKm1) ? pk.right : pk.left;

        // Replace pk in pathKm1's children with remainingChild.
        // HINT for n-ary: replace matching entry in pathKm1.children list.
        if (pathKm1.left == pk) {
            pathKm1.left = remainingChild;
        } else {
            pathKm1.right = remainingChild;
        }
        remainingChild.parent = pathKm1;
        // pk is now unreachable and will be garbage-collected.

        return newRoot;
    }

    // ── Insertion ─────────────────────────────────────────────────────────────

    /**
     * Insert taxon x at the position determined by the navigation loop.
     *
     * Two cases (mirrors WQDataCollection.java lines 325–337):
     *
     * Case 1 — stopped at a leaf (start.isLeaf()):
     *   Graft x as a sibling of start under a new internal node.
     *     start.parent → newInternal → {start, newLeaf(x)}
     *
     * Case 2 — stopped at an internal node (betterSide == anchor):
     *   x becomes a new direct child of start; c1 and c2 are wrapped under
     *   a new internal node as the other child.
     *     start → {newLeaf(x), newInternal → {c1, c2}}
     *
     * HINT for n-ary: in case 2, wrap ALL existing children under newInternal:
     *   newInternal.children = start.children; start.children = [newLeaf, newInternal].
     *
     * @param x     taxon ID to insert
     * @param start node where navigation stopped
     * @param c1    start.left as of the last loop iteration (used in case 2 only)
     * @param c2    start.right as of the last loop iteration (used in case 2 only)
     * @return the newly created leaf node for taxon x
     */
    private static TreeNode insertTaxon(int x, TreeNode start,
                                        TreeNode c1, TreeNode c2) {
        TreeNode newLeaf = new TreeNode();
        newLeaf.taxonId  = x;

        if (start.isLeaf()) {
            // Case 1: stopped at a leaf — graft as sibling.
            TreeNode newInternal = new TreeNode();
            TreeNode p           = start.parent;

            newInternal.left   = start;
            newInternal.right  = newLeaf;
            newInternal.parent = p;
            start.parent       = newInternal;
            newLeaf.parent     = newInternal;

            // p should never be null here: after rerooting the tree has at least
            // anchor on one side and start on the other, so start is never the root.
            if (p != null) {
                if (p.left == start) p.left  = newInternal;
                else                 p.right = newInternal;
            }
        } else {
            // Case 2: stopped at an internal node — push children down.
            // c1 and c2 are start.left / start.right as captured by the loop.
            TreeNode newInternal = new TreeNode();

            newInternal.left   = c1;
            newInternal.right  = c2;
            newInternal.parent = start;
            c1.parent          = newInternal;
            c2.parent          = newInternal;

            start.left  = newLeaf;
            start.right = newInternal;
            newLeaf.parent = start;
        }

        return newLeaf;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /**
     * Walk left-child pointers until a leaf; return its taxonId.
     *
     * Used both for computing c1rep/c2rep during navigation and as a utility
     * inside deepCopyNodes.
     *
     * Complexity: O(depth).
     *
     * HINT for n-ary: walk children.get(0) instead of node.left.
     */
    private static int leftmostTaxon(TreeNode node) {
        while (!node.isLeaf()) node = node.left;
        return node.taxonId;
    }

    // ── Deep copy ────────────────────────────────────────────────────────────

    /**
     * Recursively deep-copy a TreeNode subtree, also populating taxonNode[].
     *
     * The copies are fresh objects with the same taxonId/rangeStart/rangeEnd
     * but independent parent/left/right pointers — mutations to the copy do
     * not affect the original Tree's nodes (preserving originalTrees' range
     * fields used by PrefixHashArrays and PartitionTable).
     *
     * taxonNode[id] is set to the leaf copy for each leaf encountered.
     *
     * HINT for n-ary: copy node.children list here instead of left/right.
     *
     * @param src       source node
     * @param parent    parent of the copy (null for root)
     * @param taxonNode map from taxon ID → leaf node; populated for leaves
     * @return root of the copied subtree
     */
    private static TreeNode deepCopyNodes(TreeNode src, TreeNode parent,
                                          TreeNode[] taxonNode) {
        if (src == null) return null;
        TreeNode copy    = new TreeNode();
        copy.taxonId     = src.taxonId;
        copy.rangeStart  = src.rangeStart;
        copy.rangeEnd    = src.rangeEnd;
        copy.parent      = parent;
        copy.left        = deepCopyNodes(src.left,  copy, taxonNode);
        copy.right       = deepCopyNodes(src.right, copy, taxonNode);
        if (copy.isLeaf()) {
            taxonNode[copy.taxonId] = copy;
        }
        return copy;
    }

    // ── Rebuild Tree ──────────────────────────────────────────────────────────

    /** Reconstruct a Tree object from the mutated TreeNode structure. */
    private static Tree rebuildTree(int treeIndex, TreeNode root, int n) {
        int[] postorderArray = new int[n];
        int[] counter        = {0};
        assignRangesAndFill(root, postorderArray, counter);
        int leafCount = counter[0];
        postorderArray = Arrays.copyOf(postorderArray, leafCount);

        int[] positionMap = new int[n];
        Arrays.fill(positionMap, -1);
        for (int j = 0; j < leafCount; j++) positionMap[postorderArray[j]] = j;

        return new Tree(treeIndex, root, postorderArray, positionMap, leafCount, n);
    }

    private static void assignRangesAndFill(TreeNode node, int[] arr, int[] counter) {
        if (node.isLeaf()) {
            node.rangeStart  = counter[0];
            node.rangeEnd    = counter[0] + 1;
            arr[counter[0]]  = node.taxonId;
            counter[0]++;
            return;
        }
        assignRangesAndFill(node.left,  arr, counter);
        assignRangesAndFill(node.right, arr, counter);
        node.rangeStart = node.left.rangeStart;
        node.rangeEnd   = node.right.rangeEnd;
    }
}
