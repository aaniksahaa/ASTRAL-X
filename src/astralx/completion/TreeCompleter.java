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
 * Inserts missing taxa into incomplete gene trees using a distance-matrix guided
 * greedy descent.
 *
 * Algorithm for inserting taxon x into tree T:
 *   1. Navigate from root toward a leaf: at each internal node, compare
 *      D[x][leftmostLeaf(left)] vs D[x][leftmostLeaf(right)] and descend
 *      toward the smaller distance.
 *   2. Graft x as a new sibling of the arrived leaf: insert a new internal
 *      node between that leaf and its parent, with the leaf and x as children.
 *
 * Multiple missing taxa for the same tree are inserted sequentially (each
 * modifies the live TreeNode structure before the next insert).
 * Different trees are completed in parallel using the thread pool.
 *
 * After all insertions for a tree the postorderArray / positionMap / range fields
 * are rebuilt from scratch, producing a valid new Tree object.
 */
public class TreeCompleter {

    /**
     * Complete all incomplete trees in the list.
     *
     * @param trees  gene trees (may mix complete and incomplete)
     * @param dm     normalized distance matrix (D[a][b] = avg topological distance)
     * @param n      total taxon count
     * @return new list where every tree is complete; already-complete trees pass through
     */
    public static List<Tree> completeAll(List<Tree> trees, DistanceMatrix dm, int n) {
        List<Integer> incomplete = new ArrayList<>();
        for (int i = 0; i < trees.size(); i++) {
            if (!trees.get(i).isComplete) incomplete.add(i);
        }

        if (incomplete.isEmpty()) return trees;

        Logging.info("Tree completion: %d/%d trees incomplete", incomplete.size(), trees.size());

        // Mutable result array; complete trees pass through unchanged
        Tree[] result = trees.toArray(new Tree[0]);

        ProgressBar bar  = new ProgressBar("Completing incomplete gene trees", incomplete.size());
        AtomicInteger cnt = new AtomicInteger(0);

        // Each tree's completion is fully independent → safe to parallelise
        Threading.processParallel(incomplete, idx -> {
            result[idx] = completeTree(trees.get(idx), dm, n);
            bar.update(cnt.incrementAndGet());
        });
        bar.done();

        return Arrays.asList(result);
    }

    // ── Per-tree completion ───────────────────────────────────────────────────

    /** Insert every taxon missing from this tree and return a rebuilt Tree. */
    private static Tree completeTree(Tree tree, DistanceMatrix dm, int n) {
        List<Integer> missing = new ArrayList<>();
        for (int x = 0; x < n; x++) {
            if (tree.positionMap[x] == -1) missing.add(x);
        }

        // Deep-copy tree nodes before any mutation so the original Tree's nodes
        // remain unmodified.  This preserves originalTrees' rangeStart/rangeEnd
        // values, which PrefixHashArrays (prefParts) and PartitionTable rely on.
        TreeNode root = deepCopyNodes(tree.root, null);
        for (int x : missing) {
            root = insertTaxon(root, x, dm.dist, n);
        }

        return rebuildTree(tree.treeIndex, root, n);
    }

    /**
     * Recursively deep-copy a TreeNode subtree.
     * The copies are fresh objects with the same taxonId/rangeStart/rangeEnd
     * but independent parent/left/right pointers.
     */
    private static TreeNode deepCopyNodes(TreeNode src, TreeNode parent) {
        if (src == null) return null;
        TreeNode copy = new TreeNode();
        copy.taxonId    = src.taxonId;
        copy.rangeStart = src.rangeStart;
        copy.rangeEnd   = src.rangeEnd;
        copy.parent     = parent;
        copy.left  = deepCopyNodes(src.left,  copy);
        copy.right = deepCopyNodes(src.right, copy);
        return copy;
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    /**
     * Insert taxon x into the tree rooted at root using greedy distance descent.
     *
     * At each internal node the leftmost leaf of each child subtree is used as
     * a representative; we descend toward the child whose representative is
     * closer to x in the distance matrix.
     *
     * @param root current root (may change if it was a lone leaf)
     * @param x    taxon ID to insert
     * @param dist flat n×n distance matrix (dist[a*n + b] = D(a,b))
     * @param n    total taxon count
     * @return root of the updated tree
     */
    private static TreeNode insertTaxon(TreeNode root, int x, double[] dist, int n) {
        // Navigate to landing leaf
        TreeNode node = root;
        while (!node.isLeaf()) {
            int    leftRep  = leftmostTaxon(node.left);
            int    rightRep = leftmostTaxon(node.right);
            double dLeft    = dist[x * n + leftRep];
            double dRight   = dist[x * n + rightRep];
            node = (dLeft <= dRight) ? node.left : node.right;
        }

        // Save original parent before grafting
        TreeNode landedParent = node.parent;

        // Build new leaf and new internal node
        TreeNode newLeaf     = new TreeNode();
        newLeaf.taxonId = x;

        TreeNode newInternal = new TreeNode();
        newInternal.left   = node;
        newInternal.right  = newLeaf;
        node.parent        = newInternal;
        newLeaf.parent     = newInternal;
        newInternal.parent = landedParent;

        if (landedParent != null) {
            if (landedParent.left == node) landedParent.left  = newInternal;
            else                           landedParent.right = newInternal;
        }

        return (landedParent == null) ? newInternal : root;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /** Walk left-child pointers until a leaf; return its taxonId. O(depth). */
    private static int leftmostTaxon(TreeNode node) {
        while (!node.isLeaf()) node = node.left;
        return node.taxonId;
    }

    // ── Rebuild Tree ─────────────────────────────────────────────────────────

    /** Reconstruct a Tree object from the mutated TreeNode structure. */
    private static Tree rebuildTree(int treeIndex, TreeNode root, int n) {
        int[] postorderArray = new int[n];
        int[] counter = {0};
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
            node.rangeStart = counter[0];
            node.rangeEnd   = counter[0] + 1;
            arr[counter[0]] = node.taxonId;
            counter[0]++;
            return;
        }
        assignRangesAndFill(node.left,  arr, counter);
        assignRangesAndFill(node.right, arr, counter);
        node.rangeStart = node.left.rangeStart;
        node.rangeEnd   = node.right.rangeEnd;
    }
}
