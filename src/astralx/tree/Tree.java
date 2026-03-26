package astralx.tree;

import astralx.taxon.TaxonRegistry;

/**
 * A parsed rooted binary gene tree with postorder (L-to-R leaf) array
 * and inverse position map.
 */
public class Tree {
    /** Index in the gene tree list (0..k-1). */
    public final int treeIndex;

    /** Root of the tree. */
    public final TreeNode root;

    /**
     * postorderArray[pos] = taxon ID at left-to-right position pos.
     * Length = leafCount.
     */
    public final int[] postorderArray;

    /**
     * positionMap[taxonId] = position in postorderArray (-1 if absent).
     * Length = total taxa count n.
     */
    public final int[] positionMap;

    /** Number of leaves in this tree. */
    public final int leafCount;

    /** True when this tree contains all n taxa. */
    public final boolean isComplete;

    public Tree(int treeIndex, TreeNode root,
                int[] postorderArray, int[] positionMap,
                int leafCount, int totalTaxa) {
        this.treeIndex = treeIndex;
        this.root = root;
        this.postorderArray = postorderArray;
        this.positionMap = positionMap;
        this.leafCount = leafCount;
        this.isComplete = (leafCount == totalTaxa);
    }

    /** Reconstruct Newick string (no branch lengths). */
    public String toNewick(TaxonRegistry reg) {
        return nodeToNewick(root, reg) + ";";
    }

    private String nodeToNewick(TreeNode n, TaxonRegistry reg) {
        if (n.isLeaf()) return reg.getName(n.taxonId);
        return "(" + nodeToNewick(n.left, reg) + "," + nodeToNewick(n.right, reg) + ")";
    }
}
