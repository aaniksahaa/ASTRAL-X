package astralx.tree;

/**
 * Node in a rooted binary gene tree.
 * After parsing, every node has [rangeStart, rangeEnd) covering its leaf positions
 * in the tree's left-to-right postorder array.
 */
public class TreeNode {
    public TreeNode left, right, parent;

    /** Taxon ID for leaves; -1 for internal nodes. */
    public int taxonId = -1;

    /**
     * Half-open range [rangeStart, rangeEnd) in the tree's postorder array.
     * For a leaf: rangeEnd = rangeStart + 1.
     * For an internal node: spans entire descendant leaf range.
     */
    public int rangeStart = -1;
    public int rangeEnd   = -1;

    public boolean isLeaf() { return left == null; }
    public boolean isRoot() { return parent == null; }
    public int rangeSize()  { return rangeEnd - rangeStart; }

    /** The other child of our parent (null if we are root). */
    public TreeNode getSibling() {
        if (parent == null) return null;
        return (parent.left == this) ? parent.right : parent.left;
    }
}
