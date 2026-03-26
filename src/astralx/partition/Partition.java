package astralx.partition;

import astralx.cluster.ClusterHash;

/**
 * A gene-tree tripartition: three disjoint non-empty groups of taxa.
 *
 * For a non-root internal node u of a gene tree g:
 *   part1 = left-subtree  of u  (range [u.left.start, u.left.end))
 *   part2 = right-subtree of u  (range [u.right.start, u.right.end))
 *   part3 = Lg \ (part1 ∪ part2)   -- the "everything else in this tree"
 *
 * Only part1 and part2 are stored explicitly.  part3 is recovered on demand
 * since its hash = totalHash(tree) - hash(part1) - hash(part2).
 *
 * The sizes of all three parts are stored for O(1) QI computation setup.
 */
public final class Partition {

    /** Hash of part1 (left subtree exemplar). */
    public final ClusterHash hash1;

    /** Hash of part2 (right subtree exemplar). */
    public final ClusterHash hash2;

    /** Hash of part3 = Lg \ (part1 ∪ part2). */
    public final ClusterHash hash3;

    /** Sizes of the three parts. */
    public final int size1, size2, size3;

    /** Index of the gene tree this partition came from (of the first exemplar). */
    public final int treeIndex;

    /** Range of the left child in the exemplar tree. */
    public final int leftStart, leftEnd;

    /** Range of the right child in the exemplar tree. */
    public final int rightStart, rightEnd;

    public Partition(ClusterHash h1, ClusterHash h2, ClusterHash h3,
                     int sz1, int sz2, int sz3,
                     int treeIndex,
                     int leftStart, int leftEnd,
                     int rightStart, int rightEnd) {
        this.hash1 = h1; this.hash2 = h2; this.hash3 = h3;
        this.size1 = sz1; this.size2 = sz2; this.size3 = sz3;
        this.treeIndex = treeIndex;
        this.leftStart = leftStart; this.leftEnd = leftEnd;
        this.rightStart = rightStart; this.rightEnd = rightEnd;
    }

    @Override
    public String toString() {
        return String.format("Partition{sz=%d|%d|%d, t=%d, L=[%d,%d) R=[%d,%d)}",
            size1, size2, size3, treeIndex,
            leftStart, leftEnd, rightStart, rightEnd);
    }
}
