package tree;

/**
 * Mixed bipartition representation for cross-tree recombination.
 * 
 * Unlike RangeBipartition where both sides come from the same gene tree,
 * a MixedBipartition allows each side to come from different gene trees:
 * 
 *   A = ordering[i_A][l_A : r_A]   (from gene tree i_A)
 *   B = ordering[i_B][l_B : r_B]   (from gene tree i_B)
 * 
 * This enables constructing candidate bipartitions via cluster-consistent
 * cross-tree recombination, potentially finding better species tree splits
 * that are not present in any single gene tree.
 * 
 * Representation: (i_A, l_A, r_A | i_B, l_B, r_B)
 * Memory usage: O(1) per bipartition (6 integers)
 * 
 * Note: For weight computation, we need to find the closest observed
 * bipartition in UGB (gene tree bipartitions) since mixed bipartitions
 * are not directly in the frequency map.
 */
public class MixedBipartition {
    
    // Left side (A)
    public final int leftTreeIndex;   // Gene tree index for left side
    public final int leftStart;       // Start position in left tree (inclusive)
    public final int leftEnd;         // End position in left tree (exclusive)
    
    // Right side (B)
    public final int rightTreeIndex;  // Gene tree index for right side
    public final int rightStart;      // Start position in right tree (inclusive)
    public final int rightEnd;        // End position in right tree (exclusive)
    
    // Parent cluster hash (C = A ∪ B) for DP mapping
    private SideClusterHash parentClusterHash;
    
    // Cached hash for efficiency
    private int _hash = 0;
    
    /**
     * Create a mixed bipartition from two side clusters.
     */
    public MixedBipartition(SideCluster left, SideCluster right) {
        this(left.geneTreeIndex, left.start, left.end,
             right.geneTreeIndex, right.start, right.end);
    }
    
    /**
     * Create a mixed bipartition with explicit parameters.
     */
    public MixedBipartition(int leftTreeIndex, int leftStart, int leftEnd,
                           int rightTreeIndex, int rightStart, int rightEnd) {
        this.leftTreeIndex = leftTreeIndex;
        this.leftStart = leftStart;
        this.leftEnd = leftEnd;
        this.rightTreeIndex = rightTreeIndex;
        this.rightStart = rightStart;
        this.rightEnd = rightEnd;
    }
    
    /**
     * Check if this is a same-tree bipartition (both sides from same gene tree).
     * If true, this could potentially be represented as a RangeBipartition.
     */
    public boolean isSameTree() {
        return leftTreeIndex == rightTreeIndex;
    }
    
    /**
     * Check if this is a cross-tree bipartition (sides from different gene trees).
     */
    public boolean isCrossTree() {
        return leftTreeIndex != rightTreeIndex;
    }
    
    /**
     * Get the left side as a SideCluster.
     */
    public SideCluster getLeftSide() {
        return new SideCluster(leftTreeIndex, leftStart, leftEnd);
    }
    
    /**
     * Get the right side as a SideCluster.
     */
    public SideCluster getRightSide() {
        return new SideCluster(rightTreeIndex, rightStart, rightEnd);
    }
    
    /**
     * Get left side size (number of taxa).
     */
    public int leftSize() {
        return leftEnd - leftStart;
    }
    
    /**
     * Get right side size (number of taxa).
     */
    public int rightSize() {
        return rightEnd - rightStart;
    }
    
    /**
     * Get total size (number of taxa in union).
     */
    public int totalSize() {
        return leftSize() + rightSize();
    }
    
    /**
     * Set the parent cluster hash for DP mapping.
     */
    public void setParentClusterHash(SideClusterHash hash) {
        this.parentClusterHash = hash;
    }
    
    /**
     * Get the parent cluster hash.
     */
    public SideClusterHash getParentClusterHash() {
        return parentClusterHash;
    }
    
    /**
     * Convert to a RangeBipartition if both sides are from the same tree
     * and form contiguous ranges.
     * 
     * @return RangeBipartition or null if not convertible
     */
    public RangeBipartition toRangeBipartition() {
        if (!isSameTree()) {
            return null; // Cannot convert cross-tree to RangeBipartition
        }
        
        // Check if ranges are contiguous (right starts where left ends)
        if (rightStart == leftEnd) {
            return new RangeBipartition(leftTreeIndex, leftStart, leftEnd, rightStart, rightEnd);
        } else if (leftStart == rightEnd) {
            // Swap order to make contiguous
            return new RangeBipartition(leftTreeIndex, rightStart, rightEnd, leftStart, leftEnd);
        }
        
        return null; // Non-contiguous same-tree bipartition
    }
    
    /**
     * Compute the combined raw hash for this bipartition (union of both sides).
     * This is used for parent cluster matching in DP.
     */
    public SideClusterHash computeUnionHash(long[][] prefixSums, long[][] prefixXORs) {
        SideCluster left = getLeftSide();
        SideCluster right = getRightSide();
        
        SideClusterHash leftHash = left.computeRawHash(prefixSums, prefixXORs);
        SideClusterHash rightHash = right.computeRawHash(prefixSums, prefixXORs);
        
        // Union hash: sum the sums, XOR the XORs, add the sizes
        return new SideClusterHash(
            leftHash.rawSum + rightHash.rawSum,
            leftHash.rawXor ^ rightHash.rawXor,
            leftHash.size + rightHash.size
        );
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MixedBipartition)) return false;
        MixedBipartition other = (MixedBipartition) obj;
        
        // Check direct match
        boolean directMatch = 
            this.leftTreeIndex == other.leftTreeIndex &&
            this.leftStart == other.leftStart &&
            this.leftEnd == other.leftEnd &&
            this.rightTreeIndex == other.rightTreeIndex &&
            this.rightStart == other.rightStart &&
            this.rightEnd == other.rightEnd;
        
        if (directMatch) return true;
        
        // Check symmetric match (left<->right swapped)
        boolean symmetricMatch = 
            this.leftTreeIndex == other.rightTreeIndex &&
            this.leftStart == other.rightStart &&
            this.leftEnd == other.rightEnd &&
            this.rightTreeIndex == other.leftTreeIndex &&
            this.rightStart == other.leftStart &&
            this.rightEnd == other.leftEnd;
        
        return symmetricMatch;
    }
    
    @Override
    public int hashCode() {
        if (_hash == 0) {
            // Symmetric hash - same regardless of left/right order
            int leftHash = 31 * (31 * leftTreeIndex + leftStart) + leftEnd;
            int rightHash = 31 * (31 * rightTreeIndex + rightStart) + rightEnd;
            
            // Use min/max to ensure symmetric hashing
            _hash = Math.min(leftHash, rightHash) * 37 + Math.max(leftHash, rightHash);
            if (_hash == 0) _hash = 1;
        }
        return _hash;
    }
    
    @Override
    public String toString() {
        return String.format("MixedBipartition[left=(tree=%d,[%d,%d)), right=(tree=%d,[%d,%d))]",
                           leftTreeIndex, leftStart, leftEnd,
                           rightTreeIndex, rightStart, rightEnd);
    }
    
    /**
     * Detailed string for debugging.
     */
    public String toDebugString() {
        String type = isCrossTree() ? "CROSS-TREE" : "SAME-TREE";
        return String.format("%s: (i=%d, [%d,%d)) | (i=%d, [%d,%d)) [sizes: %d|%d, total: %d]",
                           type,
                           leftTreeIndex, leftStart, leftEnd,
                           rightTreeIndex, rightStart, rightEnd,
                           leftSize(), rightSize(), totalSize());
    }
}

