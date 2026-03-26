package tree;

/**
 * Represents one side of a bipartition as a cluster interval in a gene tree.
 * 
 * A SideCluster is defined by:
 * - geneTreeIndex: which gene tree this cluster comes from
 * - start: start position in the gene tree's left-to-right ordering (inclusive)
 * - end: end position in the gene tree's left-to-right ordering (exclusive)
 * 
 * This is extracted from observed bipartitions in gene trees and used for
 * cross-tree recombination in candidate extension.
 * 
 * Memory usage: O(1) per side cluster (3 integers + cached hash)
 */
public class SideCluster {
    public final int geneTreeIndex;  // Which gene tree this side comes from
    public final int start;          // Start position (inclusive)
    public final int end;            // End position (exclusive)
    
    // Cached raw hash for efficiency
    private SideClusterHash cachedHash;
    
    public SideCluster(int geneTreeIndex, int start, int end) {
        if (start < 0 || end < start) {
            throw new IllegalArgumentException(
                "Invalid side cluster range: [" + start + ", " + end + ")");
        }
        
        this.geneTreeIndex = geneTreeIndex;
        this.start = start;
        this.end = end;
    }
    
    /**
     * Get the number of taxa in this side cluster.
     */
    public int size() {
        return end - start;
    }
    
    /**
     * Check if this side cluster is empty.
     */
    public boolean isEmpty() {
        return start >= end;
    }
    
    /**
     * Compute the raw hash for this side cluster using prefix arrays.
     * Results are cached for efficiency.
     * 
     * @param prefixSums Prefix sums of hashed taxon IDs [tree][position]
     * @param prefixXORs Prefix XORs of hashed taxon IDs [tree][position]
     * @return Raw hash (unmixed sum and XOR)
     */
    public SideClusterHash computeRawHash(long[][] prefixSums, long[][] prefixXORs) {
        if (cachedHash != null) {
            return cachedHash;
        }
        
        long rawSum = computeRawSum(prefixSums);
        long rawXor = computeRawXOR(prefixXORs);
        
        cachedHash = new SideClusterHash(rawSum, rawXor, size());
        return cachedHash;
    }
    
    /**
     * Compute raw sum hash for this range using prefix sums.
     * This is the actual sum of hashed taxon IDs, not mixed.
     */
    private long computeRawSum(long[][] prefixSums) {
        if (prefixSums == null || geneTreeIndex >= prefixSums.length || geneTreeIndex < 0) {
            return computeFallbackSum();
        }
        
        long[] prefix = prefixSums[geneTreeIndex];
        if (prefix == null || prefix.length == 0) {
            return computeFallbackSum();
        }
        
        // Range sum: sum(start, end) = prefix[end-1] - prefix[start-1]
        if (end <= 0) return 0;
        
        int safeEnd = Math.min(end - 1, prefix.length - 1);
        long endSum = (safeEnd >= 0) ? prefix[safeEnd] : 0;
        long startSum = (start > 0 && start - 1 < prefix.length) ? prefix[start - 1] : 0;
        
        return endSum - startSum;
    }
    
    /**
     * Compute raw XOR hash for this range using prefix XORs.
     * This is the actual XOR of hashed taxon IDs, not mixed.
     */
    private long computeRawXOR(long[][] prefixXORs) {
        if (prefixXORs == null || geneTreeIndex >= prefixXORs.length || geneTreeIndex < 0) {
            return computeFallbackXOR();
        }
        
        long[] prefix = prefixXORs[geneTreeIndex];
        if (prefix == null || prefix.length == 0) {
            return computeFallbackXOR();
        }
        
        // Range XOR: xor(start, end) = prefix[end-1] ^ prefix[start-1]
        if (end <= 0) return 0;
        
        int safeEnd = Math.min(end - 1, prefix.length - 1);
        long endXOR = (safeEnd >= 0) ? prefix[safeEnd] : 0L;
        long startXOR = (start > 0 && start - 1 < prefix.length) ? prefix[start - 1] : 0L;
        
        return endXOR ^ startXOR;
    }
    
    /**
     * Fallback sum when prefix arrays not available.
     */
    private long computeFallbackSum() {
        // Use structural info as fallback
        return ((long) geneTreeIndex << 32) + ((long) start << 16) + end;
    }
    
    /**
     * Fallback XOR when prefix arrays not available.
     */
    private long computeFallbackXOR() {
        // Use structural info as fallback
        return ((long) geneTreeIndex << 32) ^ ((long) start << 16) ^ end;
    }
    
    /**
     * Check if this side cluster has a cached hash.
     */
    public boolean hasCachedHash() {
        return cachedHash != null;
    }
    
    /**
     * Get cached hash (or null if not computed).
     */
    public SideClusterHash getCachedHash() {
        return cachedHash;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof SideCluster)) return false;
        SideCluster other = (SideCluster) obj;
        return this.geneTreeIndex == other.geneTreeIndex &&
               this.start == other.start &&
               this.end == other.end;
    }
    
    @Override
    public int hashCode() {
        return 31 * (31 * geneTreeIndex + start) + end;
    }
    
    @Override
    public String toString() {
        return String.format("SideCluster[tree=%d, range=[%d,%d)]", geneTreeIndex, start, end);
    }
}

