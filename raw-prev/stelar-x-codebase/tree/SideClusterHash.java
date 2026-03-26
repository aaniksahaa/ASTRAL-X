package tree;

/**
 * Raw (unmixed) hash representation for a single side cluster.
 * 
 * Unlike ClusterHashPair which uses mixed hashes, this class stores the RAW
 * sum and XOR values computed directly from prefix arrays. This is essential
 * for complement computation in cross-tree recombination:
 * 
 *   h1(B) = h1(C) - h1(A)  (raw sum is additive)
 *   h2(B) = h2(C) ^ h2(A)  (raw XOR is self-inverse)
 * 
 * The raw hashes enable O(1) complement lookup for finding matching sides.
 */
public class SideClusterHash {
    public final long rawSum;   // Raw sum of hashed taxon IDs (no mixing applied)
    public final long rawXor;   // Raw XOR of hashed taxon IDs (no mixing applied)
    public final int size;      // Number of taxa in this cluster
    
    public SideClusterHash(long rawSum, long rawXor, int size) {
        this.rawSum = rawSum;
        this.rawXor = rawXor;
        this.size = size;
    }
    
    /**
     * Compute the complement hash given a parent cluster hash.
     * If this is side A and parent is C = A ∪ B, returns hash of B.
     * 
     * @param parent The parent cluster (C = A ∪ B)
     * @return The complement hash (B = C - A)
     */
    public SideClusterHash computeComplement(SideClusterHash parent) {
        return new SideClusterHash(
            parent.rawSum - this.rawSum,    // h1(B) = h1(C) - h1(A)
            parent.rawXor ^ this.rawXor,    // h2(B) = h2(C) ^ h2(A)
            parent.size - this.size         // |B| = |C| - |A|
        );
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof SideClusterHash)) return false;
        SideClusterHash other = (SideClusterHash) obj;
        return this.rawSum == other.rawSum && 
               this.rawXor == other.rawXor && 
               this.size == other.size;
    }
    
    @Override
    public int hashCode() {
        // Combine all three fields for good distribution
        int result = (int)(rawSum ^ (rawSum >>> 32));
        result = 31 * result + (int)(rawXor ^ (rawXor >>> 32));
        result = 31 * result + size;
        return result;
    }
    
    @Override
    public String toString() {
        return String.format("SideClusterHash[sum=%d, xor=%d, size=%d]", rawSum, rawXor, size);
    }
    
    public String toDebugString() {
        return String.format("SideHash[sum=0x%016x, xor=0x%016x, n=%d]", rawSum, rawXor, size);
    }
}

