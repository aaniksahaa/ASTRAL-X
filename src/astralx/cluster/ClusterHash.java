package astralx.cluster;

import astralx.hash.TaxonHasher;
import java.util.Arrays;

/**
 * Immutable hash key for a cluster (set of taxa).
 *
 * Stores 2m values: m finalized sum-hashes and m finalized XOR-hashes.
 * Used as HashMap key; equals/hashCode based on these values.
 *
 * The "finalized" (mixed) values give better bucket distribution.
 * Raw (un-mixed) sum/XOR values are NOT stored here -- they are
 * computed on-the-fly from PrefixHashArrays when needed for arithmetic.
 *
 * Also stores the cluster size for O(1) access.
 */
public final class ClusterHash {

    public final long[] sums;   // sums[s]  = finalized sum  hash under seed s
    public final long[] xors;  // xors[s]  = finalized XOR  hash under seed s
    public final int size;     // number of taxa in this cluster
    private final int cachedHashCode;

    public ClusterHash(long[] rawSums, long[] rawXors, int size, int m) {
        this.size = size;
        this.sums = new long[m];
        this.xors = new long[m];
        for (int s = 0; s < m; s++) {
            // Apply SplitMix64 finalizer to raw values for better hash-table distribution.
            // Raw values are additive; mixed values are not -- use raw for arithmetic.
            this.sums[s] = TaxonHasher.mix64(rawSums[s]);
            this.xors[s] = TaxonHasher.mix64(rawXors[s]);
        }
        // Combine into a single Java hashCode
        int h = 1;
        for (long v : this.sums) h = 31 * h + Long.hashCode(v);
        for (long v : this.xors) h = 31 * h + Long.hashCode(v);
        this.cachedHashCode = h;
    }

    /** Two ClusterHash objects are equal iff all finalized values and size match. */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ClusterHash c)) return false;
        return size == c.size
            && Arrays.equals(sums, c.sums)
            && Arrays.equals(xors, c.xors);
    }

    @Override
    public int hashCode() { return cachedHashCode; }

    @Override
    public String toString() {
        return String.format("CH{size=%d, sum0=%016x, xor0=%016x}", size, sums[0], xors[0]);
    }
}
