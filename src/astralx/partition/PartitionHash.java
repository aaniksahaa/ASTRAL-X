package astralx.partition;

import astralx.cluster.ClusterHash;

/**
 * Order-invariant hash key for a gene-tree tripartition.
 *
 * A tripartition (A|B|C) is the same partition regardless of which of A,B,C
 * we call "part1" or "part2".  We store only the two *explicit* parts (the
 * third is implicitly Lg \ A \ B for gene-tree partitions, or S \ A \ B for
 * species-tree tripartitions).  Since the two explicit parts are unordered,
 * we canonicalize by sorting them before combining.
 *
 * The combined hash is a simple integer polynomial over the two sorted
 * ClusterHash objects, giving order-invariant identity.
 */
public final class PartitionHash {

    private final int cachedHashCode;

    /**
     * The two "sorted" finalized hashes that identify this partition.
     * We use lexicographic order on (sum0, xor0, sum1, ...) to canonicalize.
     */
    private final long[] lo;   // the lexicographically smaller cluster's sums then xors
    private final long[] hi;   // the larger

    public PartitionHash(ClusterHash a, ClusterHash b) {
        // Decide ordering: compare element-by-element until difference found
        boolean aFirst = compare(a, b) <= 0;
        ClusterHash first  = aFirst ? a : b;
        ClusterHash second = aFirst ? b : a;

        int m = a.sums.length;
        lo = new long[2 * m];
        hi = new long[2 * m];
        for (int s = 0; s < m; s++) {
            lo[s]     = first.sums[s];
            lo[s + m] = first.xors[s];
            hi[s]     = second.sums[s];
            hi[s + m] = second.xors[s];
        }

        int h = 1;
        for (long v : lo) h = 31 * h + Long.hashCode(v);
        for (long v : hi) h = 31 * h + Long.hashCode(v);
        this.cachedHashCode = h;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PartitionHash p)) return false;
        if (lo.length != p.lo.length) return false;
        for (int i = 0; i < lo.length; i++) {
            if (lo[i] != p.lo[i] || hi[i] != p.hi[i]) return false;
        }
        return true;
    }

    @Override
    public int hashCode() { return cachedHashCode; }

    /** Lexicographic comparison of two ClusterHash objects (sums first, then xors). */
    private static int compare(ClusterHash a, ClusterHash b) {
        int m = a.sums.length;
        for (int s = 0; s < m; s++) {
            int c = Long.compareUnsigned(a.sums[s], b.sums[s]);
            if (c != 0) return c;
        }
        for (int s = 0; s < m; s++) {
            int c = Long.compareUnsigned(a.xors[s], b.xors[s]);
            if (c != 0) return c;
        }
        return 0;
    }
}
