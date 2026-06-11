package astralx.partition;

import astralx.cluster.ClusterHash;

/**
 * Order-invariant hash key for a gene-tree d-partition.
 *
 * BINARY (d=3): a tripartition (M1|M2|M3) is identified by the unordered pair {M1,M2}
 * plus M3.  For complete trees M3 = S\M1\M2 is determined by (M1,M2); for incomplete
 * trees two nodes can share M1,M2 but differ in M3, so h3 is included.  This binary
 * path is kept BYTE-IDENTICAL to the pre-polytomy implementation.
 *
 * POLYTOMOUS (d≥4): a d-partition M0|…|M_{d-1} is identified by the unordered MULTISET
 * of its d part-hashes (ASTRAL-MP's {@code Polytomy} sorts all d clusters before
 * storing — polytomy-design.md §3.5).  We sort the d fingerprints lexicographically and
 * hash the concatenation.
 *
 * The two representations never collide: binary partitions (d=3) and polytomous ones
 * (d≥4) come from disjoint parser paths and {@code equals} short-circuits on {@code d}.
 */
public final class PartitionHash {

    private final int cachedHashCode;
    private final int d;       // number of parts (3 for binary, k+1 for polytomous)

    // ── Binary (d=3) representation — unchanged ──
    private final long[] lo;   // lexicographically smaller of {h1,h2}; null when d≥4
    private final long[] hi;   // the larger; null when d≥4
    private final long[] m3;   // h3 (complement); null when d≥4

    // ── General (d≥4) representation ──
    private final long[] data; // sorted+flattened d fingerprints; null when d==3

    public PartitionHash(ClusterHash a, ClusterHash b, ClusterHash c) {
        // Decide ordering of the two explicit parts (a,b are interchangeable)
        boolean aFirst = compare(a, b) <= 0;
        ClusterHash first  = aFirst ? a : b;
        ClusterHash second = aFirst ? b : a;

        int m = a.sums.length;
        this.d = 3;
        this.data = null;
        lo = new long[2 * m];
        hi = new long[2 * m];
        m3 = new long[2 * m];
        for (int s = 0; s < m; s++) {
            lo[s]     = first.sums[s];
            lo[s + m] = first.xors[s];
            hi[s]     = second.sums[s];
            hi[s + m] = second.xors[s];
            m3[s]     = c.sums[s];
            m3[s + m] = c.xors[s];
        }

        int h = 1;
        for (long v : lo) h = 31 * h + Long.hashCode(v);
        for (long v : hi) h = 31 * h + Long.hashCode(v);
        for (long v : m3) h = 31 * h + Long.hashCode(v);
        this.cachedHashCode = h;
    }

    /**
     * General d-partition key (used for polytomous nodes, d≥4): order-invariant over
     * ALL d parts.  {@code parts[d-1]} is conventionally the complement, but the hash
     * is symmetric so the convention is irrelevant.
     */
    public PartitionHash(ClusterHash[] parts) {
        int dd = parts.length;
        int m = parts[0].sums.length;
        long[][] fps = new long[dd][2 * m];
        for (int i = 0; i < dd; i++) {
            for (int s = 0; s < m; s++) {
                fps[i][s]     = parts[i].sums[s];
                fps[i][s + m] = parts[i].xors[s];
            }
        }
        java.util.Arrays.sort(fps, (x, y) -> {
            for (int s = 0; s < x.length; s++) {
                int c = Long.compareUnsigned(x[s], y[s]);
                if (c != 0) return c;
            }
            return 0;
        });
        long[] flat = new long[dd * 2 * m];
        int p = 0;
        for (long[] fp : fps) for (long v : fp) flat[p++] = v;

        this.d = dd;
        this.data = flat;
        this.lo = null; this.hi = null; this.m3 = null;
        int h = 1;
        for (long v : flat) h = 31 * h + Long.hashCode(v);
        this.cachedHashCode = h;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PartitionHash p)) return false;
        if (d != p.d) return false;
        if (data != null) {
            return java.util.Arrays.equals(data, p.data);
        }
        if (lo.length != p.lo.length) return false;
        for (int i = 0; i < lo.length; i++) {
            if (lo[i] != p.lo[i] || hi[i] != p.hi[i] || m3[i] != p.m3[i]) return false;
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
