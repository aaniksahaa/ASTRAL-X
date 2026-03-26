package astralx.hash;

import java.util.Arrays;

public final class ClusterHashVector {
    public final long[] sumHash;
    public final long[] xorHash;

    public ClusterHashVector(int m) {
        this.sumHash = new long[m];
        this.xorHash = new long[m];
    }

    public ClusterHashVector(long[] sumHash, long[] xorHash) {
        this.sumHash = sumHash;
        this.xorHash = xorHash;
    }

    public ClusterHashVector copy() {
        return new ClusterHashVector(Arrays.copyOf(sumHash, sumHash.length), Arrays.copyOf(xorHash, xorHash.length));
    }

    public static ClusterHashVector subtract(ClusterHashVector a, ClusterHashVector b) {
        int m = a.sumHash.length;
        long[] sum = new long[m];
        long[] xor = new long[m];
        for (int i = 0; i < m; i++) {
            sum[i] = a.sumHash[i] - b.sumHash[i];
            xor[i] = a.xorHash[i] ^ b.xorHash[i];
        }
        return new ClusterHashVector(sum, xor);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ClusterHashVector)) {
            return false;
        }
        ClusterHashVector other = (ClusterHashVector) obj;
        return Arrays.equals(sumHash, other.sumHash) && Arrays.equals(xorHash, other.xorHash);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(sumHash) + Arrays.hashCode(xorHash);
    }
}
