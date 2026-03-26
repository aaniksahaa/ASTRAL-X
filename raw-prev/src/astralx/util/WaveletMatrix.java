package astralx.util;

/**
 * Integer wavelet matrix supporting range frequency queries.
 * Values are expected in [0, maxValueExclusive).
 */
public final class WaveletMatrix {
    private final int levels;
    private final int length;
    private final int[][] prefixOnes; // per level, prefix count of ones
    private final int[] zeroCounts;   // per level, number of zeros

    public WaveletMatrix(int[] data, int maxValueExclusive) {
        this.length = data.length;
        int maxBits = 0;
        int x = Math.max(1, maxValueExclusive - 1);
        while (x > 0) {
            maxBits++;
            x >>>= 1;
        }
        this.levels = Math.max(1, maxBits);
        this.prefixOnes = new int[levels][length + 1];
        this.zeroCounts = new int[levels];

        int[] curr = data.clone();
        int[] next = new int[length];

        for (int level = 0; level < levels; level++) {
            int bit = levels - 1 - level;

            int zeros = 0;
            for (int i = 0; i < length; i++) {
                int b = (curr[i] >>> bit) & 1;
                prefixOnes[level][i + 1] = prefixOnes[level][i] + b;
                if (b == 0) {
                    zeros++;
                }
            }
            zeroCounts[level] = zeros;

            int z = 0;
            int o = zeros;
            for (int i = 0; i < length; i++) {
                int v = curr[i];
                int b = (v >>> bit) & 1;
                if (b == 0) {
                    next[z++] = v;
                } else {
                    next[o++] = v;
                }
            }

            int[] tmp = curr;
            curr = next;
            next = tmp;
        }
    }

    // Count values in index range [l, r) with value in [lower, upperExclusive).
    public int rangeFreq(int l, int r, int lower, int upperExclusive) {
        if (l < 0) l = 0;
        if (r > length) r = length;
        if (l >= r || lower >= upperExclusive) {
            return 0;
        }
        return lessThan(l, r, upperExclusive) - lessThan(l, r, lower);
    }

    private int lessThan(int l, int r, int x) {
        int count = 0;
        for (int level = 0; level < levels; level++) {
            int bit = levels - 1 - level;
            int xb = (x >>> bit) & 1;

            int onesL = prefixOnes[level][l];
            int onesR = prefixOnes[level][r];
            int zerosL = l - onesL;
            int zerosR = r - onesR;

            if (xb == 1) {
                count += (zerosR - zerosL);
                l = zeroCounts[level] + onesL;
                r = zeroCounts[level] + onesR;
            } else {
                l = zerosL;
                r = zerosR;
            }
        }
        return count;
    }
}
