package astralx.hash;

import java.util.Random;

public final class SeededTaxonHashes {
    public final int taxaCount;
    public final int replicates;
    public final long[] seeds;
    public final long[][] hashesByTaxon; // [taxon][replicate]

    public SeededTaxonHashes(int taxaCount, int replicates, long seed) {
        this.taxaCount = taxaCount;
        this.replicates = replicates;
        this.seeds = new long[replicates];
        this.hashesByTaxon = new long[taxaCount][replicates];

        Random rng = new Random(seed);
        for (int r = 0; r < replicates; r++) {
            seeds[r] = rng.nextLong();
        }

        for (int t = 0; t < taxaCount; t++) {
            for (int r = 0; r < replicates; r++) {
                hashesByTaxon[t][r] = map(t, seeds[r]);
            }
        }
    }

    private static long mix64(long x) {
        x ^= (x >>> 30);
        x *= 0xbf58476d1ce4e5b9L;
        x ^= (x >>> 27);
        x *= 0x94d049bb133111ebL;
        x ^= (x >>> 31);
        return x;
    }

    private static long map(long taxonId, long seed) {
        return mix64(taxonId + seed);
    }
}
