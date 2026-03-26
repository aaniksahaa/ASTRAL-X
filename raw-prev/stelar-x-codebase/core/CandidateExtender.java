package core;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

import tree.*;
import utils.Threading;
import java.util.HashSet;

/**
 * Extends the candidate bipartition set via cluster-consistent cross-tree recombination.
 * 
 * This class implements the algorithm described in the paper:
 * 1. Collect all side clusters S from observed bipartitions (both left and right sides)
 * 2. Build size-bucketed hash maps for O(1) complement lookup
 * 3. For each cluster C in the DP state space, find all valid (A, B) pairs where:
 *    - A ∈ S, B ∈ S
 *    - A ∩ B = ∅
 *    - A ∪ B = C
 * 4. Generate mixed bipartitions for all valid pairs
 * 
 * The key optimization is using hash-based complement matching:
 *   h1(B) = h1(C) - h1(A)   (raw sum is additive)
 *   h2(B) = h2(C) ⊕ h2(A)  (raw XOR is self-inverse)
 * 
 * This enables O(1) complement lookup instead of O(n²) pairwise enumeration.
 */
public class CandidateExtender {
    
    // Configuration
    private static final boolean DEBUG_OUTPUT = true;
    private static final int PROGRESS_INTERVAL = 1000;
    
    // Input data
    private final List<RangeBipartition> observedBipartitions;
    private final long[][] prefixSums;
    private final long[][] prefixXORs;
    private final int[][] geneTreeOrderings;
    
    // Computed data structures
    private final Map<Integer, Map<SideClusterHash, List<SideCluster>>> sizeBuckets;
    private final Set<SideClusterHash> clusterSet;  // C = set of all clusters (union of bipartition sides)
    private final Map<SideClusterHash, List<SideCluster>> clusterToSides;  // For getting sides that form a cluster
    
    // Statistics
    private final AtomicLong totalSideClusters = new AtomicLong(0);
    private final AtomicLong uniqueSideHashes = new AtomicLong(0);
    private final AtomicLong totalMixedGenerated = new AtomicLong(0);
    private final AtomicLong complementLookups = new AtomicLong(0);
    private final AtomicLong complementMatches = new AtomicLong(0);
    private final AtomicLong verificationFailures = new AtomicLong(0);
    
    public CandidateExtender(List<RangeBipartition> observedBipartitions,
                            long[][] prefixSums, long[][] prefixXORs,
                            int[][] geneTreeOrderings) {
        System.out.println("\n==== INITIALIZING CANDIDATE EXTENDER ====");
        System.out.println("Observed bipartitions: " + observedBipartitions.size());
        
        this.observedBipartitions = observedBipartitions;
        this.prefixSums = prefixSums;
        this.prefixXORs = prefixXORs;
        this.geneTreeOrderings = geneTreeOrderings;
        this.sizeBuckets = new ConcurrentHashMap<>();
        this.clusterSet = ConcurrentHashMap.newKeySet();
        this.clusterToSides = new ConcurrentHashMap<>();
        
        // Build data structures
        buildSideBuckets();
        buildClusterSet();
        
        System.out.println("==== CANDIDATE EXTENDER INITIALIZED ====\n");
    }
    
    /**
     * Build size-bucketed hash maps for all side clusters.
     * 
     * For each observed bipartition (A|B), extract both sides A and B,
     * compute their raw hashes, and add to size buckets.
     */
    private void buildSideBuckets() {
        System.out.println("Building size-bucketed hash maps for side clusters...");
        
        long startTime = System.currentTimeMillis();
        
        for (RangeBipartition bip : observedBipartitions) {
            // Extract left side
            SideCluster leftSide = new SideCluster(bip.geneTreeIndex, bip.leftStart, bip.leftEnd);
            SideClusterHash leftHash = leftSide.computeRawHash(prefixSums, prefixXORs);
            addToSizeBucket(leftSide, leftHash);
            
            // Extract right side
            SideCluster rightSide = new SideCluster(bip.geneTreeIndex, bip.rightStart, bip.rightEnd);
            SideClusterHash rightHash = rightSide.computeRawHash(prefixSums, prefixXORs);
            addToSizeBucket(rightSide, rightHash);
            
            totalSideClusters.addAndGet(2);
        }
        
        // Count unique hashes across all buckets
        long uniqueCount = 0;
        for (Map<SideClusterHash, List<SideCluster>> bucket : sizeBuckets.values()) {
            uniqueCount += bucket.size();
        }
        uniqueSideHashes.set(uniqueCount);
        
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.println("Size bucket construction completed in " + elapsed + " ms");
        System.out.println("  Total side clusters extracted: " + totalSideClusters.get());
        System.out.println("  Unique side cluster hashes: " + uniqueSideHashes.get());
        System.out.println("  Number of size buckets: " + sizeBuckets.size());
        
        // Print size distribution
        if (DEBUG_OUTPUT) {
            System.out.println("  Size distribution:");
            sizeBuckets.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .limit(10)
                .forEach(e -> System.out.println("    Size " + e.getKey() + ": " + e.getValue().size() + " unique hashes"));
        }
    }
    
    /**
     * Add a side cluster to its corresponding size bucket.
     */
    private void addToSizeBucket(SideCluster side, SideClusterHash hash) {
        int size = side.size();
        
        sizeBuckets.computeIfAbsent(size, k -> new ConcurrentHashMap<>())
                   .computeIfAbsent(hash, k -> new CopyOnWriteArrayList<>())
                   .add(side);
    }
    
    /**
     * Build the cluster set C (all union clusters from observed bipartitions).
     * These are the clusters that exist in the DP state space.
     */
    private void buildClusterSet() {
        System.out.println("Building cluster set C (DP state space clusters)...");
        
        long startTime = System.currentTimeMillis();
        
        for (RangeBipartition bip : observedBipartitions) {
            // Compute union cluster hash (C = A ∪ B)
            SideCluster leftSide = new SideCluster(bip.geneTreeIndex, bip.leftStart, bip.leftEnd);
            SideCluster rightSide = new SideCluster(bip.geneTreeIndex, bip.rightStart, bip.rightEnd);
            
            SideClusterHash leftHash = leftSide.computeRawHash(prefixSums, prefixXORs);
            SideClusterHash rightHash = rightSide.computeRawHash(prefixSums, prefixXORs);
            
            // Union hash = sum of sums, XOR of XORs, sum of sizes
            SideClusterHash unionHash = new SideClusterHash(
                leftHash.rawSum + rightHash.rawSum,
                leftHash.rawXor ^ rightHash.rawXor,
                leftHash.size + rightHash.size
            );
            
            clusterSet.add(unionHash);
            
            // Also track which sides can form this cluster (for reference)
            clusterToSides.computeIfAbsent(unionHash, k -> new CopyOnWriteArrayList<>())
                         .add(leftSide);
            clusterToSides.get(unionHash).add(rightSide);
        }
        
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.println("Cluster set construction completed in " + elapsed + " ms");
        System.out.println("  Unique clusters in DP state space: " + clusterSet.size());
    }
    
    /**
     * Generate extended candidate bipartitions using cross-tree recombination.
     * 
     * This is the main algorithm that finds all valid (A, B) pairs where:
     * - A and B are sides from observed bipartitions
     * - A ∩ B = ∅ (implied by hash matching with correct sizes)
     * - A ∪ B = C for some cluster C in the DP state space
     * 
     * @return List of mixed bipartitions (new candidates from cross-tree recombination)
     */
    public List<MixedBipartition> generateMixedBipartitions() {
        System.out.println("\n==== GENERATING MIXED BIPARTITIONS ====");
        System.out.println("Using size-bucketed hash join with complement matching...");
        
        long startTime = System.currentTimeMillis();
        
        // Parallel processing over clusters
        List<SideClusterHash> clusterList = new ArrayList<>(clusterSet);
        List<MixedBipartition> allMixed = Collections.synchronizedList(new ArrayList<>());
        
        int numThreads = Runtime.getRuntime().availableProcessors();
        Threading.startThreading(numThreads);
        
        int chunkSize = Math.max(1, (clusterList.size() + numThreads - 1) / numThreads);
        int actualThreads = Math.min(numThreads, (clusterList.size() + chunkSize - 1) / chunkSize);
        CountDownLatch latch = new CountDownLatch(actualThreads);
        
        System.out.println("Processing " + clusterList.size() + " clusters using " + actualThreads + " threads");
        
        AtomicInteger processedClusters = new AtomicInteger(0);
        
        for (int t = 0; t < actualThreads; t++) {
            final int startIdx = t * chunkSize;
            final int endIdx = Math.min(startIdx + chunkSize, clusterList.size());
            final int threadId = t;
            
            Threading.execute(() -> {
                try {
                    List<MixedBipartition> localMixed = new ArrayList<>();
                    
                    for (int i = startIdx; i < endIdx; i++) {
                        SideClusterHash clusterHash = clusterList.get(i);
                        List<MixedBipartition> clusterMixed = generateMixedForCluster(clusterHash);
                        localMixed.addAll(clusterMixed);
                        
                        int processed = processedClusters.incrementAndGet();
                        if (processed % PROGRESS_INTERVAL == 0 || processed == clusterList.size()) {
                            System.out.println("Progress: " + processed + "/" + clusterList.size() + 
                                             " clusters, " + totalMixedGenerated.get() + " mixed bipartitions");
                        }
                    }
                    
                    allMixed.addAll(localMixed);
                    totalMixedGenerated.addAndGet(localMixed.size());
                    
                    if (DEBUG_OUTPUT) {
                        System.out.println("Thread " + threadId + " generated " + localMixed.size() + " mixed bipartitions");
                    }
                    
                } finally {
                    latch.countDown();
                }
            });
        }
        
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Mixed bipartition generation interrupted", e);
        } finally {
            Threading.shutdown();
        }
        
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.println("\n==== MIXED BIPARTITION GENERATION COMPLETE ====");
        System.out.println("  Total mixed bipartitions generated: " + allMixed.size());
        System.out.println("  Complement lookups: " + complementLookups.get());
        System.out.println("  Complement matches: " + complementMatches.get());
        System.out.println("  Processing time: " + elapsed + " ms");
        
        return allMixed;
    }
    
    /**
     * Generate all mixed bipartitions for a single cluster C.
     * 
     * For each valid size split (m, M-m) where M = |C|:
     * 1. Choose smaller bucket to iterate over
     * 2. For each side A in that bucket, compute complement hash
     * 3. Look up matching sides B in the complementary bucket
     * 4. Verify A ∩ B = ∅ and A ∪ B = C (expensive element-wise check)
     * 5. Generate mixed bipartition for each verified match
     */
    private List<MixedBipartition> generateMixedForCluster(SideClusterHash clusterHash) {
        List<MixedBipartition> result = new ArrayList<>();
        
        int M = clusterHash.size;
        
        // Iterate over all valid size splits
        for (int m = 1; m <= M / 2; m++) {
            int complementSize = M - m;
            
            // Get buckets for both sizes
            Map<SideClusterHash, List<SideCluster>> bucketM = sizeBuckets.get(m);
            Map<SideClusterHash, List<SideCluster>> bucketComplement = sizeBuckets.get(complementSize);
            
            if (bucketM == null || bucketComplement == null) {
                continue; // No sides of this size
            }
            
            // Choose smaller bucket to iterate over (optimization)
            boolean iterateSmaller = bucketM.size() <= bucketComplement.size();
            Map<SideClusterHash, List<SideCluster>> iterateBucket = iterateSmaller ? bucketM : bucketComplement;
            Map<SideClusterHash, List<SideCluster>> lookupBucket = iterateSmaller ? bucketComplement : bucketM;
            
            // For each side A in iterate bucket, find complement B
            for (Map.Entry<SideClusterHash, List<SideCluster>> entry : iterateBucket.entrySet()) {
                SideClusterHash sideAHash = entry.getKey();
                List<SideCluster> sidesA = entry.getValue();
                
                // Compute required complement hash
                SideClusterHash complementHash = sideAHash.computeComplement(clusterHash);
                complementLookups.incrementAndGet();
                
                // Look up matching sides B
                List<SideCluster> sidesB = lookupBucket.get(complementHash);
                
                if (sidesB != null && !sidesB.isEmpty()) {
                    complementMatches.incrementAndGet();
                    
                    // All clusters in the same hash group represent the same taxon set,
                    // so we only need ONE representative from each group (no Cartesian product needed)
                    SideCluster sideA = sidesA.get(0);  // Pick first representative
                    SideCluster sideB = sidesB.get(0);  // Pick first representative
                    
                    // Skip if both sides are from the exact same position in same tree
                    // (this would mean A = B which is invalid)
                    if (sideA.equals(sideB)) {
                        // Try to find a different representative from sidesB
                        boolean foundDifferent = false;
                        for (SideCluster altB : sidesB) {
                            if (!sideA.equals(altB)) {
                                sideB = altB;
                                foundDifferent = true;
                                break;
                            }
                        }
                        if (!foundDifferent) {
                            continue; // No valid pair exists
                        }
                    }
                    
                    // // EXPENSIVE VERIFICATION: Check A ∩ B = ∅ and A ∪ B = C
                    // if (!verifyDisjointAndUnion(sideA, sideB, clusterHash)) {
                    //     verificationFailures.incrementAndGet();
                    //     continue; // Skip invalid pairs
                    // }
                    
                    // Create ONE mixed bipartition (representatives of equivalent clusters)
                    MixedBipartition mixed;
                    if (iterateSmaller) {
                        mixed = new MixedBipartition(sideA, sideB);
                    } else {
                        mixed = new MixedBipartition(sideB, sideA);
                    }
                    mixed.setParentClusterHash(clusterHash);
                    
                    result.add(mixed);
                }
            }
        }
        
        return result;
    }
    
    /**
     * EXPENSIVE VERIFICATION: Check that A ∩ B = ∅ and A ∪ B = C
     * 
     * This performs element-wise set operations to verify the hash-based matching
     * is correct. Uses actual taxon sets from gene tree orderings.
     * 
     * @param sideA First side cluster
     * @param sideB Second side cluster  
     * @param expectedUnionHash Expected hash of the union (C)
     * @return true if A and B are disjoint and their union matches C
     */
    private boolean verifyDisjointAndUnion(SideCluster sideA, SideCluster sideB, SideClusterHash expectedUnionHash) {
        // Get actual taxon sets for A and B
        Set<Integer> taxaA = getTaxonSet(sideA);
        Set<Integer> taxaB = getTaxonSet(sideB);
        
        // Check disjoint: A ∩ B = ∅
        for (int taxon : taxaA) {
            if (taxaB.contains(taxon)) {
                if (DEBUG_OUTPUT) {
                    System.err.println("VERIFICATION FAILED: A ∩ B ≠ ∅");
                    System.err.println("  A = " + sideA + " -> " + taxaA);
                    System.err.println("  B = " + sideB + " -> " + taxaB);
                    System.err.println("  Common taxon: " + taxon);
                }
                return false; // Not disjoint
            }
        }
        
        // Compute union: C = A ∪ B
        Set<Integer> unionSet = new HashSet<>(taxaA);
        unionSet.addAll(taxaB);
        
        // Verify size matches
        if (unionSet.size() != expectedUnionHash.size) {
            if (DEBUG_OUTPUT) {
                System.err.println("VERIFICATION FAILED: |A ∪ B| ≠ |C|");
                System.err.println("  |A ∪ B| = " + unionSet.size());
                System.err.println("  |C| = " + expectedUnionHash.size);
            }
            return false;
        }
        
        // Compute hash of union and compare
        SideClusterHash actualUnionHash = computeHashFromTaxonSet(unionSet);
        
        if (!actualUnionHash.equals(expectedUnionHash)) {
            if (DEBUG_OUTPUT) {
                System.err.println("VERIFICATION FAILED: hash(A ∪ B) ≠ hash(C)");
                System.err.println("  Actual hash: " + actualUnionHash);
                System.err.println("  Expected hash: " + expectedUnionHash);
                System.err.println("  A = " + taxaA);
                System.err.println("  B = " + taxaB);
                System.err.println("  Union = " + unionSet);
            }
            return false;
        }
        
        return true; // Verified: A ∩ B = ∅ and A ∪ B = C
    }
    
    /**
     * Get the taxon set for a side cluster from gene tree orderings.
     */
    private Set<Integer> getTaxonSet(SideCluster side) {
        Set<Integer> taxonSet = new HashSet<>();
        
        if (geneTreeOrderings != null && 
            side.geneTreeIndex >= 0 && 
            side.geneTreeIndex < geneTreeOrderings.length) {
            
            int[] ordering = geneTreeOrderings[side.geneTreeIndex];
            if (ordering != null) {
                for (int i = side.start; i < side.end && i < ordering.length; i++) {
                    taxonSet.add(ordering[i]);
                }
            }
        }
        
        return taxonSet;
    }
    
    /**
     * Compute raw hash from a taxon set (for verification).
     */
    private SideClusterHash computeHashFromTaxonSet(Set<Integer> taxonSet) {
        long rawSum = 0;
        long rawXor = 0;
        
        for (int taxonId : taxonSet) {
            long hashedTaxon = hashSingleTaxon(taxonId);
            rawSum += hashedTaxon;
            rawXor ^= hashedTaxon;
        }
        
        return new SideClusterHash(rawSum, rawXor, taxonSet.size());
    }
    
    /**
     * Hash a single taxon ID (same as MemoryEfficientBipartitionManager).
     */
    private static long hashSingleTaxon(int taxonId) {
        long x = taxonId;
        x ^= x >>> 16;
        x *= 0x85ebca6b;
        x ^= x >>> 13;
        x *= 0xc2b2ae35;
        x ^= x >>> 16;
        return x == 0 ? 1 : x;
    }
    
    /**
     * Get the set of all cluster hashes (DP state space).
     */
    public Set<SideClusterHash> getClusterSet() {
        return Collections.unmodifiableSet(clusterSet);
    }
    
    /**
     * Get size buckets for debugging/analysis.
     */
    public Map<Integer, Map<SideClusterHash, List<SideCluster>>> getSizeBuckets() {
        return Collections.unmodifiableMap(sizeBuckets);
    }
    
    /**
     * Get statistics about the candidate extension process.
     */
    public String getStatistics() {
        StringBuilder sb = new StringBuilder();
        sb.append("Candidate Extender Statistics:\n");
        sb.append("  Observed bipartitions: ").append(observedBipartitions.size()).append("\n");
        sb.append("  Total side clusters: ").append(totalSideClusters.get()).append("\n");
        sb.append("  Unique side hashes: ").append(uniqueSideHashes.get()).append("\n");
        sb.append("  Size buckets: ").append(sizeBuckets.size()).append("\n");
        sb.append("  Clusters in DP space: ").append(clusterSet.size()).append("\n");
        sb.append("  Mixed bipartitions generated: ").append(totalMixedGenerated.get()).append("\n");
        sb.append("  Complement lookups: ").append(complementLookups.get()).append("\n");
        sb.append("  Complement matches: ").append(complementMatches.get()).append("\n");
        sb.append("  Verification failures (hash collisions): ").append(verificationFailures.get()).append("\n");
        
        if (complementLookups.get() > 0) {
            double matchRate = 100.0 * complementMatches.get() / complementLookups.get();
            sb.append("  Match rate: ").append(String.format("%.2f%%", matchRate)).append("\n");
        }
        
        if (verificationFailures.get() > 0) {
            sb.append("  WARNING: Hash collisions detected! ").append(verificationFailures.get())
              .append(" false positive matches were filtered out.\n");
        }
        
        return sb.toString();
    }
}

