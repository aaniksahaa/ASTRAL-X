package astralx.completion;

import astralx.Config;
import astralx.Logging;
import astralx.gpu.GPUSimilarityMatrix;
import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.ProgressBar;
import astralx.util.Threading;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Builds the quartet-based taxon similarity matrix from gene trees.
 *
 * Reproduces ASTRAL-MP's SimilarityMatrix.populateByQuartetDistance exactly:
 * for each tree T with kt leaves, and for each pair (a, b) both present in T,
 *   num_T(a, b) = same_side_T(a, b)
 *              = number of quartets {a, b, x, y} resolved by T where a and b
 *                fall on the same side of the bipartition
 *   den_T(a, b) = C2(kt − 2)
 * The final similarity is sim(a,b) = Σ num_T / Σ den_T.
 *
 * ── CPU path (buildCPU) ──────────────────────────────────────────────────────
 * Per-tree scatter, identical to ASTRAL-MP:
 *   For every internal node u, form three "components":
 *     left  = leaves of left  child of u
 *     right = leaves of right child of u
 *     others = leaves in T but not in subtree(u)
 *   totalPairs = C2(|left|) + C2(|right|) + C2(|others|)
 *   For each component-pair (X, Y), the per-(l ∈ X, r ∈ Y) scatter is
 *     sim = totalPairs − C2(|X|) − C2(|Y|)
 *
 * This sums contributions across every internal node on the unrooted path
 * between a and b — exactly the same-side quartet count. The earlier
 * implementation only scattered at the LCA, dropping the (path × others)
 * contributions; that was the source of the systematic mismatch.
 *
 * ── GPU path (buildGPU) ──────────────────────────────────────────────────────
 * Uses the validated bridge identity
 *   same_side_T(a, b)  =  C2(kt − 2)  −  QD_gt(a, b)
 * with an O(1) closed form for QD_gt via Euler tour + sparse-table RMQ
 * carrying (s, F) child-of-LCA payloads. See
 *   DOCS/similarity-matrix-design.md
 * for the derivation. Ordinary tours use the compact unsigned-16 RMQ; larger
 * tours use an exact blocked RMQ with wide positions and child sizes. The CPU
 * path remains the byte-for-byte-correct reference.
 */
public class SimilarityMatrixBuilder {

    private static final int COMPACT_MAX_TOUR = Character.MAX_VALUE + 1;
    private static final int MAX_JAVA_ARRAY_LENGTH = Integer.MAX_VALUE - 8;
    /**
     * Only used after the ordinary one-shot compact layout has exceeded a Java
     * array limit.  Keeping each streamed flattening batch around 4 GiB avoids
     * replacing one impossible array with several near-limit arrays while
     * retaining large enough batches to amortize JNI/GPU setup.
     */
    private static final long STREAMED_COMPACT_HOST_BATCH_BYTES = 4L << 30;
    /** Large-N only: enough to keep the observed 50k×1000 wide tree data in one GPU batch. */
    private static final int LARGE_PACKED_GPU_TREE_CAP_MIB = 8192;

    // ── Public entry points ───────────────────────────────────────────────────

    public static SimilarityMatrix buildCPU(List<Tree> trees, int n) {
        preflightPackedHeap(n);
        SimilarityMatrix sm = new SimilarityMatrix(n);

        ProgressBar bar = new ProgressBar("Building similarity matrix (CPU)", trees.size());
        int done = 0;
        for (Tree tree : trees) {
            accumulateCPU(tree, sm);
            bar.update(++done);
        }
        bar.done();

        sm.normalize();
        return sm;
    }

    public static SimilarityMatrix buildGPU(List<Tree> trees, int n) {
        int k = trees.size();
        preflightPackedHeap(n);
        if (SimilarityMatrix.requiresPacked(n)
                || Boolean.getBoolean("astralx.similarity.forcePacked")) {
            long triangle = SimilarityMatrix.triangleCellCount(n);
            Logging.info("Similarity output: exact segmented upper triangle "
                + "(%d cells, %.2f GiB per accumulator; dense n×n arrays are disabled)",
                triangle, triangle * Double.BYTES / (double)(1L << 30));
        }
        int eMaxRaw = 0;
        int logMax = 1;
        for (int i = 0; i < k; i++) {
            int len;
            try {
                len = EulerTourBuilder.tourLength(trees.get(i));
            } catch (RuntimeException e) {
                throw treeBuildFailure(i, trees.get(i), e);
            }
            eMaxRaw = Math.max(eMaxRaw, len);
            logMax = Math.max(logMax, sparseLog(len));
        }

        int ePadded = nextPowerOfTwo(eMaxRaw);
        boolean compactTourOverflow = eMaxRaw > COMPACT_MAX_TOUR;
        boolean compactLayoutOverflow = !fitsJavaArray((long)k * ePadded)
            || !fitsJavaArray((long)k * logMax * ePadded)
            || !fitsJavaArray((long)k * n);
        boolean forceWide = Boolean.getBoolean("astralx.similarity.forceWide");

        if (!forceWide && !compactTourOverflow && compactLayoutOverflow) {
            Logging.info("GPU similarity RMQ: compact unsigned-16 mode with streamed host "
                + "batches (flat layout exceeds the Java single-array limit)");
            return buildGPUCompactStreamed(trees, n, ePadded, logMax);
        }

        if (forceWide || compactTourOverflow) {
            String reason = forceWide
                ? "forced by -Dastralx.similarity.forceWide=true"
                : "maximum Euler tour " + eMaxRaw + " exceeds compact limit "
                    + COMPACT_MAX_TOUR;
            Logging.info("GPU similarity RMQ: wide blocked mode (%s)", reason);
            return buildGPUWide(trees, n, eMaxRaw);
        }

        Logging.debug("GPU similarity RMQ: compact unsigned-16 mode (maximum tour %d)", eMaxRaw);
        return buildGPUCompact(trees, n);
    }

    private static SimilarityMatrix buildGPUCompact(List<Tree> trees, int n) {
        SimilarityMatrix sm = new SimilarityMatrix(n);
        accumulateGPUCompact(trees, n, sm, 0);
        sm.normalize();
        return sm;
    }

    /**
     * Stream compact tree data through bounded Java arrays.  This path is
     * selected only when the established one-shot layout is impossible; the
     * ordinary compact path above remains unchanged for all fitting inputs.
     */
    private static SimilarityMatrix buildGPUCompactStreamed(List<Tree> trees, int n,
                                                             int ePadded, int logMax) {
        int batchTrees = compactBatchTreeCount(trees.size(), n, ePadded, logMax,
            MAX_JAVA_ARRAY_LENGTH, STREAMED_COMPACT_HOST_BATCH_BYTES);
        int batches = (int)((trees.size() + (long)batchTrees - 1L) / batchTrees);
        Logging.info("Streaming %d compact similarity trees in %d host batches "
                + "(up to %d trees per batch; original tree order preserved)",
            trees.size(), batches, batchTrees);

        SimilarityMatrix sm = new SimilarityMatrix(n);
        for (int from = 0, batch = 1; from < trees.size(); batch++) {
            int to = (int)Math.min(trees.size(), from + (long)batchTrees);
            Logging.info("Compact similarity host batch %d/%d: trees %d..%d",
                batch, batches, from, to - 1);
            accumulateGPUCompact(trees.subList(from, to), n, sm, from);
            from = to;
        }
        sm.normalize();
        return sm;
    }

    /** Flatten and accumulate one compact host batch without normalizing it. */
    private static void accumulateGPUCompact(List<Tree> trees, int n, SimilarityMatrix sm,
                                             int treeIndexBase) {
        int k = trees.size();
        Logging.info("Building FullTourData for %d trees (parallel CPU)", k);

        // ── Step 1: Build per-tree full tour data in parallel ─────────────────
        EulerTourBuilder.FullTourData[] tours = new EulerTourBuilder.FullTourData[k];
        ProgressBar eulerBar    = new ProgressBar("FullTour + RMQ build", k);
        AtomicInteger eulerDone = new AtomicInteger(0);

        Threading.processRangeParallel(k, i -> {
            try {
                tours[i] = EulerTourBuilder.buildFull(trees.get(i), n);
                eulerBar.update(eulerDone.incrementAndGet());
            } catch (RuntimeException e) {
                throw treeBuildFailure(treeIndexBase + i, trees.get(i), e);
            }
        });
        eulerBar.done();

        // ── Step 2: Compute flat-array layout constants ───────────────────────
        int eMaxRaw = 0, logMaxRaw = 0;
        for (EulerTourBuilder.FullTourData td : tours) {
            if (td.tourLen > eMaxRaw)   eMaxRaw   = td.tourLen;
            if (td.log     > logMaxRaw) logMaxRaw = td.log;
        }
        int ePadded = 1;
        while (ePadded < eMaxRaw) ePadded <<= 1;
        final int E_max   = ePadded;
        final int LOG_max = logMaxRaw;

        // Per-position arrays:
        //   eulerDepths (short, 2B), eulerLeftChildS/RightChildS (short, 2B each),
        //   eulerF, eulerLeftChildF, eulerRightChildF (double, 8B each)
        // → 2 + 2 + 2 + 8 + 8 + 8 = 30 bytes/pos
        double euler_mb  = (double)k * E_max * 30 / 1e6;
        // Sparse table: one unsigned-16 left-biased argmin Euler position.
        // The kernel fetches depth and child payloads from the base Euler arrays.
        double sparse_mb = (double)k * LOG_max * E_max * Character.BYTES / 1e6;
        double leaf_mb   = (double)k * n * 4 / 1e6;
        Logging.info("  E_max=%d  LOG=%d  euler %.1f MB  sparse %.1f MB  leaf %.1f MB",
            E_max, LOG_max, euler_mb, sparse_mb, leaf_mb);

        // ── Step 3: Flatten into contiguous Java arrays ───────────────────────
        long edSize = (long)k * E_max;
        long spSize = (long)k * LOG_max * E_max;
        long ldSize = (long)k * n;

        short[]  eulerDepths       = new short [(int)edSize];
        double[] eulerF            = new double[(int)edSize];
        short[]  eulerLeftChildS   = new short [(int)edSize];
        double[] eulerLeftChildF   = new double[(int)edSize];
        short[]  eulerRightChildS  = new short [(int)edSize];
        double[] eulerRightChildF  = new double[(int)edSize];

        char[] sparseArgmin = new char[(int)spSize];

        int[]    firstOcc          = new int   [(int)ldSize];
        int[]    eulerLen          = new int   [k];
        int[]    leafCount         = new int   [k];

        Arrays.fill(firstOcc, -1);

        ProgressBar flatBar    = new ProgressBar("Flattening similarity tour data", k);
        AtomicInteger flatDone = new AtomicInteger(0);

        Threading.processRangeParallel(k, i -> {
            EulerTourBuilder.FullTourData td = tours[i];
            int len   = td.tourLen;
            int treeN = td.firstOcc.length;

            long edOff = (long)i * E_max;
            long spOff = (long)i * LOG_max * E_max;
            long ldOff = (long)i * n;

            for (int p = 0; p < len; p++) {
                int dst = (int)(edOff + p);
                eulerDepths      [dst] = td.depths[p];
                eulerF           [dst] = td.eulerF[p];
                eulerLeftChildS  [dst] = td.eulerLeftChildS[p];
                eulerLeftChildF  [dst] = td.eulerLeftChildF[p];
                eulerRightChildS [dst] = td.eulerRightChildS[p];
                eulerRightChildF [dst] = td.eulerRightChildF[p];
            }

            for (int lvl = 0; lvl < td.log; lvl++) {
                int rowLen = Math.max(0, len - (1 << lvl) + 1);
                long dst   = spOff + (long)lvl * E_max;
                for (int p = 0; p < rowLen; p++) {
                    int idx = (int)(dst + p);
                    sparseArgmin[idx] = td.sparseArgmin[lvl][p];
                }
            }

            for (int a = 0; a < treeN; a++) {
                int fo = td.firstOcc[a];
                if (fo >= 0) firstOcc[(int)(ldOff + a)] = fo;
            }

            eulerLen [i] = len;
            leafCount[i] = td.leafCount;
            flatBar.update(flatDone.incrementAndGet());
        });
        flatBar.done();

        // ── Step 4: Call GPU kernel ───────────────────────────────────────────
        Config cfg = Config.getInstance();
        int    tileSizeB        = cfg.getGpuDistTileSizeB();
        double progressInterval = cfg.getGpuDpProgressInterval();
        int    progressMaxSteps = cfg.getGpuDpProgressMaxSteps();

        int treeCapMiB = effectiveTreeCapMiB(sm, cfg);
        GPUSimilarityMatrix.computeSimilarityGPU(
            eulerDepths,
            eulerF,
            eulerLeftChildS,  eulerLeftChildF,
            eulerRightChildS, eulerRightChildF,
            sparseArgmin,
            firstOcc, eulerLen, leafCount,
            k, n, E_max, LOG_max,
            tileSizeB, treeCapMiB,
            progressInterval, progressMaxSteps,
            sm.numSum, sm.denSum,
            sm.isPacked() ? sm.packedNumeratorSegments() : null,
            sm.isPacked() ? sm.packedDenominatorSegments() : null,
            sm.packedSegmentShift()
        );
    }

    private static SimilarityMatrix buildGPUWide(List<Tree> trees, int n, int eMaxRaw) {
        int k = trees.size();
        int E_max = Math.max(1, eMaxRaw); // deliberately not power-of-two padded
        int blockSize = EulerTourBuilder.WIDE_BLOCK_SIZE;
        int microLog = EulerTourBuilder.WIDE_MICRO_LOG;
        int blockMax = (E_max + blockSize - 1) / blockSize;
        int macroLog = sparseLog(blockMax);

        int edSize = checkedLength((long)k * E_max, "wide Euler arrays");
        int microSize = checkedLength((long)k * microLog * E_max,
            "wide micro-RMQ array");
        int macroSize = checkedLength((long)k * macroLog * blockMax,
            "wide macro-RMQ array");
        int leafSize = checkedLength((long)k * n, "wide first-occurrence array");

        double baseGiB = (double)edSize * (Integer.BYTES * 3L + Double.BYTES * 3L)
            / (1L << 30);
        double rmqGiB = ((double)microSize + (double)macroSize * Integer.BYTES)
            / (1L << 30);
        Logging.info("Building wide FullTourData for %d trees (parallel CPU)", k);
        Logging.info("  E_max=%d  blocks=%d  microLOG=%d  macroLOG=%d  "
                + "Euler/payload %.2f GiB  RMQ %.2f GiB",
            E_max, blockMax, microLog, macroLog, baseGiB, rmqGiB);

        int[] eulerDepths = new int[edSize];
        double[] eulerF = new double[edSize];
        int[] eulerLeftChildS = new int[edSize];
        double[] eulerLeftChildF = new double[edSize];
        int[] eulerRightChildS = new int[edSize];
        double[] eulerRightChildF = new double[edSize];
        byte[] microArgmin = new byte[microSize];
        int[] macroArgmin = new int[macroSize];
        int[] firstOcc = new int[leafSize];
        int[] eulerLen = new int[k];
        int[] leafCount = new int[k];
        Arrays.fill(firstOcc, -1);

        ProgressBar bar = new ProgressBar("Wide FullTour + blocked RMQ build", k);
        AtomicInteger done = new AtomicInteger(0);
        Threading.processRangeParallel(k, i -> {
            Tree tree = trees.get(i);
            try {
                EulerTourBuilder.WideTourData td = EulerTourBuilder.buildWide(tree, n);
                int edOff = i * E_max;
                System.arraycopy(td.depths, 0, eulerDepths, edOff, td.tourLen);
                System.arraycopy(td.eulerF, 0, eulerF, edOff, td.tourLen);
                System.arraycopy(td.eulerLeftChildS, 0, eulerLeftChildS, edOff, td.tourLen);
                System.arraycopy(td.eulerLeftChildF, 0, eulerLeftChildF, edOff, td.tourLen);
                System.arraycopy(td.eulerRightChildS, 0, eulerRightChildS, edOff, td.tourLen);
                System.arraycopy(td.eulerRightChildF, 0, eulerRightChildF, edOff, td.tourLen);

                int microTreeOff = i * microLog * E_max;
                for (int lvl = 0; lvl < microLog; lvl++) {
                    System.arraycopy(td.microArgmin[lvl], 0, microArgmin,
                        microTreeOff + lvl * E_max, td.tourLen);
                }
                int macroTreeOff = i * macroLog * blockMax;
                for (int lvl = 0; lvl < td.macroLog; lvl++) {
                    int rowLen = Math.max(0, td.blockCount - (1 << lvl) + 1);
                    System.arraycopy(td.macroArgmin[lvl], 0, macroArgmin,
                        macroTreeOff + lvl * blockMax, rowLen);
                }
                System.arraycopy(td.firstOcc, 0, firstOcc, i * n, n);
                eulerLen[i] = td.tourLen;
                leafCount[i] = td.leafCount;
                bar.update(done.incrementAndGet());
            } catch (RuntimeException e) {
                throw treeBuildFailure(i, tree, e);
            }
        });
        bar.done();

        Config cfg = Config.getInstance();
        SimilarityMatrix sm = new SimilarityMatrix(n);
        int treeCapMiB = effectiveTreeCapMiB(sm, cfg);
        GPUSimilarityMatrix.computeSimilarityGPUWide(
            eulerDepths, eulerF,
            eulerLeftChildS, eulerLeftChildF,
            eulerRightChildS, eulerRightChildF,
            microArgmin, macroArgmin,
            firstOcc, eulerLen, leafCount,
            k, n, E_max, microLog, blockSize, blockMax, macroLog,
            cfg.getGpuDistTileSizeB(), treeCapMiB,
            cfg.getGpuDpProgressInterval(), cfg.getGpuDpProgressMaxSteps(),
            sm.numSum, sm.denSum,
            sm.isPacked() ? sm.packedNumeratorSegments() : null,
            sm.isPacked() ? sm.packedDenominatorSegments() : null,
            sm.packedSegmentShift());
        sm.normalize();
        return sm;
    }

    static int effectiveTreeCapMiB(SimilarityMatrix sm, Config cfg) {
        int configured = cfg.getGpuSimilarityVramCapMiB();
        if (!sm.isPacked() || cfg.isGpuSimilarityVramCapExplicit()
                || configured >= LARGE_PACKED_GPU_TREE_CAP_MIB) return configured;
        Logging.info("Large-N similarity path: raising the tree-data batching ceiling "
            + "from %d MiB to %d MiB (still clamped to currently free VRAM)",
            configured, LARGE_PACKED_GPU_TREE_CAP_MIB);
        return LARGE_PACKED_GPU_TREE_CAP_MIB;
    }

    private static void preflightPackedHeap(int n) {
        if (!SimilarityMatrix.requiresPacked(n)
                && !Boolean.getBoolean("astralx.similarity.forcePacked")) return;
        long triangle = SimilarityMatrix.triangleCellCount(n);
        if (triangle > Long.MAX_VALUE / (2L * Double.BYTES)) {
            throw new IllegalArgumentException("Similarity matrix size overflows byte accounting");
        }
        long accumulatorBytes = triangle * 2L * Double.BYTES;
        Runtime rt = Runtime.getRuntime();
        long used = rt.totalMemory() - rt.freeMemory();
        long available = Math.max(0L, rt.maxMemory() - used);
        long safety = 256L << 20;
        if (available < accumulatorBytes + safety) {
            throw new IllegalArgumentException("Exact packed similarity accumulators for " + n
                + " taxa require at least "
                + String.format(java.util.Locale.ROOT, "%.2f", accumulatorBytes / (double)(1L << 30))
                + " GiB of currently available Java heap before similarity preprocessing; available "
                + String.format(java.util.Locale.ROOT, "%.2f", available / (double)(1L << 30))
                + " GiB. Increase --xmx or reduce the dataset size.");
        }
    }

    private static IllegalStateException treeBuildFailure(int index, Tree tree,
                                                           RuntimeException cause) {
        return new IllegalStateException("Similarity preprocessing failed for tree " + index
            + " (leaves=" + tree.leafCount + ", complete=" + tree.isComplete + ")", cause);
    }

    private static int sparseLog(int length) {
        if (length <= 1) return 1;
        return 32 - Integer.numberOfLeadingZeros(length);
    }

    private static int nextPowerOfTwo(int value) {
        if (value <= 1) return 1;
        if (value > (1 << 30)) return Integer.MAX_VALUE;
        return 1 << (32 - Integer.numberOfLeadingZeros(value - 1));
    }

    private static boolean fitsJavaArray(long length) {
        return length >= 0 && length <= MAX_JAVA_ARRAY_LENGTH;
    }

    /**
     * Choose a safe streamed compact batch size from both Java's per-array
     * element bound and a total flattened-host-memory bound.
     *
     * Package-private with explicit limits so boundary arithmetic can be unit
     * tested without allocating multi-gigabyte arrays.
     */
    static int compactBatchTreeCount(int treeCount, int n, int ePadded, int logMax,
                                     long maxArrayLength, long maxBatchBytes) {
        if (treeCount <= 0) return 1;
        if (n <= 0 || ePadded <= 0 || logMax <= 0
                || maxArrayLength <= 0 || maxBatchBytes <= 0) {
            throw new IllegalArgumentException("invalid compact similarity batch dimensions");
        }

        long sparseCellsPerTree = Math.multiplyExact((long)logMax, ePadded);
        long flattenedBytesPerTree = Math.addExact(
            Math.addExact(Math.multiplyExact((long)ePadded, 30L),
                Math.multiplyExact(sparseCellsPerTree, Character.BYTES)),
            Math.addExact(Math.multiplyExact((long)n, Integer.BYTES), 2L * Integer.BYTES));

        long byEulerArray = maxArrayLength / ePadded;
        long bySparseArray = maxArrayLength / sparseCellsPerTree;
        long byFirstOccurrence = maxArrayLength / n;
        long byHostBytes = maxBatchBytes / flattenedBytesPerTree;
        long safe = Math.min(Math.min(byEulerArray, bySparseArray),
            Math.min(byFirstOccurrence, byHostBytes));
        if (safe < 1) {
            throw new IllegalArgumentException("one compact similarity tree cannot fit in the "
                + "configured streamed host-array bounds");
        }
        return (int)Math.min(treeCount, Math.min(safe, Integer.MAX_VALUE));
    }

    private static int checkedLength(long length, String label) {
        if (!fitsJavaArray(length)) {
            throw new IllegalArgumentException(label + " requires " + length
                + " elements; Java arrays support at most " + MAX_JAVA_ARRAY_LENGTH
                + ". Reduce the number of trees per run or use a future streamed layout.");
        }
        return (int)length;
    }

    // ── CPU accumulation (per-tree, scatter mirroring ASTRAL-MP) ─────────────

    /**
     * Scatter same-side quartet counts to all (l, r) pairs at every internal
     * node, plus accumulate the per-pair denominator C2(kt − 2).
     */
    private static void accumulateCPU(Tree tree, SimilarityMatrix sm) {
        int kt = tree.leafCount;
        if (kt < 4) {
            // C2(kt-2) = 0 → tree contributes nothing to numerator or denominator
            // (matches ASTRAL-MP, which skips by virtue of sim and den both being 0).
            return;
        }

        long denPerPair = EulerTourBuilder.c2(kt - 2);
        int n = sm.n;
        int[] postArr = tree.postorderArray;

        // ── Numerator: scatter across all internal nodes ─────────────────────
        if (sm.isPacked()) scatterAtNodePacked(tree.root, tree, kt, sm);
        else               scatterAtNodeDense(tree.root, tree, kt, sm);

        // ── Denominator: every pair (a, b) co-occurring in T gets += C2(kt-2) ─
        // ASTRAL-MP accumulates 2·C2(kt-2) in dn[l][r] (doubled by the
        // mirror-write pattern), then normalizes by dn/2. Equivalent to a
        // single sum here; the constant factor 2 cancels.
        if (sm.isPacked()) {
            for (int i = 0; i < kt; i++) {
                int a = postArr[i];
                for (int j = i + 1; j < kt; j++) {
                    sm.addPackedDenominator(a, postArr[j], denPerPair);
                }
            }
        } else {
            for (int i = 0; i < kt; i++) {
                int a = postArr[i];
                long rowOff = (long) a * n;
                for (int j = 0; j < kt; j++) {
                    if (i == j) continue;
                    int b = postArr[j];
                    sm.denSum[(int)(rowOff + b)] += denPerPair;
                }
            }
        }
    }

    /**
     * Post-order scatter at one internal node.
     *
     * Components at u:
     *   left   = subtree of u.left   (rangeStart_L .. rangeEnd_L)
     *   right  = subtree of u.right  (rangeStart_R .. rangeEnd_R)
     *   others = leaves of T outside subtree(u)
     *            (i.e. positions [0, u.rangeStart) ∪ [u.rangeEnd, kt) in postArr)
     *
     * For each ordered component-pair, scatter
     *   sim = totalPairs − C2(|X|) − C2(|Y|)
     * to every (l ∈ X, r ∈ Y) leaf pair, symmetrically.
     */
    private static void scatterAtNodeDense(TreeNode node, Tree tree, int kt,
                                           SimilarityMatrix sm) {
        if (node.isLeaf()) return;
        scatterAtNodeDense(node.left,  tree, kt, sm);
        scatterAtNodeDense(node.right, tree, kt, sm);

        int subL = node.left.rangeEnd  - node.left.rangeStart;
        int subR = node.right.rangeEnd - node.right.rangeStart;
        int subU = node.rangeEnd       - node.rangeStart;
        int subO = kt - subU;                                 // "others" size

        long cL = EulerTourBuilder.c2(subL);
        long cR = EulerTourBuilder.c2(subR);
        long cO = EulerTourBuilder.c2(subO);
        long totalPairs = cL + cR + cO;                       // only positive comps survive

        int n = sm.n;
        int[] postArr = tree.postorderArray;

        // ── (left × right): always present ───────────────────────────────────
        long simLR = totalPairs - cL - cR;
        if (simLR != 0) {
            scatterRangeRange(postArr,
                node.left.rangeStart, node.left.rangeEnd,
                node.right.rangeStart, node.right.rangeEnd,
                simLR, sm.numSum, n);
        }

        // ── (left × others) and (right × others): only when u is non-root ────
        if (subO > 0) {
            long simLO = totalPairs - cL - cO;
            if (simLO != 0) {
                scatterRangeOthers(postArr, kt,
                    node.left.rangeStart, node.left.rangeEnd,
                    node.rangeStart, node.rangeEnd, simLO, sm.numSum, n);
            }
            long simRO = totalPairs - cR - cO;
            if (simRO != 0) {
                scatterRangeOthers(postArr, kt,
                    node.right.rangeStart, node.right.rangeEnd,
                    node.rangeStart, node.rangeEnd, simRO, sm.numSum, n);
            }
        }
    }

    private static void scatterAtNodePacked(TreeNode node, Tree tree, int kt,
                                            SimilarityMatrix sm) {
        if (node.isLeaf()) return;
        scatterAtNodePacked(node.left, tree, kt, sm);
        scatterAtNodePacked(node.right, tree, kt, sm);

        int subL = node.left.rangeEnd - node.left.rangeStart;
        int subR = node.right.rangeEnd - node.right.rangeStart;
        int subU = node.rangeEnd - node.rangeStart;
        int subO = kt - subU;
        long cL = EulerTourBuilder.c2(subL);
        long cR = EulerTourBuilder.c2(subR);
        long cO = EulerTourBuilder.c2(subO);
        long totalPairs = cL + cR + cO;
        int[] postArr = tree.postorderArray;

        long simLR = totalPairs - cL - cR;
        if (simLR != 0) {
            scatterRangeRangePacked(postArr,
                node.left.rangeStart, node.left.rangeEnd,
                node.right.rangeStart, node.right.rangeEnd, simLR, sm);
        }
        if (subO > 0) {
            long simLO = totalPairs - cL - cO;
            if (simLO != 0) {
                scatterRangeOthersPacked(postArr, kt,
                    node.left.rangeStart, node.left.rangeEnd,
                    node.rangeStart, node.rangeEnd, simLO, sm);
            }
            long simRO = totalPairs - cR - cO;
            if (simRO != 0) {
                scatterRangeOthersPacked(postArr, kt,
                    node.right.rangeStart, node.right.rangeEnd,
                    node.rangeStart, node.rangeEnd, simRO, sm);
            }
        }
    }

    /** Scatter `sim` to every (a ∈ [aLo,aHi)) × (b ∈ [bLo,bHi)) leaf-pair, both directions. */
    private static void scatterRangeRange(int[] postArr,
                                           int aLo, int aHi,
                                           int bLo, int bHi,
                                           long sim, double[] numSum, int n) {
        double sd = (double) sim;
        for (int pi = aLo; pi < aHi; pi++) {
            int a = postArr[pi];
            long rowA = (long) a * n;
            for (int pj = bLo; pj < bHi; pj++) {
                int b = postArr[pj];
                numSum[(int)(rowA + b)] += sd;
                numSum[b * n + a]       += sd;
            }
        }
    }

    private static void scatterRangeRangePacked(int[] postArr,
                                                 int aLo, int aHi,
                                                 int bLo, int bHi,
                                                 long sim, SimilarityMatrix sm) {
        double sd = (double)sim;
        for (int pi = aLo; pi < aHi; pi++) {
            int a = postArr[pi];
            for (int pj = bLo; pj < bHi; pj++) {
                sm.addPackedNumerator(a, postArr[pj], sd);
            }
        }
    }

    /**
     * Scatter `sim` to every (a ∈ [aLo,aHi)) × (b ∈ others) leaf-pair, where
     * others = leaves in T outside [subLo, subHi).
     */
    private static void scatterRangeOthers(int[] postArr, int kt,
                                            int aLo, int aHi,
                                            int subLo, int subHi,
                                            long sim, double[] numSum, int n) {
        double sd = (double) sim;
        for (int pi = aLo; pi < aHi; pi++) {
            int a = postArr[pi];
            long rowA = (long) a * n;
            for (int pj = 0; pj < subLo; pj++) {
                int b = postArr[pj];
                numSum[(int)(rowA + b)] += sd;
                numSum[b * n + a]       += sd;
            }
            for (int pj = subHi; pj < kt; pj++) {
                int b = postArr[pj];
                numSum[(int)(rowA + b)] += sd;
                numSum[b * n + a]       += sd;
            }
        }
    }

    private static void scatterRangeOthersPacked(int[] postArr, int kt,
                                                  int aLo, int aHi,
                                                  int subLo, int subHi,
                                                  long sim, SimilarityMatrix sm) {
        double sd = (double)sim;
        for (int pi = aLo; pi < aHi; pi++) {
            int a = postArr[pi];
            for (int pj = 0; pj < subLo; pj++) {
                sm.addPackedNumerator(a, postArr[pj], sd);
            }
            for (int pj = subHi; pj < kt; pj++) {
                sm.addPackedNumerator(a, postArr[pj], sd);
            }
        }
    }
}
