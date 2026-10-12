package astralx.support;

import astralx.Logging;
import astralx.taxon.TaxonRegistry;
import astralx.tree.Tree;
import astralx.tree.TreeNode;
import astralx.util.ProgressBar;
import astralx.util.Threading;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Quartet-frequency branch support (local posterior probability) for a fixed
 * species tree, computed exactly as ASTRAL's {@code -q ... -t N} scoring does.
 *
 * <p>For every internal branch the four surrounding clusters form a
 * quadripartition (c1, c2 | sister, remaining).  For each gene tree g the number
 * of quartets with one taxon drawn from each cluster that are resolved as
 * c1c2|sr, c1s|c2r and c1r|c2s is counted with one post-order pass, each count
 * is divided by the number of such quartets present in g, and the per-gene
 * fractions are summed over the genes in which at least one such quartet is
 * present (the effective number of genes).  ASTRAL's quadripartition rules,
 * cluster roles, missing-data normalisation and gene-tree polytomy handling are
 * mirrored exactly; the resulting (f1, f2, f3, n) feed {@link Posterior}.
 *
 * <p>Cost is O(branches × total gene-tree size) with a few dozen integer
 * operations per gene-tree node and no per-node allocation.  Branches are
 * processed in parallel; each worker owns its scratch arrays.  Nothing here
 * touches the inference or scoring code paths.
 */
public final class BranchSupport {

    private BranchSupport() {}

    /** Annotation styles, matching ASTRAL's {@code -t} values. */
    public enum Format {
        /** ASTRAL -t 0: no labels, posterior branch lengths only. */
        NONE,
        /** ASTRAL -t 1: main-topology quartet support in percent. */
        Q1,
        /** ASTRAL -t 10: polytomy-test p-value (five decimals, "NA" when undefined). */
        PVALUE,
        /** ASTRAL -t 4: '[pp1=..;pp2=..;pp3=..]' (default). */
        PP3,
        /** ASTRAL -t 3: main-topology posterior as a plain label. */
        PP,
        /** ASTRAL -t 8: '[q1=..;q2=..;q3=..]' normalised quartet frequencies. */
        Q3,
        /** ASTRAL -t 2: full annotation with raw frequencies, QC and EN. */
        FULL,
        /** Integer percentage of pp1 (what the cleaning scripts derive from -t 4). */
        PERCENT;

        public static Format parse(String s) {
            switch (s.trim().toLowerCase(java.util.Locale.ROOT)) {
                case "none": case "0": return NONE;
                case "q1": case "1": return Q1;
                case "pvalue": case "10": return PVALUE;
                case "pp3": case "4": return PP3;
                case "pp": case "3": return PP;
                case "q3": case "8": return Q3;
                case "full": case "2": return FULL;
                case "percent": case "support": return PERCENT;
                default: throw new IllegalArgumentException(
                    "Unknown --support-format '" + s + "' (pp3|pp|q3|full|percent|q1|pvalue|none)");
            }
        }
    }

    public static final class Result {
        public final String annotatedNewick;
        public final int annotatedBranches;
        public final int lowEffectiveN;
        public final double logPosterior;
        Result(String newick, int branches, int lowEffn, double logPosterior) {
            this.annotatedNewick = newick;
            this.annotatedBranches = branches;
            this.lowEffectiveN = lowEffn;
            this.logPosterior = logPosterior;
        }
    }

    // ------------------------------------------------------------------ public entry

    /**
     * @param speciesNewick species tree text (any rooting; polytomies allowed)
     * @param geneTrees     native gene trees over {@code registry}
     * @param registry      taxon universe; every species-tree leaf must be in it and
     *                      every registered taxon must appear in the species tree
     * @param lambda        Yule prior rate (ASTRAL default 0.5)
     * @param format        annotation style
     */
    public static Result compute(String speciesNewick, List<Tree> geneTrees,
                                 TaxonRegistry registry, double lambda, Format format) {
        return compute(speciesNewick, geneTrees, registry, lambda, format, false, 512);
    }

    /**
     * @param useGpu        try the CUDA kernel first (falls back to CPU, same results)
     * @param gpuVramCapMiB cap on per-batch device memory for the kernel
     */
    public static Result compute(String speciesNewick, List<Tree> geneTrees,
                                 TaxonRegistry registry, double lambda, Format format,
                                 boolean useGpu, int gpuVramCapMiB) {
        long t0 = System.nanoTime();
        int n = registry.size();
        int W = (n + 63) >>> 6;

        SupportNewick.Node root = SupportNewick.parse(speciesNewick);
        List<SupportNewick.Node> nodes = SupportNewick.postOrder(root);
        resolveLeaves(nodes, registry, n, W);

        if (format == Format.NONE) {
            // ASTRAL -t 0 performs no branch scoring: the tree is re-emitted with
            // internal labels and internal lengths dropped and nothing added.
            String out = SupportNewick.write(root);
            Logging.info("Branch support: format none; species tree re-emitted without annotation");
            return new Result(out, 0, 0, 0.0);
        }

        GeneData gd = buildGeneData(geneTrees, n, W);
        Logging.info("Branch support: %d taxa, %d gene trees, %d gene-tree tokens, "
            + "max stack %d", n, gd.numTrees, gd.tokens.length, gd.maxFrontier);

        List<Branch> branches = enumerateBranches(nodes, root, W);
        Logging.info("Branch support: %d internal branch(es) to score", branches.size());

        boolean gpuDone = false;
        if (useGpu && astralx.gpu.GPUBranchSupport.isLoaded() && !branches.isEmpty()) {
            gpuDone = scoreBranchesGPU(branches, gd, W, gpuVramCapMiB);
            if (!gpuDone) Logging.warn("Branch support: GPU path unavailable, using CPU");
        } else if (useGpu) {
            Logging.info("Branch support: GPU requested but kernel not loaded (%s); using CPU",
                astralx.gpu.GPUBranchSupport.getStatus());
        }
        if (!gpuDone) {
            ProgressBar bar = new ProgressBar("Scoring branches (quartet frequencies)", branches.size());
            AtomicInteger done = new AtomicInteger();
            ThreadLocal<Scratch> scratchTL = ThreadLocal.withInitial(() -> new Scratch(n, W, gd.maxFrontier));
            Threading.processRangeParallel(branches.size(), i -> {
                scoreBranch(branches.get(i), gd, scratchTL.get(), W);
                bar.update(done.incrementAndGet());
            });
            bar.done();
        }

        int lowEffn = 0;
        double logPost = 0.0;
        DecimalFormat df = decimalFormat();
        DecimalFormat df5 = decimalFormat();
        df5.setMaximumFractionDigits(5);
        for (Branch b : branches) {
            if (b.effn < 20) {
                lowEffn++;
                Logging.debug("Branch support: effective number of genes is only %d for the "
                    + "branch above a %d-taxon cluster", b.effn, b.node.size);
            }
            logPost += annotate(b, lambda, format, df, df5);
        }
        if (lowEffn > 0) {
            Logging.warn("Branch support: %d branch(es) have fewer than 20 effective genes; "
                + "their posterior probabilities are unreliable (run with -vv to list them)",
                lowEffn);
        }
        String out = SupportNewick.write(root);
        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Branch support: annotated %d branch(es); log local posterior = %.6f  (%d ms)",
            branches.size(), logPost, ms);
        return new Result(out, branches.size(), lowEffn, logPost);
    }

    // ------------------------------------------------------------------ species tree

    private static void resolveLeaves(List<SupportNewick.Node> nodes, TaxonRegistry registry,
                                      int n, int W) {
        boolean[] seen = new boolean[n];
        int leaves = 0;
        for (SupportNewick.Node nd : nodes) {
            if (nd.isLeaf()) {
                int id = registry.findId(nd.label);
                if (id < 0) {
                    throw new IllegalArgumentException("Species tree taxon '" + nd.label
                        + "' does not occur in the gene trees");
                }
                if (seen[id]) {
                    throw new IllegalArgumentException("Species tree lists taxon '" + nd.label
                        + "' more than once");
                }
                seen[id] = true;
                leaves++;
                nd.taxonId = id;
                nd.bits = new long[W];
                nd.bits[id >>> 6] |= 1L << (id & 63);
                nd.size = 1;
            } else {
                nd.bits = new long[W];
                for (SupportNewick.Node c : nd.children) {
                    for (int w = 0; w < W; w++) nd.bits[w] |= c.bits[w];
                    nd.size += c.size;
                }
            }
        }
        if (leaves != n) {
            List<String> missing = new ArrayList<>();
            for (int i = 0; i < n && missing.size() < 10; i++) if (!seen[i]) missing.add(registry.getName(i));
            throw new IllegalArgumentException("Species tree has " + leaves + " of the " + n
                + " gene-tree taxa; missing e.g. " + missing);
        }
    }

    /** One annotatable branch: the edge above {@code node}. */
    private static final class Branch {
        final SupportNewick.Node node;
        final long[][] cluster = new long[4][];   // 0=c1, 1=c2, 2=sister, 3=remaining
        final int[] size = new int[4];
        double f1, f2, f3;
        int effn;
        long quartCount;
        Branch(SupportNewick.Node node) { this.node = node; }
    }

    /**
     * ASTRAL's branch selection: leaves, the root, polytomous nodes, children of
     * polytomies (except under an unrooted three-way root), and the root-adjacent
     * node whose sibling is not binary are not annotated.
     */
    private static boolean skipNode(SupportNewick.Node node) {
        if (node.isLeaf() || node.isRoot() || node.children.size() > 2) return true;
        SupportNewick.Node parent = node.parent;
        int pc = parent.children.size();
        if (pc > 3) return true;
        if (pc > 2 && !parent.isRoot()) return true;
        if (parent.isRoot() && pc == 2) {
            SupportNewick.Node sib = parent.children.get(0) == node
                ? parent.children.get(1) : parent.children.get(0);
            if (sib.children.size() != 2) return true;
        }
        return false;
    }

    private static List<Branch> enumerateBranches(List<SupportNewick.Node> nodes,
                                                  SupportNewick.Node root, int W) {
        List<Branch> out = new ArrayList<>();
        for (SupportNewick.Node node : nodes) {
            if (skipNode(node)) continue;
            Branch b = new Branch(node);
            // ASTRAL pops children off a post-order stack: c1 = last child, c2 = first.
            SupportNewick.Node c1 = node.children.get(1), c2 = node.children.get(0);
            SupportNewick.Node parent = node.parent;
            SupportNewick.Node sibling = null;
            for (SupportNewick.Node c : parent.children) { if (c != node) { sibling = c; break; } }
            long[] sisterBits, remainingBits;
            int sisterSize, remainingSize;
            if (parent.isRoot() && parent.children.size() == 3) {
                SupportNewick.Node third = null;
                for (SupportNewick.Node c : parent.children) {
                    if (c != node && c != sibling) { third = c; break; }
                }
                sisterBits = sibling.bits; sisterSize = sibling.size;
                remainingBits = third.bits; remainingSize = third.size;
            } else if (parent.isRoot() && parent.children.size() == 2) {
                SupportNewick.Node n1 = sibling.children.get(0), n2 = sibling.children.get(1);
                sisterBits = n1.bits; sisterSize = n1.size;
                remainingBits = n2.bits; remainingSize = n2.size;
            } else {
                sisterBits = sibling.bits; sisterSize = sibling.size;
                remainingBits = new long[W];
                for (int w = 0; w < W; w++) remainingBits[w] = root.bits[w] & ~parent.bits[w];
                remainingSize = root.size - parent.size;
            }
            b.cluster[0] = c1.bits; b.size[0] = c1.size;
            b.cluster[1] = c2.bits; b.size[1] = c2.size;
            b.cluster[2] = sisterBits; b.size[2] = sisterSize;
            b.cluster[3] = remainingBits; b.size[3] = remainingSize;
            b.quartCount = (long) c1.size * c2.size * sisterSize * remainingSize;
            out.add(b);
        }
        return out;
    }

    // ------------------------------------------------------------------ gene trees

    private static final class GeneData {
        int numTrees;
        int[] tokens;        // post-order: leaf = taxon id, internal = -childCount
        int[] offset;        // numTrees + 1
        long[] presentBits;  // numTrees * W
        int maxFrontier;
    }

    private static GeneData buildGeneData(List<Tree> trees, int n, int W) {
        GeneData d = new GeneData();
        d.numTrees = trees.size();
        d.offset = new int[d.numTrees + 1];
        for (int g = 0; g < d.numTrees; g++) {
            long tok = (long) d.offset[g] + countTokens(trees.get(g).root);
            if (tok > Integer.MAX_VALUE) {
                throw new IllegalStateException("Gene-tree token stream too large for one int[]");
            }
            d.offset[g + 1] = (int) tok;
        }
        d.tokens = new int[d.offset[d.numTrees]];
        if ((long) d.numTrees * W > Integer.MAX_VALUE) {
            throw new IllegalStateException("Gene-tree presence bitsets too large for one long[]");
        }
        d.presentBits = new long[d.numTrees * W];
        int[] frontier = new int[d.numTrees];
        Threading.processRangeParallel(d.numTrees, g -> {
            Tree t = trees.get(g);
            int end = fillTokens(t.root, d.tokens, d.offset[g]);
            if (end != d.offset[g + 1]) {
                throw new IllegalStateException("Gene tree " + g + " token count mismatch");
            }
            int base = g * W;
            for (int pos = 0; pos < t.leafCount; pos++) {
                int id = t.postorderArray[pos];
                d.presentBits[base + (id >>> 6)] |= 1L << (id & 63);
            }
            int top = 0, peak = 1;
            for (int i = d.offset[g]; i < d.offset[g + 1]; i++) {
                int tok = d.tokens[i];
                if (tok >= 0) { top++; if (top > peak) peak = top; }
                else top = top + tok + 1;
            }
            frontier[g] = peak;
        });
        d.maxFrontier = 1;
        for (int f : frontier) d.maxFrontier = Math.max(d.maxFrontier, f);
        return d;
    }

    private static int countTokens(TreeNode node) {
        if (node.isLeaf()) return 1;
        int c = 1;
        if (node.isPolytomous()) { for (TreeNode ch : node.children) c += countTokens(ch); }
        else { c += countTokens(node.left) + countTokens(node.right); }
        return c;
    }

    private static int fillTokens(TreeNode node, int[] out, int pos) {
        if (node.isLeaf()) { out[pos] = node.taxonId; return pos + 1; }
        int k;
        if (node.isPolytomous()) {
            k = node.children.length;
            for (TreeNode ch : node.children) pos = fillTokens(ch, out, pos);
        } else {
            k = 2;
            pos = fillTokens(node.left, out, pos);
            pos = fillTokens(node.right, out, pos);
        }
        out[pos] = -k;
        return pos + 1;
    }

    // ------------------------------------------------------------------ GPU path

    /** Same arithmetic as {@link #scoreBranch} on the GPU; returns false to fall back. */
    private static boolean scoreBranchesGPU(List<Branch> branches, GeneData d, int W, int vramCapMiB) {
        long t0 = System.nanoTime();
        int nb = branches.size();
        if ((long) nb * 4 * W > Integer.MAX_VALUE) return false;
        long[] clusterBits = new long[nb * 4 * W];
        for (int i = 0; i < nb; i++) {
            Branch b = branches.get(i);
            for (int k = 0; k < 4; k++) {
                System.arraycopy(b.cluster[k], 0, clusterBits, (i * 4 + k) * W, W);
            }
        }
        double[] out = astralx.gpu.GPUBranchSupport.computeSupportGPU(
            clusterBits, d.presentBits, d.tokens, d.offset, nb, d.numTrees, W, d.maxFrontier, vramCapMiB);
        if (out == null || out.length != nb * 4) return false;
        for (int i = 0; i < nb; i++) {
            Branch b = branches.get(i);
            b.f1 = out[i * 4]; b.f2 = out[i * 4 + 1]; b.f3 = out[i * 4 + 2];
            b.effn = (int) Math.round(out[i * 4 + 3]);
        }
        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Branch support: %d branch(es) scored on the GPU (%d ms)", nb, ms);
        return true;
    }

    // ------------------------------------------------------------------ per-branch walk

    private static final class Scratch {
        final byte[] partOf;
        final int[] stack;      // 4 ints per entry: counts in c1, c2, sister, remaining
        Scratch(int n, int W, int maxFrontier) {
            partOf = new byte[n];
            stack = new int[4 * Math.max(1, maxFrontier)];
        }
    }

    private static void scoreBranch(Branch b, GeneData d, Scratch s, int W) {
        byte[] partOf = s.partOf;
        Arrays.fill(partOf, (byte) 4);
        for (int k = 0; k < 4; k++) {
            long[] bits = b.cluster[k];
            for (int w = 0; w < W; w++) {
                long word = bits[w];
                while (word != 0) {
                    int bit = Long.numberOfTrailingZeros(word);
                    partOf[(w << 6) + bit] = (byte) k;
                    word &= word - 1;
                }
            }
        }
        int[] stack = s.stack;
        int[] tokens = d.tokens;
        double w1 = 0, w2 = 0, w3 = 0;
        int effn = 0;
        for (int g = 0; g < d.numTrees; g++) {
            int gb = g * W;
            int t0 = popAnd(b.cluster[0], d.presentBits, gb, W);
            int t1 = popAnd(b.cluster[1], d.presentBits, gb, W);
            int t2 = popAnd(b.cluster[2], d.presentBits, gb, W);
            int t3 = popAnd(b.cluster[3], d.presentBits, gb, W);
            long mi = (long) t0 * t1 * t2 * t3;
            if (mi == 0) continue;
            effn++;
            long f1 = 0, f2 = 0, f3 = 0;
            int top = 0;
            for (int i = d.offset[g], end = d.offset[g + 1]; i < end; i++) {
                int tok = tokens[i];
                if (tok >= 0) {
                    int e = top << 2;
                    stack[e] = 0; stack[e + 1] = 0; stack[e + 2] = 0; stack[e + 3] = 0;
                    int p = partOf[tok];
                    if (p < 4) stack[e + p] = 1;
                    top++;
                    continue;
                }
                int k = -tok;
                int base = top - k;
                int bs = base << 2;
                // Subtree sums over the k children.
                int u0 = 0, u1 = 0, u2 = 0, u3 = 0;
                for (int j = 0; j < k; j++) {
                    int e = bs + (j << 2);
                    u0 += stack[e]; u1 += stack[e + 1]; u2 += stack[e + 2]; u3 += stack[e + 3];
                }
                // Parts of this node's partition: the k children plus the remaining
                // taxa of the gene tree (an all-zero remaining part contributes 0).
                int r0 = t0 - u0, r1 = t1 - u1, r2 = t2 - u2, r3 = t3 - u3;
                long p01 = (long) r0 * r1, p23 = (long) r2 * r3;
                long p02 = (long) r0 * r2, p13 = (long) r1 * r3;
                long p03 = (long) r0 * r3, p12 = (long) r1 * r2;
                for (int j = 0; j < k; j++) {
                    int e = bs + (j << 2);
                    long a = stack[e], bb = stack[e + 1], c = stack[e + 2], dd = stack[e + 3];
                    p01 += a * bb; p23 += c * dd;
                    p02 += a * c;  p13 += bb * dd;
                    p03 += a * dd; p12 += bb * c;
                }
                // Pairing (x,y | z,w): a part j holding the pair {x,y} with z and w in
                // two distinct other parts, or the pair {z,w} with x and y so placed.
                f1 += pairingTerm(r0, r1, r2, r3, t0, t1, t2, t3, p01, p23);
                f2 += pairingTerm(r0, r2, r1, r3, t0, t2, t1, t3, p02, p13);
                f3 += pairingTerm(r0, r3, r1, r2, t0, t3, t1, t2, p03, p12);
                for (int j = 0; j < k; j++) {
                    int e = bs + (j << 2);
                    int a = stack[e], bb = stack[e + 1], c = stack[e + 2], dd = stack[e + 3];
                    f1 += pairingTerm(a, bb, c, dd, t0, t1, t2, t3, p01, p23);
                    f2 += pairingTerm(a, c, bb, dd, t0, t2, t1, t3, p02, p13);
                    f3 += pairingTerm(a, dd, bb, c, t0, t3, t1, t2, p03, p12);
                }
                stack[bs] = u0; stack[bs + 1] = u1; stack[bs + 2] = u2; stack[bs + 3] = u3;
                top = base + 1;
            }
            double denom = 2.0 * mi;
            w1 += f1 / denom;
            w2 += f2 / denom;
            w3 += f3 / denom;
        }
        b.f1 = w1; b.f2 = w2; b.f3 = w3; b.effn = effn;
    }

    /**
     * Contribution of one part with counts (x, y, z, w) to the pairing xy|zw:
     * x·y·#{(z',w') in distinct other parts} + z·w·#{(x',y') in distinct other parts},
     * where the "distinct other parts" sums are obtained from the totals
     * (sx, sy, sz, sw) and the within-part pair sums (pxy, pzw).
     */
    private static long pairingTerm(long x, long y, long z, long w,
                                    long sx, long sy, long sz, long sw,
                                    long pxy, long pzw) {
        return x * y * ((sz - z) * (sw - w) - pzw + z * w)
             + z * w * ((sx - x) * (sy - y) - pxy + x * y);
    }

    private static int popAnd(long[] cluster, long[] present, int base, int W) {
        int c = 0;
        for (int w = 0; w < W; w++) c += Long.bitCount(cluster[w] & present[base + w]);
        return c;
    }

    // ------------------------------------------------------------------ annotation

    static DecimalFormat decimalFormat() {
        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(2);
        DecimalFormatSymbols dfs = DecimalFormatSymbols.getInstance();
        dfs.setDecimalSeparator('.');
        df.setDecimalFormatSymbols(dfs);
        return df;
    }

    /** Mirrors ASTRAL's annotation step; returns log(pp1) for the summary. */
    private static double annotate(Branch b, double lambda, Format format, DecimalFormat df,
                                   DecimalFormat df5) {
        double f1 = b.f1, f2 = b.f2, f3 = b.f3;
        double effni = b.effn;
        if (Math.abs((f1 + f2 + f3) - effni) > 0.00000000001) {
            // Only possible with gene-tree polytomies (unresolved quartets).
            effni = f1 + f2 + f3;
        }
        if (effni == 0) {
            Logging.warn("Branch support: a branch has effective N == 0 and so no quartet "
                + "resolutions (cluster of %d taxa)", b.node.size);
        }
        Posterior post = new Posterior(f1, f2, f3, effni, lambda);
        double bl = post.branchLength();
        b.node.newLength = bl;
        double pp1 = post.getPost();
        double ret = Math.log(pp1);
        SupportNewick.Node node = b.node;
        switch (format) {
            case Q1 -> node.annotation = effni != 0 ? df.format((f1 + .0) / effni * 100) : "";
            case PVALUE -> {
                double pval = post.getPvalue();
                node.annotation = pval < 0 ? "NA" : df5.format(pval);
            }
            case PP -> node.annotation = df.format(pp1);
            case PERCENT -> node.annotation = Long.toString(
                Math.round(Double.parseDouble(df.format(pp1)) * 100.0));
            default -> {
                double pp2 = new Posterior(f2, f1, f3, effni, lambda).getPost();
                double pp3 = new Posterior(f3, f1, f2, effni, lambda).getPost();
                switch (format) {
                    case FULL -> node.annotation = "'[q1=" + (f1) / effni + ";q2=" + (f2) / effni
                        + ";q3=" + (f3) / effni + ";f1=" + f1 + ";f2=" + f2 + ";f3=" + f3
                        + ";pp1=" + pp1 + ";pp2=" + pp2 + ";pp3=" + pp3
                        + ";QC=" + b.quartCount + ";EN=" + effni + "]'";
                    case Q3 -> node.annotation = "'[q1=" + df.format((f1) / effni)
                        + ";q2=" + df.format((f2) / effni)
                        + ";q3=" + df.format((f3) / effni) + "]'";
                    default -> node.annotation = "'[pp1=" + df.format(pp1) + ";pp2=" + df.format(pp2)
                        + ";pp3=" + df.format(pp3) + "]'";
                }
            }
        }
        return ret;
    }
}
