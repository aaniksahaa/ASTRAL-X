package astralx.tree;

import astralx.Logging;
import astralx.taxon.TaxonRegistry;
import astralx.util.ProgressBar;
import astralx.util.Threading;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Newick parser for rooted or unrooted gene trees with optional native polytomies.
 *
 * Two-pass design:
 *   Pass 1  -- collectTaxonNames(): scan every Newick string, register all names.
 *   Pass 2  -- parseNewick(): build Tree objects with postorder arrays + node ranges.
 *
 * Parsing is intentionally lenient: any number of children is allowed during the
 * stack-based parse phase.  The conversion step then applies the selected input
 * policy and roots the in-memory representation:
 *
 *   Root node with 2 children → already a rooted binary tree, keep as-is.
 *   Root node with 3 children → unrooted binary tree; arbitrarily rooted here
 *                                (ASTRAL is rooting-agnostic, so any choice is fine).
 *   Default                            → deterministic first-pair binary refinement.
 *   --keep-polytomy internal degree≥3 → native polytomous node.
 *
 * After parsing every node has a half-open range [rangeStart, rangeEnd) that indexes
 * into the tree's postorderArray (left-to-right leaf ordering).
 */
public class TreeParser {

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    public static List<Tree> parseGeneTrees(String inputFile,
                                             TaxonRegistry registry) throws IOException {
        return parseGeneTrees(inputFile, registry, false);
    }

    /**
     * Parse gene trees and, by default, apply the same deterministic first-pair
     * binary refinement available in {@code clean.py --deterministic}.  Passing
     * {@code keepPolytomy=true} retains native multifurcations for downstream
     * unresolved-quartet scoring.
     */
    public static List<Tree> parseGeneTrees(String inputFile,
                                             TaxonRegistry registry,
                                             boolean keepPolytomy) throws IOException {
        long t0 = System.nanoTime();

        // Read all non-empty lines
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
            String ln;
            while ((ln = br.readLine()) != null) {
                ln = ln.trim();
                if (!ln.isEmpty()) lines.add(ln);
            }
        }
        Logging.info("Read %d lines from %s", lines.size(), inputFile);

        // Pass 1 – register taxon names
        for (String ln : lines) collectTaxonNames(ln, registry);
        registry.lock();
        int n = registry.size();
        Logging.info("Registered %d unique taxa", n);

        // Pass 2 is independent after the registry is locked.  In normal CLI runs
        // Threading is already initialized, so parsing + optional refinement uses
        // the configured worker pool while preserving input order in parsed[].
        Tree[] parsed = new Tree[lines.size()];
        int[][] perTreeCounts = new int[lines.size()][6];
        ProgressBar parseBar = new ProgressBar("Parsing trees", lines.size());
        AtomicInteger completed = new AtomicInteger(0);
        if (lines.size() > 1 && Threading.isStarted() && Threading.getNumThreads() > 1) {
            Threading.processRangeParallel(lines.size(), i -> {
                parsed[i] = parseNewick(lines.get(i), i, registry, perTreeCounts[i], keepPolytomy);
                parseBar.update(completed.incrementAndGet());
            });
        } else {
            for (int i = 0; i < lines.size(); i++) {
                parsed[i] = parseNewick(lines.get(i), i, registry, perTreeCounts[i], keepPolytomy);
                parseBar.update(i + 1);
            }
        }
        parseBar.done();
        List<Tree> trees = new ArrayList<>(Arrays.asList(parsed));

        int[] totals = new int[6];
        int treesWithPolytomies = 0;
        int treesRefined = 0;
        for (int[] counts : perTreeCounts) {
            for (int j = 0; j < totals.length; j++) totals[j] += counts[j];
            if (counts[4] > 0) treesWithPolytomies++;
            if (counts[3] > 0) treesRefined++;
        }

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Parsed %d gene trees in %d ms", trees.size(), ms);
        if (keepPolytomy && (totals[0] > 0 || totals[1] > 0)) {
            Logging.info("Rooted unrooted input at the root: %d tree(s) with a 3-furcation "
                + "(unrooted binary), %d tree(s) with a ≥4-furcation polytomy — rooted arbitrarily "
                + "(ASTRAL is rooting-agnostic).", totals[0], totals[1]);
        }
        if (keepPolytomy) {
            Logging.info("Input polytomies: keeping %d node(s) across %d tree(s); "
                + "native downstream handling enabled", totals[4], treesWithPolytomies);
        } else if (totals[3] > 0) {
            Logging.info("Input binary refinement: detected %d polytomy node(s); resolved "
                + "%d multifurcation(s) across %d tree(s) by deterministic first-pair pairing "
                + "(%s; includes %d unrooted 3-way root(s))",
                totals[4], totals[3], treesRefined,
                lines.size() > 1 && Threading.isStarted() && Threading.getNumThreads() > 1
                    ? "parallel" : "serial",
                totals[5]);
        } else {
            Logging.info("Input binary refinement: no multifurcations detected");
        }

        // Per-tree debug log -- cap at 5 trees to avoid flooding on large inputs
        if (Logging.isDebug()) {
            int cap = Math.min(5, trees.size());
            for (int i = 0; i < cap; i++) {
                Tree t = trees.get(i);
                Logging.debug("  Tree %d: %d leaves, complete=%b  postorder=%s",
                    i, t.leafCount, t.isComplete,
                    Logging.isTrace() ? Arrays.toString(t.postorderArray) : "(use -vvv to see)");
            }
            if (trees.size() > cap)
                Logging.debug("  ... (%d more trees not shown)", trees.size() - cap);
        }

        return trees;
    }

    /**
     * Parse one supplied species tree against an already-locked gene-tree taxon
     * registry. The tree must contain exactly the same taxa as the gene-tree
     * input; unknown, missing, or duplicate taxa are rejected.
     */
    public static Tree parseSpeciesTree(String inputFile,
                                        TaxonRegistry registry) throws IOException {
        if (!registry.isLocked()) {
            throw new IllegalArgumentException("Species-tree parsing requires a locked taxon registry");
        }

        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
            String ln;
            while ((ln = br.readLine()) != null) {
                ln = ln.trim();
                if (!ln.isEmpty()) lines.add(ln);
            }
        }
        if (lines.isEmpty()) {
            throw new IllegalArgumentException("Species tree file is empty: " + inputFile);
        }
        if (lines.size() != 1) {
            throw new IllegalArgumentException("Species tree file must contain exactly one Newick tree: " + inputFile);
        }

        int[] rootingCounts = new int[6];
        Tree tree = parseNewick(lines.get(0), 0, registry, rootingCounts, true);
        validateCompleteTaxonSet(tree, registry, inputFile);
        Logging.info("Parsed supplied species tree: %d leaves", tree.leafCount);
        if (rootingCounts[0] > 0 || rootingCounts[1] > 0) {
            Logging.info("Rooted supplied unrooted species tree at the root "
                + "(%d binary 3-furcation, %d polytomy root).",
                rootingCounts[0], rootingCounts[1]);
        }
        return tree;
    }

    // -------------------------------------------------------------------------
    // Pass 1 – name collection
    // -------------------------------------------------------------------------

    /**
     * Walk the Newick string and register every taxon name.
     * Tracks whether the last structural token was ')': if so, the next label is
     * a bootstrap/internal value (skip it); otherwise it is a taxon name (register it).
     * This correctly handles both named taxa (strings) and integer-labelled taxa.
     */
    private static void collectTaxonNames(String s, TaxonRegistry reg) {
        int i = 0, n = s.length();
        // true if the most recent structural character was ')'
        boolean afterCloseParen = false;
        while (i < n) {
            char c = s.charAt(i);
            if (c == '(') { afterCloseParen = false; i++; continue; }
            if (c == ',') { afterCloseParen = false; i++; continue; }
            if (c == ')') { afterCloseParen = true;  i++; continue; }
            if (c == ';') break;
            if (c == ':') { i = skipBranchLen(s, i + 1, n); continue; }
            if (c == '[') { // NHX or comment: skip to ']'
                while (i < n && s.charAt(i) != ']') i++;
                if (i < n) i++;
                continue;
            }
            // Token: taxon name (after '(' or ',') or internal label (after ')')
            int start = i;
            while (i < n && !isDelim(s.charAt(i))) i++;
            String tok = s.substring(start, i).trim();
            if (!tok.isEmpty() && !afterCloseParen) {
                reg.register(tok);
            }
            afterCloseParen = false;
        }
    }

    // -------------------------------------------------------------------------
    // Temporary multi-child node for lenient parsing
    // -------------------------------------------------------------------------

    /**
     * Internal node used only during parsing — supports any number of children.
     * Converted to binary TreeNode after validation.
     */
    private static class RawNode {
        int taxonId = -1;                          // leaf: taxon ID; internal: -1
        final List<RawNode> children = new ArrayList<>();
        boolean isLeaf() { return children.isEmpty(); }
    }

    // -------------------------------------------------------------------------
    // Pass 2 – full parse
    // -------------------------------------------------------------------------

    /** Sentinel object pushed onto the stack to mark an open parenthesis. */
    private static final Object SENTINEL = new Object();

    private static Tree parseNewick(String s, int treeIdx, TaxonRegistry reg,
                                    int[] rootingCounts, boolean keepPolytomy) {
        int n = s.length(), totalTaxa = reg.size();
        Deque<Object> stack = new ArrayDeque<>();   // contains RawNode or SENTINEL
        int i = 0;

        while (i < n) {
            char c = s.charAt(i);

            if (c == '(') {
                stack.push(SENTINEL);
                i++;

            } else if (c == ')') {
                // Collect all children pushed since the matching '('
                List<RawNode> children = new ArrayList<>();
                while (stack.peek() != SENTINEL) children.add((RawNode) stack.pop());
                stack.pop();   // remove sentinel

                // children were pushed left-to-right, popped right-to-left; restore order
                Collections.reverse(children);

                RawNode node = new RawNode();
                node.children.addAll(children);
                stack.push(node);

                i++;
                // skip optional internal label (e.g. bootstrap) then branch length
                i = skipLabelAndBranchLen(s, i, n);

            } else if (c == ',') {
                i++;

            } else if (c == ';') {
                break;

            } else if (c == ':') {
                // shouldn't appear at top level, but be safe
                i = skipBranchLen(s, i + 1, n);

            } else {
                // Leaf taxon name
                int start = i;
                while (i < n && !isDelim(s.charAt(i))) i++;
                String name = s.substring(start, i).trim();
                if (!name.isEmpty()) {
                    RawNode leaf = new RawNode();
                    leaf.taxonId = reg.getId(name);
                    stack.push(leaf);
                }
                i = skipBranchLen(s, i, n);   // skip ':length' if present
            }
        }

        if (stack.size() != 1) {
            throw new RuntimeException("Tree " + treeIdx
                + ": malformed Newick, stack size=" + stack.size());
        }
        RawNode rawRoot = (RawNode) stack.pop();
        if (rawRoot.isLeaf()) {
            throw new RuntimeException("Tree " + treeIdx + ": root is a leaf");
        }

        // Validate arity and root unrooted trees; convert RawNode → TreeNode
        int polytomyCountBefore = rootingCounts[2];
        TreeNode root = validateAndConvert(
            rawRoot, treeIdx, true, rootingCounts, keepPolytomy);
        boolean hasPolytomy = rootingCounts[2] != polytomyCountBefore;

        // Assign ranges and build postorderArray in one left-to-right DFS
        int[] postorderArray = new int[reg.size()]; // upper bound; trimmed below
        int[] counter = {0};
        assignRangesAndFillArray(root, postorderArray, counter);
        int leafCount = counter[0];
        postorderArray = Arrays.copyOf(postorderArray, leafCount);

        // Build inverse map
        int[] positionMap = new int[totalTaxa];
        Arrays.fill(positionMap, -1);
        for (int j = 0; j < leafCount; j++) positionMap[postorderArray[j]] = j;

        return new Tree(treeIdx, root, postorderArray, positionMap, leafCount, totalTaxa,
            hasPolytomy);
    }

    /** Exact child-order semantics of one clean.py deterministic first-pair refinement. */
    private static void resolveNodeFirstPair(RawNode node) {
        ArrayDeque<RawNode> work = new ArrayDeque<>(node.children);
        while (work.size() > 2) {
            RawNode first = work.removeFirst();
            RawNode second = work.removeFirst();
            RawNode joined = new RawNode();
            joined.children.add(first);
            joined.children.add(second);
            work.addLast(joined);
        }
        node.children.clear();
        node.children.addAll(work);
    }

    /**
     * Recursively validates a RawNode tree and converts it to a TreeNode (binary or
     * polytomous — see DOCS/polytomy-design.md §3.2):
     *
     *   isRoot=true,  2 children  → rooted binary root, recurse normally.
     *   isRoot=true,  3 children  → unrooted binary input; root arbitrarily by isolating
     *                               the first child and joining the other two. Logs a message.
     *   isRoot=true,  ≥4 children → unrooted polytomy at root; isolate child[0] as the
     *                               root's left and make children[1..nc-1] a polytomous
     *                               right child (the inner node recovers the full nc-partition
     *                               via its nc-1 children + complement = child[0]).
     *   isRoot=false, 2 children  → normal binary internal node.
     *   isRoot=false, ≥3 children → polytomous internal node (children[] array).
     *   leaf                      → leaf node.
     */
    private static TreeNode validateAndConvert(RawNode raw, int treeIdx, boolean isRoot,
                                               int[] rootingCounts, boolean keepPolytomy) {
        if (raw.isLeaf()) {
            TreeNode leaf = new TreeNode();
            leaf.taxonId = raw.taxonId;
            return leaf;
        }

        int nc = raw.children.size();
        if (nc > (isRoot ? 3 : 2)) rootingCounts[4]++;
        if (isRoot && nc == 3) rootingCounts[5]++;
        if (!keepPolytomy && nc > 2) {
            rootingCounts[3]++;
            resolveNodeFirstPair(raw);
            nc = 2;
        }

        if (nc == 2) {
            TreeNode node = new TreeNode();
            node.left  = validateAndConvert(
                raw.children.get(0), treeIdx, false, rootingCounts, keepPolytomy);
            node.right = validateAndConvert(
                raw.children.get(1), treeIdx, false, rootingCounts, keepPolytomy);
            node.left.parent  = node;
            node.right.parent = node;
            return node;

        } else if (nc == 3 && isRoot) {
            // Unrooted binary tree: 3-furcation at root.
            // Root by isolating children[0] as left and joining children[1]+children[2]
            // into a new binary right node.  Rooting-agnostic ⇒ any choice is fine.
            rootingCounts[0]++;   // tallied; summarized once in parseGeneTrees (no per-tree log)

            TreeNode c0 = validateAndConvert(
                raw.children.get(0), treeIdx, false, rootingCounts, keepPolytomy);
            TreeNode c1 = validateAndConvert(
                raw.children.get(1), treeIdx, false, rootingCounts, keepPolytomy);
            TreeNode c2 = validateAndConvert(
                raw.children.get(2), treeIdx, false, rootingCounts, keepPolytomy);

            TreeNode inner = new TreeNode();
            inner.left  = c1;
            inner.right = c2;
            c1.parent = inner;
            c2.parent = inner;

            TreeNode root = new TreeNode();
            root.left  = c0;
            root.right = inner;
            c0.parent    = root;
            inner.parent = root;
            return root;

        } else if (nc >= 4 && isRoot) {
            // Unrooted polytomy at root (degree ≥ 4).  Isolate child[0] as the root's
            // left; the remaining nc-1 children form a polytomous right child whose
            // complement is exactly child[0] — recovering the full nc-way partition.
            rootingCounts[1]++;   // tallied; summarized once in parseGeneTrees (no per-tree log)
            rootingCounts[2]++;

            TreeNode c0 = validateAndConvert(
                raw.children.get(0), treeIdx, false, rootingCounts, keepPolytomy);

            TreeNode inner = new TreeNode();
            inner.children = new TreeNode[nc - 1];
            for (int j = 1; j < nc; j++) {
                TreeNode cj = validateAndConvert(
                    raw.children.get(j), treeIdx, false, rootingCounts, keepPolytomy);
                inner.children[j - 1] = cj;
                cj.parent = inner;
            }
            inner.left  = inner.children[0];
            inner.right = inner.children[nc - 2];

            TreeNode root = new TreeNode();
            root.left  = c0;
            root.right = inner;
            c0.parent    = root;
            inner.parent = root;
            return root;

        } else {
            // nc >= 3 && !isRoot  → polytomous internal node.
            rootingCounts[2]++;
            TreeNode node = new TreeNode();
            node.children = new TreeNode[nc];
            for (int j = 0; j < nc; j++) {
                TreeNode cj = validateAndConvert(
                    raw.children.get(j), treeIdx, false, rootingCounts, keepPolytomy);
                node.children[j] = cj;
                cj.parent = node;
            }
            node.left  = node.children[0];
            node.right = node.children[nc - 1];
            return node;
        }
    }

    /**
     * Single left-to-right DFS:
     *   - Leaf: assign rangeStart=counter, rangeEnd=counter+1, increment counter,
     *           write taxonId into postorderArray[counter].
     *   - Internal (binary or polytomous): recurse into children in order; range
     *     spans from leftmost (left) to rightmost (right) child.
     */
    private static void assignRangesAndFillArray(TreeNode node,
                                                  int[] arr, int[] counter) {
        if (node.isLeaf()) {
            node.rangeStart = counter[0];
            node.rangeEnd   = counter[0] + 1;
            arr[counter[0]] = node.taxonId;
            counter[0]++;
            return;
        }
        if (node.isPolytomous()) {
            for (TreeNode child : node.children) assignRangesAndFillArray(child, arr, counter);
        } else {
            assignRangesAndFillArray(node.left,  arr, counter);
            assignRangesAndFillArray(node.right, arr, counter);
        }
        node.rangeStart = node.left.rangeStart;
        node.rangeEnd   = node.right.rangeEnd;
    }

    private static void validateCompleteTaxonSet(Tree tree, TaxonRegistry reg, String inputFile) {
        int n = reg.size();
        if (tree.leafCount != n) {
            throw new IllegalArgumentException("Species tree taxon count (" + tree.leafCount
                + ") does not match gene-tree taxon count (" + n + "): " + inputFile);
        }

        boolean[] seen = new boolean[n];
        for (int taxon : tree.postorderArray) {
            if (seen[taxon]) {
                throw new IllegalArgumentException("Species tree contains duplicate taxon: "
                    + reg.getName(taxon));
            }
            seen[taxon] = true;
        }
        for (int i = 0; i < n; i++) {
            if (!seen[i]) {
                throw new IllegalArgumentException("Species tree is missing taxon: " + reg.getName(i));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** True for characters that delimit a name or branch-length token. */
    private static boolean isDelim(char c) {
        return c == '(' || c == ')' || c == ',' || c == ':' || c == ';';
    }

    /** Skip digits/dots/e/+/- that make up a branch length value. */
    private static int skipBranchLen(String s, int i, int n) {
        if (i < n && s.charAt(i) == ':') i++;
        while (i < n && !isDelim(s.charAt(i))) i++;
        return i;
    }

    /**
     * After a ')' we may have: optional label (bootstrap or name), optional ':len'.
     * Skip both.
     */
    private static int skipLabelAndBranchLen(String s, int i, int n) {
        // skip optional label (anything that is not a structural delimiter or ':')
        while (i < n && s.charAt(i) != ':' && !isDelim(s.charAt(i))) i++;
        return skipBranchLen(s, i, n);
    }
}
