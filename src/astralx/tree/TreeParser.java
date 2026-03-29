package astralx.tree;

import astralx.Logging;
import astralx.taxon.TaxonRegistry;
import astralx.util.ProgressBar;

import java.io.*;
import java.util.*;

/**
 * Newick parser for binary gene trees — supports both rooted and unrooted input.
 *
 * Two-pass design:
 *   Pass 1  -- collectTaxonNames(): scan every Newick string, register all names.
 *   Pass 2  -- parseNewick(): build Tree objects with postorder arrays + node ranges.
 *
 * Parsing is intentionally lenient: any number of children is allowed during the
 * stack-based parse phase.  A separate validation+rooting step then checks:
 *
 *   Root node with 2 children → already a rooted binary tree, keep as-is.
 *   Root node with 3 children → unrooted binary tree; arbitrarily rooted here
 *                                (ASTRAL is rooting-agnostic, so any choice is fine).
 *   Any internal node with exactly 2 children → valid binary node.
 *   Any other arity → polytomy error (not yet supported).
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

        // Pass 2 – parse each tree
        List<Tree> trees = new ArrayList<>(lines.size());
        ProgressBar parseBar = new ProgressBar("Parsing trees", lines.size());
        for (int i = 0; i < lines.size(); i++) {
            trees.add(parseNewick(lines.get(i), i, registry));
            parseBar.update(i + 1);
        }
        parseBar.done();

        long ms = (System.nanoTime() - t0) / 1_000_000;
        Logging.info("Parsed %d gene trees in %d ms", trees.size(), ms);

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

    private static Tree parseNewick(String s, int treeIdx, TaxonRegistry reg) {
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

        // Validate arity and root unrooted trees; convert RawNode → binary TreeNode
        TreeNode root = validateAndConvert(rawRoot, treeIdx, true);

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

        return new Tree(treeIdx, root, postorderArray, positionMap, leafCount, totalTaxa);
    }

    /**
     * Recursively validates a RawNode tree and converts it to binary TreeNode:
     *
     *   isRoot=true, 2 children  → rooted binary root, recurse normally.
     *   isRoot=true, 3 children  → unrooted input; root arbitrarily by isolating
     *                              the first child and making a new internal node
     *                              from the remaining two. Logs a message.
     *   isRoot=false, 2 children → normal binary internal node.
     *   leaf                     → leaf node.
     *   any other arity          → RuntimeException (polytomy not supported).
     */
    private static TreeNode validateAndConvert(RawNode raw, int treeIdx, boolean isRoot) {
        if (raw.isLeaf()) {
            TreeNode leaf = new TreeNode();
            leaf.taxonId = raw.taxonId;
            return leaf;
        }

        int nc = raw.children.size();

        if (nc == 2) {
            TreeNode node = new TreeNode();
            node.left  = validateAndConvert(raw.children.get(0), treeIdx, false);
            node.right = validateAndConvert(raw.children.get(1), treeIdx, false);
            node.left.parent  = node;
            node.right.parent = node;
            return node;

        } else if (nc == 3 && isRoot) {
            // Unrooted tree: 3-furcation at root
            // Root by isolating children[0] as left and joining children[1]+children[2]
            // into a new internal right node.  Any choice gives a valid rooted binary
            // tree equivalent under ASTRAL's rooting-agnostic scoring.
            Logging.info("Tree %d: unrooted input (3-furcation at root) — rooting arbitrarily", treeIdx);

            TreeNode c0 = validateAndConvert(raw.children.get(0), treeIdx, false);
            TreeNode c1 = validateAndConvert(raw.children.get(1), treeIdx, false);
            TreeNode c2 = validateAndConvert(raw.children.get(2), treeIdx, false);

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

        } else {
            String where = isRoot ? "root" : "internal node";
            throw new RuntimeException("Tree " + treeIdx + ": " + nc
                + "-furcation at " + where + " — polytomy not supported");
        }
    }

    /**
     * Single left-to-right DFS:
     *   - Leaf: assign rangeStart=counter, rangeEnd=counter+1, increment counter,
     *           write taxonId into postorderArray[counter].
     *   - Internal: recurse into left, then right; range spans children.
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
        assignRangesAndFillArray(node.left,  arr, counter);
        assignRangesAndFillArray(node.right, arr, counter);
        node.rangeStart = node.left.rangeStart;
        node.rangeEnd   = node.right.rangeEnd;
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
