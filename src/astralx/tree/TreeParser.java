package astralx.tree;

import astralx.Logging;
import astralx.taxon.TaxonRegistry;

import java.io.*;
import java.util.*;

/**
 * Newick parser for rooted binary gene trees.
 *
 * Two-pass design:
 *   Pass 1  -- collectTaxonNames(): scan every Newick string, register all names.
 *   Pass 2  -- parseNewick(): build Tree objects with postorder arrays + node ranges.
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
        for (int i = 0; i < lines.size(); i++) {
            trees.add(parseNewick(lines.get(i), i, registry));
        }

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
     * We only care about characters that look like taxon names (not digits/symbols
     * that appear as bootstrap support inside ")" labels).
     */
    private static void collectTaxonNames(String s, TaxonRegistry reg) {
        int i = 0, n = s.length();
        while (i < n) {
            char c = s.charAt(i);
            if (c == '(' || c == ')' || c == ',') { i++; continue; }
            if (c == ';') break;
            if (c == ':') { i = skipBranchLen(s, i + 1, n); continue; }
            // potential name token
            int start = i;
            while (i < n && !isDelim(s.charAt(i))) i++;
            String tok = s.substring(start, i).trim();
            // Accept as taxon name only if it does NOT start with a digit or '['
            // (bootstrap values / comments appear after ')' and start with digits)
            if (!tok.isEmpty() && !Character.isDigit(tok.charAt(0)) && tok.charAt(0) != '[') {
                reg.register(tok);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pass 2 – full parse
    // -------------------------------------------------------------------------

    /** Sentinel node pushed onto the stack to mark an open parenthesis. */
    private static final TreeNode SENTINEL = new TreeNode();

    private static Tree parseNewick(String s, int treeIdx, TaxonRegistry reg) {
        int n = s.length(), totalTaxa = reg.size();
        Deque<TreeNode> stack = new ArrayDeque<>();
        int i = 0;

        while (i < n) {
            char c = s.charAt(i);

            if (c == '(') {
                stack.push(SENTINEL);   // marks open-paren
                i++;

            } else if (c == ')') {
                // Pop children (pushed left-to-right, popped right-to-left)
                List<TreeNode> children = new ArrayList<>();
                while (stack.peek() != SENTINEL) children.add(stack.pop());
                stack.pop();   // remove sentinel

                if (children.size() != 2) {
                    throw new RuntimeException("Tree " + treeIdx
                        + ": non-binary node with " + children.size() + " children near pos " + i);
                }
                TreeNode node = new TreeNode();
                // children were pushed L then R, so popped: [R, L]
                node.right = children.get(0);
                node.left  = children.get(1);
                node.left.parent  = node;
                node.right.parent = node;
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
                    TreeNode leaf = new TreeNode();
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
        TreeNode root = stack.pop();
        if (root.isLeaf()) {
            throw new RuntimeException("Tree " + treeIdx + ": root is a leaf");
        }

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
