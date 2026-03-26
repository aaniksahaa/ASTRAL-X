package astralx.parse;

import astralx.model.GeneTree;
import astralx.model.TaxonRegistry;
import astralx.model.TreeNode;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public final class NewickParser {
    public GeneTree parseGeneTree(String line, int treeIndex, TaxonRegistry taxa) {
        String s = line.trim();
        if (s.isEmpty()) {
            throw new IllegalArgumentException("Empty Newick line");
        }

        Stack<TreeNode> stack = new Stack<>();
        int i = 0;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(null);
                i++;
            } else if (c == ')') {
                List<TreeNode> children = new ArrayList<>();
                while (!stack.isEmpty() && stack.peek() != null) {
                    children.add(0, stack.pop());
                }
                if (stack.isEmpty()) {
                    throw new IllegalArgumentException("Malformed Newick: unmatched ')' at " + i);
                }
                stack.pop();
                TreeNode internal = new TreeNode();
                for (TreeNode child : children) {
                    internal.children.add(child);
                    child.parent = internal;
                }
                stack.push(internal);
                i = skipLabelAndBranch(s, i + 1);
            } else if (c == ',' || c == ';' || Character.isWhitespace(c)) {
                i++;
            } else {
                int start = i;
                while (i < s.length()) {
                    char t = s.charAt(i);
                    if (t == ',' || t == ')' || t == ':' || t == ';') {
                        break;
                    }
                    i++;
                }
                String label = s.substring(start, i).trim();
                if (!label.isEmpty()) {
                    TreeNode leaf = new TreeNode();
                    leaf.taxonId = taxa.getOrCreate(label);
                    stack.push(leaf);
                }
                i = skipBranch(s, i);
            }
        }

        if (stack.size() != 1 || stack.peek() == null) {
            throw new IllegalArgumentException("Malformed Newick: expected single root");
        }
        TreeNode root = stack.pop();
        validateBinary(root);

        List<TreeNode> postorder = new ArrayList<>();
        collectPostorder(root, postorder);
        return new GeneTree(treeIndex, root, postorder);
    }

    private static int skipLabelAndBranch(String s, int i) {
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == ':' || c == ',' || c == ')' || c == ';') {
                break;
            }
            i++;
        }
        return skipBranch(s, i);
    }

    private static int skipBranch(String s, int i) {
        if (i < s.length() && s.charAt(i) == ':') {
            i++;
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ',' || c == ')' || c == ';') {
                    break;
                }
                i++;
            }
        }
        return i;
    }

    private static void validateBinary(TreeNode node) {
        if (node.isLeaf()) {
            return;
        }
        if (node.children.size() != 2) {
            throw new IllegalArgumentException("Only rooted binary trees are supported for now");
        }
        for (TreeNode child : node.children) {
            validateBinary(child);
        }
    }

    private static void collectPostorder(TreeNode node, List<TreeNode> out) {
        for (TreeNode child : node.children) {
            collectPostorder(child, out);
        }
        out.add(node);
    }
}
