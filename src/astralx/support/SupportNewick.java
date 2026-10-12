package astralx.support;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal structure-preserving Newick reader/writer used for branch annotation.
 *
 * <p>The species tree is annotated in exactly the shape it was supplied (child
 * order, rooting, unrooted three-way roots), as ASTRAL does, so the annotated
 * output is a drop-in replacement for the input file.  Leaf labels are read the
 * same way as ASTRAL-X's gene-tree parser (raw text up to a structural
 * delimiter, trimmed) so they resolve against the same taxon registry.
 */
public final class SupportNewick {

    private SupportNewick() {}

    public static final class Node {
        public final List<Node> children = new ArrayList<>();
        public Node parent;
        public String label = "";        // leaf name or original internal label
        public String length;            // original branch-length text, or null
        public int taxonId = -1;         // leaves only
        public long[] bits;              // taxa under this node
        public int size;                 // |bits|
        public String annotation;        // written in place of the internal label
        public Double newLength;         // written in place of the original length

        public boolean isLeaf() { return children.isEmpty(); }
        public boolean isRoot() { return parent == null; }
    }

    // ------------------------------------------------------------------ parsing

    public static Node parse(String newick) {
        Parser p = new Parser(newick.trim());
        Node root = p.parseNode();
        p.skipWs();
        if (p.pos < p.s.length() && p.s.charAt(p.pos) == ';') p.pos++;
        p.skipWs();
        if (p.pos != p.s.length()) {
            throw new IllegalArgumentException("Unexpected trailing text in species tree at "
                + "position " + p.pos + ": " + snippet(p.s, p.pos));
        }
        return root;
    }

    private static String snippet(String s, int pos) {
        return s.substring(pos, Math.min(s.length(), pos + 30));
    }

    private static final class Parser {
        final String s;
        int pos = 0;
        Parser(String s) { this.s = s; }

        void skipWs() { while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) pos++; }

        static boolean isDelim(char c) {
            return c == '(' || c == ')' || c == ',' || c == ':' || c == ';' || c == '[';
        }

        Node parseNode() {
            Node n = new Node();
            skipWs();
            if (pos < s.length() && s.charAt(pos) == '(') {
                pos++;
                while (true) {
                    Node child = parseNode();
                    child.parent = n;
                    n.children.add(child);
                    skipWs();
                    if (pos < s.length() && s.charAt(pos) == ',') { pos++; continue; }
                    break;
                }
                if (pos >= s.length() || s.charAt(pos) != ')') {
                    throw new IllegalArgumentException("Expected ')' in species tree at position "
                        + pos + ": " + snippet(s, pos));
                }
                pos++;
            }
            int start = pos;
            while (pos < s.length() && !isDelim(s.charAt(pos))) pos++;
            n.label = s.substring(start, pos).trim();
            skipComment();
            skipWs();
            if (pos < s.length() && s.charAt(pos) == ':') {
                pos++;
                int ls = pos;
                while (pos < s.length() && !isDelim(s.charAt(pos))) pos++;
                n.length = s.substring(ls, pos).trim();
                skipComment();
            }
            if (n.isLeaf() && n.label.isEmpty()) {
                throw new IllegalArgumentException("Empty leaf label in species tree at position "
                    + start + ": " + snippet(s, start));
            }
            return n;
        }

        void skipComment() {
            skipWs();
            while (pos < s.length() && s.charAt(pos) == '[') {
                int end = s.indexOf(']', pos);
                if (end < 0) throw new IllegalArgumentException("Unterminated '[' comment in species tree");
                pos = end + 1;
                skipWs();
            }
        }
    }

    // ------------------------------------------------------------------ writing

    /**
     * Serialise the tree exactly as ASTRAL prints a scored species tree:
     * internal labels and internal input lengths are dropped, annotated nodes
     * print their annotation and posterior length, leaves keep their name and
     * their input length re-serialised through {@link Double#toString}, the root
     * carries nothing.  (ASTRAL's stdout line additionally ends with a space,
     * which the caller appends when printing.)
     */
    public static String write(Node root) {
        StringBuilder sb = new StringBuilder();
        write(root, sb);
        sb.append(';');
        return sb.toString();
    }

    private static void write(Node n, StringBuilder sb) {
        if (!n.isLeaf()) {
            sb.append('(');
            for (int i = 0; i < n.children.size(); i++) {
                if (i > 0) sb.append(',');
                write(n.children.get(i), sb);
            }
            sb.append(')');
            if (n.isRoot()) return;
            if (n.annotation != null) {
                sb.append(n.annotation);   // may be "" (lengths only)
                if (n.newLength != null) sb.append(':').append(Double.toString(n.newLength));
            }
        } else {
            sb.append(n.label);
            if (n.isRoot()) return;
            if (n.length != null && !n.length.isEmpty()) {
                sb.append(':').append(reserialise(n.length));
            }
        }
    }

    /** ASTRAL parses lengths into doubles and prints them with Double.toString. */
    private static String reserialise(String length) {
        try {
            return Double.toString(Double.parseDouble(length));
        } catch (NumberFormatException e) {
            return length;
        }
    }

    // ------------------------------------------------------------------ traversal helpers

    /** Post-order list of all nodes (children before parents, Newick order). */
    public static List<Node> postOrder(Node root) {
        List<Node> out = new ArrayList<>();
        collect(root, out);
        return out;
    }

    private static void collect(Node n, List<Node> out) {
        for (Node c : n.children) collect(c, out);
        out.add(n);
    }
}
