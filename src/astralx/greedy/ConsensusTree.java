package astralx.greedy;

import astralx.taxon.TaxonRegistry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.TreeSet;

/**
 * Frozen snapshot of a {@link LaminarForest}'s state at a given threshold.
 *
 * Part I (this file) provides:
 *   - {@link #toNewick(TaxonRegistry)} for head-to-head comparison against
 *     ASTRAL-MP's {@code allGreedies[i]} via diff
 *   - {@link #canonicalLeafSets()} for set-equality cross-checks against
 *     {@link LaminarOracle#canonicalLeafSets()}
 *
 * Part II will layer A_cons (postorder leaf array) + prefix-scan hash arrays
 * on top of this same structure.
 */
public final class ConsensusTree {

    /** Snapshot-internal node. */
    static final class SNode {
        final int taxonId;            // -1 if internal
        final List<SNode> children;   // empty for leaves

        SNode(int taxonId, List<SNode> children) {
            this.taxonId = taxonId;
            this.children = children;
        }

        boolean isLeaf() { return taxonId >= 0; }
    }

    private final SNode root;
    private final int numTaxa;
    private final int numInternalNodes;     // excluding virtual root
    private final int numPolytomies;        // internal nodes with > 2 children

    private ConsensusTree(SNode root, int numTaxa,
                          int numInternalNodes, int numPolytomies) {
        this.root = root;
        this.numTaxa = numTaxa;
        this.numInternalNodes = numInternalNodes;
        this.numPolytomies = numPolytomies;
    }

    /**
     * Take a snapshot of the current laminar-forest state.  Deep-copies the
     * children structure so subsequent INSERTs into the source forest do not
     * mutate the snapshot.
     */
    static ConsensusTree snapshot(LaminarForest forest) {
        int[] counts = new int[2];   // [0] internal nodes, [1] polytomies
        SNode root = copyRec(forest.virtualRoot, counts);
        return new ConsensusTree(root, forest.numTaxa, counts[0], counts[1]);
    }

    private static SNode copyRec(LaminarNode src, int[] counts) {
        if (src.isLeaf()) {
            return new SNode(src.taxonId, Collections.emptyList());
        }
        List<SNode> kids = new ArrayList<>(src.children.size());
        for (LaminarNode c : src.children) kids.add(copyRec(c, counts));
        // Don't count the virtual root itself
        if (src.parent >= 0) {
            counts[0]++;
            if (kids.size() > 2) counts[1]++;
        }
        return new SNode(-1, kids);
    }

    public int numTaxa()           { return numTaxa; }
    public int numInternalNodes()  { return numInternalNodes; }
    public int numPolytomies()     { return numPolytomies; }

    /** Newick string using taxon names; no branch lengths. */
    public String toNewick(TaxonRegistry registry) {
        StringBuilder sb = new StringBuilder();
        writeNewick(root, registry, sb);
        sb.append(';');
        return sb.toString();
    }

    private void writeNewick(SNode n, TaxonRegistry reg, StringBuilder sb) {
        if (n.isLeaf()) {
            sb.append(reg.getName(n.taxonId));
            return;
        }
        sb.append('(');
        for (int i = 0; i < n.children.size(); i++) {
            if (i > 0) sb.append(',');
            writeNewick(n.children.get(i), reg, sb);
        }
        sb.append(')');
    }

    /**
     * Canonical "{1,4,7}\n{2,5}\n..." dump — one line per non-root internal
     * node, leaf ids sorted, lines sorted lexicographically.
     */
    public String canonicalLeafSets() {
        List<String> lines = new ArrayList<>();
        TreeSet<Integer> scratch = new TreeSet<>();
        collectLeafSets(root, /*isRoot=*/true, lines, scratch);
        Collections.sort(lines);
        return String.join("\n", lines);
    }

    private void collectLeafSets(SNode n, boolean isRoot,
                                  List<String> out, TreeSet<Integer> scratch) {
        if (n.isLeaf()) return;
        if (!isRoot) {
            scratch.clear();
            collectLeaves(n, scratch);
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (int t : scratch) {
                if (!first) sb.append(',');
                sb.append(t);
                first = false;
            }
            sb.append('}');
            out.add(sb.toString());
        }
        for (SNode c : n.children) collectLeafSets(c, false, out, scratch);
    }

    private void collectLeaves(SNode n, TreeSet<Integer> out) {
        if (n.isLeaf()) { out.add(n.taxonId); return; }
        for (SNode c : n.children) collectLeaves(c, out);
    }
}
