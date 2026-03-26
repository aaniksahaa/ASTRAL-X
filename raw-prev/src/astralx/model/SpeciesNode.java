package astralx.model;

public final class SpeciesNode {
    public SpeciesNode left;
    public SpeciesNode right;
    public int taxonId = -1;

    public static SpeciesNode leaf(int taxonId) {
        SpeciesNode n = new SpeciesNode();
        n.taxonId = taxonId;
        return n;
    }

    public static SpeciesNode internal(SpeciesNode left, SpeciesNode right) {
        SpeciesNode n = new SpeciesNode();
        n.left = left;
        n.right = right;
        return n;
    }

    public boolean isLeaf() {
        return taxonId >= 0;
    }
}
