package astralx.util;

import astralx.model.SpeciesNode;
import astralx.model.TaxonRegistry;

public final class NewickWriter {
    private NewickWriter() {}

    public static String toNewick(SpeciesNode root, TaxonRegistry taxa) {
        return render(root, taxa) + ";";
    }

    private static String render(SpeciesNode node, TaxonRegistry taxa) {
        if (node.isLeaf()) {
            return taxa.getLabel(node.taxonId);
        }
        return "(" + render(node.left, taxa) + "," + render(node.right, taxa) + ")";
    }
}
