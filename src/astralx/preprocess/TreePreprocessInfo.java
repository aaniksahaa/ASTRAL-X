package astralx.preprocess;

import astralx.model.TreeNode;

import java.util.IdentityHashMap;
import java.util.Map;

public final class TreePreprocessInfo {
    public final int treeIndex;
    public final int[] taxaByPostorderLeaf;   // length n, padded with -1
    public final int[] positionByTaxon;       // length n, -1 if taxon missing in tree
    public final int presentTaxaCount;
    public final Map<TreeNode, IntRange> subtreeLeafRanges;

    public TreePreprocessInfo(int treeIndex, int[] taxaByPostorderLeaf, int[] positionByTaxon,
                              int presentTaxaCount, Map<TreeNode, IntRange> subtreeLeafRanges) {
        this.treeIndex = treeIndex;
        this.taxaByPostorderLeaf = taxaByPostorderLeaf;
        this.positionByTaxon = positionByTaxon;
        this.presentTaxaCount = presentTaxaCount;
        this.subtreeLeafRanges = subtreeLeafRanges;
    }

    public static final class IntRange {
        public final int left;
        public final int right;

        public IntRange(int left, int right) {
            this.left = left;
            this.right = right;
        }
    }

    public static Map<TreeNode, IntRange> newRangeMap() {
        return new IdentityHashMap<>();
    }
}
