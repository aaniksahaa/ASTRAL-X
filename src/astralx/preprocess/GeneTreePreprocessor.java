package astralx.preprocess;

import astralx.model.GeneTree;
import astralx.model.TreeNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public final class GeneTreePreprocessor {
    public PreprocessedGeneTrees preprocess(List<GeneTree> trees, int totalTaxa) {
        List<TreePreprocessInfo> infos = new ArrayList<>(trees.size());
        for (GeneTree tree : trees) {
            infos.add(preprocessSingle(tree, totalTaxa));
        }
        return new PreprocessedGeneTrees(totalTaxa, trees, infos);
    }

    private TreePreprocessInfo preprocessSingle(GeneTree tree, int totalTaxa) {
        int[] taxaByPostorder = new int[totalTaxa];
        Arrays.fill(taxaByPostorder, -1);
        int[] positionByTaxon = new int[totalTaxa];
        Arrays.fill(positionByTaxon, -1);
        Map<TreeNode, TreePreprocessInfo.IntRange> ranges = TreePreprocessInfo.newRangeMap();

        int[] writePos = new int[]{0};
        computeRanges(tree.root, taxaByPostorder, positionByTaxon, ranges, writePos);
        return new TreePreprocessInfo(tree.index, taxaByPostorder, positionByTaxon, writePos[0], ranges);
    }

    private TreePreprocessInfo.IntRange computeRanges(
            TreeNode node,
            int[] taxaByPostorder,
            int[] positionByTaxon,
            Map<TreeNode, TreePreprocessInfo.IntRange> ranges,
            int[] writePos) {

        if (node.isLeaf()) {
            int idx = writePos[0]++;
            taxaByPostorder[idx] = node.taxonId;
            positionByTaxon[node.taxonId] = idx;
            TreePreprocessInfo.IntRange r = new TreePreprocessInfo.IntRange(idx, idx);
            ranges.put(node, r);
            return r;
        }

        int leftMost = Integer.MAX_VALUE;
        int rightMost = Integer.MIN_VALUE;
        for (TreeNode child : node.children) {
            TreePreprocessInfo.IntRange childRange = computeRanges(child, taxaByPostorder, positionByTaxon, ranges, writePos);
            leftMost = Math.min(leftMost, childRange.left);
            rightMost = Math.max(rightMost, childRange.right);
        }

        TreePreprocessInfo.IntRange r = new TreePreprocessInfo.IntRange(leftMost, rightMost);
        ranges.put(node, r);
        return r;
    }
}
