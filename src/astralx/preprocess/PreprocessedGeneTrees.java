package astralx.preprocess;

import astralx.model.GeneTree;

import java.util.List;

public final class PreprocessedGeneTrees {
    public final int totalTaxa;
    public final List<GeneTree> geneTrees;
    public final List<TreePreprocessInfo> treeInfos;

    public PreprocessedGeneTrees(int totalTaxa, List<GeneTree> geneTrees, List<TreePreprocessInfo> treeInfos) {
        this.totalTaxa = totalTaxa;
        this.geneTrees = geneTrees;
        this.treeInfos = treeInfos;
    }
}
