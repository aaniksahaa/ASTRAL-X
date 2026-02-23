package astralx.cluster;

import astralx.hash.ClusterHashVector;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.preprocess.TreePreprocessInfo;

public final class Cluster {
    public final int id;
    public final int sourceTreeIndex; // -1 for all taxa cluster
    public final int left;
    public final int right;
    public final boolean localComplement;
    public final boolean globalComplement;
    public final boolean allTaxa;
    public final ClusterHashVector hash;
    public final int size;

    public Cluster(int id, int sourceTreeIndex, int left, int right, boolean localComplement,
                   boolean globalComplement, boolean allTaxa, ClusterHashVector hash, int size) {
        this.id = id;
        this.sourceTreeIndex = sourceTreeIndex;
        this.left = left;
        this.right = right;
        this.localComplement = localComplement;
        this.globalComplement = globalComplement;
        this.allTaxa = allTaxa;
        this.hash = hash;
        this.size = size;
    }

    public boolean containsTaxon(int taxonId, PreprocessedGeneTrees prep) {
        if (allTaxa) {
            return true;
        }
        TreePreprocessInfo info = prep.treeInfos.get(sourceTreeIndex);
        int pos = info.positionByTaxon[taxonId];
        boolean local = pos >= 0 && pos >= left && pos <= right;
        if (localComplement) {
            local = !local && pos >= 0;
        }
        if (globalComplement) {
            return !local;
        }
        return local;
    }

    @Override
    public String toString() {
        if (allTaxa) {
            return "ALL_TAXA";
        }
        return "Cluster{id=" + id + ",tree=" + sourceTreeIndex + ",range=[" + left + "," + right + "]"
                + ",localComp=" + localComplement + ",globalComp=" + globalComplement + ",size=" + size + "}";
    }
}
