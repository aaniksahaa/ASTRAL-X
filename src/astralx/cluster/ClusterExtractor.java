package astralx.cluster;

import astralx.hash.ClusterHashVector;
import astralx.hash.PrefixHashIndex;
import astralx.model.GeneTree;
import astralx.model.TreeNode;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.preprocess.TreePreprocessInfo;

import java.util.ArrayList;
import java.util.List;

public final class ClusterExtractor {
    private int nextId = 0;

    public static final class Result {
        public final ClusterTable table;
        public final Cluster allTaxaCluster;

        public Result(ClusterTable table, Cluster allTaxaCluster) {
            this.table = table;
            this.allTaxaCluster = allTaxaCluster;
        }
    }

    public Result build(PreprocessedGeneTrees prep, PrefixHashIndex hashIndex, boolean treatInputAsUnrooted) {
        long startNs = System.nanoTime();
        ClusterTable table = new ClusterTable();

        Cluster allTaxa = new Cluster(nextId++, -1, -1, -1, false, false, true,
                hashIndex.allTaxaTotal().copy(), prep.totalTaxa);

        int processedTrees = 0;
        for (GeneTree tree : prep.geneTrees) {
            TreePreprocessInfo info = prep.treeInfos.get(tree.index);
            traverse(tree.root, tree, info, prep, hashIndex, table, treatInputAsUnrooted);
            processedTrees++;
            if (processedTrees % 50 == 0) {
                System.out.printf("Cluster extraction progress: %d/%d trees%n", processedTrees, prep.geneTrees.size());
                System.out.flush();
            }
        }

        List<Cluster> original = new ArrayList<>(table.uniqueClusters());
        for (Cluster c : original) {
            if (c.allTaxa || c.globalComplement) {
                continue;
            }
            Cluster superComplement = createClusterFromDescriptor(
                    c.sourceTreeIndex,
                    c.left,
                    c.right,
                    c.localComplement,
                    true,
                    prep,
                    hashIndex
            );
            table.upsert(superComplement);
        }

        double seconds = (System.nanoTime() - startNs) / 1_000_000_000.0;
        System.out.printf("Cluster extraction done in %.2fs (unique=%d)%n", seconds, table.uniqueClusters().size());
        System.out.flush();
        return new Result(table, allTaxa);
    }

    private void traverse(TreeNode node, GeneTree tree, TreePreprocessInfo info, PreprocessedGeneTrees prep,
                          PrefixHashIndex hashIndex, ClusterTable table, boolean treatInputAsUnrooted) {
        if (node != tree.root) {
            TreePreprocessInfo.IntRange range = info.subtreeLeafRanges.get(node);
            Cluster subtree = createClusterFromDescriptor(tree.index, range.left, range.right, false, false, prep, hashIndex);
            table.upsert(subtree);

            Cluster subtreeComplement = createClusterFromDescriptor(tree.index, range.left, range.right, true, false, prep, hashIndex);
            table.upsert(subtreeComplement);

            if (treatInputAsUnrooted && !node.isLeaf()) {
                for (TreeNode child : node.children) {
                    TreePreprocessInfo.IntRange childRange = info.subtreeLeafRanges.get(child);
                    table.upsert(createClusterFromDescriptor(tree.index, childRange.left, childRange.right, false, false, prep, hashIndex));
                    table.upsert(createClusterFromDescriptor(tree.index, childRange.left, childRange.right, true, false, prep, hashIndex));
                }
            }
        }

        for (TreeNode child : node.children) {
            traverse(child, tree, info, prep, hashIndex, table, treatInputAsUnrooted);
        }
    }

    public Cluster createClusterFromDescriptor(int treeIndex, int left, int right, boolean localComplement,
                                               boolean globalComplement, PreprocessedGeneTrees prep,
                                               PrefixHashIndex hashIndex) {
        ClusterHashVector rangeHash = hashIndex.rangeHash(treeIndex, left, right);
        ClusterHashVector local;
        if (!localComplement) {
            local = rangeHash;
        } else {
            local = ClusterHashVector.subtract(hashIndex.treePrefix(treeIndex).treeTotal, rangeHash);
        }

        ClusterHashVector finalHash = local;
        if (globalComplement) {
            finalHash = ClusterHashVector.subtract(hashIndex.allTaxaTotal(), local);
        }

        int rangeSize = right - left + 1;
        int localSize = localComplement
                ? prep.treeInfos.get(treeIndex).presentTaxaCount - rangeSize
                : rangeSize;
        int size = globalComplement ? prep.totalTaxa - localSize : localSize;

        return new Cluster(nextId++, treeIndex, left, right, localComplement, globalComplement, false, finalHash, size);
    }
}
