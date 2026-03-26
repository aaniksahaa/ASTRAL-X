package astralx.partition;

import astralx.cluster.Cluster;
import astralx.cluster.ClusterExtractor;
import astralx.cluster.ClusterTable;
import astralx.hash.PrefixHashIndex;
import astralx.model.GeneTree;
import astralx.model.TreeNode;
import astralx.preprocess.PreprocessedGeneTrees;
import astralx.preprocess.TreePreprocessInfo;

import java.util.Arrays;

public final class TripartitionExtractor {
    public PartitionTable extract(
            PreprocessedGeneTrees prep,
            PrefixHashIndex hashIndex,
            ClusterExtractor clusterExtractor,
            ClusterTable clusterTable,
            boolean treatInputAsUnrooted,
            boolean includePolytomies) {

        PartitionTable table = new PartitionTable();
        if (includePolytomies) {
            throw new UnsupportedOperationException("Polytomy partition extraction is planned but not implemented in this baseline.");
        }

        int processedTrees = 0;
        for (GeneTree tree : prep.geneTrees) {
            TreePreprocessInfo info = prep.treeInfos.get(tree.index);
            traverse(tree.root, tree, info, prep, hashIndex, clusterExtractor, clusterTable, table, treatInputAsUnrooted);
            processedTrees++;
            if (processedTrees % 50 == 0) {
                System.out.printf("Tripartition extraction progress: %d/%d trees%n", processedTrees, prep.geneTrees.size());
                System.out.flush();
            }
        }
        return table;
    }

    private void traverse(TreeNode node, GeneTree tree, TreePreprocessInfo info, PreprocessedGeneTrees prep,
                          PrefixHashIndex hashIndex, ClusterExtractor clusterExtractor, ClusterTable clusterTable,
                          PartitionTable partitionTable,
                          boolean treatInputAsUnrooted) {

        for (TreeNode child : node.children) {
            traverse(child, tree, info, prep, hashIndex, clusterExtractor, clusterTable, partitionTable, treatInputAsUnrooted);
        }

        if (node == tree.root || node.isLeaf()) {
            return;
        }

        TreeNode leftChild = node.children.get(0);
        TreeNode rightChild = node.children.get(1);

        TreePreprocessInfo.IntRange lr = info.subtreeLeafRanges.get(leftChild);
        TreePreprocessInfo.IntRange rr = info.subtreeLeafRanges.get(rightChild);

        Cluster a = clusterTable.upsert(clusterExtractor.createClusterFromDescriptor(tree.index, lr.left, lr.right, false, false, prep, hashIndex));
        Cluster b = clusterTable.upsert(clusterExtractor.createClusterFromDescriptor(tree.index, rr.left, rr.right, false, false, prep, hashIndex));

        partitionTable.upsert(new Partition(Arrays.asList(a, b), false, tree.index));

        if (treatInputAsUnrooted) {
            Cluster ac = clusterTable.upsert(clusterExtractor.createClusterFromDescriptor(tree.index, lr.left, lr.right, true, false, prep, hashIndex));
            Cluster bc = clusterTable.upsert(clusterExtractor.createClusterFromDescriptor(tree.index, rr.left, rr.right, true, false, prep, hashIndex));
            partitionTable.upsert(new Partition(Arrays.asList(ac, b), false, tree.index));
            partitionTable.upsert(new Partition(Arrays.asList(a, bc), false, tree.index));
        }
    }
}
