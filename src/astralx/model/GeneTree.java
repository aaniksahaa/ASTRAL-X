package astralx.model;

import java.util.ArrayList;
import java.util.List;

public final class GeneTree {
    public final int index;
    public final TreeNode root;
    public final List<TreeNode> postorderNodes;

    public GeneTree(int index, TreeNode root, List<TreeNode> postorderNodes) {
        this.index = index;
        this.root = root;
        this.postorderNodes = new ArrayList<>(postorderNodes);
    }
}
