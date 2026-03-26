package astralx.model;

import java.util.ArrayList;
import java.util.List;

public final class TreeNode {
    public int taxonId = -1;
    public final List<TreeNode> children = new ArrayList<>();
    public TreeNode parent;

    public boolean isLeaf() {
        return children.isEmpty();
    }
}
