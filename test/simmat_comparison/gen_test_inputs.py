#!/usr/bin/env python3
"""
gen_test_inputs.py — Generate similarity matrix comparison test inputs.

Creates:
  input/tc_sm1.tre  — 6 taxa, 20 trees, mixed (from tc5_heavy_incomplete.tre)
  input/tc_sm2.tre  — 8 taxa, 30 trees, ~30% leaves missing per tree
  input/tc_sm3.tre  — 10 taxa, 50 trees, ~40% leaves missing per tree
  input/tc_sm4.tre  — 15 taxa, 80 trees, ~30% leaves missing per tree

All trees are valid Newick with semicolons. Fixed seed for reproducibility.
"""

import random
import os
import shutil

SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, "input")
ASTRALX_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
TC5_PATH = os.path.join(ASTRALX_ROOT, "test", "input", "tc5_heavy_incomplete.tre")


# ── Tree generation helpers ───────────────────────────────────────────────────

def balanced_tree(taxa):
    """Build a balanced binary species tree over taxa list (returns nested list)."""
    if len(taxa) == 1:
        return taxa[0]
    mid = len(taxa) // 2
    return (balanced_tree(taxa[:mid]), balanced_tree(taxa[mid:]))


def newick(node):
    """Convert nested-list tree to Newick string (no branch lengths)."""
    if isinstance(node, str):
        return node
    left, right = node
    return f"({newick(left)},{newick(right)})"


def random_nni(tree, rng):
    """Apply one random NNI swap on a nested-list tree, return new tree."""
    # Collect all internal nodes (non-leaf)
    nodes = []

    def collect(t):
        if isinstance(t, str):
            return
        nodes.append(t)
        collect(t[0])
        collect(t[1])

    collect(tree)
    if len(nodes) < 2:
        return tree

    # Pick a random internal node and swap one of its children with a nephew
    target = rng.choice(nodes)
    left, right = target
    if isinstance(left, tuple) and isinstance(right, str):
        # swap right with left's left child
        new_left = (right, left[1])
        new_right = left[0]
        return _replace_node(tree, target, (new_left, new_right))
    elif isinstance(right, tuple) and isinstance(left, str):
        new_right = (left, right[0])
        new_left  = right[1]
        return _replace_node(tree, target, (new_left, new_right))
    elif isinstance(left, tuple) and isinstance(right, tuple):
        # swap left's left child with right's left child
        new_left  = (left[0], right[1])
        new_right = (right[0], left[1])
        return _replace_node(tree, target, (new_left, new_right))
    return tree


def _replace_node(tree, old, new_node):
    """Replace exactly the node `old` (by identity) with `new_node`."""
    if tree is old:
        return new_node
    if isinstance(tree, str):
        return tree
    left = _replace_node(tree[0], old, new_node)
    right = _replace_node(tree[1], old, new_node)
    return (left, right)


def get_leaves(tree):
    if isinstance(tree, str):
        return [tree]
    return get_leaves(tree[0]) + get_leaves(tree[1])


def prune_tree(tree, keep_set):
    """Remove leaves not in keep_set; return pruned tree or None if empty."""
    if isinstance(tree, str):
        return tree if tree in keep_set else None
    left  = prune_tree(tree[0], keep_set)
    right = prune_tree(tree[1], keep_set)
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return (left, right)


def generate_gene_trees(species_tree, num_trees, missing_rate, rng, min_taxa=3):
    """
    Generate num_trees gene trees by:
    1. Randomly applying 1-3 NNI swaps to the species tree.
    2. Pruning leaves at the given missing_rate.
    Trees with fewer than min_taxa leaves are retried.
    """
    all_taxa = get_leaves(species_tree)
    n = len(all_taxa)
    trees = []
    while len(trees) < num_trees:
        # Apply random NNIs for ILS simulation
        gt = species_tree
        num_nni = rng.randint(0, 3)
        for _ in range(num_nni):
            gt = random_nni(gt, rng)

        # Apply missing data pruning
        keep = [t for t in all_taxa if rng.random() > missing_rate]
        if len(keep) < min_taxa:
            # Keep at least min_taxa random taxa
            keep = rng.sample(all_taxa, min_taxa)
        keep_set = set(keep)
        pruned = prune_tree(gt, keep_set)
        if pruned is None or isinstance(pruned, str):
            continue
        nwk = newick(pruned) + ";"
        trees.append(nwk)
    return trees


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # tc_sm1: copy tc5_heavy_incomplete.tre (6 taxa, 20 trees)
    tc_sm1_path = os.path.join(INPUT_DIR, "tc_sm1.tre")
    shutil.copy(TC5_PATH, tc_sm1_path)
    print(f"tc_sm1: copied from {TC5_PATH}")

    # tc_sm2: 8 taxa, 30 trees, 30% missing
    taxa8 = [f"t{i}" for i in range(8)]
    sp2 = balanced_tree(taxa8)
    trees2 = generate_gene_trees(sp2, 30, 0.30, rng)
    path2 = os.path.join(INPUT_DIR, "tc_sm2.tre")
    with open(path2, "w") as f:
        f.write("\n".join(trees2) + "\n")
    print(f"tc_sm2: 8 taxa, 30 trees -> {path2}")

    # tc_sm3: 10 taxa, 50 trees, 40% missing
    taxa10 = [f"t{i}" for i in range(10)]
    sp3 = balanced_tree(taxa10)
    trees3 = generate_gene_trees(sp3, 50, 0.40, rng)
    path3 = os.path.join(INPUT_DIR, "tc_sm3.tre")
    with open(path3, "w") as f:
        f.write("\n".join(trees3) + "\n")
    print(f"tc_sm3: 10 taxa, 50 trees -> {path3}")

    # tc_sm4: 15 taxa, 80 trees, 30% missing
    taxa15 = [f"t{i}" for i in range(15)]
    sp4 = balanced_tree(taxa15)
    trees4 = generate_gene_trees(sp4, 80, 0.30, rng)
    path4 = os.path.join(INPUT_DIR, "tc_sm4.tre")
    with open(path4, "w") as f:
        f.write("\n".join(trees4) + "\n")
    print(f"tc_sm4: 15 taxa, 80 trees -> {path4}")

    print("\nAll test inputs generated.")


if __name__ == "__main__":
    main()
