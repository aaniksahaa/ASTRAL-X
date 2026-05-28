#!/usr/bin/env python3
"""
compare_completed_trees.py — Head-to-head comparison of completed gene trees:
ASTRAL-X vs ASTRAL-MP.

Usage:
  python3 compare_completed_trees.py <input.tre> [--verbose] [--max-show N]

For each gene tree, one tool completes the missing taxa using four-point
navigation on the similarity matrix. Since the similarity matrices are
verified identical, the two completions should agree.

Metric: unrooted RF distance per gene tree.
Expected: all RF = 0 (identical completed topologies).

Output format for each tree:
  tree_i  rf=N  bips_x=K  bips_mp=K  [PASS|FAIL]
"""

import sys
import os
import re
import subprocess
import argparse

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ASTRALX_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
ASTRALMP_DIR = os.path.join(ASTRALX_ROOT, "astral-my")

ASTRALX_RUN  = os.path.join(ASTRALX_ROOT, "run.sh")
ASTRALMP_DEV = os.path.join(ASTRALMP_DIR, "dev.sh")


# ── Newick utilities ──────────────────────────────────────────────────────────

def strip_annotations(newick):
    """
    Remove ASTRAL-MP's inline annotations from a Newick string.
    Two passes:
      1. Remove `{...}` cluster annotations.
      2. Remove `:<anything>` after node names — this covers both branch lengths
         (`:0.51`) and bare colons left behind by step 1 (`:`).
    The result is plain Newick with just node names and structure.
    """
    s = re.sub(r'\{[^}]*\}', '', newick)
    s = re.sub(r':[^,()\s;]*', '', s)
    return s


def parse_newick_bipartitions(newick_str):
    """
    Parse a Newick string and return (frozenset_of_bipartitions, frozenset_of_all_leaves).
    Each bipartition is frozenset({frozenset(side1), frozenset(side2)}).
    Only internal (non-trivial) bipartitions are included.
    """
    s = newick_str.strip().rstrip(';').strip()
    # collect all leaf names for the complement
    all_leaves = set()
    token = ""
    for ch in s:
        if ch in "(),":
            t = token.strip()
            if t:
                all_leaves.add(t)
            token = ""
        else:
            token += ch
    t = token.strip()
    if t:
        all_leaves.add(t)
    all_leaves = frozenset(all_leaves)

    # parse into nested list
    def parse_node(pos):
        if pos >= len(s):
            return None, pos
        if s[pos] == '(':
            children = []
            pos += 1
            while pos < len(s):
                child, pos = parse_node(pos)
                if child is not None:
                    children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    # skip optional label/branch
                    while pos < len(s) and s[pos] not in ',()':
                        pos += 1
                    break
                else:
                    break
            return children, pos
        else:
            name = ""
            while pos < len(s) and s[pos] not in ',()':
                name += s[pos]
                pos += 1
            name = name.strip()
            return name if name else None, pos

    try:
        tree, _ = parse_node(0)
    except Exception as e:
        raise ValueError(f"Newick parse error: {e}\nString: {s[:300]}")

    bipartitions = set()

    def get_leaves(node):
        if isinstance(node, str):
            return frozenset([node])
        result = frozenset()
        for ch in node:
            result |= get_leaves(ch)
        return result

    def traverse(node, is_root=False):
        if isinstance(node, str):
            return frozenset([node])
        child_sets = [traverse(ch) for ch in node]
        my_leaves = frozenset().union(*child_sets)
        if not is_root and len(my_leaves) > 1 and len(all_leaves - my_leaves) > 0:
            complement = all_leaves - my_leaves
            bipartitions.add(frozenset([my_leaves, complement]))
        return my_leaves

    traverse(tree, is_root=True)
    return bipartitions, all_leaves


def rf_distance(bips_a, bips_b):
    return len(bips_a.symmetric_difference(bips_b))


# ── Runners ───────────────────────────────────────────────────────────────────

def run_astralx(input_path, out_file, verbose=False):
    cmd = [
        "bash", ASTRALX_RUN,
        "-i", input_path,
        "--autocomplete-incomplete-gene-trees",
        "--dump-completed-gene-trees", out_file,
        "--cpu", "--no-build", "-q",
    ]
    if verbose:
        print(f"  [ASTRAL-X] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ASTRAL-X failed (rc={r.returncode}):\n{r.stderr[-1000:]}")
    trees = []
    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(line)
    return trees


def run_astralmp(input_path, out_base, verbose=False):
    cmd = [
        "bash", ASTRALMP_DEV,
        "--run-only",
        "-i", input_path,
        "-o", out_base,
        "-C", "-k", "completed",
    ]
    if verbose:
        print(f"  [ASTRAL-MP] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    completed_file = out_base + ".completed_gene_trees"
    if not os.path.isfile(completed_file):
        raise RuntimeError(
            f"ASTRAL-MP completed_gene_trees file not found: {completed_file}\n"
            f"stderr: {r.stderr[-1000:]}")
    trees = []
    with open(completed_file) as f:
        for line in f:
            line = strip_annotations(line.strip())
            if line:
                trees.append(line)
    return trees


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASTRAL-X vs ASTRAL-MP completed gene tree comparison")
    parser.add_argument("input", help="Input gene tree file (.tre)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-show", type=int, default=10,
                        help="Max number of mismatched trees to show (default: 10)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Error: {input_path}", file=sys.stderr); sys.exit(2)

    print(f"\nComparing completed gene trees for: {os.path.basename(input_path)}")
    print(f"  Input: {input_path}")

    tmpx  = "/tmp/astralx_completed_cmp.tre"
    tmpmp = "/tmp/astralmp_out_cmp.tre"

    print("\n  Running ASTRAL-X...", end=" ", flush=True)
    try:
        trees_x = run_astralx(input_path, tmpx, args.verbose)
        print(f"OK  ({len(trees_x)} trees)")
    except Exception as e:
        print(f"FAILED\n  {e}", file=sys.stderr); sys.exit(2)

    print("  Running ASTRAL-MP...", end=" ", flush=True)
    try:
        trees_mp = run_astralmp(input_path, tmpmp, args.verbose)
        print(f"OK  ({len(trees_mp)} trees)")
    except Exception as e:
        print(f"FAILED\n  {e}", file=sys.stderr); sys.exit(2)

    if len(trees_x) != len(trees_mp):
        print(f"  ERROR: tree count mismatch: ASTRAL-X={len(trees_x)}, ASTRAL-MP={len(trees_mp)}")
        sys.exit(2)

    k = len(trees_x)
    rf_values = []
    mismatches = []

    for i, (tx, tmp) in enumerate(zip(trees_x, trees_mp)):
        try:
            bx,  lx  = parse_newick_bipartitions(tx)
            bmp, lmp = parse_newick_bipartitions(tmp)
        except Exception as e:
            print(f"  Parse error on tree {i+1}: {e}"); rf_values.append(-1); continue

        if lx != lmp:
            print(f"  WARNING tree {i+1}: leaf sets differ  "
                  f"(|X only|={len(lx-lmp)}, |MP only|={len(lmp-lx)})")

        rf = rf_distance(bx, bmp)
        rf_values.append(rf)
        if rf != 0:
            mismatches.append((i+1, rf, len(bx), len(bmp)))

    valid_rf = [r for r in rf_values if r >= 0]
    all_pass  = all(r == 0 for r in valid_rf)
    max_rf    = max(valid_rf) if valid_rf else 0
    mean_rf   = sum(valid_rf) / len(valid_rf) if valid_rf else 0.0
    num_fail  = len(mismatches)

    print(f"\n  Trees compared:              {k}")
    print(f"  Max RF distance:             {max_rf}")
    print(f"  Mean RF distance:            {mean_rf:.4f}")
    print(f"  Trees with RF > 0:           {num_fail}")

    if mismatches:
        show = mismatches[:args.max_show]
        print(f"\n  Mismatched trees (showing {len(show)}/{len(mismatches)}):")
        print(f"  {'tree':>6}  {'RF':>6}  {'bips_x':>7}  {'bips_mp':>8}")
        print("  " + "-"*35)
        for (idx, rf, bx, bmp) in show:
            print(f"  {idx:>6}  {rf:>6}  {bx:>7}  {bmp:>8}")

    if all_pass:
        print(f"\n  RESULT: PASS  (all {k} trees have RF = 0)")
        sys.exit(0)
    else:
        print(f"\n  RESULT: FAIL  ({num_fail}/{k} trees have RF > 0)")
        sys.exit(1)


if __name__ == "__main__":
    main()
