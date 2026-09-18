#!/usr/bin/env python3
"""
compare_upgma.py — Head-to-head UPGMA guide tree comparison: ASTRAL-X vs ASTRAL-MP.

Usage:
  python3 compare_upgma.py <input.tre> [--verbose]

Checks:
  1. Both tools are run and produce a UPGMA guide tree.
  2. The set of bipartitions (unrooted) is extracted from each tree.
  3. RF distance = |only_in_X| + |only_in_MP| (normalised to [0,1] optionally).
  4. PASS if RF = 0 (identical bipartition sets).

Because UPGMA is deterministic given identical similarity matrices (which we have
already verified match), the UPGMA trees should be identical — RF = 0 is expected.
"""

import sys
import os
import subprocess
import argparse
from collections import defaultdict

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ASTRALX_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
ASTRALMP_DIR = os.path.join(ASTRALX_ROOT, "astral-my")

ASTRALX_RUN  = os.path.join(ASTRALX_ROOT, "astralx")
ASTRALMP_DEV = os.path.join(ASTRALMP_DIR, "dev.sh")


# ── Newick parser ─────────────────────────────────────────────────────────────

def parse_newick_leaves(newick_str):
    """Return the set of leaf names in a Newick string (quick and correct)."""
    # strip trailing semicolon / whitespace
    s = newick_str.strip().rstrip(";").strip()
    leaves = set()
    token = ""
    for ch in s:
        if ch in "(),;":
            t = token.strip()
            if t:
                leaves.add(t)
            token = ""
        else:
            token += ch
    t = token.strip()
    if t:
        leaves.add(t)
    return leaves


def parse_newick_bipartitions(newick_str):
    """
    Parse a Newick string and return frozenset of bipartitions.
    Each bipartition is a frozenset of two frozensets (the two sides).
    Only internal (non-root) bipartitions are included.
    """
    s = newick_str.strip().rstrip(";").strip()
    all_leaves = parse_newick_leaves(s)

    # We'll build a tree as nested lists via a simple stack parser, then
    # extract bipartitions by DFS.
    #
    # Returns a list-tree: each node is either a string (leaf) or
    # a list of children (internal).

    def parse_node(pos):
        """Parse one subtree starting at pos, return (node, next_pos)."""
        if s[pos] == '(':
            children = []
            pos += 1  # skip '('
            while True:
                child, pos = parse_node(pos)
                children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1  # skip ','
                elif pos < len(s) and s[pos] == ')':
                    pos += 1  # skip ')'
                    break
                else:
                    break
            # skip optional label / branch length after ')'
            while pos < len(s) and s[pos] not in ',()':
                pos += 1
            return children, pos
        else:
            # leaf: read until ',', ')', or end
            name = ""
            while pos < len(s) and s[pos] not in ',()':
                name += s[pos]
                pos += 1
            return name.strip(), pos

    try:
        tree, _ = parse_node(0)
    except Exception as e:
        raise ValueError(f"Newick parse error: {e}\nString: {s[:200]}")

    bipartitions = set()

    def collect_leaves(node):
        if isinstance(node, str):
            return frozenset([node])
        result = frozenset()
        for child in node:
            result = result | collect_leaves(child)
        return result

    def traverse(node, is_root=False):
        if isinstance(node, str):
            return frozenset([node])
        child_sets = [traverse(child) for child in node]
        my_leaves = frozenset().union(*child_sets)
        if not is_root:
            complement = frozenset(all_leaves - my_leaves)
            if len(my_leaves) > 0 and len(complement) > 0:
                bipartitions.add(frozenset([my_leaves, complement]))
        return my_leaves

    traverse(tree, is_root=True)
    return bipartitions, all_leaves


# ── ASTRAL-X runner ───────────────────────────────────────────────────────────

def run_astralx_upgma(input_path, verbose=False):
    """Run ASTRAL-X --verify-upgma and parse bipartitions from stdout."""
    cmd = [
        "bash", ASTRALX_RUN,
        "-i", input_path,
        "--verify-upgma",
        "--cpu",
        "--no-build",
    ]
    if verbose:
        print(f"  [ASTRAL-X] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse UPGMA_BIPARTITIONS format
    lines = result.stdout.splitlines()
    in_block = False
    taxa = None
    bipartitions = set()

    for line in lines:
        line = line.strip()
        if line == "UPGMA_BIPARTITIONS":
            in_block = True
            continue
        if not in_block:
            continue
        if line.startswith("taxa="):
            taxa = frozenset(line[5:].split(","))
        elif line.startswith("bipartition="):
            leaves = frozenset(line[12:].split(","))
            complement = frozenset(taxa - leaves)
            if leaves and complement:
                bipartitions.add(frozenset([leaves, complement]))

    if taxa is None:
        raise ValueError("Could not parse ASTRAL-X UPGMA bipartitions.\n"
                         "Stdout:\n" + result.stdout[:2000])
    return bipartitions, taxa


# ── ASTRAL-MP runner ──────────────────────────────────────────────────────────

def run_astralmp_upgma(input_path, verbose=False):
    """Run ASTRAL-MP with dumpUpgmaTree and parse bipartitions from stderr."""
    env = os.environ.copy()
    env["ASTRAL_JVM_OPTS"] = "-DdumpUpgmaTree=1"
    cmd = [
        "bash", ASTRALMP_DEV,
        "--run-only",
        "-i", input_path,
        "-o", "/tmp/astralmp_upgma_comparison_out.tre",
        "-C",
    ]
    if verbose:
        print(f"  [ASTRAL-MP] Running: ASTRAL_JVM_OPTS=-DdumpUpgmaTree=1 {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    lines = result.stderr.splitlines()
    newick = None
    in_block = False
    for line in lines:
        line = line.strip()
        if line == "ASTRALMP_UPGMA_BEGIN":
            in_block = True
            continue
        if line == "ASTRALMP_UPGMA_END":
            in_block = False
            continue
        if in_block and line:
            newick = line

    if newick is None:
        raise ValueError("Could not parse ASTRAL-MP UPGMA Newick from stderr.\n"
                         "Stderr snippet:\n" + result.stderr[:2000])

    bipartitions, taxa = parse_newick_bipartitions(newick)
    return bipartitions, taxa


# ── RF distance ───────────────────────────────────────────────────────────────

def rf_distance(bips_x, taxa_x, bips_mp, taxa_mp):
    """
    Compute unrooted RF distance between two bipartition sets.
    Bipartitions are matched by leaf-name identity.
    Returns (rf, only_in_x_count, only_in_mp_count, shared_count, max_possible).
    """
    only_x  = bips_x  - bips_mp
    only_mp = bips_mp - bips_x
    shared  = bips_x  & bips_mp
    rf = len(only_x) + len(only_mp)
    max_rf = len(bips_x) + len(bips_mp)
    return rf, len(only_x), len(only_mp), len(shared), max_rf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASTRAL-X vs ASTRAL-MP UPGMA guide tree comparison (RF distance)")
    parser.add_argument("input", help="Input gene tree file (.tre)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    print(f"\nComparing UPGMA guide trees for: {os.path.basename(input_path)}")
    print(f"  Input: {input_path}")

    print("\n  Running ASTRAL-X (--verify-upgma)...", end=" ", flush=True)
    try:
        bips_x, taxa_x = run_astralx_upgma(input_path, verbose=args.verbose)
        print(f"OK  (n={len(taxa_x)}, bipartitions={len(bips_x)})")
    except Exception as e:
        print(f"FAILED\n  Error: {e}", file=sys.stderr)
        sys.exit(2)

    print("  Running ASTRAL-MP (dumpUpgmaTree)...", end=" ", flush=True)
    try:
        bips_mp, taxa_mp = run_astralmp_upgma(input_path, verbose=args.verbose)
        print(f"OK  (n={len(taxa_mp)}, bipartitions={len(bips_mp)})")
    except Exception as e:
        print(f"FAILED\n  Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Warn on taxon set mismatch
    missing_x  = taxa_mp - taxa_x
    missing_mp = taxa_x  - taxa_mp
    if missing_x:
        print(f"  WARNING: taxa in ASTRAL-MP but not ASTRAL-X: {sorted(missing_x)[:5]}...")
    if missing_mp:
        print(f"  WARNING: taxa in ASTRAL-X but not ASTRAL-MP: {sorted(missing_mp)[:5]}...")

    rf, only_x, only_mp, shared, max_rf = rf_distance(bips_x, taxa_x, bips_mp, taxa_mp)
    norm_rf = rf / max_rf if max_rf > 0 else 0.0

    print(f"\n  Bipartitions in ASTRAL-X:            {len(bips_x)}")
    print(f"  Bipartitions in ASTRAL-MP:           {len(bips_mp)}")
    print(f"  Shared bipartitions:                 {shared}")
    print(f"  Only in ASTRAL-X:                    {only_x}")
    print(f"  Only in ASTRAL-MP:                   {only_mp}")
    print(f"  RF distance (raw):                   {rf}")
    print(f"  RF distance (normalised, /max_rf):   {norm_rf:.4f}")

    if args.verbose and only_x:
        print("\n  Bipartitions only in ASTRAL-X (first 5):")
        for bp in list(only_x)[:5] if isinstance(only_x, set) else list(bips_x - bips_mp)[:5]:
            sides = sorted(sorted(s) for s in bp)
            print(f"    {sides[0][:5]}... | {sides[1][:5]}...")
    if args.verbose and only_mp:
        print("\n  Bipartitions only in ASTRAL-MP (first 5):")
        for bp in list(bips_mp - bips_x)[:5]:
            sides = sorted(sorted(s) for s in bp)
            print(f"    {sides[0][:5]}... | {sides[1][:5]}...")

    if rf == 0:
        print(f"\n  RESULT: PASS  (RF = 0, UPGMA trees identical)")
        sys.exit(0)
    else:
        print(f"\n  RESULT: FAIL  (RF = {rf})")
        sys.exit(1)


if __name__ == "__main__":
    main()
