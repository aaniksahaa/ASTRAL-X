#!/usr/bin/env python3
"""
compare_with_astralmp.py — Head-to-head comparison of ASTRAL-X vs ASTRAL-MP output.

Usage:
    python3 compare_with_astralmp.py \
        --clusters-astralx  /tmp/tc1_astralx_clusters.txt \
        --clusters-astralmp /tmp/tc1_astralmp_clusters.txt \
        [--tree-astralx  /tmp/tc1_astralx_tree.tre] \
        [--tree-astralmp /tmp/tc1_astralmp_tree.tre] \
        [--label TC1]

Exit code: 0 = all checks passed, 1 = at least one mismatch.

Cluster dump format (produced by both tools):
    {A,B,C}           one cluster per line, taxa sorted alphabetically
    Lines are also sorted lexicographically.
    Each line is a bipartition: the complement bipartition is NOT listed
    separately — both tools include BOTH a bipartition and its complement,
    so the full symmetric set is present and the comparison is symmetric.
"""

import sys
import argparse
import re
import subprocess
import os

RF_PY = os.path.join(os.path.dirname(__file__), '..', 'rf.py')


# ── helpers ───────────────────────────────────────────────────────────────────

def load_clusters(path: str) -> set:
    """Read a cluster dump file and return a set of frozensets of taxon names."""
    clusters = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # expect format: {A,B,C}
            m = re.fullmatch(r'\{([^}]+)\}', line)
            if not m:
                raise ValueError(f"Unexpected line format: {line!r}")
            taxa = frozenset(t.strip() for t in m.group(1).split(','))
            clusters.add(taxa)
    return clusters


def rf_distance(tree1: str, tree2: str) -> float:
    """
    Compute Robinson-Foulds distance between two Newick files using rf.py.
    Returns the RF distance (0.0 = identical topology).
    Raises RuntimeError if rf.py fails.
    """
    rf_script = os.path.normpath(RF_PY)
    result = subprocess.run(
        [sys.executable, rf_script, tree1, tree2],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    # Parse "Robinson-Foulds distance: 0.0000"
    m = re.search(r'Robinson-Foulds distance:\s*([\d.]+)', result.stdout)
    if not m:
        raise RuntimeError(f"Unexpected rf.py output: {result.stdout!r}")
    return float(m.group(1))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='ASTRAL-X vs ASTRAL-MP comparison')
    ap.add_argument('--clusters-astralx',  required=True, help='ASTRAL-X cluster dump')
    ap.add_argument('--clusters-astralmp', required=True, help='ASTRAL-MP cluster dump')
    ap.add_argument('--tree-astralx',  default=None, help='ASTRAL-X output tree (optional)')
    ap.add_argument('--tree-astralmp', default=None, help='ASTRAL-MP output tree (optional)')
    ap.add_argument('--label', default='', help='Test case label for display')
    args = ap.parse_args()

    label = f"[{args.label}] " if args.label else ''
    all_pass = True

    # ── Cluster comparison ────────────────────────────────────────────────────
    cx = load_clusters(args.clusters_astralx)
    cm = load_clusters(args.clusters_astralmp)

    only_x  = cx - cm   # in ASTRAL-X but not ASTRAL-MP
    only_mp = cm - cx   # in ASTRAL-MP but not ASTRAL-X
    common  = cx & cm

    print(f"{label}Cluster set X comparison:")
    print(f"  ASTRAL-X  : {len(cx):5d} clusters")
    print(f"  ASTRAL-MP : {len(cm):5d} clusters")
    print(f"  Common    : {len(common):5d}")
    print(f"  Only in X : {len(only_x):5d}")
    print(f"  Only in MP: {len(only_mp):5d}")

    if only_x or only_mp:
        print(f"  MISMATCH — cluster sets differ")
        all_pass = False
        # Print up to 10 discrepancies in each direction
        for i, c in enumerate(sorted(only_x, key=lambda s: sorted(s))):
            if i >= 10:
                print(f"    ... ({len(only_x) - 10} more only-in-X clusters)")
                break
            print(f"    only-in-X : {{{','.join(sorted(c))}}}")
        for i, c in enumerate(sorted(only_mp, key=lambda s: sorted(s))):
            if i >= 10:
                print(f"    ... ({len(only_mp) - 10} more only-in-MP clusters)")
                break
            print(f"    only-in-MP: {{{','.join(sorted(c))}}}")
    else:
        print(f"  PASS — cluster sets are identical")

    # ── Tree comparison via RF distance (optional) ────────────────────────────
    if args.tree_astralx and args.tree_astralmp:
        print(f"\n{label}Final species tree comparison (RF distance):")
        try:
            rf = rf_distance(args.tree_astralx, args.tree_astralmp)
            if rf == 0.0:
                print(f"  PASS — RF distance = 0 (topologically identical)")
            else:
                print(f"  MISMATCH — RF distance = {rf}")
                all_pass = False
        except Exception as e:
            print(f"  SKIP — RF check failed: {e}")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
