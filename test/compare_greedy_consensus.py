#!/usr/bin/env python3
"""
compare_greedy_consensus.py — Head-to-head comparison of the 7 greedy-consensus
trees produced by ASTRAL-X (Part I) against ASTRAL-MP (astral-my, patched to
dump allGreedies via [GREEDY_DUMP_T*] sentinel lines).

For each threshold T[0]..T[6]:
  * Parse both Newicks under a shared taxon namespace.
  * Treat both as unrooted.
  * Extract the set of non-trivial unrooted bipartitions (canonicalized as the
    sorted side not containing taxon 0).
  * Report set-equality, |A \\ B|, |B \\ A|, and Robinson-Foulds distance.

Exit code: 0 if all 7 trees agree as unrooted bipartition sets, else 1.

Usage:
  python3 compare_greedy_consensus.py \\
      --astralx /tmp/astralx_gc_37.log \\
      --astralmp /tmp/astral_mp_gc_37.log \\
      [--label 37taxon]

Each input file is the verbatim run log; lines matching the sentinel pattern
are extracted.  ASTRAL-X sentinel: "T[i]_threshold_X.XXXX: (...);"
ASTRAL-MP sentinel: "[GREEDY_DUMP_Ti_threshold_X.XXXX] (...);"
"""

import argparse
import re
import sys

try:
    import dendropy
except ImportError:
    print("Install dendropy: pip install dendropy", file=sys.stderr)
    sys.exit(2)


ASTRALX_RE  = re.compile(r"^T\[(\d)\]_threshold_[\d.]+:\s*(.+);")
ASTRALMP_RE = re.compile(r"^\[GREEDY_DUMP_T(\d)_threshold_[\d.]+\]\s*(.+);+")


def extract_newicks(path, pattern):
    """Return dict {ti: newick_str} for ti in 0..6."""
    out = {}
    with open(path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                out[int(m.group(1))] = m.group(2).rstrip(";") + ";"
    return out


def bipartition_set(newick, tns):
    """Set of frozensets of taxa-not-containing-taxon-0 (canonical side)."""
    t = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tns)
    t.is_rooted = False
    t.encode_bipartitions()
    # Pick the canonical side: the side NOT containing taxon at namespace index 0
    canon_taxon = tns[0]
    bs = set()
    all_taxa = frozenset(tx.label for tx in tns)
    for edge in t.preorder_edge_iter():
        bm = edge.bipartition.leafset_bitmask
        # Map bitmask -> set of taxon labels
        leaves = frozenset(
            tns[i].label for i in range(len(tns))
            if (bm >> i) & 1
        )
        if not leaves or leaves == all_taxa:
            continue
        size = len(leaves)
        # Skip trivial (single taxon and its complement — n-1)
        if size == 1 or size == len(tns) - 1:
            continue
        # Canonical: side NOT containing canon_taxon (else the complement)
        if canon_taxon.label in leaves:
            leaves = all_taxa - leaves
        bs.add(leaves)
    return bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astralx",  required=True)
    ap.add_argument("--astralmp", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    xt = extract_newicks(args.astralx,  ASTRALX_RE)
    mt = extract_newicks(args.astralmp, ASTRALMP_RE)

    print(f"=== Head-to-head greedy consensus comparison  {args.label} ===")
    print(f"  ASTRAL-X  : extracted {len(xt)} snapshot trees from {args.astralx}")
    print(f"  ASTRAL-MP : extracted {len(mt)} snapshot trees from {args.astralmp}")

    if set(xt.keys()) != set(mt.keys()):
        print(f"FAIL: differing snapshot indices  X={sorted(xt.keys())} MP={sorted(mt.keys())}")
        sys.exit(1)

    all_pass = True
    for ti in sorted(xt.keys()):
        # Shared namespace per-pair so bipartition bitmasks align
        tns = dendropy.TaxonNamespace()
        bx = bipartition_set(xt[ti], tns)
        bm = bipartition_set(mt[ti], tns)
        same = (bx == bm)
        x_only = bx - bm
        m_only = bm - bx

        try:
            tx = dendropy.Tree.get(data=xt[ti], schema="newick", taxon_namespace=tns)
            tm = dendropy.Tree.get(data=mt[ti], schema="newick", taxon_namespace=tns)
            tx.is_rooted = False; tm.is_rooted = False
            tx.encode_bipartitions(); tm.encode_bipartitions()
            rf = dendropy.calculate.treecompare.symmetric_difference(tx, tm)
        except Exception:
            rf = -1

        verdict = "PASS" if same else "FAIL"
        print(f"  T[{ti}]  |bp_X|={len(bx):3d}  |bp_MP|={len(bm):3d}  "
              f"X\\MP={len(x_only):2d}  MP\\X={len(m_only):2d}  RF={rf:3d}  → {verdict}")
        if not same:
            all_pass = False
            for s in sorted(x_only, key=lambda x: (len(x), sorted(x)))[:5]:
                print(f"       + X only:  {{{','.join(sorted(s))}}}")
            for s in sorted(m_only, key=lambda x: (len(x), sorted(x)))[:5]:
                print(f"       - MP only: {{{','.join(sorted(s))}}}")

    print()
    print("ALL PASS" if all_pass else "MISMATCHES PRESENT")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
