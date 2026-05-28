#!/usr/bin/env python3
"""
compare_step_a.py — Head-to-head comparison of Step A (polytomy resolution via
UPGMA on group similarity matrix) emissions between ASTRAL-X and ASTRAL-MP.

Both sources emit lines of the form "[STEPA_EMIT] ti=I size=S {taxa}" (MP, by
patched dump) and "[STEPA] ti=I  size=S  {taxa}" (X verifier output).  We parse
both, canonicalize each emission to its frozen taxa set, dedupe within each
source, and compare set-equality.

Exit code 0 = sets identical; 1 = differ.
"""

import argparse
import re
import sys

ASTRALX_RE  = re.compile(r"\[STEPA\]\s+ti=\d+\s+size=\d+\s+\{([^}]*)\}")
ASTRALMP_RE = re.compile(r"\[STEPA_EMIT\]\s+ti=\d+\s+size=\d+\s+\{([^}]*)\}")


def extract_sets(path, pattern, all_taxa=None):
    """Emit a set of canonicalized taxa-frozensets — always the SIDE not
    containing the lex-smallest taxon (canonical), so complementary emissions
    from the two tools collapse to the same bipartition representative."""
    seen = set()
    raw = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if not m: continue
            taxa = frozenset(t.strip() for t in m.group(1).split(",") if t.strip())
            raw.append(taxa)
    if all_taxa is None:
        all_taxa = frozenset().union(*raw) if raw else frozenset()
    for taxa in raw:
        comp = all_taxa - taxa
        # Canonical = the side that does NOT contain min(all_taxa)
        canon = (comp if min(all_taxa) in taxa else taxa)
        seen.add(canon)
    return seen, all_taxa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astralx", required=True)
    ap.add_argument("--astralmp", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    # Build all_taxa from union of both sources so canonicalization is consistent
    _, x_all = extract_sets(args.astralx,  ASTRALX_RE)
    _, m_all = extract_sets(args.astralmp, ASTRALMP_RE)
    all_taxa = x_all | m_all
    xs, _ = extract_sets(args.astralx,  ASTRALX_RE,  all_taxa)
    ms, _ = extract_sets(args.astralmp, ASTRALMP_RE, all_taxa)

    print(f"=== Step A head-to-head  {args.label} ===")
    print(f"  ASTRAL-X  : {len(xs):3d} unique bipartitions  ({args.astralx})")
    print(f"  ASTRAL-MP : {len(ms):3d} unique bipartitions  ({args.astralmp})")

    x_only = xs - ms
    m_only = ms - xs
    common = xs & ms

    print(f"  Common    : {len(common)}")
    print(f"  X only    : {len(x_only)}")
    print(f"  MP only   : {len(m_only)}")

    for s in sorted(x_only, key=lambda x: (len(x), sorted(x)))[:10]:
        print(f"    + X only:  size={len(s)}  {{{','.join(sorted(s))}}}")
    for s in sorted(m_only, key=lambda x: (len(x), sorted(x)))[:10]:
        print(f"    - MP only: size={len(s)}  {{{','.join(sorted(s))}}}")

    if xs == ms:
        print(f"  → PASS  (Step A emissions identical)")
        sys.exit(0)
    print(f"  → FAIL  (Step A emissions differ)")
    sys.exit(1)


if __name__ == "__main__":
    main()
