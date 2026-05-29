#!/usr/bin/env python3
"""
compare_step_b.py — Head-to-head comparison of Step B emissions (polytomy
resolution via sampleAndResolve / resolveLinearly) between ASTRAL-X and the
patched astral-my.

Tag formats:
   ASTRAL-X  : "[STEPB] ti=I  size=S  {taxa}"
   ASTRAL-MP : "[STEPB_EMIT] size=S {taxa}"

Each emission is canonicalized to the side NOT containing the lex-smallest
taxon so complementary emissions collapse.

Exit code 0 = identical sets, 1 = differ.
"""

import argparse
import re
import sys

ASTRALX_RE  = re.compile(r"\[STEPB\]\s+ti=\d+\s+size=\d+\s+\{([^}]*)\}")
ASTRALMP_RE = re.compile(r"\[STEPB_EMIT\]\s+size=\d+\s+\{([^}]*)\}")


def extract_sets(path, pattern, all_taxa=None):
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
        canon = (comp if min(all_taxa) in taxa else taxa)
        seen.add(canon)
    return seen, all_taxa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astralx",  required=True)
    ap.add_argument("--astralmp", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    _, x_all = extract_sets(args.astralx,  ASTRALX_RE)
    _, m_all = extract_sets(args.astralmp, ASTRALMP_RE)
    all_taxa = x_all | m_all
    xs, _ = extract_sets(args.astralx,  ASTRALX_RE,  all_taxa)
    ms, _ = extract_sets(args.astralmp, ASTRALMP_RE, all_taxa)

    print(f"=== Step B head-to-head  {args.label} ===")
    print(f"  ASTRAL-X  : {len(xs):3d} unique bipartitions  ({args.astralx})")
    print(f"  ASTRAL-MP : {len(ms):3d} unique bipartitions  ({args.astralmp})")

    common = xs & ms
    x_only = xs - ms
    m_only = ms - xs

    print(f"  Common    : {len(common)}")
    print(f"  X only    : {len(x_only)}")
    print(f"  MP only   : {len(m_only)}")

    for s in sorted(x_only, key=lambda x: (len(x), sorted(x)))[:10]:
        print(f"    + X only:  size={len(s)}  {{{','.join(sorted(s))}}}")
    for s in sorted(m_only, key=lambda x: (len(x), sorted(x)))[:10]:
        print(f"    - MP only: size={len(s)}  {{{','.join(sorted(s))}}}")

    if xs == ms:
        print(f"  → PASS")
        sys.exit(0)
    print(f"  → FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
