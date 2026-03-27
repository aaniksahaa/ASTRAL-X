#!/usr/bin/env python3
"""
Reference ASTRAL-X weight verifier.

Independently recomputes:
  1. Gene-tree tripartitions (M1|M2|M3) with frequencies.
  2. Candidate clusters (sub(u) and S\\sub(u) for every non-root u).
  3. QI weight for each (candidate_split, tripartition) pair using:
       - lgA = |A ∩ Lg_GT|         (row sum, correct for incomplete trees)
       - c2  = sz3 - a2 - b2       (column M3 constraint, correct formula)
  4. Final split scores and inferred species tree.

Matches ASTRAL-X exactly.  Any discrepancy indicates a bug in either.
"""

import sys, re
from collections import defaultdict
from itertools import permutations as iperms

# ─── Newick parser ────────────────────────────────────────────────────────────

def _top_commas(s):
    """Positions of ',' at parenthesis depth 1."""
    depth, pos = 0, []
    for i, c in enumerate(s):
        if   c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == ',' and depth == 1: pos.append(i)
    return pos

def parse_newick(s):
    s = s.strip().rstrip(';').strip()
    # strip branch lengths / internal labels at this level
    s = re.sub(r':[^,)]*', '', s)   # remove ':0.12' annotations
    if not s.startswith('('):
        return frozenset([s.strip()])
    commas = _top_commas(s)
    if len(commas) != 1:
        raise ValueError(f"Non-binary node (commas={len(commas)}): {s[:60]}")
    c = commas[0]
    left  = parse_newick(s[1:c])
    right = parse_newick(s[c+1:-1])
    return (left, right, leaves(left) | leaves(right))  # (left_leaves, right_leaves, own_leaves)

def leaves(node):
    if isinstance(node, frozenset): return node
    return node[2]

def subtrees(node):
    """All (left, right, own_leaves) internal nodes in the tree, post-order."""
    if isinstance(node, frozenset): return []
    result = subtrees(node[0]) + subtrees(node[1]) + [node]
    return result

# ─── Tripartitions ────────────────────────────────────────────────────────────

def extract_tripartitions(parsed_trees):
    """
    For each non-root internal node u of gene tree g (leaf set Lg):
      M1 = left child's leaves
      M2 = right child's leaves
      M3 = Lg - M1 - M2
    Returns: dict{ (M1,M2,M3) -> frequency }  with M1 <= M2 canonically.
    """
    triparts = defaultdict(int)
    for (tree, lg) in parsed_trees:
        nodes = subtrees(tree)
        root_leaves = lg
        for node in nodes:
            own = node[2]
            if own == root_leaves:
                continue   # skip root
            m1, m2 = node[0], node[1]
            if isinstance(m1, tuple): m1 = m1[2]
            if isinstance(m2, tuple): m2 = m2[2]
            m3 = lg - m1 - m2
            if not m3:
                continue   # shouldn't happen for non-root
            key = tuple(sorted([m1, m2], key=lambda x: sorted(x))) + (m3,)
            triparts[key] += 1
    return triparts

# ─── Clusters ─────────────────────────────────────────────────────────────────

def extract_clusters(parsed_trees, S):
    """
    For every non-root internal node u in every gene tree:
      register sub(u)   and   S\\sub(u).
    Also: singletons are always valid (leaf DP nodes).
    Returns: dict{ frozenset -> count }
    """
    clusters = defaultdict(int)
    # Singletons are always in the candidate set
    for t in S:
        clusters[frozenset([t])] += 0  # ensure present with count>=0
    for (tree, lg) in parsed_trees:
        nodes = subtrees(tree)
        root_leaves = lg
        for node in nodes:
            own = node[2]
            if own == root_leaves: continue
            # sub(u)
            clusters[own] += 1
            # super-complement S \ sub(u)
            sc = S - own
            if sc: clusters[sc] += 1
        # also register Lg and S\Lg for incomplete trees (Type 3 DP)
        if lg != S:
            clusters[lg]       += 1
            clusters[S - lg]   += 1
    return clusters

# ─── QI computation ───────────────────────────────────────────────────────────

_PERMS6 = list(iperms([0, 1, 2]))

def two_qi(a, b, c):
    """2*QI = sum_{(i,j,k) perm} a[i]*b[j]*c[k]*(a[i]+b[j]+c[k]-3)."""
    res = 0
    for (i, j, k) in _PERMS6:
        ai, bj, ck = a[i], b[j], c[k]
        s = ai + bj + ck - 3
        if s > 0:
            res += ai * bj * ck * s
    return res


def score_split(A_set, B_set, S, triparts, verbose=False):
    """
    Score the species-tree split (A_set | B_set), where C = S - A - B.

    For each tripartition (M1, M2, M3) with frequency f:
      a[i] = |A ∩ Mi|,  b[i] = |B ∩ Mi|
      lgA   = |A ∩ (M1∪M2∪M3)|          ← row sum for A (correct for incomplete tGT)
      lgB   = |B ∩ (M1∪M2∪M3)|
      a2    = lgA - a0 - a1
      b2    = lgB - b0 - b1
      c0    = sz1 - a0 - b0
      c1    = sz2 - a1 - b1
      c2    = sz3 - a2 - b2              ← column M3 (CORRECT, NOT sz3-c0-c1)

    Returns: score = (1/2) * sum_P freq * 2*QI
    """
    two_score = 0
    for (m1, m2, m3), freq in triparts.items():
        lg  = m1 | m2 | m3
        sz1, sz2, sz3 = len(m1), len(m2), len(m3)

        a0 = len(A_set & m1);  a1 = len(A_set & m2)
        b0 = len(B_set & m1);  b1 = len(B_set & m2)

        lgA = len(A_set & lg)
        lgB = len(B_set & lg)

        a2 = lgA - a0 - a1
        b2 = lgB - b0 - b1
        c0 = sz1 - a0 - b0
        c1 = sz2 - a1 - b1
        c2 = sz3 - a2 - b2   # CORRECT column M3 formula

        if a2 < 0 or b2 < 0 or c0 < 0 or c1 < 0 or c2 < 0:
            if verbose:
                print(f"    SKIP  sz={sz1}|{sz2}|{sz3} lgA={lgA} lgB={lgB} "
                      f"a=[{a0},{a1},{a2}] b=[{b0},{b1},{b2}] c=[{c0},{c1},{c2}]")
            continue

        tqi = two_qi([a0,a1,a2], [b0,b1,b2], [c0,c1,c2])
        if verbose:
            print(f"    PART  sz={sz1}|{sz2}|{sz3} lgA={lgA} lgB={lgB} "
                  f"a=[{a0},{a1},{a2}] b=[{b0},{b1},{b2}] c=[{c0},{c1},{c2}] "
                  f"2*QI={tqi} freq={freq}")
        two_score += freq * tqi

    return two_score // 2

# ─── DP ───────────────────────────────────────────────────────────────────────

def infer(S, clusters, scores_map):
    """
    ASTRAL inference DP.
    dp[cluster] = max over all valid splits B|R of cluster:
                    score(B, R) + dp[B] + dp[R]
    A valid split of cluster P: B ⊂ P, R = P-B, both B and R in clusters ∪ {S}.
    """
    all_cl = set(clusters) | {S}

    # Build adjacency: parent → list of (B, R)
    adj = defaultdict(list)
    for P in all_cl:
        for B in clusters:
            if not (B < P): continue
            R = P - B
            if R not in all_cl: continue
            if len(B) > len(R): continue    # canonical: smaller or lex-first
            elif len(B) == len(R) and sorted(B) > sorted(R): continue
            adj[P].append((B, R))

    memo = {}
    best = {}

    def dp(P):
        if P in memo: return memo[P]
        if len(P) == 1:
            memo[P] = 0; return 0
        best_sc = -1
        for (B, R) in adj.get(P, []):
            key = (min(B,R,key=lambda x:sorted(x)), max(B,R,key=lambda x:sorted(x)))
            sc = scores_map.get(key, 0) + dp(B) + dp(R)
            if sc > best_sc:
                best_sc = sc
                best[P] = (B, R)
        memo[P] = max(best_sc, 0)
        return memo[P]

    total = dp(S)

    def newick(P):
        if len(P) == 1: return next(iter(P))
        if P not in best: return '(' + ','.join(sorted(P)) + ')'
        B, R = best[P]
        return '(' + newick(B) + ',' + newick(R) + ')'

    return total, newick(S) + ';'

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='Gene-tree file (one Newick per line)')
    ap.add_argument('--dump-splits',  action='store_true')
    ap.add_argument('--dump-triparts', action='store_true')
    ap.add_argument('--dump-clusters', action='store_true')
    ap.add_argument('--verbose-score', action='store_true',
                    help='Print per-(split,partition) QI values')
    ap.add_argument('--compare', metavar='TRUE_NEWICK',
                    help='Compute RF against this tree string')
    args = ap.parse_args()

    # --- parse ---
    raw = [l.strip() for l in open(args.input) if l.strip() and not l.startswith('#')]
    parsed = []
    for line in raw:
        try:
            t = parse_newick(line)
            lg = leaves(t)
            parsed.append((t, lg))
        except Exception as e:
            print(f"[warn] skip: {e}", file=sys.stderr)
    if not parsed: sys.exit("No trees parsed")

    S = frozenset().union(*(lg for _, lg in parsed))
    n = len(S)
    print(f"=== {len(parsed)} gene trees, {n} taxa: {sorted(S)} ===")

    # --- tripartitions ---
    triparts = extract_tripartitions(parsed)
    print(f"[Tripartitions] {len(triparts)} unique")
    if args.dump_triparts:
        for (m1,m2,m3), f in sorted(triparts.items(), key=lambda x: (-x[1], sorted(x[0][0]))):
            print(f"  freq={f:3d}  {sorted(m1)} | {sorted(m2)} | {sorted(m3)}  sz={len(m1)}|{len(m2)}|{len(m3)}")

    # --- clusters ---
    clusters = extract_clusters(parsed, S)
    print(f"[Clusters] {len(clusters)} unique")
    if args.dump_clusters:
        for cl, f in sorted(clusters.items(), key=lambda x:(len(x[0]),sorted(x[0]))):
            print(f"  sz={len(cl):2d}  freq={f:3d}  {sorted(cl)}")

    # --- score all candidate splits ---
    all_cl = set(clusters) | {S}
    scores_map = {}
    n_scored = 0
    for P in all_cl:
        for B in clusters:
            if not (B < P): continue
            R = P - B
            if R not in all_cl: continue
            if len(B) > len(R): continue
            if len(B) == len(R) and sorted(B) > sorted(R): continue
            key = (min(B,R,key=lambda x:sorted(x)), max(B,R,key=lambda x:sorted(x)))
            if key in scores_map: continue
            if args.verbose_score:
                print(f"SPLIT sz={len(B)}|{len(R)}  {sorted(B)} | {sorted(R)}")
            sc = score_split(B, R, S, triparts, verbose=args.verbose_score)
            if args.verbose_score:
                print(f"  => score={sc}")
            scores_map[key] = sc
            n_scored += 1

    print(f"[Scores] {n_scored} candidate splits scored")
    if args.dump_splits:
        for (B, R), sc in sorted(scores_map.items(), key=lambda x: -x[1]):
            print(f"  score={sc:8d}  {sorted(B)} | {sorted(R)}")

    # --- DP ---
    total, tree = infer(S, clusters, scores_map)
    print(f"[Inference] quartet score = {total}")
    print(f"Species tree: {tree}")

    # --- RF ---
    if args.compare:
        try:
            import dendropy
            from dendropy.calculate import treecompare
            tns = dendropy.TaxonNamespace()
            t1 = dendropy.Tree.get(data=args.compare, schema='newick', taxon_namespace=tns)
            t2 = dendropy.Tree.get(data=tree, schema='newick', taxon_namespace=tns)
            rf = treecompare.symmetric_difference(t1, t2)
            mx = max(1, 2*(n-3))
            print(f"RF = {rf}  (norm={rf/mx:.4f}, similarity={1-rf/mx:.1%})")
        except ImportError:
            print("(dendropy not available)")

if __name__ == '__main__':
    main()
