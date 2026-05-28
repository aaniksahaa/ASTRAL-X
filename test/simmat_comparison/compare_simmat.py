#!/usr/bin/env python3
"""
compare_simmat.py — Head-to-head similarity matrix comparison between ASTRAL-X and ASTRAL-MP.

Usage:
  python3 compare_simmat.py <input.tre> [--tol 1e-5] [--verbose]

What it does:
  1. Run ASTRAL-X --verify-similarity-matrix on input.tre, parse matrix from stdout.
  2. Run ASTRAL-MP with ASTRAL_JVM_OPTS=-DdumpSimMatrix=1, parse matrix from stderr.
  3. Build name→index maps for each tool.
  4. Compare sim[i][j] values by name pair, within tolerance.
  5. Report max/mean absolute diff, number of pairs exceeding tol.
  6. Exit 0 if all pairs within tol, else exit 1.

Both tools now use the same algorithm: for each tree T containing both a and b,
  num_T(a,b) = same_side_T(a,b)   // # quartets where a,b are on the same side
  den_T(a,b) = C2(k_t − 2)
ASTRAL-X (CPU) replicates ASTRAL-MP's scatter exactly via the (left, right,
others) component decomposition. ASTRAL-X (GPU) uses the validated bridge
identity same_side = C2(kt-2) - QD with an O(1) Euler-tour + RMQ kernel.
Matrices match byte-for-byte modulo float vs double rounding (max diff ~1e-8).
"""

import sys
import os
import subprocess
import argparse

# Paths (resolved relative to this script's location)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ASTRALX_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
ASTRALMP_DIR = os.path.join(ASTRALX_ROOT, "astral-my")

ASTRALX_RUN  = os.path.join(ASTRALX_ROOT, "run.sh")
ASTRALMP_DEV = os.path.join(ASTRALMP_DIR, "dev.sh")


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_astralx_matrix(stdout_text):
    """
    Parse ASTRAL-X similarity matrix output from stdout.
    Expected format:
      SIMILARITY_MATRIX
      n=<count>
      taxa=name0,name1,...
      sim_row0=v00,v01,...
      sim_row1=...
    Returns (taxa_list, matrix_2d_list).
    """
    lines = stdout_text.splitlines()
    in_block = False
    taxa = None
    n = None
    rows = []

    for line in lines:
        line = line.strip()
        if line == "SIMILARITY_MATRIX":
            in_block = True
            continue
        if not in_block:
            continue
        if line.startswith("n="):
            n = int(line[2:])
        elif line.startswith("taxa="):
            taxa = line[5:].split(",")
        elif line.startswith("sim_row"):
            eq = line.index("=")
            vals = [float(x) for x in line[eq+1:].split(",")]
            rows.append(vals)

    if taxa is None or len(rows) == 0:
        raise ValueError("Could not parse ASTRAL-X similarity matrix from stdout.")
    return taxa, rows


def parse_astralmp_matrix(stderr_text):
    """
    Parse ASTRAL-MP similarity matrix from stderr.
    Expected format (between BEGIN/END markers):
      ASTRALMP_SIMMAT_BEGIN
      n=<count>
      taxa=name0,name1,...
      sim_row0=v00,v01,...
      ...
      ASTRALMP_SIMMAT_END
    Returns (taxa_list, matrix_2d_list).
    """
    lines = stderr_text.splitlines()
    in_block = False
    taxa = None
    n = None
    rows = []

    for line in lines:
        line = line.strip()
        if line == "ASTRALMP_SIMMAT_BEGIN":
            in_block = True
            continue
        if line == "ASTRALMP_SIMMAT_END":
            in_block = False
            continue
        if not in_block:
            continue
        if line.startswith("n="):
            n = int(line[2:])
        elif line.startswith("taxa="):
            taxa = line[5:].split(",")
        elif line.startswith("sim_row"):
            eq = line.index("=")
            vals = [float(x) for x in line[eq+1:].split(",")]
            rows.append(vals)

    if taxa is None or len(rows) == 0:
        raise ValueError("Could not parse ASTRAL-MP similarity matrix from stderr.\n"
                         "Stderr snippet:\n" + stderr_text[:2000])
    return taxa, rows


# ── Runners ───────────────────────────────────────────────────────────────────

def run_astralx(input_path, verbose=False, mode="cpu"):
    """Run ASTRAL-X with --verify-similarity-matrix and return parsed matrix."""
    cmd = [
        "bash", ASTRALX_RUN,
        "-i", input_path,
        "--verify-similarity-matrix",
        "--" + mode,
        "--no-build",
    ]
    if verbose:
        print(f"  [ASTRAL-X] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        stderr_lines = [l for l in result.stderr.splitlines() if l.strip()]
        for l in stderr_lines[:5]:
            print(f"    stderr: {l}")
    return parse_astralx_matrix(result.stdout)


def run_astralmp(input_path, verbose=False):
    """Run ASTRAL-MP with dumpSimMatrix system property and return parsed matrix."""
    env = os.environ.copy()
    env["ASTRAL_JVM_OPTS"] = "-DdumpSimMatrix=1"
    cmd = [
        "bash", ASTRALMP_DEV,
        "--run-only",
        "-i", input_path,
        "-o", "/tmp/astralmp_simmat_comparison_out.tre",
        "-C",
    ]
    if verbose:
        print(f"  [ASTRAL-MP] Running: ASTRAL_JVM_OPTS=-DdumpSimMatrix=1 {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return parse_astralmp_matrix(result.stderr)


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(taxa_x, mat_x, taxa_mp, mat_mp, tol, verbose=False):
    """
    Compare two matrices by (name_i, name_j) pairs.
    Returns (all_pass, max_diff, mean_diff, fail_count, total_count).
    """
    # Build name → index maps
    idx_x  = {name: i for i, name in enumerate(taxa_x)}
    idx_mp = {name: i for i, name in enumerate(taxa_mp)}

    # Intersection of taxa names
    common = sorted(set(taxa_x) & set(taxa_mp))
    if not common:
        raise ValueError(f"No common taxa between ASTRAL-X ({taxa_x}) and ASTRAL-MP ({taxa_mp}).")

    missing_x  = set(taxa_mp) - set(taxa_x)
    missing_mp = set(taxa_x) - set(taxa_mp)
    if missing_x:
        print(f"  WARNING: Taxa in ASTRAL-MP but not ASTRAL-X: {sorted(missing_x)}")
    if missing_mp:
        print(f"  WARNING: Taxa in ASTRAL-X but not ASTRAL-MP: {sorted(missing_mp)}")

    n = len(common)
    diffs = []
    failures = []

    for i, ni in enumerate(common):
        for j, nj in enumerate(common):
            vx  = mat_x [idx_x [ni]][idx_x [nj]]
            vmp = mat_mp[idx_mp[ni]][idx_mp[nj]]
            d   = abs(vx - vmp)
            diffs.append(d)
            if d > tol:
                failures.append((ni, nj, vx, vmp, d))

    max_diff  = max(diffs) if diffs else 0.0
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    fail_count = len(failures)
    total_count = len(diffs)
    all_pass = (fail_count == 0)

    if verbose and failures:
        print(f"\n  Pairs exceeding tolerance (tol={tol}):")
        hdr = f"  {'taxon_i':>12}  {'taxon_j':>12}  {'astralx':>12}  {'astralmp':>12}  {'abs_diff':>12}  status"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for ni, nj, vx, vmp, d in failures[:50]:
            status = "FAIL"
            print(f"  {ni:>12}  {nj:>12}  {vx:12.8f}  {vmp:12.8f}  {d:12.2e}  {status}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")

    return all_pass, max_diff, mean_diff, fail_count, total_count


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASTRAL-X vs ASTRAL-MP similarity matrix comparison")
    parser.add_argument("input", help="Input gene tree file (.tre)")
    parser.add_argument("--tol", type=float, default=1e-5, help="Tolerance for float comparison (default: 1e-5)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu", help="ASTRAL-X compute mode (default: cpu)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    tol = args.tol
    verbose = args.verbose

    print(f"\nComparing similarity matrices for: {os.path.basename(input_path)}")
    print(f"  Input:     {input_path}")
    print(f"  Tolerance: {tol:.1e}")

    # Run ASTRAL-X
    print(f"\n  Running ASTRAL-X ({args.mode.upper()})...", end=" ", flush=True)
    try:
        taxa_x, mat_x = run_astralx(input_path, verbose=verbose, mode=args.mode)
        print(f"OK  (n={len(taxa_x)}, taxa: {','.join(taxa_x[:4])}{',...' if len(taxa_x) > 4 else ''})")
    except Exception as e:
        print(f"FAILED\n  Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Run ASTRAL-MP
    print("  Running ASTRAL-MP...", end=" ", flush=True)
    try:
        taxa_mp, mat_mp = run_astralmp(input_path, verbose=verbose)
        print(f"OK  (n={len(taxa_mp)}, taxa: {','.join(taxa_mp[:4])}{',...' if len(taxa_mp) > 4 else ''})")
    except Exception as e:
        print(f"FAILED\n  Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Compare
    print("\n  Comparing by taxon name...")
    try:
        all_pass, max_diff, mean_diff, fail_count, total_count = compare(
            taxa_x, mat_x, taxa_mp, mat_mp, tol, verbose=verbose
        )
    except Exception as e:
        print(f"  Comparison error: {e}", file=sys.stderr)
        sys.exit(2)

    # Rank correlation (Spearman) as supplemental metric
    try:
        idx_x_c  = {name: i for i, name in enumerate(taxa_x)}
        idx_mp_c = {name: i for i, name in enumerate(taxa_mp)}
        common = sorted(set(taxa_x) & set(taxa_mp))
        vals_x  = []
        vals_mp = []
        for ni in common:
            for nj in common:
                if ni < nj:
                    vals_x.append(mat_x[idx_x_c[ni]][idx_x_c[nj]])
                    vals_mp.append(mat_mp[idx_mp_c[ni]][idx_mp_c[nj]])

        def spearman(a, b):
            n = len(a)
            def rank(v):
                s = sorted(range(n), key=lambda i: v[i])
                r = [0]*n
                for rank_i, idx in enumerate(s):
                    r[idx] = rank_i + 1
                return r
            ra, rb = rank(a), rank(b)
            d2 = sum((ra[i]-rb[i])**2 for i in range(n))
            return 1 - 6*d2 / (n*(n**2-1)) if n > 1 else 1.0

        rho = spearman(vals_x, vals_mp)
        print(f"  Spearman rank corr:    {rho:.6f}  (1.0=identical ordering)")
    except Exception:
        pass

    # Report
    print(f"\n  Pairs compared:        {total_count}")
    print(f"  Max absolute diff:     {max_diff:.2e}")
    print(f"  Mean absolute diff:    {mean_diff:.2e}")
    print(f"  Pairs exceeding tol:   {fail_count}")
    print(f"\n  NOTE: Systematic differences are EXPECTED between ASTRAL-X and ASTRAL-MP")
    print(f"        because they use different similarity formulas (see module docstring).")

    if all_pass:
        print(f"\n  RESULT: PASS (all {total_count} pairs within tol={tol:.1e})")
        sys.exit(0)
    else:
        print(f"\n  RESULT: FAIL ({fail_count}/{total_count} pairs exceed tol={tol:.1e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
