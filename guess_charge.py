"""Heuristic charge guesser for generated Ln complexes.

Inspects elemental composition of the generated ligand to guess the
overall complex charge.  For Eu(TMMA)2(NO3)3 the context is net-neutral
(Eu3+ balanced by 3 NO3-), so the total charge equals the generated
ligand's formal charge.

This is a rough heuristic — ~20% of structures may have incorrect charge.
The guess is logged but xTB defaults to charge=0.

Usage:
    from guess_charge import guess_charge
    charge, reason = guess_charge("path/to/structure.xyz")
"""
import re
from pathlib import Path
from collections import Counter


# Typical anionic functional-group signatures (element-count heuristics)
# These are very rough — proper assignment needs bond topology.
_ANION_HINTS = [
    # carboxylate COO-: at least 2 O near a C
    ("carboxylate", lambda c: c.get("O", 0) >= 2 and c.get("C", 0) >= 1, -1),
    # phenoxide / alkoxide: lone O without enough H
    ("alkoxide", lambda c: c.get("O", 0) >= 1 and c.get("H", 0) < c.get("O", 0), -1),
    # thiolate
    ("thiolate", lambda c: c.get("S", 0) >= 1 and c.get("H", 0) < c.get("S", 0), -1),
]


def _parse_gen_ligand(xyz_path: str) -> Counter:
    """Extract elemental composition of the generated (non-context) ligand."""
    with open(xyz_path) as f:
        lines = f.readlines()
    try:
        n_context = int(lines[1].strip())
    except ValueError:
        # Already-clean xyz (comment on line 2)
        return Counter()

    gen_lines = lines[n_context + 2:]  # skip header + context atoms
    counts = Counter()
    for line in gen_lines:
        parts = line.split()
        if len(parts) >= 4:
            counts[parts[0]] += 1
    return counts


def guess_charge(xyz_path: str, verbose: bool = False) -> tuple[int, str]:
    """Return (charge, reason_string) for the complex.

    Default is 0 (neutral).  Adjusts -1 per detected anionic group in
    the generated ligand.
    """
    comp = _parse_gen_ligand(xyz_path)
    if not comp:
        return 0, "could not parse generated ligand; defaulting to 0"

    charge = 0
    reasons = []
    for name, test, delta in _ANION_HINTS:
        if test(comp):
            charge += delta
            reasons.append(f"{name} ({delta:+d})")

    if not reasons:
        reason = f"no anionic groups detected in generated ligand ({dict(comp)}); charge=0"
    else:
        reason = f"detected {', '.join(reasons)} -> charge={charge}  (composition: {dict(comp)})"

    if verbose:
        print(f"  {Path(xyz_path).name}: {reason}")

    return charge, reason


if __name__ == "__main__":
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="xyz files or glob patterns")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p))

    for f in sorted(files):
        ch, reason = guess_charge(f, verbose=True)
