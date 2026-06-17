"""Re-score existing .xyz structures (and the pristine reference) through the
generation validity gate, after the charge-aware fixes (prompts 01-02).

The decisive no-GPU experiment from code-review Finding 2: apply the *same* gate
that runs inside ``generate.generate_ligand`` -- ``sanitycheck`` (molSimplify
overlap + ligand_breakdown) followed by ``BasicLigandMetrics`` (build_mol ->
reset_dative_bonds -> SanitizeMol with ADJUSTHS off) -- offline, against arbitrary
.xyz files.  Nothing here is re-implemented: the gate functions are imported from
``src.molecule_builder`` so the verdict is exactly what the generator would give.

Finding 2's hypothesis is that part of the headline ``0 / 6300`` validity was
*instrumental* -- a charge-blind gate rejecting valid charged donors (e.g. nitrate) --
rather than a failure of the model.  This script measures the *current* verdict; that
hypothesis is settled by a before/after comparison (this same script run against the
pre-fix gate, or the re-scored generated sets), never by the current pass alone.

NB (measured 2026-06-16): at crystal geometry ``eu_tmma_cis.xyz`` decomposes into
C7N2O2 x2 + N1O3 x3, and all five fragments -- the three nitrates included -- already
PASS the *pre-fix* gate.  OpenBabel perceives all-single N-O bonds there, so the
nitrates carry formal charge 0 and need none; stripping charges (old code) changes
nothing.  For this pristine reference the result is therefore *not* instrumental -- the
charge-preservation fix bites only on distorted (generated) geometries where OpenBabel
assigns a double bond and hence a formal charge.  The decisive instrumental signal, if
any, is in the re-scored generated sets, not the reference.

Usage:
  python rescore_validity.py --reference eu_tmma_cis.xyz \
      --inputs generated/eu_tmma_mask1_epoch48/noH design_test_runs/maskall_*/maskall/noH \
      --out metrics/rescore.csv
"""
import argparse
import csv
import glob
import os
from collections import Counter, OrderedDict

import torch

from src import const
from src.molecule_builder import BasicLigandMetrics, build_mol, sanitycheck

# The five gate verdicts, in the order generate_ligand can emit them.
REASONS = ['ok', 'overlap', 'atom_count', 'sanitize', 'disconnected']

BOLD = '\033[1m'
RESET = '\033[0m'


def fragment_formula(atom_types, lig):
    """Hill-ish formula string for one ligand fragment, e.g. 'C7N2O2' or 'N1O3'."""
    syms = Counter(const.IDX2ATOM[atom_types[i].item()] for i in lig)
    return ''.join(f'{e}{syms[e]}' for e in sorted(syms))


def read_xyz_heavy(path):
    """Read an .xyz the way ``generate.parse_complex``/``read_molecule`` do.

    Conventions reused verbatim from generate.py:
      * the metal sits at index 0 (first coordinate line); we locate it by symbol
        so a file whose metal is not first is still handled, matching ``sort_pos``;
      * hydrogens are dropped -- the gate operates on the heavy-atom skeleton;
      * heavy atoms map through ``const.ATOM2IDX`` and the metal to its nuclear
        charge ``const.CHARGES`` (the scalar ``write_xyz_file`` turns back into a
        metal symbol via ``const.idx2metals``).

    Returns (positions [N,3] float, atom_types [N] long, metal nuclear-charge scalar).
    """
    with open(path) as fh:
        lines = fh.readlines()
    atom_lines = [ln for ln in lines[2:] if len(ln.split()) >= 4]
    if not atom_lines:
        raise ValueError(f'no atom lines parsed from {path}')

    # Locate the metal center (first atom whose symbol is a supported metal).
    metal_idx = next((k for k, ln in enumerate(atom_lines)
                      if ln.split()[0] in const.metals), None)
    if metal_idx is None:
        raise ValueError(f'no supported metal found in {path}')
    metal_line = atom_lines.pop(metal_idx)
    metal_symbol = metal_line.split()[0]

    # Metal at index 0; its atom_type is a placeholder (write_xyz_file overrides
    # index 0 with the metal symbol, and ligand_breakdown excludes the metal).
    pos = [[float(v) for v in metal_line.split()[1:4]]]
    atom_types = [0]
    for ln in atom_lines:
        sym = ln.split()[0]
        if sym == 'H':            # heavy-atom skeleton only, as in parse_complex
            continue
        atom_types.append(const.ATOM2IDX[sym])
        pos.append([float(v) for v in ln.split()[1:4]])

    positions = torch.tensor(pos, dtype=torch.float32)
    atom_types = torch.tensor(atom_types, dtype=torch.long)
    metal = torch.tensor(const.CHARGES[metal_symbol])
    return positions, atom_types, metal


def score_xyz(path):
    """Run one .xyz file through the generation validity gate.

    Mirrors the per-structure block of ``generate.generate_ligand`` exactly, except
    it grades *every* ligand fragment (generate_ligand only scores the freshly
    generated ones).  Grading every fragment is the whole point of Finding 2: the
    reference's three nitrates are context, not generated, yet they are the fragments
    Finding 2 puts on trial as the suspected instrumental failure.
    """
    positions, atom_types, metal = read_xyz_heavy(path)
    natoms = positions.shape[0]

    overlapping, liglist = sanitycheck(positions, atom_types, metal)
    formulas = [fragment_formula(atom_types, lig) for lig in liglist]

    def verdict(reason):
        return {'valid': int(reason == 'ok'), 'reason': reason,
                'nlig': len(liglist), 'formulas': formulas}

    total_atoms = sum(len(lig) for lig in liglist) + 1  # +1: the metal center
    if overlapping:
        return verdict('overlap')
    if total_atoms != natoms:
        return verdict('atom_count')

    rdmols = [build_mol(positions[lig], atom_types[lig]) for lig in liglist]
    metrics = BasicLigandMetrics()
    valid = metrics.compute_validity(rdmols)
    if len(valid) != len(rdmols):
        return verdict('sanitize')
    if len(metrics.compute_connectivity(valid)) != len(rdmols):
        return verdict('disconnected')
    return verdict('ok')


def grade_one(path, setname, rows, tallies):
    """Score one file, record a CSV row + tally entry, never raise."""
    try:
        res = score_xyz(path)
    except Exception as exc:  # a single bad file must not abort the whole sweep
        res = {'valid': 0, 'reason': 'error'}
        print(f'  ! {path}: {type(exc).__name__}: {exc}')
    rows.append((setname, path, res['valid'], res['reason']))
    tallies.setdefault(setname, Counter())[res['reason']] += 1
    return res


def collect_inputs(patterns):
    """Expand each --inputs entry (dir, glob, or .xyz file) into a flat, de-duped,
    sorted list of .xyz paths.  Tolerates non-matching globs and /dev/null."""
    files, seen = [], set()
    for pat in patterns:
        matches = glob.glob(pat)
        if not matches and os.path.exists(pat):
            matches = [pat]
        for m in matches:
            if os.path.isdir(m):
                found = sorted(glob.glob(os.path.join(m, '*.xyz')))
            elif m.endswith('.xyz') and os.path.isfile(m):
                found = [m]
            else:
                found = []  # /dev/null, stray non-xyz, etc.
            for f in found:
                if f not in seen:
                    seen.add(f)
                    files.append(f)
    return files


def write_csv(out, rows):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['set', 'file', 'valid', 'reason'])
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {out}')


def print_tallies(tallies):
    cats = REASONS + (['error'] if any(c.get('error') for c in tallies.values()) else [])
    print('\nper-set tally (' + ' / '.join(REASONS) + '):')
    for setname, c in tallies.items():
        total = sum(c.values())
        counts = '  '.join(f'{cat}={c.get(cat, 0)}' for cat in cats)
        print(f'  {setname}\n      n={total}  valid={c.get("ok", 0)}/{total}  |  {counts}')


def print_headline(reference, ref_result):
    """The Finding 2 assertion, reported factually: does the pristine reference
    itself pass the *current* gate?  The instrumental conclusion is NOT asserted from
    the current pass alone -- it requires a before/after comparison (see module docstring)."""
    print()
    if not reference:
        print(f'{BOLD}HEADLINE: no --reference supplied; the Finding 2 sanity test was skipped.{RESET}')
        return
    if ref_result is None or ref_result.get('reason') == 'error':
        print(f'{BOLD}HEADLINE: the pristine reference {reference} could not be scored '
              f'(missing or unreadable) -- Finding 2 inconclusive.{RESET}')
        return
    frags = Counter(ref_result.get('formulas', []))
    frag_str = ', '.join(f'{f}x{n}' for f, n in frags.most_common()) or 'no fragments'
    if ref_result['valid'] == 1:
        print(f'{BOLD}HEADLINE: the pristine reference {reference} PASSES the current validity gate '
              f'-- {ref_result.get("nlig", "?")} fragments, all valid [{frag_str}].{RESET}')
        print('Finding 2 check: the reference and all its fragments (the nitrates among them) '
              'sanitize under the current charge-aware gate. Whether that is a *change* -- the '
              'instrumental question -- is proven only by a before/after, never by this pass alone: '
              're-run this script against the pre-fix gate, or read the re-scored generated sets above. '
              'Where a generated structure now passes that the original 0/6300 run rejected, that gap '
              'was instrumental (a charge-blind gate); where the reference passes both gates, it is not.')
    else:
        print(f'{BOLD}HEADLINE: the pristine reference {reference} FAILS the current validity gate '
              f'(reason: {ref_result["reason"]}; fragments [{frag_str}]).{RESET}')
        print('The gate rejects the ground truth itself -- investigate this before trusting any '
              'validity number from the generator.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', help='pristine reference .xyz to grade (the headline test)')
    ap.add_argument('--inputs', nargs='*', default=[],
                    help='dirs / globs / .xyz files of existing generated structures')
    ap.add_argument('--out', default='metrics/rescore.csv', help='output CSV path')
    args = ap.parse_args()

    rows = []
    tallies = OrderedDict()

    # --- the headline: grade the pristine reference first ---
    ref_result = None
    if args.reference:
        if os.path.isfile(args.reference):
            print(f'scoring reference: {args.reference}')
            ref_result = grade_one(args.reference, 'reference', rows, tallies)
        else:
            print(f'reference not found locally: {args.reference} '
                  f'(generated dirs are gitignored / live on the cluster)')

    # --- re-score every existing generated structure ---
    files = collect_inputs(args.inputs)
    print(f'scoring {len(files)} input file(s) from {len(args.inputs)} --inputs pattern(s)')
    for n, path in enumerate(files, 1):
        grade_one(path, os.path.dirname(path) or '.', rows, tallies)
        if n % 200 == 0:
            print(f'  ... {n}/{len(files)} scored')

    write_csv(args.out, rows)
    print_tallies(tallies)
    print_headline(args.reference, ref_result)


if __name__ == '__main__':
    main()
