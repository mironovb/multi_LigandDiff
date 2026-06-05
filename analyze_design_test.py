#!/usr/bin/env python
"""Analyze the design-ability sweep produced by ``generate_design_test.py``.

Two products, both traceable to primary files (no hand-entered numbers):

1. **Scaffold-degradation curve.** For each mask level it reports attempts,
   valid count and yield. Valid counts come from the saved structures
   (``mask{K}/noH/*.xyz``); attempts come from the ``design_test mask_k=... ``
   summary lines the generator prints (parsed from the supplied logs), falling
   back to the directory contents when a log is absent.

2. **Rejection-mechanism tally.** Greps the SLURM ``.err``/``.out`` logs for
   RDKit sanitization failures and classifies them, quantifying the claim that
   de-novo failures are *chemical* (nitrogen explicit-valence violations), not
   geometric near-misses.

Usage::

    python analyze_design_test.py --runs design_test_runs/maskall_14344725 \\
                                          design_test_runs/14292188 \\
                                  --logs logs/ln_design_*.err logs/ln_maskall_*.err \\
                                  --out metrics/design_test

Writes ``design_degradation.csv`` and ``rejection_tally.csv`` under ``--out``
and, if matplotlib is available, ``design_degradation.png``.
"""

import argparse
import csv
import glob
import os
import re
from collections import Counter, defaultdict

MASK_DIR_RE = re.compile(r'mask(\d+|all)$')
SUMMARY_RE = re.compile(
    r'design_test\s+mask_k=(\w+)\s+context=(\S+)\s+attempts=(\d+)\s+valid=(\d+)')
# Older free-text log form: "maskall  context=0/5  attempts=6300  valid=0 ..."
SUMMARY_RE_ALT = re.compile(
    r'mask(\w+)\s+context=(\S+)\s+attempts=(\d+)\s+valid=(\d+)')

# RDKit / sanitization rejection patterns.
VALENCE_RE = re.compile(
    r'Explicit valence for atom #\s*\d+\s+([A-Z][a-z]?),\s*(\d+),\s*is greater than permitted')
KEKULIZE_RE = re.compile(r'[Cc]ould not [Kk]ekulize')
NAN_RE = re.compile(r'FoundNaN|nan', re.IGNORECASE)


def count_valid(mask_dir):
    noh = os.path.join(mask_dir, 'noH')
    if not os.path.isdir(noh):
        return 0
    return len([f for f in os.listdir(noh) if f.endswith('.xyz')])


def discover_mask_dirs(run_roots):
    """Return {mask_label: path} for every mask*/ directory under the run roots."""
    found = {}
    for root in run_roots:
        if not os.path.isdir(root):
            print(f'  [warn] run root not found: {root}')
            continue
        # the root itself may be a mask dir, or contain them
        candidates = [root] + [os.path.join(root, d) for d in os.listdir(root)]
        for c in candidates:
            if not os.path.isdir(c):
                continue
            m = MASK_DIR_RE.search(os.path.basename(c.rstrip('/')))
            if m and os.path.isdir(os.path.join(c, 'noH')) or (m and os.path.isdir(c)):
                if m:
                    found[m.group(1)] = c
    return found


def parse_attempts_from_logs(log_paths):
    """{mask_label: (attempts, valid_logged)} parsed from generator stdout lines."""
    out = {}
    for path in log_paths:
        try:
            with open(path, 'r', errors='ignore') as f:
                text = f.read()
        except OSError:
            continue
        for rx in (SUMMARY_RE, SUMMARY_RE_ALT):
            for m in rx.finditer(text):
                label, _ctx, attempts, valid = m.groups()
                out[label] = (int(attempts), int(valid))
    return out


def tally_rejections(log_paths):
    valence_by_element = Counter()
    other = Counter()
    total_lines = 0
    for path in log_paths:
        try:
            with open(path, 'r', errors='ignore') as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            total_lines += 1
            m = VALENCE_RE.search(line)
            if m:
                valence_by_element[m.group(1)] += 1
            elif KEKULIZE_RE.search(line):
                other['kekulize'] += 1
            elif NAN_RE.search(line) and 'FoundNaN' in line:
                other['nan'] += 1
    return valence_by_element, other, total_lines


def mask_sort_key(label):
    return (1, 0) if label == 'all' else (0, int(label))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', default=[],
                   help='run roots containing mask*/ output directories')
    p.add_argument('--logs', nargs='+', default=[],
                   help='SLURM .out/.err logs (globs allowed)')
    p.add_argument('--out', default='metrics/design_test',
                   help='output directory for CSVs / figure')
    args = p.parse_args()

    log_paths = []
    for g in args.logs:
        log_paths.extend(sorted(glob.glob(g)))

    os.makedirs(args.out, exist_ok=True)

    mask_dirs = discover_mask_dirs(args.runs)
    logged = parse_attempts_from_logs(log_paths)

    rows = []
    labels = sorted(set(mask_dirs) | set(logged), key=mask_sort_key)
    for label in labels:
        valid_files = count_valid(mask_dirs[label]) if label in mask_dirs else None
        attempts_log, valid_log = logged.get(label, (None, None))
        valid = valid_files if valid_files is not None else valid_log
        attempts = attempts_log
        yld = (100.0 * valid / attempts) if (attempts and valid is not None) else None
        rows.append({
            'mask_level': label,
            'attempts': attempts if attempts is not None else '',
            'valid': valid if valid is not None else '',
            'valid_from_files': valid_files if valid_files is not None else '',
            'valid_from_log': valid_log if valid_log is not None else '',
            'yield_pct': f'{yld:.2f}' if yld is not None else '',
        })

    deg_csv = os.path.join(args.out, 'design_degradation.csv')
    with open(deg_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['mask_level', 'attempts', 'valid',
                            'valid_from_files', 'valid_from_log', 'yield_pct'])
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {deg_csv}')
    print('\nScaffold-degradation curve:')
    print(f"  {'mask':>6} {'attempts':>10} {'valid':>7} {'yield%':>8}")
    for r in rows:
        print(f"  {r['mask_level']:>6} {str(r['attempts']):>10} "
              f"{str(r['valid']):>7} {str(r['yield_pct']):>8}")

    valence, other, total = tally_rejections(log_paths)
    rej_csv = os.path.join(args.out, 'rejection_tally.csv')
    with open(rej_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'detail', 'count'])
        for elem, n in valence.most_common():
            w.writerow(['explicit_valence', elem, n])
        for k, n in other.most_common():
            w.writerow(['other', k, n])
    print(f'\nWrote {rej_csv}')
    total_valence = sum(valence.values())
    print('\nRejection mechanism (from logs):')
    print(f'  explicit-valence errors: {total_valence}'
          + (f"  ({100.0*valence.get('N',0)/total_valence:.0f}% nitrogen)"
             if total_valence else ''))
    for elem, n in valence.most_common():
        print(f'    {elem}: {n}')
    for k, n in other.most_common():
        print(f'  {k}: {n}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_rows = [r for r in rows if r['valid'] != '']
        if plot_rows:
            xs = [r['mask_level'] for r in plot_rows]
            ys = [int(r['valid']) for r in plot_rows]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(range(len(xs)), ys, 'o-', color='#b2182b')
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([f'mask {x}' for x in xs])
            ax.set_ylabel('valid structures')
            ax.set_xlabel('ligands hidden (scaffold shrinking ->)')
            ax.set_title('De-novo design collapses as context shrinks')
            for i, y in enumerate(ys):
                ax.annotate(str(y), (i, y), textcoords='offset points',
                            xytext=(0, 6), ha='center', fontsize=9)
            fig.tight_layout()
            fig_path = os.path.join(args.out, 'design_degradation.png')
            fig.savefig(fig_path, dpi=200)
            print(f'\nWrote {fig_path}')
    except ImportError:
        print('\n[info] matplotlib not available; skipped figure.')


if __name__ == '__main__':
    main()
