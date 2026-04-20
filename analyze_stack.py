#!/usr/bin/env python
"""Analyze 2x2 projection x RePaint stack experiment.

Compares four cells:
  (proj OFF, r=1)   — baseline from Prompt 4 sweep
  (proj OFF, r=10)  — RePaint only from Prompt 4 sweep
  (proj ON,  r=1)   — projection only (this experiment)
  (proj ON,  r=10)  — the stack (this experiment)

Produces:
  - stack_summary.csv
  - stack_figure.png  (2x2 grid)
  - console summary with decision recommendation
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.taxonomy import (
    analyze_structure, analyze_reference, collect_files, load_xyz
)

REFERENCE = str(PROJECT_ROOT / "eu_tmma_cis.xyz")

# 2x2 grid: (label, directory, n_attempted)
CELLS = [
    dict(label="OFF / r=1",  proj=False, r=1,
         dir="generated/repaint_r1_epoch48/noH",  n_attempted=500),
    dict(label="ON / r=1",   proj=True,  r=1,
         dir="generated/proj_r1_epoch48/noH",      n_attempted=500),
    dict(label="OFF / r=10", proj=False, r=10,
         dir="generated/repaint_r10_epoch48/noH",  n_attempted=500),
    dict(label="ON / r=10",  proj=True,  r=10,
         dir="generated/proj_r10_epoch48/noH",     n_attempted=500),
]


def run_taxonomy(input_dir):
    """Run taxonomy metrics on all xyz files in input_dir."""
    files = collect_files(input_dir)
    if not files:
        return []
    results = []
    for fpath in files:
        try:
            r = analyze_structure(fpath, REFERENCE, PROJECT_ROOT)
            results.append(r)
        except Exception as e:
            print(f"  WARN: {os.path.basename(fpath)}: {e}", file=sys.stderr)
    return results


def bootstrap_ci(n_valid, n_total, n_boot=10000):
    """Bootstrap 95% CI for valid_rate."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    rate = n_valid / n_total
    draws = np.random.binomial(n_total, rate, size=n_boot) / n_total
    return rate, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyze_cell(cell):
    """Analyze one cell of the 2x2 grid."""
    d = PROJECT_ROOT / cell["dir"]
    attempted = cell["n_attempted"]

    if not d.exists():
        print(f"  {cell['label']}: directory {d} not found, skipping")
        return None

    results = run_taxonomy(str(d))
    n_valid = len(results)
    rate, ci_lo, ci_hi = bootstrap_ci(n_valid, attempted)

    bridges = [r["bridging"]["n_bridges"] for r in results]
    mean_bridges = float(np.mean(bridges)) if bridges else 0.0
    frac_zero = (sum(1 for b in bridges if b == 0) / n_valid) if n_valid else 0.0

    perturbs = [r["perturbation"]["mean_A"] for r in results
                if r["perturbation"]["mean_A"] is not None]
    mean_perturb = float(np.mean(perturbs)) if perturbs else 0.0

    mutated = [r["mutation"]["n_mutated"] for r in results]
    mean_mutated = float(np.mean(mutated)) if mutated else 0.0

    print(f"  {cell['label']}: {n_valid}/{attempted} valid "
          f"({100*rate:.2f}% [{100*ci_lo:.2f}-{100*ci_hi:.2f}%]), "
          f"bridges={mean_bridges:.2f}, perturb={mean_perturb:.4f} A")

    return dict(
        label=cell["label"],
        projection=cell["proj"],
        resample_r=cell["r"],
        n_attempted=attempted,
        n_valid=n_valid,
        valid_rate=round(rate, 5),
        valid_rate_ci_lo=round(ci_lo, 5),
        valid_rate_ci_hi=round(ci_hi, 5),
        mean_bridges_per_struct=round(mean_bridges, 3),
        frac_with_zero_bridges=round(frac_zero, 4),
        mean_context_perturb=round(mean_perturb, 4),
        mean_ligands_mutated_per_struct=round(mean_mutated, 3),
    )


def write_csv(rows, path):
    """Write stack_summary.csv."""
    fields = [
        "label", "projection", "resample_r",
        "n_attempted", "n_valid", "valid_rate",
        "mean_bridges_per_struct", "frac_with_zero_bridges",
        "mean_context_perturb", "mean_ligands_mutated_per_struct",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nWrote {path}")


def make_figure(rows, path):
    """Produce 2x2 comparison figure with grouped bars."""
    # Organize data: group by r, compare proj OFF vs ON
    r_vals = sorted(set(r["resample_r"] for r in rows))
    proj_off = {r["resample_r"]: r for r in rows if not r["projection"]}
    proj_on = {r["resample_r"]: r for r in rows if r["projection"]}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Projection x RePaint Stack: Eu(TMMA)$_2$(NO$_3$)$_3$",
                 fontsize=14, y=0.98)

    x = np.arange(len(r_vals))
    width = 0.35

    def _vals(d, key):
        return [d[rv][key] if rv in d else 0 for rv in r_vals]

    # (a) Valid rate
    ax = axes[0, 0]
    off_rates = [100 * (proj_off[rv]["valid_rate"] if rv in proj_off else 0)
                 for rv in r_vals]
    on_rates = [100 * (proj_on[rv]["valid_rate"] if rv in proj_on else 0)
                for rv in r_vals]
    bars_off = ax.bar(x - width/2, off_rates, width, label="Proj OFF",
                      color="tab:blue", alpha=0.8)
    bars_on = ax.bar(x + width/2, on_rates, width, label="Proj ON",
                     color="tab:orange", alpha=0.8)
    # Error bars from CI
    for i, rv in enumerate(r_vals):
        if rv in proj_off:
            r = proj_off[rv]
            ax.errorbar(i - width/2, 100*r["valid_rate"],
                        yerr=[[100*(r["valid_rate"] - r["valid_rate_ci_lo"])],
                              [100*(r["valid_rate_ci_hi"] - r["valid_rate"])]],
                        fmt="none", color="black", capsize=3)
        if rv in proj_on:
            r = proj_on[rv]
            ax.errorbar(i + width/2, 100*r["valid_rate"],
                        yerr=[[100*(r["valid_rate"] - r["valid_rate_ci_lo"])],
                              [100*(r["valid_rate_ci_hi"] - r["valid_rate"])]],
                        fmt="none", color="black", capsize=3)
    ax.set_ylabel("Valid rate (%)")
    ax.set_title("(a) Valid complex rate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={rv}" for rv in r_vals])
    ax.legend()

    # (b) Mean bridges per structure
    ax = axes[0, 1]
    ax.bar(x - width/2, _vals(proj_off, "mean_bridges_per_struct"), width,
           label="Proj OFF", color="tab:blue", alpha=0.8)
    ax.bar(x + width/2, _vals(proj_on, "mean_bridges_per_struct"), width,
           label="Proj ON", color="tab:orange", alpha=0.8)
    ax.set_ylabel("Mean bridges / structure")
    ax.set_title("(b) Inter-ligand bridges")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={rv}" for rv in r_vals])
    ax.legend()

    # (c) Zero-bridge fraction
    ax = axes[1, 0]
    ax.bar(x - width/2,
           [100 * (proj_off[rv]["frac_with_zero_bridges"] if rv in proj_off else 0)
            for rv in r_vals],
           width, label="Proj OFF", color="tab:blue", alpha=0.8)
    ax.bar(x + width/2,
           [100 * (proj_on[rv]["frac_with_zero_bridges"] if rv in proj_on else 0)
            for rv in r_vals],
           width, label="Proj ON", color="tab:orange", alpha=0.8)
    ax.set_ylabel("Structures w/ zero bridges (%)")
    ax.set_title("(c) Zero-bridge fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={rv}" for rv in r_vals])
    ax.legend()

    # (d) Context perturbation
    ax = axes[1, 1]
    ax.bar(x - width/2, _vals(proj_off, "mean_context_perturb"), width,
           label="Proj OFF", color="tab:blue", alpha=0.8)
    ax.bar(x + width/2, _vals(proj_on, "mean_context_perturb"), width,
           label="Proj ON", color="tab:orange", alpha=0.8)
    ax.set_ylabel("Mean perturbation (\u00c5)")
    ax.set_title("(d) Context perturbation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={rv}" for rv in r_vals])
    ax.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def sanity_check_bridges(rows):
    """Verify projection-ON cells have zero bridges (by construction)."""
    print("\n--- SANITY CHECK: projection-ON bridge count ---")
    ok = True
    for r in rows:
        if not r["projection"]:
            continue
        if r["mean_bridges_per_struct"] > 0:
            print(f"  FAIL: {r['label']} has mean_bridges={r['mean_bridges_per_struct']:.3f}")
            print("  -> BUG in src/projection.py — debug before proceeding")
            ok = False
        else:
            print(f"  OK:   {r['label']} — zero bridges")
    if ok:
        print("  All projection-ON cells pass bridge sanity check.")
    return ok


def print_summary(rows):
    """Print summary table and decision recommendation."""
    print("\n" + "=" * 80)
    print("PROJECTION x REPAINT STACK SUMMARY")
    print("=" * 80)
    hdr = (f"{'Cell':>14} {'proj':>5} {'r':>4} {'tried':>6} {'valid':>6} "
           f"{'rate%':>8} {'bridges':>9} {'0-br%':>7} {'perturb':>9} {'mutated':>8}")
    print(hdr)
    print("-" * 80)
    for r in rows:
        proj_str = "ON" if r["projection"] else "OFF"
        print(f"{r['label']:>14} {proj_str:>5} {r['resample_r']:>4} "
              f"{r['n_attempted']:>6} {r['n_valid']:>6} "
              f"{100*r['valid_rate']:>7.2f}% {r['mean_bridges_per_struct']:>9.2f} "
              f"{100*r['frac_with_zero_bridges']:>6.1f}% "
              f"{r['mean_context_perturb']:>8.4f} "
              f"{r['mean_ligands_mutated_per_struct']:>8.2f}")
    print("=" * 80)

    # Find the stack cell (proj ON, r=10)
    stack = next((r for r in rows if r["projection"] and r["resample_r"] == 10), None)
    baseline = next((r for r in rows if not r["projection"] and r["resample_r"] == 1), None)

    if stack:
        stack_rate = stack["valid_rate"]
        stack_pct = 100 * stack_rate
        print(f"\nStack (proj ON, r=10) valid rate: {stack_pct:.2f}%")

        if baseline:
            base_rate = baseline["valid_rate"]
            ratio = stack_rate / base_rate if base_rate > 0 else float("inf")
            print(f"Baseline (proj OFF, r=1) valid rate: {100*base_rate:.2f}%")
            print(f"Stack improvement over baseline: {ratio:.1f}x")

        # Projection effect at fixed r
        for rv in sorted(set(r["resample_r"] for r in rows)):
            off = next((r for r in rows if not r["projection"] and r["resample_r"] == rv), None)
            on = next((r for r in rows if r["projection"] and r["resample_r"] == rv), None)
            if off and on:
                delta = 100 * (on["valid_rate"] - off["valid_rate"])
                bridge_drop = (off["mean_bridges_per_struct"] - on["mean_bridges_per_struct"])
                print(f"\n  r={rv}: proj ON vs OFF -> valid_rate delta = {delta:+.2f}pp, "
                      f"bridge reduction = {bridge_drop:+.2f}")

        print("\n--- DECISION RECOMMENDATION ---")
        if stack_pct > 10:
            print(f"  valid_rate = {stack_pct:.2f}% (>10%)")
            print("  -> PROCEED to Prompt 6 (sparse context ablation)")
        elif stack_pct >= 5:
            print(f"  valid_rate = {stack_pct:.2f}% (5-10%)")
            print("  -> SUFFICIENT for paper. Proceed to Prompt 9 (paper writing)")
            print("     Optionally run Prompt 6 (sparse context ablation) in parallel")
        else:
            print(f"  valid_rate = {stack_pct:.2f}% (<5%)")
            print("  -> INSUFFICIENT. Need soft-LJ DPS guidance (not yet implemented).")
            print("     Raise as new task before proceeding.")


def main():
    print("Projection x RePaint Stack Analysis")
    print(f"Reference: {REFERENCE}")
    print()

    analyze_reference(REFERENCE)

    rows = []
    for cell in CELLS:
        print(f"Analyzing {cell['label']}...")
        result = analyze_cell(cell)
        if result is not None:
            rows.append(result)

    if not rows:
        sys.exit("No results found. Have the sbatch jobs completed?")

    write_csv(rows, PROJECT_ROOT / "stack_summary.csv")
    make_figure(rows, PROJECT_ROOT / "stack_figure.png")
    sanity_check_bridges(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
