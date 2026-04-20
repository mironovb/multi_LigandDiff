#!/usr/bin/env python
"""Analyze RePaint r-sweep experiment results.

Runs metrics/taxonomy.py on each r-value output directory and produces:
  - sweep_results.csv   (quantitative comparison)
  - sweep_figure.png    (4-panel figure)
  - console summary with go/no-go recommendation
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path so we can import taxonomy
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.taxonomy import (
    analyze_structure, analyze_reference, collect_files, load_xyz
)

REFERENCE = str(PROJECT_ROOT / "eu_tmma_cis.xyz")
R_VALUES = [1, 5, 10, 20]
DIR_TEMPLATE = "generated/repaint_r{}_epoch48/noH"
BASELINE_R1_VALID_RATE = 0.005  # ~0.5% from earlier runs


def run_taxonomy(input_dir):
    """Run taxonomy metrics on all xyz files in input_dir, return list of results."""
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


def n_attempted(r_val):
    """Return the expected sample count for a given r value."""
    return 250 if r_val == 20 else 500


def bootstrap_valid_rate(n_valid, n_total, n_boot=10000):
    """Bootstrap CI for valid_rate = n_valid / n_total."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    rate = n_valid / n_total
    draws = np.random.binomial(n_total, rate, size=n_boot) / n_total
    lo = np.percentile(draws, 2.5)
    hi = np.percentile(draws, 97.5)
    return rate, lo, hi


def analyze_one_r(r_val):
    """Analyze results for one r value. Returns dict of aggregate metrics."""
    d = PROJECT_ROOT / DIR_TEMPLATE.format(r_val)
    attempted = n_attempted(r_val)

    if not d.exists():
        print(f"  r={r_val}: directory {d} not found, skipping")
        return None

    results = run_taxonomy(str(d))
    n_valid = len(results)

    rate, ci_lo, ci_hi = bootstrap_valid_rate(n_valid, attempted)

    bridges = [r["bridging"]["n_bridges"] for r in results]
    zero_bridges = sum(1 for b in bridges if b == 0)
    mean_bridges = float(np.mean(bridges)) if bridges else 0.0
    frac_zero = zero_bridges / n_valid if n_valid else 0.0

    perturbs = [r["perturbation"]["mean_A"] for r in results
                if r["perturbation"]["mean_A"] is not None]
    mean_perturb = float(np.mean(perturbs)) if perturbs else 0.0

    mutated_per = [r["mutation"]["n_mutated"] for r in results]
    mean_mutated = float(np.mean(mutated_per)) if mutated_per else 0.0

    print(f"  r={r_val}: {n_valid}/{attempted} valid "
          f"({100*rate:.2f}% [{100*ci_lo:.2f}-{100*ci_hi:.2f}%]), "
          f"bridges={mean_bridges:.2f}, perturb={mean_perturb:.4f} A")

    return dict(
        resample_r=r_val,
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
    """Write sweep_results.csv."""
    fields = [
        "resample_r", "n_attempted", "n_valid", "valid_rate",
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
    """Produce 4-panel sweep figure."""
    rs = [r["resample_r"] for r in rows]
    rates = [100 * r["valid_rate"] for r in rows]
    rate_lo = [100 * r["valid_rate_ci_lo"] for r in rows]
    rate_hi = [100 * r["valid_rate_ci_hi"] for r in rows]
    bridges = [r["mean_bridges_per_struct"] for r in rows]
    zero_br = [100 * r["frac_with_zero_bridges"] for r in rows]
    perturb = [r["mean_context_perturb"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("RePaint r-Sweep: Eu(TMMA)$_2$(NO$_3$)$_3$", fontsize=14)

    # (a) valid rate vs r
    ax = axes[0, 0]
    yerr_lo = [rates[i] - rate_lo[i] for i in range(len(rs))]
    yerr_hi = [rate_hi[i] - rates[i] for i in range(len(rs))]
    ax.errorbar(rs, rates, yerr=[yerr_lo, yerr_hi],
                marker="o", capsize=4, color="tab:blue")
    ax.set_xlabel("Resample r")
    ax.set_ylabel("Valid rate (%)")
    ax.set_title("(a) Valid complex rate")
    ax.set_xticks(rs)

    # (b) mean bridges per structure vs r
    ax = axes[0, 1]
    ax.plot(rs, bridges, marker="s", color="tab:red")
    ax.set_xlabel("Resample r")
    ax.set_ylabel("Mean bridges / structure")
    ax.set_title("(b) Inter-ligand bridges")
    ax.set_xticks(rs)

    # (c) fraction with zero bridges vs r
    ax = axes[1, 0]
    ax.plot(rs, zero_br, marker="^", color="tab:green")
    ax.set_xlabel("Resample r")
    ax.set_ylabel("Structures w/ zero bridges (%)")
    ax.set_title("(c) Zero-bridge fraction")
    ax.set_xticks(rs)

    # (d) mean context perturbation vs r
    ax = axes[1, 1]
    ax.plot(rs, perturb, marker="D", color="tab:orange")
    ax.set_xlabel("Resample r")
    ax.set_ylabel("Mean perturbation (\u00c5)")
    ax.set_title("(d) Context perturbation")
    ax.set_xticks(rs)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def print_summary(rows):
    """Print summary table and go/no-go recommendation."""
    print("\n" + "=" * 72)
    print("REPAINT R-SWEEP SUMMARY")
    print("=" * 72)
    hdr = f"{'r':>4} {'attempted':>10} {'valid':>8} {'rate%':>8} {'bridges':>10} {'0-bridge%':>10} {'perturb':>10} {'mutated':>10}"
    print(hdr)
    print("-" * 72)
    for r in rows:
        print(f"{r['resample_r']:>4} {r['n_attempted']:>10} {r['n_valid']:>8} "
              f"{100*r['valid_rate']:>7.2f}% {r['mean_bridges_per_struct']:>10.2f} "
              f"{100*r['frac_with_zero_bridges']:>9.1f}% "
              f"{r['mean_context_perturb']:>9.4f} "
              f"{r['mean_ligands_mutated_per_struct']:>10.2f}")
    print("=" * 72)

    # Find baseline and r=10
    baseline = next((r for r in rows if r["resample_r"] == 1), None)
    r10 = next((r for r in rows if r["resample_r"] == 10), None)

    if baseline and r10:
        base_rate = baseline["valid_rate"]
        r10_rate = r10["valid_rate"]
        ratio = r10_rate / base_rate if base_rate > 0 else float("inf")

        print(f"\nBaseline (r=1) valid rate: {100*base_rate:.2f}%")
        print(f"r=10 valid rate:           {100*r10_rate:.2f}%")
        print(f"Improvement ratio:         {ratio:.1f}x")

        if r10_rate >= 0.05:
            print("\nResearch prediction (5-15% at r=10): CONFIRMED")
        elif r10_rate >= 0.015:
            print(f"\nResearch prediction (5-15% at r=10): PARTIALLY confirmed "
                  f"({100*r10_rate:.2f}% < 5% but significant improvement)")
        else:
            print(f"\nResearch prediction (5-15% at r=10): NOT confirmed "
                  f"({100*r10_rate:.2f}%)")

        print("\n--- GO / NO-GO ---")
        if ratio >= 3.0 and r10_rate >= 0.015:
            print("GO: RePaint is a significant mitigation (>3x baseline).")
            print("NEXT: Proceed to Prompt 6 (sparse context conditioning).")
        else:
            print("NO-GO: RePaint alone is insufficient for f-block.")
            print("NEXT: Proceed to Prompt 5 (stack geometric projection on top of RePaint).")

        # Bridge metric insight
        if r10["mean_bridges_per_struct"] < baseline["mean_bridges_per_struct"] * 0.7:
            print(f"\nBridge reduction: {baseline['mean_bridges_per_struct']:.2f} -> "
                  f"{r10['mean_bridges_per_struct']:.2f} "
                  f"({100*(1 - r10['mean_bridges_per_struct']/baseline['mean_bridges_per_struct']):.0f}% drop). "
                  f"RePaint is reducing boundary disharmony as expected.")
        else:
            print(f"\nBridge reduction: {baseline['mean_bridges_per_struct']:.2f} -> "
                  f"{r10['mean_bridges_per_struct']:.2f}. "
                  f"Modest or no improvement in boundary disharmony.")


def main():
    print("RePaint r-Sweep Analysis")
    print(f"Reference: {REFERENCE}")
    print()

    # Pre-cache reference analysis
    analyze_reference(REFERENCE)

    rows = []
    for r_val in R_VALUES:
        print(f"Analyzing r={r_val}...")
        result = analyze_one_r(r_val)
        if result is not None:
            rows.append(result)

    if not rows:
        sys.exit("No results found. Has the sweep sbatch completed?")

    write_csv(rows, PROJECT_ROOT / "sweep_results.csv")
    make_figure(rows, PROJECT_ROOT / "sweep_figure.png")
    print_summary(rows)


if __name__ == "__main__":
    main()
