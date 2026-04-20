"""
Analysis and plotting for xTB optimisation results.

Reads summary.csv files produced by xtb_pipeline.py and generates:
  1. Histogram of final xTB energies (converged structures only)
  2. Bar chart of convergence rates across runs
  3. Scatter: context perturbation distance vs xTB final energy

Usage:
    python xtb_summary_plot.py --results-dir xtb_results
    # produces plots in xtb_results/plots/
"""
import argparse, csv, os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir: str) -> list[dict]:
    """Load summary_all.csv or individual summary.csv files."""
    base = Path(results_dir)
    all_csv = base / "summary_all.csv"
    if all_csv.exists():
        with open(all_csv) as f:
            return list(csv.DictReader(f))

    rows = []
    for csv_file in sorted(base.rglob("summary.csv")):
        run_name = csv_file.parent.name
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                row["run"] = run_name
                rows.append(row)
    return rows


def plot_energy_histogram(rows: list[dict], plot_dir: Path):
    """Histogram of final energies for converged structures."""
    energies_by_run = defaultdict(list)
    for r in rows:
        if r.get("xtb_success", "").lower() == "true" and r.get("final_energy_hartree"):
            try:
                e = float(r["final_energy_hartree"])
                energies_by_run[r.get("run", "default")].append(e)
            except ValueError:
                pass

    if not energies_by_run:
        print("No converged energies to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_e = []
    for run, es in sorted(energies_by_run.items()):
        ax.hist(es, bins=20, alpha=0.6, label=f"{run} (n={len(es)})")
        all_e.extend(es)

    ax.set_xlabel("Final xTB Energy (Hartree)")
    ax.set_ylabel("Count")
    ax.set_title("xTB Optimised Energies — Converged Structures")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "energy_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  energy_histogram.png  ({len(all_e)} structures)")


def plot_convergence_bar(rows: list[dict], plot_dir: Path):
    """Bar chart of convergence rates per run."""
    stats = defaultdict(lambda: {"total": 0, "converged": 0})
    for r in rows:
        run = r.get("run", "default")
        stats[run]["total"] += 1
        if r.get("xtb_success", "").lower() == "true":
            stats[run]["converged"] += 1

    runs = sorted(stats.keys())
    rates = [100 * stats[r]["converged"] / stats[r]["total"] if stats[r]["total"] else 0
             for r in runs]
    totals = [stats[r]["total"] for r in runs]

    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.2), 5))
    bars = ax.bar(range(len(runs)), rates, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("xTB Convergence Rates by Run")
    ax.set_ylim(0, 105)

    for i, (bar, rate, total) in enumerate(zip(bars, rates, totals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%\n(n={total})", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(plot_dir / "convergence_bar.png", dpi=150)
    plt.close(fig)

    # Also print paper-ready table
    print("\n=== xTB Convergence Rates (paper table) ===")
    print(f"{'Run':<35} {'Conv':>6} {'Total':>6} {'Rate':>7}")
    print("-" * 60)
    for r in runs:
        s = stats[r]
        rate = 100 * s["converged"] / s["total"] if s["total"] else 0
        print(f"{r:<35} {s['converged']:>6} {s['total']:>6} {rate:>6.1f}%")


def plot_perturbation_vs_energy(rows: list[dict], plot_dir: Path):
    """Scatter: context perturbation distance vs xTB energy.

    Perturbation distance is encoded in structure names when available
    (e.g., from metrics CSVs produced by earlier prompts).  If not
    available, we skip this plot.
    """
    # Try to load perturbation metrics from analysis CSVs
    base = plot_dir.parent
    metric_files = list(base.rglob("*metrics*.csv")) + list(Path(".").glob("analysis/*metrics*.csv"))

    perturb_map = {}
    for mf in metric_files:
        try:
            with open(mf) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("name", row.get("structure", ""))
                    for key in ["mean_perturbation", "context_rmsd", "perturbation_angstrom"]:
                        if key in row and row[key]:
                            try:
                                perturb_map[name] = float(row[key])
                            except ValueError:
                                pass
        except Exception:
            pass

    if not perturb_map:
        print("  No perturbation metrics found — skipping scatter plot.")
        return

    xs, ys, labels = [], [], []
    for r in rows:
        if r.get("xtb_success", "").lower() != "true":
            continue
        name = r.get("name", "")
        e = r.get("final_energy_hartree")
        if name in perturb_map and e:
            try:
                xs.append(perturb_map[name])
                ys.append(float(e))
                labels.append(name)
            except ValueError:
                pass

    if len(xs) < 3:
        print(f"  Only {len(xs)} matched structures — skipping scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=20, alpha=0.7, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("Context Perturbation (A)")
    ax.set_ylabel("Final xTB Energy (Hartree)")
    ax.set_title("Perturbation vs Optimised Energy")

    # Trend line
    if len(xs) >= 5:
        z = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.5,
                label=f"slope={z[0]:.2f} Ha/A")
        ax.legend()

    fig.tight_layout()
    fig.savefig(plot_dir / "perturbation_vs_energy.png", dpi=150)
    plt.close(fig)
    print(f"  perturbation_vs_energy.png  ({len(xs)} points)")


def plot_failure_breakdown(rows: list[dict], plot_dir: Path):
    """Stacked bar chart of failure modes per run."""
    from collections import Counter
    failure_by_run = defaultdict(Counter)
    for r in rows:
        run = r.get("run", "default")
        if r.get("xtb_success", "").lower() == "true":
            failure_by_run[run]["converged"] += 1
        else:
            mode = r.get("failure_mode", "unknown") or "unknown"
            failure_by_run[run][mode] += 1

    runs = sorted(failure_by_run.keys())
    all_modes = set()
    for c in failure_by_run.values():
        all_modes |= set(c.keys())
    modes = sorted(all_modes)

    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.2), 5))
    bottom = np.zeros(len(runs))
    colors = plt.cm.Set3(np.linspace(0, 1, len(modes)))

    for mode, color in zip(modes, colors):
        vals = [failure_by_run[r][mode] for r in runs]
        ax.bar(range(len(runs)), vals, bottom=bottom, label=mode,
               color=color, edgecolor="black", linewidth=0.3)
        bottom += vals

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("xTB Outcome Breakdown by Run")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / "failure_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  failure_breakdown.png")


def main():
    ap = argparse.ArgumentParser(description="xTB result analysis and plots")
    ap.add_argument("--results-dir", default="xtb_results",
                    help="Base directory containing run subdirectories")
    args = ap.parse_args()

    rows = load_all_results(args.results_dir)
    if not rows:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(rows)} records from {args.results_dir}")

    plot_dir = Path(args.results_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_energy_histogram(rows, plot_dir)
    plot_convergence_bar(rows, plot_dir)
    plot_perturbation_vs_energy(rows, plot_dir)
    plot_failure_breakdown(rows, plot_dir)

    print(f"\nPlots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
