#!/usr/bin/env python
"""Analyze sparse-to-dense context ablation (Experiment 2).

Compares generation quality across four reference structures with
monotonically increasing context density:
  1. Bare Eu           (1 atom,  CN_c=0, total generation)
  2. Eu(H2O)9          (10 atoms, CN_c=8, mask-one water)
  3. Eu(NO3)3(H2O)3    (16 atoms, CN_c=7-8, mask-one ligand)
  4. Eu(TMMA)2(NO3)3   (35 atoms, CN_c=6-8, mask-one ligand)

Produces:
  - context_ablation_results.csv
  - context_ablation_figure.{png,pdf}   (headline validity vs density plot)
  - aqua_seed_figure.{png,pdf}          (water preservation analysis)
  - console interpretation
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.taxonomy import (
    analyze_structure, analyze_reference, collect_files, load_xyz,
    match_context, _dist,
)

# ── experiment configuration ───────────────────────────────────────────

CONTEXTS = [
    dict(
        name="eu_bare",
        ref="refs/eu_bare.xyz",
        gen_dir="generated/ctx_eu_bare_epoch48/noH",
        n_attempted=3000,       # 6 LD_g partitions x 500 samples
        n_heavy=1,
        label="Bare Eu$^{3+}$",
        description="total generation (CN_c=0)",
    ),
    dict(
        name="eu_aqua9",
        ref="refs/eu_aqua9.xyz",
        gen_dir="generated/ctx_eu_aqua9_epoch48/noH",
        n_attempted=500,        # 9 mask-one scenarios x ~500/9 effective
        n_heavy=10,
        label="Eu(H$_2$O)$_9$",
        description="mask-one water",
    ),
    dict(
        name="eu_no3_3_h2o_3",
        ref="refs/eu_no3_3_h2o_3.xyz",
        gen_dir="generated/ctx_eu_no3_3_h2o_3_epoch48/noH",
        n_attempted=500,
        n_heavy=16,
        label="Eu(NO$_3$)$_3$(H$_2$O)$_3$",
        description="mask-one ligand",
    ),
    dict(
        name="eu_tmma_cis",
        ref="refs/eu_tmma_cis.xyz",
        gen_dir="generated/ctx_eu_tmma_cis_epoch48/noH",
        n_attempted=500,
        n_heavy=35,
        label="Eu(TMMA)$_2$(NO$_3$)$_3$",
        description="mask-one ligand",
    ),
]


def bootstrap_ci(n_valid, n_total, n_boot=10000):
    """Bootstrap 95% CI for valid_rate = n_valid / n_total."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    rate = n_valid / n_total
    draws = np.random.binomial(n_total, rate, size=n_boot) / n_total
    return rate, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def run_taxonomy(input_dir, ref_path):
    """Run taxonomy metrics on all xyz files in input_dir."""
    files = collect_files(input_dir)
    if not files:
        return []
    results = []
    for fpath in files:
        try:
            r = analyze_structure(fpath, ref_path, PROJECT_ROOT)
            results.append(r)
        except Exception as e:
            print(f"  WARN: {os.path.basename(fpath)}: {e}", file=sys.stderr)
    return results


def analyze_one_context(ctx):
    """Analyze results for one context level. Returns dict of aggregate metrics."""
    d = PROJECT_ROOT / ctx["gen_dir"]
    ref = str(PROJECT_ROOT / ctx["ref"])
    attempted = ctx["n_attempted"]

    if not d.exists():
        print(f"  {ctx['name']}: directory {d} not found, skipping")
        return None

    # Pre-cache reference
    analyze_reference(ref)

    results = run_taxonomy(str(d), ref)
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

    # Denticity recovery
    dent_matches = [r["denticity"]["match"] for r in results
                    if r["denticity"]["match"] is not None]
    dent_recovery = (sum(dent_matches) / len(dent_matches)
                     if dent_matches else None)

    print(f"  {ctx['name']}: {n_valid}/{attempted} valid "
          f"({100*rate:.2f}% [{100*ci_lo:.2f}-{100*ci_hi:.2f}%]), "
          f"bridges={mean_bridges:.2f}, perturb={mean_perturb:.4f} A, "
          f"dent_recovery={100*dent_recovery:.1f}%" if dent_recovery else "")

    return dict(
        name=ctx["name"],
        label=ctx["label"],
        description=ctx["description"],
        n_heavy=ctx["n_heavy"],
        n_attempted=attempted,
        n_valid=n_valid,
        valid_rate=round(rate, 5),
        valid_rate_ci_lo=round(ci_lo, 5),
        valid_rate_ci_hi=round(ci_hi, 5),
        mean_bridges_per_struct=round(mean_bridges, 3),
        frac_with_zero_bridges=round(frac_zero, 4),
        mean_context_perturb=round(mean_perturb, 4),
        mean_ligands_mutated_per_struct=round(mean_mutated, 3),
        denticity_recovery=round(dent_recovery, 4) if dent_recovery is not None else None,
        # Stash raw results for aqua-seed analysis
        _results=results,
    )


# ── CSV writer ─────────────────────────────────────────────────────────

def write_csv(rows, path):
    """Write context_ablation_results.csv."""
    fields = [
        "name", "n_heavy", "n_attempted", "n_valid", "valid_rate",
        "valid_rate_ci_lo", "valid_rate_ci_hi",
        "mean_bridges_per_struct", "frac_with_zero_bridges",
        "mean_context_perturb", "mean_ligands_mutated_per_struct",
        "denticity_recovery",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"\nWrote {path}")


# ── Headline figure: validity vs context density ───────────────────────

def make_headline_figure(rows, path_png, path_pdf):
    """Produce the key 'validity vs context atom count' plot.

    Publication-ready: >=300 dpi, serif font, no gridlines, readable axes.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

    fig, ax = plt.subplots(figsize=(6, 4.5))

    x = [r["n_heavy"] for r in rows]
    y = [100 * r["valid_rate"] for r in rows]
    yerr_lo = [100 * (r["valid_rate"] - r["valid_rate_ci_lo"]) for r in rows]
    yerr_hi = [100 * (r["valid_rate_ci_hi"] - r["valid_rate"]) for r in rows]
    labels = [r["label"] for r in rows]

    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi],
                marker="o", markersize=8, capsize=5, capthick=1.5,
                linewidth=2, color="#2c3e50", ecolor="#7f8c8d",
                markerfacecolor="#3498db", markeredgecolor="#2c3e50",
                markeredgewidth=1.5, zorder=5)

    # Annotate each point
    offsets = [(15, 10), (15, -15), (15, 10), (15, -15)]
    for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(lab, (xi, yi), textcoords="offset points",
                    xytext=(ox, oy), fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    ax.set_xlabel("Context atom count (heavy atoms)", fontsize=13)
    ax.set_ylabel("Valid complex rate (%)", fontsize=13)
    ax.set_title("Generative Quality vs. Coordination Context Density",
                 fontsize=13, fontweight="bold")

    # No gridlines per spec
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path_png} and {path_pdf}")


# ── Aqua-seed analysis ────────────────────────────────────────────────

def analyze_aqua_seed(aqua_row, ref_path):
    """Analyze context water preservation in Eu(H2O)9 generations.

    For each generated structure, check how many of the 8 context water
    oxygens are preserved vs mutated (element changed or significantly
    displaced).
    """
    results = aqua_row.get("_results", [])
    if not results:
        return None

    ref_info = analyze_reference(ref_path)
    n_preserved_list = []
    n_mutated_list = []

    for r in results:
        mut = r["mutation"]
        n_ctx_lig = mut["n_context_ligands"]
        n_mut = mut["n_mutated"]
        n_pres = n_ctx_lig - n_mut
        n_preserved_list.append(n_pres)
        n_mutated_list.append(n_mut)

    if not n_preserved_list:
        return None

    total_ctx_lig = np.mean([r["mutation"]["n_context_ligands"] for r in results])
    mean_preserved = np.mean(n_preserved_list)
    mean_mutated = np.mean(n_mutated_list)
    mutation_rate = mean_mutated / total_ctx_lig if total_ctx_lig > 0 else 0

    return dict(
        n_structures=len(results),
        mean_context_waters=round(total_ctx_lig, 2),
        mean_preserved=round(mean_preserved, 2),
        mean_mutated=round(mean_mutated, 2),
        mutation_rate=round(mutation_rate, 3),
        preserved_frac=round(1 - mutation_rate, 3),
        preserved_list=n_preserved_list,
        mutated_list=n_mutated_list,
    )


def make_aqua_seed_figure(aqua_stats, path_png, path_pdf):
    """Plot water preservation vs mutation in the aqua case."""
    if aqua_stats is None:
        print("  No aqua-seed data to plot")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) Histogram of preserved waters per structure
    ax = axes[0]
    preserved = aqua_stats["preserved_list"]
    bins = np.arange(-0.5, max(preserved) + 1.5, 1)
    ax.hist(preserved, bins=bins, color="#27ae60", edgecolor="white",
            alpha=0.85, rwidth=0.85)
    ax.set_xlabel("Preserved context waters", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("(a) Context water preservation", fontsize=12)
    ax.axvline(np.mean(preserved), color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(preserved):.1f}")
    ax.legend(fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Pie chart: preserved vs mutated
    ax = axes[1]
    pres_frac = aqua_stats["preserved_frac"]
    mut_frac = aqua_stats["mutation_rate"]
    sizes = [pres_frac, mut_frac]
    labels_pie = [f"Preserved\n({100*pres_frac:.1f}%)",
                  f"Mutated\n({100*mut_frac:.1f}%)"]
    colors = ["#27ae60", "#e74c3c"]
    ax.pie(sizes, labels=labels_pie, colors=colors, startangle=90,
           textprops={"fontsize": 11},
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("(b) Overall mutation rate", fontsize=12)

    fig.suptitle("Aqua-Seed Experiment: Eu(H$_2$O)$_9$ Context Fidelity",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path_png} and {path_pdf}")


# ── Summary + interpretation ──────────────────────────────────────────

def print_summary(rows, aqua_stats):
    """Print summary table and 2-paragraph interpretation."""
    print("\n" + "=" * 85)
    print("CONTEXT ABLATION SUMMARY (Experiment 2)")
    print("=" * 85)
    hdr = (f"{'Name':>20} {'atoms':>6} {'tried':>6} {'valid':>6} "
           f"{'rate%':>8} {'bridges':>9} {'perturb':>9} "
           f"{'mutated':>8} {'dent_rec':>9}")
    print(hdr)
    print("-" * 85)
    for r in rows:
        dr = f"{100*r['denticity_recovery']:.1f}%" if r['denticity_recovery'] is not None else "N/A"
        print(f"{r['name']:>20} {r['n_heavy']:>6} {r['n_attempted']:>6} "
              f"{r['n_valid']:>6} {100*r['valid_rate']:>7.2f}% "
              f"{r['mean_bridges_per_struct']:>9.2f} "
              f"{r['mean_context_perturb']:>8.4f} "
              f"{r['mean_ligands_mutated_per_struct']:>8.2f} "
              f"{dr:>9}")
    print("=" * 85)

    # ── Interpretation ──
    print("\n" + "=" * 85)
    print("INTERPRETATION")
    print("=" * 85)

    # Sort by n_heavy for trend analysis
    sorted_rows = sorted(rows, key=lambda r: r["n_heavy"])

    # (i) Is crowding the primary failure driver?
    rates = [(r["name"], r["n_heavy"], r["valid_rate"]) for r in sorted_rows]
    print("\n(i) Is crowding the primary failure driver?")
    if len(rates) >= 2:
        sparse = rates[0]  # bare or aqua
        dense = rates[-1]   # TMMA

        # Check if validity decreases monotonically with atom count
        monotonic = all(rates[i][2] >= rates[i+1][2]
                        for i in range(len(rates)-1))

        if monotonic and dense[2] < sparse[2] * 0.5:
            print(f"  YES: validity drops monotonically from {sparse[0]} "
                  f"({100*sparse[2]:.2f}%) to {dense[0]} ({100*dense[2]:.2f}%). "
                  f"Coordination crowding is the dominant failure mode.")
        elif dense[2] < sparse[2]:
            print(f"  PARTIALLY: validity decreases from {sparse[0]} "
                  f"({100*sparse[2]:.2f}%) to {dense[0]} ({100*dense[2]:.2f}%), "
                  f"but the relationship is not strictly monotonic. "
                  f"Crowding contributes but other factors (ligand complexity, "
                  f"charge distribution) also play a role.")
        else:
            print(f"  NO: validity at {dense[0]} ({100*dense[2]:.2f}%) is "
                  f"comparable to or higher than {sparse[0]} ({100*sparse[2]:.2f}%). "
                  f"The failure mode is model-intrinsic, not crowding-driven.")

    # (ii) Does aqua-seed elaboration work?
    print("\n(ii) Does aqua-seed elaboration work?")
    if aqua_stats:
        mr = aqua_stats["mutation_rate"]
        if mr < 0.20:
            print(f"  The model preserves context waters well (mutation rate = "
                  f"{100*mr:.1f}% < 20%). The aqua-seed paradigm does NOT apply: "
                  f"the model correctly treats waters as fixed context, not as "
                  f"growable placeholders. Single-ligand replacement from an aqua "
                  f"scaffold is viable.")
        elif mr < 0.50:
            print(f"  Moderate context mutation (rate = {100*mr:.1f}%). "
                  f"The model partially respects water identity but shows "
                  f"significant perturbation. The aqua-seed approach works "
                  f"with caveats: some context waters may be altered.")
        else:
            print(f"  High mutation rate ({100*mr:.1f}% > 50%): the model "
                  f"effectively uses waters as growable placeholders, supporting "
                  f"the 'aqua-seed elaboration' paradigm. This is a novel "
                  f"finding: the diffusion model treats monodentate waters as "
                  f"malleable seeds rather than fixed context. This has "
                  f"implications for scaffold-hopping in metal complex design.")
    else:
        print("  (no aqua-seed data available)")

    print()


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("Context Ablation Analysis (Experiment 2)")
    print(f"Working directory: {PROJECT_ROOT}")
    print()

    rows = []
    aqua_row = None
    for ctx in CONTEXTS:
        print(f"Analyzing {ctx['name']}...")
        result = analyze_one_context(ctx)
        if result is not None:
            rows.append(result)
            if ctx["name"] == "eu_aqua9":
                aqua_row = result

    if not rows:
        sys.exit("No results found. Have the sbatch jobs completed?")

    # Write CSV
    write_csv(rows, PROJECT_ROOT / "context_ablation_results.csv")

    # Headline figure
    make_headline_figure(
        rows,
        PROJECT_ROOT / "context_ablation_figure.png",
        PROJECT_ROOT / "context_ablation_figure.pdf",
    )

    # Aqua-seed analysis
    aqua_stats = None
    if aqua_row:
        aqua_ref = str(PROJECT_ROOT / "refs/eu_aqua9.xyz")
        aqua_stats = analyze_aqua_seed(aqua_row, aqua_ref)
        if aqua_stats:
            print(f"\nAqua-seed stats:")
            print(f"  Mean context waters: {aqua_stats['mean_context_waters']}")
            print(f"  Mean preserved: {aqua_stats['mean_preserved']}")
            print(f"  Mean mutated: {aqua_stats['mean_mutated']}")
            print(f"  Mutation rate: {100*aqua_stats['mutation_rate']:.1f}%")

        make_aqua_seed_figure(
            aqua_stats,
            PROJECT_ROOT / "aqua_seed_figure.png",
            PROJECT_ROOT / "aqua_seed_figure.pdf",
        )

    # Summary + interpretation
    print_summary(rows, aqua_stats)


if __name__ == "__main__":
    main()
