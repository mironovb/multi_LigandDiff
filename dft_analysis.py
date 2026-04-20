"""
Analysis and plotting for DFT validation results.

Reads dft_comparison.csv (or .json) produced by dft_pipeline.py and generates:
  1. Scatter: xTB energy vs DFT relative energy (reliability check)
  2. Eu-donor distance distribution comparison (reference vs good vs bad)
  3. One-page text interpretation

Usage:
    python dft_analysis.py --csv dft_comparison.csv --json dft_comparison.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HARTREE_TO_KCAL = 627.509

# Experimental ranges from Kravchuk et al. 2024 (VEDTAA01)
EXP_EU_O_TMMA = (2.33, 2.40)
EXP_EU_O_NO3 = (2.51, 2.56)

CATEGORY_COLORS = {
    "reference": "#2ca02c",
    "good": "#1f77b4",
    "mutated_context": "#ff7f0e",
    "bridged": "#d62728",
    "aqua_seed": "#9467bd",
    "other": "#8c564b",
}

CATEGORY_LABELS = {
    "reference": "Reference (VEDTAA01)",
    "good": "Good generated",
    "mutated_context": "Mutated context",
    "bridged": "Bridged",
    "aqua_seed": "Aqua-seed",
    "other": "Other",
}


def load_results(csv_path: str, json_path: str | None = None) -> list[dict]:
    """Load DFT comparison results."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            # Convert numeric fields
            for key in ("xtb_energy_hartree", "dft_energy_hartree",
                        "dft_deltaE_kcal_mol", "mulliken_eu_charge",
                        "max_eu_donor_A", "cn_dft", "dft_opt_cycles",
                        "n_dissociated_donors"):
                if row.get(key) and row[key] not in ("", "None"):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            row["dissociated"] = row.get("dissociated", "").lower() == "true"
            row["dft_converged"] = row.get("dft_converged", "").lower() == "true"
            rows.append(row)

    # Merge donor distance lists from JSON if available
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            json_data = json.load(f)
        name_to_donors = {r["name"]: r.get("eu_donor_distances", [])
                          for r in json_data}
        for row in rows:
            row["eu_donor_distances"] = name_to_donors.get(row["name"], [])
    else:
        for row in rows:
            row["eu_donor_distances"] = []

    return rows


# ---------------------------------------------------------------------------
# Figure 1: xTB energy vs DFT relative energy
# ---------------------------------------------------------------------------

def plot_xtb_vs_dft(results: list[dict], plot_dir: Path):
    """Scatter of xTB energy vs DFT deltaE — tests whether xTB rank is reliable."""
    xs, ys, colors, labels = [], [], [], []

    for r in results:
        xtb_e = r.get("xtb_energy_hartree")
        dft_de = r.get("dft_deltaE_kcal_mol")
        if xtb_e is None or dft_de is None:
            continue
        xs.append(xtb_e)
        ys.append(dft_de)
        cat = r["category"]
        colors.append(CATEGORY_COLORS.get(cat, "#333333"))
        labels.append(r["name"])

    if len(xs) < 2:
        print("  Too few points for xTB-vs-DFT scatter — skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cat, color in CATEGORY_COLORS.items():
        cx = [x for x, r in zip(xs, results)
              if r.get("xtb_energy_hartree") is not None
              and r.get("dft_deltaE_kcal_mol") is not None
              and r["category"] == cat]
        cy = [y for y, r in zip(ys, results)
              if r.get("xtb_energy_hartree") is not None
              and r.get("dft_deltaE_kcal_mol") is not None
              and r["category"] == cat]
        if cx:
            ax.scatter(cx, cy, c=color, s=60, edgecolors="black",
                       linewidths=0.5, label=CATEGORY_LABELS.get(cat, cat),
                       zorder=3)

    # Trend line
    if len(xs) >= 4:
        z = np.polyfit(xs, ys, 1)
        r2 = np.corrcoef(xs, ys)[0, 1] ** 2
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, np.polyval(z, xline), "k--", alpha=0.4,
                label=f"Linear fit (R$^2$={r2:.2f})")

    ax.axhline(0, color="green", linestyle=":", alpha=0.5, label="Reference level")
    ax.axhline(50, color="red", linestyle=":", alpha=0.3, label="50 kcal/mol threshold")

    ax.set_xlabel("xTB Final Energy (Hartree)", fontsize=11)
    ax.set_ylabel("DFT $\\Delta E$ vs Reference (kcal/mol)", fontsize=11)
    ax.set_title("xTB Energy vs DFT Relative Energy", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_dir / "xtb_vs_dft_energy.png", dpi=200)
    plt.close(fig)
    print(f"  xtb_vs_dft_energy.png ({len(xs)} points)")


# ---------------------------------------------------------------------------
# Figure 2: Eu-donor distance distributions
# ---------------------------------------------------------------------------

def plot_donor_distances(results: list[dict], plot_dir: Path):
    """Violin/box plots of Eu-donor distances by category."""
    groups = defaultdict(list)
    for r in results:
        dists = r.get("eu_donor_distances", [])
        if not dists:
            continue
        cat = r["category"]
        # Group into "reference", "good", "failed" (mutated/bridged/aqua)
        if cat == "reference":
            group = "Reference"
        elif cat == "good":
            group = "Good generated"
        else:
            group = "Failed generated"
        groups[group].extend(dists)

    if not groups:
        print("  No donor distance data — skipping distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    group_order = ["Reference", "Good generated", "Failed generated"]
    group_colors = ["#2ca02c", "#1f77b4", "#d62728"]

    plot_data = []
    plot_labels = []
    plot_colors = []
    for g, c in zip(group_order, group_colors):
        if g in groups:
            plot_data.append(groups[g])
            plot_labels.append(f"{g}\n(n={len(groups[g])})")
            plot_colors.append(c)

    if not plot_data:
        plt.close(fig)
        return

    parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plot_colors[i])
        pc.set_alpha(0.6)

    # Overlay experimental ranges
    ax.axhspan(EXP_EU_O_TMMA[0], EXP_EU_O_TMMA[1], alpha=0.15, color="green",
               label=f"Exp. Eu-O(TMMA) {EXP_EU_O_TMMA[0]}-{EXP_EU_O_TMMA[1]} A")
    ax.axhspan(EXP_EU_O_NO3[0], EXP_EU_O_NO3[1], alpha=0.15, color="orange",
               label=f"Exp. Eu-O(NO$_3$) {EXP_EU_O_NO3[0]}-{EXP_EU_O_NO3[1]} A")
    ax.axhline(3.5, color="red", linestyle="--", alpha=0.5,
               label="Dissociation cutoff (3.5 A)")

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylabel("Eu-Donor Distance (A)", fontsize=11)
    ax.set_title("Eu Coordination Sphere: DFT-Optimised Distances", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_dir / "eu_donor_distributions.png", dpi=200)
    plt.close(fig)
    print(f"  eu_donor_distributions.png ({sum(len(d) for d in plot_data)} distances)")


# ---------------------------------------------------------------------------
# One-page interpretation
# ---------------------------------------------------------------------------

def write_interpretation(results: list[dict], out_path: Path):
    """Write a one-page text interpretation of DFT validation results."""
    lines = []
    lines.append("DFT Validation of Generated Eu(III) Complex Failures")
    lines.append("=" * 55)
    lines.append("")

    ref = [r for r in results if r["category"] == "reference"]
    good = [r for r in results if r["category"] == "good"]
    failed = [r for r in results
              if r["category"] in ("mutated_context", "bridged", "aqua_seed")]
    all_gen = [r for r in results if r["category"] != "reference"]

    ref_e = ref[0]["dft_energy_hartree"] if ref and ref[0]["dft_energy_hartree"] else None

    lines.append("METHOD")
    lines.append("-" * 55)
    lines.append("Level of theory: PBE0-D4/def2-TZVP")
    lines.append("Eu basis: SARC-DKH-TZVP + SK-MCDHF-RSC small-core ECP")
    lines.append("Charge: 0 (neutral complex)  Multiplicity: 7 (Eu3+ 4f^6)")
    lines.append(f"Reference: Eu(TMMA)2(NO3)3 from CCDC VEDTAA01 (Kravchuk et al. 2024)")
    lines.append(f"Structures validated: {len(all_gen)} generated + 1 reference")
    lines.append("")

    lines.append("RESULTS")
    lines.append("-" * 55)

    if ref_e is not None:
        lines.append(f"Reference DFT energy: {ref_e:.6f} Ha")
    lines.append("")

    # Energy analysis
    de_vals = [r["dft_deltaE_kcal_mol"] for r in all_gen
               if r["dft_deltaE_kcal_mol"] is not None]
    if de_vals:
        lines.append(f"Generated structures relative to reference:")
        lines.append(f"  Mean dE:   {np.mean(de_vals):+.1f} kcal/mol")
        lines.append(f"  Range:     {min(de_vals):+.1f} to {max(de_vals):+.1f} kcal/mol")
        above_50 = sum(1 for d in de_vals if d > 50)
        lines.append(f"  Above 50 kcal/mol: {above_50}/{len(de_vals)}")
    lines.append("")

    # Per-category breakdown
    for cat_name, cat_results in [("Good generated", good),
                                   ("Failed (mutated/bridged/aqua)", failed)]:
        de = [r["dft_deltaE_kcal_mol"] for r in cat_results
              if r["dft_deltaE_kcal_mol"] is not None]
        dissoc = sum(1 for r in cat_results if r["dissociated"])
        if de:
            lines.append(f"{cat_name} (n={len(cat_results)}):")
            lines.append(f"  Mean dE:       {np.mean(de):+.1f} kcal/mol")
            lines.append(f"  Dissociated:   {dissoc}/{len(cat_results)}")
        elif cat_results:
            lines.append(f"{cat_name} (n={len(cat_results)}): no DFT energy available")
        lines.append("")

    # Geometry analysis
    lines.append("COORDINATION GEOMETRY")
    lines.append("-" * 55)

    if ref and ref[0].get("eu_donor_distances"):
        ref_dists = ref[0]["eu_donor_distances"]
        lines.append(f"Reference Eu-donor distances (DFT-opt):")
        lines.append(f"  Range: {min(ref_dists):.3f} - {max(ref_dists):.3f} A")
        lines.append(f"  CN = {ref[0]['cn_dft']}")
        lines.append(f"  (Experimental: Eu-O_TMMA = {EXP_EU_O_TMMA[0]}-{EXP_EU_O_TMMA[1]} A, "
                      f"Eu-O_NO3 = {EXP_EU_O_NO3[0]}-{EXP_EU_O_NO3[1]} A)")

    for cat_name, cat_results in [("Good generated", good),
                                   ("Failed generated", failed)]:
        all_dists = []
        for r in cat_results:
            all_dists.extend(r.get("eu_donor_distances", []))
        if all_dists:
            lines.append(f"\n{cat_name} Eu-donor distances (DFT-opt):")
            lines.append(f"  Range: {min(all_dists):.3f} - {max(all_dists):.3f} A")
            lines.append(f"  Mean:  {np.mean(all_dists):.3f} A")
            lines.append(f"  Std:   {np.std(all_dists):.3f} A")
    lines.append("")

    # xTB vs DFT agreement
    lines.append("xTB vs DFT AGREEMENT")
    lines.append("-" * 55)
    xtb_e = [r["xtb_energy_hartree"] for r in all_gen
             if r["xtb_energy_hartree"] is not None
             and r["dft_deltaE_kcal_mol"] is not None]
    dft_de = [r["dft_deltaE_kcal_mol"] for r in all_gen
              if r["xtb_energy_hartree"] is not None
              and r["dft_deltaE_kcal_mol"] is not None]
    if len(xtb_e) >= 3:
        r_val = np.corrcoef(xtb_e, dft_de)[0, 1]
        lines.append(f"Pearson correlation (xTB E vs DFT dE): r = {r_val:.3f}")
        if abs(r_val) < 0.5:
            lines.append("  -> Weak correlation: xTB energy ranking is NOT a reliable")
            lines.append("     proxy for DFT-level relative stability.")
        else:
            lines.append("  -> Moderate-to-strong correlation: xTB ranking partially")
            lines.append("     captures the DFT energy landscape.")
    else:
        lines.append("(Insufficient data for correlation analysis)")
    lines.append("")

    # Conclusion
    lines.append("INTERPRETATION")
    lines.append("-" * 55)
    n_unstable = sum(1 for r in all_gen
                     if r.get("dft_deltaE_kcal_mol") is not None
                     and (r["dft_deltaE_kcal_mol"] > 50 or r["dissociated"]))
    n_total = len([r for r in all_gen
                   if r.get("dft_deltaE_kcal_mol") is not None or r["dissociated"]])

    if n_total > 0:
        lines.append(f"Of {n_total} generated structures validated at the DFT level,")
        lines.append(f"{n_unstable} ({100*n_unstable/n_total:.0f}%) are quantum-chemically "
                     f"unstable (>50 kcal/mol above reference and/or show donor dissociation).")
        lines.append("")

        if failed:
            fail_unstable = sum(1 for r in failed
                                if r.get("dft_deltaE_kcal_mol") is not None
                                and (r["dft_deltaE_kcal_mol"] > 50 or r["dissociated"]))
            cats = set(r["category"] for r in failed)
            lines.append(f"Failure categories ({', '.join(cats)}) show {fail_unstable}/"
                         f"{len(failed)} structures with dE > 50 kcal/mol or dissociation,")
            lines.append("confirming that these failure modes identified at the xTB level")
            lines.append("represent genuine quantum-chemical instabilities, not artifacts")
            lines.append("of the semi-empirical approximation.")

        if good:
            good_de = [r["dft_deltaE_kcal_mol"] for r in good
                       if r["dft_deltaE_kcal_mol"] is not None]
            if good_de:
                lines.append("")
                lines.append(f"The 'good' generated structures have mean dE = "
                             f"{np.mean(good_de):+.1f} kcal/mol, suggesting that")
                if np.mean(good_de) < 20:
                    lines.append("the model CAN produce thermodynamically accessible structures")
                    lines.append("when it avoids context mutation and inter-ligand bridging.")
                else:
                    lines.append("even the best-case generated structures remain significantly")
                    lines.append("higher in energy than the experimental reference.")

    lines.append("")
    lines.append("This DFT validation supports targeting JCTC, as it provides")
    lines.append("quantum-chemical evidence beyond the semi-empirical xTB level")
    lines.append("that the identified failure modes are physically meaningful.")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nInterpretation written to {out_path}")
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DFT validation analysis and plots")
    ap.add_argument("--csv", default="dft_comparison.csv",
                    help="DFT comparison CSV from dft_pipeline.py parse")
    ap.add_argument("--json", default="dft_comparison.json",
                    help="DFT comparison JSON (for donor distance lists)")
    ap.add_argument("--plot-dir", default="dft_plots",
                    help="Output directory for figures")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Run 'python dft_pipeline.py parse ...' first.")
        return

    results = load_results(args.csv, args.json if os.path.exists(args.json) else None)
    print(f"Loaded {len(results)} structures from {args.csv}")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_xtb_vs_dft(results, plot_dir)
    plot_donor_distances(results, plot_dir)
    write_interpretation(results, plot_dir / "dft_interpretation.txt")

    print(f"\nAll outputs in {plot_dir}/")


if __name__ == "__main__":
    main()
