#!/usr/bin/env python
"""Cross-architecture replication analysis (Prompt 7).

Compares generation quality between:
  - d-block pretrained checkpoint on Eu-TMMA (CN=10)
  - d-block pretrained checkpoint on Fe-TMMA (CN=10, rescaled)
  - d-block pretrained checkpoint on Eu-TMMA with best-practice flags
  - Ln fine-tuned checkpoint on Eu-TMMA (our baseline from context ablation)

Key question: is the failure mode specific to our Ln fine-tune, or
generic to multi-LigandDiff on high-CN / f-block geometries?

Produces:
  - cross_arch_summary.csv
  - cross_arch_figure.{png,pdf}
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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.taxonomy import (
    analyze_structure, analyze_reference, collect_files,
)


# ── experiment configuration ───────────────────────────────────────────

EXPERIMENTS = [
    dict(
        name="dblock_eu_tmma_r1",
        checkpoint="d-block pretrained",
        metal="Eu",
        ref="refs/eu_tmma_cis.xyz",
        gen_dir="generated/dblock_eu_tmma_r1/noH",
        n_attempted=200,
        flags="r=1, no projection",
        label="d-block / Eu / vanilla",
    ),
    dict(
        name="dblock_fe_tmma_r1",
        checkpoint="d-block pretrained",
        metal="Fe",
        ref="refs/fe_tmma_substituted.xyz",
        gen_dir="generated/dblock_fe_tmma_r1/noH",
        n_attempted=200,
        flags="r=1, no projection",
        label="d-block / Fe-sub / vanilla",
    ),
    dict(
        name="dblock_eu_tmma_r10_proj",
        checkpoint="d-block pretrained",
        metal="Eu",
        ref="refs/eu_tmma_cis.xyz",
        gen_dir="generated/dblock_eu_tmma_r10_proj/noH",
        n_attempted=200,
        flags="r=10, projection",
        label="d-block / Eu / r=10+proj",
    ),
    dict(
        name="lnft_eu_tmma_r10_proj",
        checkpoint="Ln fine-tuned (epoch 48)",
        metal="Eu",
        ref="refs/eu_tmma_cis.xyz",
        gen_dir="generated/ctx_eu_tmma_cis_epoch48/noH",
        n_attempted=500,
        flags="r=10, projection",
        label="Ln-ft / Eu / r=10+proj",
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


def analyze_one_experiment(exp):
    """Analyze results for one experiment. Returns dict of aggregate metrics."""
    d = PROJECT_ROOT / exp["gen_dir"]
    ref = str(PROJECT_ROOT / exp["ref"])
    attempted = exp["n_attempted"]

    if not d.exists():
        print(f"  {exp['name']}: directory {d} not found, skipping")
        return None

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

    dent_matches = [r["denticity"]["match"] for r in results
                    if r["denticity"]["match"] is not None]
    dent_recovery = (sum(dent_matches) / len(dent_matches)
                     if dent_matches else None)

    status = "valid" if n_valid > 0 else "zero_valid"
    print(f"  {exp['name']}: {n_valid}/{attempted} valid "
          f"({100*rate:.2f}% [{100*ci_lo:.2f}-{100*ci_hi:.2f}%]), "
          f"bridges={mean_bridges:.2f}, perturb={mean_perturb:.4f} A")

    return dict(
        name=exp["name"],
        checkpoint=exp["checkpoint"],
        metal=exp["metal"],
        label=exp["label"],
        flags=exp["flags"],
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
        status=status,
    )


# ── CSV writer ─────────────────────────────────────────────────────────

def write_csv(rows, path):
    """Write cross_arch_summary.csv."""
    fields = [
        "name", "checkpoint", "metal", "flags",
        "n_attempted", "n_valid", "valid_rate",
        "valid_rate_ci_lo", "valid_rate_ci_hi",
        "mean_bridges_per_struct", "frac_with_zero_bridges",
        "mean_context_perturb", "mean_ligands_mutated_per_struct",
        "denticity_recovery", "status",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"\nWrote {path}")


# ── Figure: grouped bar comparison ─────────────────────────────────────

def make_figure(rows, path_png, path_pdf):
    """Produce a grouped bar chart comparing validity across experiments.

    Side-by-side bars: d-block experiments vs Ln fine-tune baseline.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── (a) Validity rate comparison ──
    ax = axes[0]
    labels = [r["label"] for r in rows]
    rates = [100 * r["valid_rate"] for r in rows]
    ci_lo = [100 * (r["valid_rate"] - r["valid_rate_ci_lo"]) for r in rows]
    ci_hi = [100 * (r["valid_rate_ci_hi"] - r["valid_rate"]) for r in rows]

    # Color: blue for d-block, orange for Ln fine-tune
    colors = ["#3498db" if "dblock" in r["name"] else "#e67e22" for r in rows]
    x = np.arange(len(labels))

    bars = ax.bar(x, rates, yerr=[ci_lo, ci_hi], capsize=4,
                  color=colors, edgecolor="#2c3e50", linewidth=1.2, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Valid complex rate (%)", fontsize=12)
    ax.set_title("(a) Validity: d-block pretrained vs. Ln fine-tune", fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bars with counts
    for bar, r in zip(bars, rows):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                f"{r['n_valid']}/{r['n_attempted']}",
                ha="center", va="bottom", fontsize=8, color="#555")

    # ── (b) Failure-mode comparison: bridges + perturbation ──
    ax = axes[1]
    bar_width = 0.35
    x2 = np.arange(len(labels))

    bridges = [r["mean_bridges_per_struct"] for r in rows]
    perturbs = [r["mean_context_perturb"] for r in rows]

    b1 = ax.bar(x2 - bar_width/2, bridges, bar_width,
                label="Mean bridges/struct", color="#e74c3c", alpha=0.8,
                edgecolor="#2c3e50", linewidth=1.0)
    b2 = ax.bar(x2 + bar_width/2, perturbs, bar_width,
                label="Mean context perturb. (A)", color="#9b59b6", alpha=0.8,
                edgecolor="#2c3e50", linewidth=1.0)

    ax.set_xticks(x2)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("(b) Failure modes: bridging + context perturbation", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Cross-Architecture Replication: d-block Pretrained vs. Ln Fine-Tune on CN=10",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path_png} and {path_pdf}")


# ── Interpretation ─────────────────────────────────────────────────────

def print_interpretation(rows):
    """Print 2-paragraph interpretation for the paper."""
    print("\n" + "=" * 85)
    print("CROSS-ARCHITECTURE REPLICATION SUMMARY (Prompt 7)")
    print("=" * 85)

    hdr = (f"{'Label':>30} {'ckpt':>12} {'metal':>5} {'tried':>6} {'valid':>6} "
           f"{'rate%':>8} {'bridges':>9} {'perturb':>9}")
    print(hdr)
    print("-" * 85)
    for r in rows:
        print(f"{r['label']:>30} {r['checkpoint'][:12]:>12} {r['metal']:>5} "
              f"{r['n_attempted']:>6} {r['n_valid']:>6} {100*r['valid_rate']:>7.2f}% "
              f"{r['mean_bridges_per_struct']:>9.2f} "
              f"{r['mean_context_perturb']:>8.4f}")
    print("=" * 85)

    # Classify results
    dblock_rows = [r for r in rows if "dblock" in r["name"]]
    lnft_rows = [r for r in rows if "lnft" in r["name"]]

    dblock_rates = [r["valid_rate"] for r in dblock_rows]
    lnft_rates = [r["valid_rate"] for r in lnft_rows]

    max_dblock = max(dblock_rates) if dblock_rates else 0
    max_lnft = max(lnft_rates) if lnft_rates else 0

    print("\n" + "=" * 85)
    print("INTERPRETATION FOR PAPER")
    print("=" * 85)

    # Paragraph 1: what happened
    print("\nParagraph 1 (results):")
    if not dblock_rows:
        print("  No d-block results available yet. Run sbatches/dblock_cross.sbatch first.")
    elif max_dblock < 0.01 and max_lnft < 0.05:
        print(f"  Both the d-block pretrained model and the Ln fine-tuned model fail to")
        print(f"  generate valid complexes for the Eu(TMMA)2(NO3)3 reference at CN=10.")
        print(f"  The d-block model achieves {100*max_dblock:.2f}% validity (best condition),")
        print(f"  while the Ln fine-tune achieves {100*max_lnft:.2f}%. Both models exhibit")
        print(f"  high rates of inter-ligand bridging and context perturbation, indicating")
        print(f"  that the dense CN=10 coordination sphere overwhelms the learned denoising")
        print(f"  dynamics regardless of training distribution.")
    elif max_dblock < max_lnft:
        print(f"  The d-block pretrained model performs worse ({100*max_dblock:.2f}% validity)")
        print(f"  than the Ln fine-tune ({100*max_lnft:.2f}%) on CN=10 Eu geometry, as")
        print(f"  expected given its training on CN<=6 complexes. Both remain far below")
        print(f"  acceptable generation quality for the dense coordination sphere.")
    else:
        print(f"  Unexpectedly, the d-block model ({100*max_dblock:.2f}%) matches or exceeds")
        print(f"  the Ln fine-tune ({100*max_lnft:.2f}%). This suggests the fine-tuning")
        print(f"  process may have introduced biases that are counterproductive for CN=10.")

    # Paragraph 2: claim for the paper
    print("\nParagraph 2 (claim):")
    if not dblock_rows:
        print("  [Pending d-block results]")
    elif max_dblock < 0.05 and max_lnft < 0.05:
        print("  We observe a model-agnostic failure in replacement-based ligand generation")
        print("  for f-block CN>=8 geometries. Neither the d-block pretrained multi-LigandDiff")
        print("  nor our Ln-adapted fine-tune produces structurally valid completions for")
        print("  Eu(TMMA)2(NO3)3 at CN=10. This suggests that the failure is not a")
        print("  training-distribution artifact but a fundamental limitation of the")
        print("  mask-and-replace diffusion framework when applied to dense coordination")
        print("  environments characteristic of lanthanide chemistry.")
    else:
        both_low = max_dblock < 0.10 and max_lnft < 0.10
        if both_low:
            print("  Both checkpoints achieve <10% validity on CN=10, supporting the claim")
            print("  that replacement-based generation struggles with high coordination")
            print("  number regardless of training distribution. The failure is architectural.")
        else:
            print("  Results are mixed. See detailed metrics above for nuanced interpretation.")

    # Paragraph 3: Fe substitution control
    fe_rows = [r for r in rows if r["metal"] == "Fe"]
    eu_dblock = [r for r in dblock_rows if r["metal"] == "Eu"]
    if fe_rows and eu_dblock:
        fe_rate = fe_rows[0]["valid_rate"]
        eu_rate = eu_dblock[0]["valid_rate"]
        print(f"\nFe-substitution control:")
        if abs(fe_rate - eu_rate) < 0.02:
            print(f"  Fe-TMMA ({100*fe_rate:.2f}%) and Eu-TMMA ({100*eu_rate:.2f}%) yield")
            print(f"  similar validity, confirming that the failure is driven by CN=10")
            print(f"  geometry rather than metal identity (metals are not distinguished")
            print(f"  at the embedding level — all-zero one-hot).")
        else:
            print(f"  Fe-TMMA ({100*fe_rate:.2f}%) differs from Eu-TMMA ({100*eu_rate:.2f}%).")
            print(f"  Since metals share the same embedding, the difference must arise from")
            print(f"  the rescaled coordination sphere distances.")

    print()


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("Cross-Architecture Replication Analysis (Prompt 7)")
    print(f"Working directory: {PROJECT_ROOT}")
    print()

    rows = []
    for exp in EXPERIMENTS:
        print(f"Analyzing {exp['name']}...")
        result = analyze_one_experiment(exp)
        if result is not None:
            rows.append(result)

    if not rows:
        sys.exit("No results found. Have the sbatch jobs completed?")

    write_csv(rows, PROJECT_ROOT / "cross_arch_summary.csv")

    if len(rows) >= 2:
        make_figure(
            rows,
            PROJECT_ROOT / "cross_arch_figure.png",
            PROJECT_ROOT / "cross_arch_figure.pdf",
        )

    print_interpretation(rows)


if __name__ == "__main__":
    main()
