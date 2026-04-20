#!/usr/bin/env python3
"""Figure 4: Validity vs context crowding — the key novel plot.

x-axis: context heavy-atom count (1, 10, 16, 35).
y-axis: valid_complex rate with 95% bootstrap CI error bars.

This is the first published quantification of generative quality degradation
as a function of metal coordination complexity.

Data source: Prompt 6 context ablation (analyze_context_ablation.py output).
Load from paper/tables/table2_context_ablation.csv or re-derive from raw data.
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# ── Load data from Table 2 ──────────────────────────────────────────────────
def load_table2():
    """Load context ablation data from table CSV."""
    path = "paper/tables/table2_context_ablation.csv"
    contexts, n_heavy, valid_rates, ci_lo, ci_hi = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(filter(lambda row: not row.startswith("#"), f))
        for row in reader:
            contexts.append(row["context"])
            n_heavy.append(int(row["context_heavy_atoms"]))
            valid_rates.append(float(row["valid_rate"]))
            ci_lo.append(float(row["valid_rate_ci_lo"]))
            ci_hi.append(float(row["valid_rate_ci_hi"]))
    return contexts, np.array(n_heavy), np.array(valid_rates), \
           np.array(ci_lo), np.array(ci_hi)


contexts, n_heavy, valid_rates, ci_lo, ci_hi = load_table2()
err_lo = valid_rates - ci_lo
err_hi = ci_hi - valid_rates

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.errorbar(n_heavy, valid_rates * 100, yerr=[err_lo * 100, err_hi * 100],
            fmt="o-", color="#5B8FBE", markersize=8, linewidth=1.8,
            capsize=5, capthick=1.5, markeredgecolor="#3A6A9A",
            markerfacecolor="#5B8FBE", ecolor="#888888", zorder=5)

# Label each point
for i, (ctx, nh, vr) in enumerate(zip(contexts, n_heavy, valid_rates)):
    offset_y = 1.8 if i < 2 else -2.5
    ax.annotate(ctx, (nh, vr * 100), textcoords="offset points",
                xytext=(0, offset_y + (8 if i < 2 else -8)),
                ha="center", fontsize=7.5, color="#444444",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5)
                if abs(offset_y) > 3 else None)

# d-block reference line
ax.axhline(89, color="#7BAE7F", linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(33, 90.5, "d-block CN=6 ref (89%)", fontsize=7.5, color="#5A8A5E",
        ha="right")

ax.set_xlabel("Context Heavy-Atom Count", fontsize=11)
ax.set_ylabel("Valid Complex Rate (%)", fontsize=11)
ax.set_title("Generative Quality Degrades with\nCoordination Crowding",
             fontsize=12, fontweight="bold")

ax.set_xlim(-2, 40)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Shaded region to emphasize the degradation
ax.fill_between([0, 40], [0, 0], [5, 5], color="#FFE0E0", alpha=0.3, zorder=0)
ax.text(38, 2, "< 5% validity", fontsize=7, ha="right", color="#CC5555",
        style="italic")

plt.tight_layout()
plt.savefig("paper/figures/fig4_validity_vs_crowding.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.savefig("paper/figures/fig4_validity_vs_crowding.pdf",
            bbox_inches="tight", facecolor="white")
plt.close()
print("Fig 4 saved: paper/figures/fig4_validity_vs_crowding.{png,pdf}")
