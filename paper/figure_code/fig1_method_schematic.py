#!/usr/bin/env python3
"""Figure 1: Method schematic — CSD → fine-tuning → sampler variants → validation.

Clean line-art diagram in the style of Jin & Merz 2024 JCTC Fig 1.
Three-panel layout: (Left) data pipeline, (Center) sampler variants, (Right) validation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Muted color palette ──────────────────────────────────────────────────────
C_DATA   = "#5B8FBE"   # steel blue
C_MODEL  = "#7BAE7F"   # sage green
C_SAMPLE = "#D4A574"   # warm tan
C_VALID  = "#C47A7A"   # muted rose
C_BG     = "#FAFAFA"
C_ARROW  = "#555555"
C_TEXT   = "#333333"

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.2, 1]})
for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
fig.patch.set_facecolor("white")

def box(ax, x, y, w, h, text, color, fontsize=8, bold=False):
    """Draw a rounded box with centered text."""
    fb = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle="round,pad=0.15", facecolor=color,
                        edgecolor="#444444", linewidth=1.0, alpha=0.85)
    ax.add_patch(fb)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=C_TEXT, weight=weight, wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.15",
                                color=C_ARROW, lw=1.2))

# ── Panel A: Data pipeline ───────────────────────────────────────────────────
ax = axes[0]
ax.set_title("(a) Data & Fine-Tuning", fontsize=10, fontweight="bold", pad=10)

box(ax, 5, 9.0, 6, 1.0, "CSD Ln database\n95,279 complexes (CN 7\u201310)", C_DATA, bold=True)
arrow(ax, 5, 8.45, 5, 7.55)
box(ax, 5, 7.0, 5.5, 0.9, "Curation\nfilter, partition, split", C_DATA)
arrow(ax, 5, 6.5, 5, 5.6)
box(ax, 5, 5.0, 5.5, 1.0, "d-block pretrained\nEGNN + GVP backbone", C_MODEL)
arrow(ax, 5, 4.45, 5, 3.55)
box(ax, 5, 3.0, 5.5, 0.9, "Fine-tune 63 epochs\nval_loss: 977 \u2192 50", C_MODEL)
arrow(ax, 5, 2.5, 5, 1.6)
box(ax, 5, 1.0, 5.5, 1.0, "Ln-adapted\nmulti-LigandDiff", C_MODEL, bold=True)

# ── Panel B: Sampler variants ────────────────────────────────────────────────
ax = axes[1]
ax.set_title("(b) Sampler Variants", fontsize=10, fontweight="bold", pad=10)

box(ax, 5, 9.0, 7, 1.0, "Reverse diffusion\nx\u209c \u2192 x\u2080", C_SAMPLE, bold=True)

# Three branches
arrow(ax, 2.5, 8.45, 2.0, 7.55)
arrow(ax, 5, 8.45, 5, 7.55)
arrow(ax, 7.5, 8.45, 8.0, 7.55)

box(ax, 2.0, 7.0, 3.0, 0.9, "Vanilla\nDDPM", C_SAMPLE)
box(ax, 5.0, 7.0, 3.0, 0.9, "RePaint\nr resamples/step", C_SAMPLE)
box(ax, 8.0, 7.0, 3.0, 0.9, "Projected\nd_min shell", C_SAMPLE)

# Combine
arrow(ax, 5, 6.5, 5, 5.6)
arrow(ax, 8, 6.5, 6.5, 5.6)
box(ax, 5.5, 5.0, 5, 1.0, "RePaint + Projection\n(best combined)", C_SAMPLE, bold=True)

arrow(ax, 5.5, 4.45, 5.5, 3.55)
box(ax, 5.5, 3.0, 6, 0.9, "Key parameters\nr=10, d_min: 1.5\u21921.3 \u00c5", "#E8E0D6")

# Annotations
ax.text(2.0, 5.8, "\u2717 bridges\n\u2717 mutations", ha="center", fontsize=6.5,
        color="#AA3333", style="italic")
ax.text(8.5, 5.0, "\u2713 no bridges\n\u2713 fewer mutations", ha="center", fontsize=6.5,
        color="#338833", style="italic")

# ── Panel C: Validation pipeline ─────────────────────────────────────────────
ax = axes[2]
ax.set_title("(c) Validation Pipeline", fontsize=10, fontweight="bold", pad=10)

box(ax, 5, 9.0, 6, 1.0, "Generated complex\n(xyz, no H)", C_VALID, bold=True)
arrow(ax, 5, 8.45, 5, 7.55)
box(ax, 5, 7.0, 5.5, 0.9, "Sanity checks\nvalidity, connectivity", C_VALID)
arrow(ax, 5, 6.5, 5, 5.6)
box(ax, 5, 5.0, 5.5, 1.0, "Failure taxonomy\nbridges, mutations,\nperturbation, denticity", "#D4A0A0")
arrow(ax, 5, 4.45, 5, 3.55)
box(ax, 5, 3.0, 5.5, 0.9, "GFN2-xTB\ngeometry optimization", C_VALID)
arrow(ax, 5, 2.5, 5, 1.6)
box(ax, 5, 1.0, 5.5, 1.0, "PBE0-D4 / def2-TZVP\nORCA DFT validation", C_VALID)

plt.tight_layout(w_pad=1.5)
plt.savefig("paper/figures/fig1_method_schematic.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.savefig("paper/figures/fig1_method_schematic.pdf", bbox_inches="tight",
            facecolor="white")
plt.close()
print("Fig 1 saved: paper/figures/fig1_method_schematic.{png,pdf}")
