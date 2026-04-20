#!/usr/bin/env python3
"""Figure 5: Generated structure showcase (3x2 grid).

Six panels showing reference, best result, aqua-seed, xTB-relaxed, and two
failure modes.  Uses ASE for simple ball-and-stick rendering.

NOTE: If ASE is not available, generates placeholder panels with labeled text.
When real xyz files are available, update the paths below.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ── Structure paths (update when real data is available) ─────────────────────
STRUCTURES = {
    "(a) Reference\nEu(TMMA)\u2082(NO\u2083)\u2083":
        "refs/eu_tmma_cis.xyz",
    "(b) Best RePaint+proj\n(chemically plausible)":
        "xtb_opt/125_20_eu_tmma_cis_[2, 2, 2, 2]_[2]/xtbopt.xyz",
    "(c) Aqua-seed context\nEu(H\u2082O)\u2089 \u2192 generated":
        "refs/eu_aqua9.xyz",
    "(d) xTB-relaxed\nEu coordination preserved":
        "xtb_opt/29_22_eu_tmma_cis_[2, 2, 2, 2]_[2]/xtbopt.xyz",
    "(e) Failure: bridged O\u2013O\n(vanilla sampler)":
        None,  # placeholder
    "(f) Failure: context mutation\n(ligand identity lost)":
        None,  # placeholder
}

# ── Element colors ───────────────────────────────────────────────────────────
ELEM_COLORS = {
    "Eu": "#FF6B6B", "O": "#CC3333", "N": "#3366CC", "C": "#666666",
    "H": "#CCCCCC", "S": "#CCCC33", "P": "#FF9933", "Cl": "#33CC33",
}
ELEM_RADII = {
    "Eu": 0.35, "O": 0.15, "N": 0.16, "C": 0.14, "H": 0.08,
}

def parse_xyz(path):
    """Parse xyz file into elements and coordinates."""
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    elements, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return elements, np.array(coords)

def plot_structure(ax, elements, coords, title):
    """Simple 2D projection of 3D coordinates."""
    if elements is None:
        ax.text(0.5, 0.5, "Structure\nnot available\n(run on cluster)",
                ha="center", va="center", fontsize=9, color="#888888",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return

    # Center and project onto xy
    coords = coords - coords.mean(axis=0)
    x, y = coords[:, 0], coords[:, 1]

    # Draw bonds (simple distance-based)
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < 2.8:  # generous cutoff for metal-ligand bonds
                lw = 0.8 if d > 2.2 else 1.2
                alpha = 0.3 if d > 2.2 else 0.6
                ax.plot([x[i], x[j]], [y[i], y[j]], "-", color="#AAAAAA",
                        linewidth=lw, alpha=alpha, zorder=1)

    # Draw atoms
    for i, elem in enumerate(elements):
        color = ELEM_COLORS.get(elem, "#999999")
        radius = ELEM_RADII.get(elem, 0.12)
        size = radius * 800
        zorder = 10 if elem == "Eu" else 5
        ax.scatter(x[i], y[i], s=size, c=color, edgecolors="#333333",
                   linewidth=0.3, zorder=zorder, alpha=0.9)
        if elem == "Eu":
            ax.text(x[i], y[i], "Eu", ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white", zorder=11)

    margin = 1.5
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")


# ── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.patch.set_facecolor("white")

for ax, (title, path) in zip(axes.flat, STRUCTURES.items()):
    if path is not None:
        elements, coords = parse_xyz(path)
    else:
        elements, coords = None, None
    plot_structure(ax, elements, coords, title)

plt.tight_layout(pad=1.5)
plt.savefig("paper/figures/fig5_structure_showcase.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.savefig("paper/figures/fig5_structure_showcase.pdf",
            bbox_inches="tight", facecolor="white")
plt.close()
print("Fig 5 saved: paper/figures/fig5_structure_showcase.{png,pdf}")
