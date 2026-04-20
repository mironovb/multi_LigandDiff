#!/usr/bin/env python
"""Prepare a transition-metal-substituted reference from Eu(TMMA)2(NO3)3.

Replaces Eu with Fe and rescales the coordination sphere from Ln-O
typical distances (~2.45 A) to TM-O typical distances (~2.0 A).

This creates a chemically implausible structure (Fe at CN=10 with TMMA
diamide ligands), but the purpose is to test whether the d-block
pretrained multi-LigandDiff model can handle CN=10 geometry at all.

Usage:
    python prepare_tm_reference.py
"""

import numpy as np
import os

# Configuration
INPUT_XYZ = "refs/eu_tmma_cis.xyz"
OUTPUT_XYZ = "refs/fe_tmma_substituted.xyz"
SUBSTITUTE_METAL = "Fe"
# Rescale factor: TM-O typical / Ln-O typical
SCALE = 2.0 / 2.45


def main():
    with open(INPUT_XYZ, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()

    # Line 2 (index 2) is the metal atom
    metal_parts = lines[2].split()
    metal_pos = np.array([float(x) for x in metal_parts[1:4]])

    # Parse all other atoms
    elements = []
    positions = []
    for line in lines[3:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elements.append(parts[0])
        positions.append(np.array([float(x) for x in parts[1:4]]))

    positions = np.array(positions)

    # Rescale: shrink coordination sphere around metal center
    new_positions = metal_pos + (positions - metal_pos) * SCALE

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_XYZ), exist_ok=True)
    with open(OUTPUT_XYZ, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{comment}\n")
        # Metal line (position unchanged)
        f.write(f"{SUBSTITUTE_METAL:2s}  {metal_pos[0]:12.5f}  {metal_pos[1]:12.5f}  {metal_pos[2]:12.5f}\n")
        # Ligand atoms with rescaled positions
        for elem, pos in zip(elements, new_positions):
            f.write(f"{elem:2s}  {pos[0]:12.5f}  {pos[1]:12.5f}  {pos[2]:12.5f}\n")

    print(f"Wrote {OUTPUT_XYZ}")
    print(f"  Metal: Eu -> {SUBSTITUTE_METAL}")
    print(f"  Scale factor: {SCALE:.5f} (shrinks Ln-O ~2.45 A -> TM-O ~2.0 A)")
    print(f"  Atoms: {n_atoms}")

    # Report distance statistics
    dists_orig = np.linalg.norm(positions - metal_pos, axis=1)
    dists_new = np.linalg.norm(new_positions - metal_pos, axis=1)
    print(f"\n  Original M-X distances: mean={np.mean(dists_orig):.3f}, "
          f"min={np.min(dists_orig):.3f}, max={np.max(dists_orig):.3f} A")
    print(f"  Rescaled M-X distances: mean={np.mean(dists_new):.3f}, "
          f"min={np.min(dists_new):.3f}, max={np.max(dists_new):.3f} A")


if __name__ == "__main__":
    main()
