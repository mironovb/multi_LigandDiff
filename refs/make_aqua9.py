#!/usr/bin/env python
"""Build Eu(H2O)9 capped-square-antiprism reference structure.

Places 9 oxygens at ~2.45 A from Eu in a capped square antiprism geometry:
  - Upper square (4 O) at z = +1.2, rotated 45 deg
  - Lower square (4 O) at z = -1.2
  - 1 axial cap at z = +2.45
All positions normalized to 2.45 A Eu-O distance.
"""

import numpy as np
from pathlib import Path

positions = []
# Upper square (at z = 1.2)
for i in range(4):
    angle = np.pi / 2 * i + np.pi / 4
    positions.append([np.cos(angle) * 1.8, np.sin(angle) * 1.8, 1.2])
# Lower square (at z = -1.2, rotated 45 deg from upper)
for i in range(4):
    angle = np.pi / 2 * i
    positions.append([np.cos(angle) * 1.8, np.sin(angle) * 1.8, -1.2])
# Cap
positions.append([0, 0, 2.45])

# Normalize all to ~2.45 A from origin (Eu position)
positions = np.array(positions)
norms = np.linalg.norm(positions, axis=1, keepdims=True)
positions = positions / norms * 2.45

outpath = Path(__file__).resolve().parent / "eu_aqua9.xyz"
with open(outpath, "w") as f:
    f.write("10\n\n")
    f.write("Eu  0.0000  0.0000  0.0000\n")
    for p in positions:
        f.write(f"O  {p[0]:.4f}  {p[1]:.4f}  {p[2]:.4f}\n")

print(f"Wrote {outpath}")
print(f"Eu-O distances: {np.linalg.norm(positions, axis=1).round(4)}")
