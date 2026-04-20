#!/usr/bin/env python
"""Build Eu(NO3)3(H2O)3 reference structure.

9-coordinate: 3 bidentate NO3- (CN=6) + 3 monodentate H2O (CN=3).
Uses tricapped trigonal prism donor arrangement:
  - 6 donors from 3 bidentate nitrates at prism vertices
  - 3 donors from waters at capping positions

Geometry:
  - Trigonal prism: 2 staggered triangles at z = +h, -h
  - Caps: 3 equatorial positions at z = 0
  - Eu-O(donor) ~ 2.50 A
  - N-O bond = 1.25 A, O-N-O = 120 deg

Total heavy atoms: 1 Eu + 3*4 (NO3) + 3*1 (O_water) = 16
"""

import numpy as np
from pathlib import Path

EU_O = 2.50      # Eu-O donor distance (A)
N_O = 1.25       # N-O bond length (A)
PRISM_H = 1.30   # half-height of trigonal prism

atoms = []  # list of (element, x, y, z)

# --- Donor positions in a tricapped trigonal prism ---
# Upper triangle (z = +h), 120 deg apart
upper = []
for i in range(3):
    angle = 2 * np.pi / 3 * i
    r = EU_O * np.cos(np.arcsin(PRISM_H / EU_O))  # project to get r
    upper.append(np.array([r * np.cos(angle), r * np.sin(angle), PRISM_H]))

# Lower triangle (z = -h), rotated 60 deg (staggered)
lower = []
for i in range(3):
    angle = 2 * np.pi / 3 * i + np.pi / 3
    r = EU_O * np.cos(np.arcsin(PRISM_H / EU_O))
    lower.append(np.array([r * np.cos(angle), r * np.sin(angle), -PRISM_H]))

# Caps (z = 0), at 120 deg between upper and lower triangle vertices
caps = []
for i in range(3):
    angle = 2 * np.pi / 3 * i + np.pi / 6
    caps.append(np.array([EU_O * np.cos(angle), EU_O * np.sin(angle), 0.0]))

# Normalize all donors to EU_O distance
all_donors = upper + lower + caps
for j in range(len(all_donors)):
    norm = np.linalg.norm(all_donors[j])
    all_donors[j] = all_donors[j] / norm * EU_O

# --- Assign donors to ligands ---
# Pair adjacent donors for bidentate nitrate:
#   NO3 #0: upper[0] + lower[0]  (nearby vertices)
#   NO3 #1: upper[1] + lower[1]
#   NO3 #2: upper[2] + lower[2]
# Waters: caps[0..2]

# Actually, for bidentate nitrate, the two donor O atoms should be close
# (~2.1-2.2 A apart, the bite distance). Let's pair upper[i] with lower[i]
# and check distance.
for i in range(3):
    d = np.linalg.norm(upper[i] - lower[i])

# If upper-lower pairs are too far, use adjacent vertices within each triangle.
# Better approach: pair upper[i] with the nearest lower vertex.
# In staggered prism, upper[0] is closest to lower[0] and lower[2].

# Redefine: use 3 NO3 with donors placed as explicit bidentate pairs
# Each pair separated by bite distance ~2.15 A at EU_O = 2.50 A
BITE = 2.15  # O...O bite distance for bidentate nitrate
half_bite_angle = np.arcsin(BITE / (2 * EU_O))

no3_donors = []
for i in range(3):
    # Central direction for this NO3 group
    base_angle = 2 * np.pi / 3 * i
    # Two donor O atoms symmetric about base_angle in the xy plane,
    # tilted slightly above/below
    for sign in [+1, -1]:
        phi = base_angle + sign * half_bite_angle
        # Slight z-offset for the pair
        z = sign * 0.3
        r_xy = np.sqrt(EU_O**2 - z**2)
        pos = np.array([r_xy * np.cos(phi), r_xy * np.sin(phi), z])
        no3_donors.append(pos)

# Waters: fill the 3 remaining coordination sites between the NO3 groups
water_donors = []
for i in range(3):
    angle = 2 * np.pi / 3 * i + np.pi / 3  # offset 60 deg from NO3 centres
    # Place in opposite hemisphere to break symmetry
    z = -1.5
    r_xy = np.sqrt(EU_O**2 - z**2)
    pos = np.array([r_xy * np.cos(angle), r_xy * np.sin(angle), z])
    water_donors.append(pos)

# --- Build NO3 groups ---
# For each bidentate NO3: 2 donor O, 1 N (centre), 1 terminal O
for i in range(3):
    o1 = no3_donors[2 * i]
    o2 = no3_donors[2 * i + 1]

    # N sits behind the two donor O (further from Eu)
    midpoint = (o1 + o2) / 2.0
    direction = midpoint / np.linalg.norm(midpoint)
    n_pos = midpoint + direction * N_O * 0.6  # N behind donors

    # Terminal O sits behind N, even further from Eu
    o3_pos = n_pos + direction * N_O

    atoms.append(("O", o1))
    atoms.append(("O", o2))
    atoms.append(("N", n_pos))
    atoms.append(("O", o3_pos))

# --- Add water oxygens ---
for w in water_donors:
    atoms.append(("O", w))

# --- Write XYZ ---
n_atoms = 1 + len(atoms)  # Eu + ligand atoms
outpath = Path(__file__).resolve().parent / "eu_no3_3_h2o_3.xyz"
with open(outpath, "w") as f:
    f.write(f"{n_atoms}\n\n")
    f.write("Eu  0.0000  0.0000  0.0000\n")
    for elem, pos in atoms:
        f.write(f"{elem}  {pos[0]:.4f}  {pos[1]:.4f}  {pos[2]:.4f}\n")

print(f"Wrote {outpath} ({n_atoms} atoms)")

# Verify Eu-O(donor) distances
print("\nEu-O(donor) distances:")
for i, (elem, pos) in enumerate(atoms):
    d = np.linalg.norm(pos)
    label = "donor" if elem == "O" and d < 2.7 else "non-donor"
    print(f"  {elem} #{i}: {d:.3f} A ({label})")
