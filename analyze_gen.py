"""Analyze a generated xyz — coordination sphere, bond distances, atom types."""
import sys
import os
import numpy as np
from collections import Counter

def analyze(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    
    total = int(lines[0].strip())
    n_context = int(lines[1].strip())
    
    atoms = []
    for line in lines[2:]:
        p = line.split()
        if len(p) >= 4:
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    
    print(f"\n=== {os.path.basename(filepath)} ===")
    print(f"Total: {total} | Context: {n_context} | Generated: {len(atoms) - n_context}")
    
    eu = atoms[0]
    
    # Coordination sphere (within 3.0 Å)
    coord = []
    for i, a in enumerate(atoms[1:], 1):
        d = np.sqrt(sum((a[k+1]-eu[k+1])**2 for k in range(3)))
        if d < 3.0:
            role = "CTX" if i <= n_context - 1 else "GEN"
            coord.append((d, a[0], role, i))
    coord.sort()
    
    print(f"Eu coordination sphere (r<3.0 Å): CN={len(coord)}")
    for d, sym, role, i in coord:
        print(f"  [{role}] {sym} atom #{i}  d={d:.3f} Å")
    
    # Generated atoms detail
    print(f"\nGenerated atoms:")
    for i in range(n_context, len(atoms)):
        a = atoms[i]
        d_eu = np.sqrt(sum((a[k+1]-eu[k+1])**2 for k in range(3)))
        # Nearest neighbor
        nearest = min(
            ((np.sqrt(sum((a[k+1]-b[k+1])**2 for k in range(3))), j, b[0]) 
             for j, b in enumerate(atoms) if j != i),
            key=lambda x: x[0]
        )
        print(f"  [{i}] {a[0]} at ({a[1]:.3f}, {a[2]:.3f}, {a[3]:.3f})")
        print(f"      d_Eu = {d_eu:.3f} Å")
        print(f"      nearest: {nearest[2]} atom #{nearest[1]} at d={nearest[0]:.3f} Å")
    
    # Composition of generated part
    gen_elements = Counter(a[0] for a in atoms[n_context:])
    print(f"\nGenerated composition: {dict(gen_elements)}")

if __name__ == '__main__':
    for f in sys.argv[1:]:
        analyze(f)
