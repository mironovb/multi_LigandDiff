"""Compare pre/post xTB optimization — bond lengths, Eu-donor distances."""
import os, glob
import numpy as np
from pathlib import Path

def load_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    atoms = []
    for line in lines[2:]:
        p = line.split()
        if len(p) >= 4:
            try:
                atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                continue
    return atoms

def dist(a, b):
    return np.sqrt(sum((a[i+1]-b[i+1])**2 for i in range(3)))

def analyze(name, pre_path, post_path, n_context):
    pre = load_xyz(pre_path)
    post = load_xyz(post_path)
    if len(pre) != len(post):
        print(f"  WARNING: atom count mismatch {len(pre)} vs {len(post)}")
        return
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"  Atoms: {len(pre)}  (context: {n_context}, generated: {len(pre)-n_context})")
    
    pre_eu = pre[0]
    post_eu = post[0]
    
    print(f"\n  Eu coordination sphere (within 3.2 Å in either pre or post):")
    print(f"  {'Atom':<10} {'Role':<5} {'Pre (Å)':<10} {'Post (Å)':<10} {'Δ':<10}")
    for i in range(1, len(pre)):
        d_pre = dist(pre_eu, pre[i])
        d_post = dist(post_eu, post[i])
        if d_pre < 3.2 or d_post < 3.2:
            role = "CTX" if i < n_context else "GEN"
            change = d_post - d_pre
            marker = "  *" if role == "GEN" else ""
            print(f"  {pre[i][0]:<3} #{i:<4} {role:<5} {d_pre:<10.3f} {d_post:<10.3f} {change:+.3f}{marker}")
    
    if len(pre) > n_context:
        print(f"\n  Generated atoms detail (pre-optimization bond problems):")
        for i in range(n_context, len(pre)):
            d_eu_pre = dist(pre[0], pre[i])
            d_eu_post = dist(post[0], post[i])
            dists_pre = sorted([(dist(pre[i], pre[j]), j, pre[j][0]) for j in range(len(pre)) if j != i])
            dists_post = sorted([(dist(post[i], post[j]), j, post[j][0]) for j in range(len(post)) if j != i])
            print(f"    {pre[i][0]} #{i}:  Eu-dist  pre: {d_eu_pre:.3f}  post: {d_eu_post:.3f}")
            print(f"      pre nearest:  {dists_pre[0][2]}#{dists_pre[0][1]} at {dists_pre[0][0]:.3f} Å")
            print(f"      post nearest: {dists_post[0][2]}#{dists_post[0][1]} at {dists_post[0][0]:.3f} Å")

# Walk all completed xtb_opt directories
for d in sorted(glob.glob('xtb_opt/*/')):
    name = Path(d).name
    pre_path = f'top10/{name}.xyz'
    post_path = f'{d}xtbopt.xyz'
    if not os.path.exists(post_path):
        print(f"\n{name}: NOT CONVERGED — skipping")
        continue
    with open(pre_path) as f:
        n_context = int(f.readlines()[1].strip())
    analyze(name, pre_path, post_path, n_context)
