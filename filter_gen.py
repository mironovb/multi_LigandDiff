"""Filter generated complexes by geometric plausibility.
Report top candidates with clean donor geometry and no atom overlaps."""
import os, glob, sys
import numpy as np
from collections import Counter

# Reasonable bond/coord ranges (Å)
MIN_NONBOND = 1.3    # any 2 non-bonded atoms should be >1.3 Å apart
MIN_BOND = 1.1       # shortest real bond (e.g., C-H)
MAX_BOND = 1.9       # longest single bond we care about (C-P ~1.87)
EU_O_RANGE = (2.2, 2.8)
EU_N_RANGE = (2.4, 2.8)
EU_P_RANGE = (2.8, 3.2)

def load_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n_context = int(lines[1].strip())
    atoms = []
    for line in lines[2:]:
        p = line.split()
        if len(p) >= 4:
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms, n_context

def dist(a, b):
    return np.sqrt(sum((a[k+1]-b[k+1])**2 for k in range(3)))

def score(path):
    atoms, n_ctx = load_xyz(path)
    eu = atoms[0]
    generated = atoms[n_ctx:]
    n_gen = len(generated)
    
    if n_gen == 0:
        return None
    
    issues = []
    score = 100
    
    # 1. Check each generated atom's distance to Eu
    donor_count = 0
    for i, a in enumerate(generated):
        d_eu = dist(a, eu)
        elem = a[0]
        is_donor = False
        
        if elem == 'O' and EU_O_RANGE[0] <= d_eu <= EU_O_RANGE[1]:
            is_donor = True
            donor_count += 1
        elif elem == 'N' and EU_N_RANGE[0] <= d_eu <= EU_N_RANGE[1]:
            is_donor = True
            donor_count += 1
        elif elem == 'P' and EU_P_RANGE[0] <= d_eu <= EU_P_RANGE[1]:
            is_donor = True
            donor_count += 1
        elif d_eu > 6.0:
            issues.append(f"{elem}#{n_ctx+i+1} floating (d_Eu={d_eu:.2f})")
            score -= 20
        elif d_eu < 2.0:
            issues.append(f"{elem}#{n_ctx+i+1} too close to Eu (d={d_eu:.2f})")
            score -= 30
    
    # 2. Check for atom overlaps (any pair < MIN_NONBOND after excluding covalent bonds)
    for i in range(n_ctx, len(atoms)):
        for j in range(len(atoms)):
            if i == j:
                continue
            d = dist(atoms[i], atoms[j])
            if d < MIN_BOND:
                issues.append(f"Overlap: {atoms[i][0]}#{i+1}-{atoms[j][0]}#{j+1} d={d:.2f}")
                score -= 40
                break
    
    # 3. Reward coord sphere completeness
    if donor_count == 0:
        issues.append("No valid donor atoms generated")
        score -= 50
    
    return {
        'path': path,
        'n_gen': n_gen,
        'gen_composition': dict(Counter(a[0] for a in generated)),
        'donor_count': donor_count,
        'issues': issues,
        'score': max(0, score),
    }

# Score all generated files
results = []
for d in ['generated/eu_tmma_real_epoch48/noH', 'generated/eu_tmma_mask1_epoch48/noH']:
    for f in sorted(glob.glob(f'{d}/*.xyz')):
        r = score(f)
        if r:
            results.append(r)

# Sort by score
results.sort(key=lambda x: -x['score'])

print(f"Total structures analyzed: {len(results)}\n")

print("=" * 80)
print("TOP 10 BY SCORE")
print("=" * 80)
for i, r in enumerate(results[:10]):
    print(f"\n{i+1}. Score={r['score']}  {os.path.basename(r['path'])}")
    print(f"   Generated: {r['n_gen']} atoms, composition: {r['gen_composition']}")
    print(f"   Valid donors: {r['donor_count']}")
    if r['issues']:
        for iss in r['issues'][:3]:
            print(f"   ⚠ {iss}")

print("\n" + "=" * 80)
print("COMPOSITION SUMMARY ACROSS ALL STRUCTURES")
print("=" * 80)
all_comp = Counter()
for r in results:
    for elem, n in r['gen_composition'].items():
        all_comp[elem] += n
print(f"Total atoms generated across {len(results)} structures: {sum(all_comp.values())}")
print(f"By element: {dict(all_comp)}")

score_bins = Counter()
for r in results:
    if r['score'] >= 80:
        score_bins['excellent (80+)'] += 1
    elif r['score'] >= 60:
        score_bins['good (60-79)'] += 1
    elif r['score'] >= 40:
        score_bins['mediocre (40-59)'] += 1
    else:
        score_bins['poor (<40)'] += 1
print(f"\nScore distribution: {dict(score_bins)}")
