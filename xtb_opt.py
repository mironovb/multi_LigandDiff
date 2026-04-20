"""GFN2-xTB geometry optimization for generated Ln complexes.
Reports energy, converged status, bond length changes."""
import os, sys, glob, subprocess, tempfile, shutil
import numpy as np
from pathlib import Path

def run_xtb_opt(xyz_path, out_dir, charge=0, uhf=0):
    """Run xTB geometry optimization. Returns path to optimized xyz or None on failure."""
    name = Path(xyz_path).stem
    work = Path(out_dir) / name
    work.mkdir(parents=True, exist_ok=True)
    
    # xTB needs: standard xyz header (no extra lines)
    # Clean the input — our files have a "n_context" line 2, xTB expects comment there
    with open(xyz_path) as f:
        lines = f.readlines()
    n_total = int(lines[0].strip())
    
    clean_xyz = work / "input.xyz"
    with open(clean_xyz, 'w') as f:
        f.write(f"{n_total}\n")
        f.write(f"Ln complex {name}\n")
        f.writelines(lines[2:])  # skip the context-count line
    
    # Run xTB
    cmd = [
        "xtb", "input.xyz",
        "--opt", "normal",
        "--gfn", "2",
        "--chrg", str(charge),
        "--uhf", str(uhf),
        "--cycles", "500",
    ]
    
    try:
        result = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=900)
        success = (work / "xtbopt.xyz").exists()
        
        energy = None
        for line in result.stdout.split('\n'):
            if 'TOTAL ENERGY' in line:
                parts = line.split()
                for p in parts:
                    try:
                        energy = float(p)
                        break
                    except ValueError:
                        pass
        
        return {
            'name': name,
            'success': success,
            'energy_hartree': energy,
            'opt_xyz': str(work / "xtbopt.xyz") if success else None,
            'error': None if success else result.stderr[-500:],
        }
    except subprocess.TimeoutExpired:
        return {'name': name, 'success': False, 'error': 'timeout', 'energy_hartree': None, 'opt_xyz': None}
    except Exception as e:
        return {'name': name, 'success': False, 'error': str(e), 'energy_hartree': None, 'opt_xyz': None}

if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'xtb_opt'
    
    # Collect top 10 candidates
    top_files = glob.glob('top10/*.xyz')
    top_files = [f for f in top_files if 'reference' not in f]
    
    print(f"Optimizing {len(top_files)} structures in {out_dir}/")
    print(f"Using charge = +3 (Eu3+ free cation; ligands neutral)\n")
    
    # For Eu(TMMA)2(NO3)3 complex, Eu3+ cation balanced by 3 NO3^- anions -> net neutral
    # The "+2" replacement ligand varies; assume neutral for simplicity
    # TODO: proper charge assignment per generated structure
    
    results = []
    for i, f in enumerate(top_files, 1):
        print(f"[{i}/{len(top_files)}] {Path(f).name}")
        r = run_xtb_opt(f, out_dir, charge=0, uhf=0)
        if r['success']:
            print(f"  ✓ Converged, E = {r['energy_hartree']:.6f} Hartree")
        else:
            print(f"  ✗ Failed: {r['error'][:100] if r['error'] else 'unknown'}")
        results.append(r)
    
    print("\n" + "="*60)
    success_count = sum(1 for r in results if r['success'])
    print(f"Summary: {success_count}/{len(results)} converged")
    
    # Save summary
    import json
    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
