#!/usr/bin/env python
"""Design-ability test: sweep how many ligands are masked at once.

This is the *design* counterpart to ``generate_mask1.py`` (which only tests
single-ligand *completion*). It controls the mask size exactly:

    --mask_k 1     hide 1 of N ligands (completion control; known to work)
    --mask_k 2     hide 2 of N ligands (partial design)
    --mask_k 3     hide 3 of N ligands (partial design)
    --mask_k all   hide ALL ligands, leaving only the metal as context
                   (true de-novo generation of the whole coordination sphere)

For a complex with N ligands and a chosen ``mask_k``, every size-``mask_k``
subset of ligands is masked in turn; ``--n_samples`` independent seeds are drawn
per subset, and ``reform_data`` then expands each masked context over all legal
denticity partitions of the remaining coordination number. The generator,
RePaint resampling and exclusion-shell projection are reused verbatim from
``generate.py`` so the only thing this wrapper changes is *which* ligands are
hidden.

Outputs are written to ``{outdir}/mask{K}/noH/`` and a single machine-readable
summary line is printed per mask level, e.g.::

    design_test mask_k=all  context=0/5  attempts=6300  valid=0  yield=0.00%

so a downstream analysis script (``analyze_design_test.py``) can reconstruct the
degradation curve directly from stdout, and the N-valence rejection mechanism
from the RDKit warnings on stderr.

Reproduces the headline result of the lanthanide adaptation
(mask1 -> mask2 -> mask3 -> maskall = 126 -> 4 -> 0 -> 0 valid on
Eu(TMMA)2(NO3)3); see RESEARCH_PLAN.md.
"""

import argparse
import os
from itertools import combinations

import torch

from src import const
from torch_geometric.data import Data

from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown

# Reuse the verified generation machinery from generate.py rather than
# duplicating it (single source of truth for sampling / RePaint / projection).
from generate import (
    sort_pos,
    reform_data,
    generate_ligand,
    add_H,
    atom2idx,
    charges,
    metal_list,
)


def parse_complex_maskk(filename, mask_k):
    """Parse an .xyz complex and build one masked context per size-``mask_k``
    ligand subset.

    ``mask_k`` is an int, or the string 'all' to mask every ligand (leaving only
    the metal as context).
    """
    label = filename[:-4]
    data_list = []
    ele = []
    pos = []
    nuclear_charges = []
    noH_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i in lines[3:]:
        if i.split()[0] == 'H':
            continue
        noH_list.append(i)
        ele.append(atom2idx[i.split()[0]])
        nuclear_charges.append(charges[i.split()[0]])
        pos.append([float(j) for j in i.split()[1:]])
    noH_list.insert(0, lines[2])
    pos.insert(0, [float(j) for j in lines[2].split()[1:]])         # metal position
    nuclear_charges.insert(0, charges[lines[2].split()[0]])          # metal charge
    one_hot = torch.zeros(len(ele), 8)
    one_hot[range(len(ele)), ele] = 1
    one_hot = torch.cat([torch.zeros(8).view(1, -1), one_hot], dim=0)
    num_atoms = len(pos)
    pos = torch.tensor(pos)
    nuclear_charges = torch.tensor(nuclear_charges)

    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
    with open(f'{tmp_file}.xyz', 'w') as file:
        file.write(f"{num_atoms}\n\n")
        file.writelines(noH_list)

    mol = mol3D()
    mol.readfromxyz(f'{tmp_file}.xyz')
    liglist, ligdents, ligcon = ligand_breakdown(
        mol, silent=True, BondedOct=False, transition_metals_only=False)

    f_group = torch.zeros(num_atoms)
    for i in range(len(liglist)):
        f_group[liglist[i]] = i + 1
    ligand_group = torch.zeros((num_atoms, const.MAX_LIGANDS + 1))
    ligand_group[range(len(f_group.long())), f_group.long()] = 1

    anchor_group = torch.zeros(num_atoms)
    for i in range(len(ligcon)):
        anchor_group[ligcon[i]] = i + 1
    anchors_group = torch.zeros((num_atoms, const.MAX_LIGANDS + 1))
    anchors_group[range(len(anchor_group.long())), anchor_group.long()] = 1
    coord_site = anchors_group[:, 1:].any(dim=1).to(torch.int)

    n_ligands = len(liglist)
    if mask_k == 'all':
        k = n_ligands
    else:
        k = int(mask_k)
    if k < 1 or k > n_ligands:
        raise ValueError(
            f"mask_k={mask_k} invalid for a complex with {n_ligands} ligands "
            f"(must be 1..{n_ligands} or 'all').")

    # Mask exactly k ligands at a time: every size-k subset of ligand indices.
    lig_idx = list(range(n_ligands))
    for subset in combinations(lig_idx, k):
        ligand = torch.zeros(num_atoms)
        for li in subset:
            for atom in liglist[li]:
                ligand[atom] = 1
        context = 1 - ligand
        data = Data(pos=pos, label=label, context=context,
                    nuclear_charges=nuclear_charges, ligand_diff=ligand,
                    num_atoms=num_atoms, one_hot=one_hot,
                    ligand_group=ligand_group[:, 1:], coord_site=coord_site)
        data_list.append(data)

    print(f'Complex denticities: {ligdents}; n_ligands={n_ligands}; '
          f'mask_k={mask_k} -> {len(data_list)} masked subset(s)')
    return data_list, n_ligands


def read_molecule_maskk(filename, mask_k):
    if not filename.endswith('.xyz'):
        raise Exception('Unknown file extension, only .xyz file is supported')
    with open(filename, 'r') as file:
        metal = file.readlines()[2]
    if metal.split()[0] not in metal_list:
        if sort_pos(filename):
            print(f'Metal not first; rearranged to {filename[:-4]}_re.xyz')
            return parse_complex_maskk(f'{filename[:-4]}_re.xyz', mask_k)
        raise ValueError('Metal not in supported metals list (src/const.py).')
    return parse_complex_maskk(filename, mask_k)


def count_valid(outdir):
    noh = os.path.join(outdir, 'noH')
    if not os.path.isdir(noh):
        return 0
    return len([f for f in os.listdir(noh) if f.endswith('.xyz')])


def run_mask_level(complex_path, model, outdir, mask_k, n_samples, batch_size,
                   ligand_size, resample_r, project_enabled, d_min_start,
                   d_min_end, add_Hs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list, n_ligands = read_molecule_maskk(complex_path, mask_k)
    dataset = data_list * n_samples
    data = reform_data(dataset, device, ligand_size=ligand_size)
    attempts = len(data)
    print(f'{attempts} sampling attempts will be generated for mask_k={mask_k}')
    if attempts == 0:
        print(f'design_test mask_k={mask_k}  attempts=0  valid=0  yield=0.00%')
        return
    bs = min(batch_size, attempts)
    os.makedirs(outdir, exist_ok=True)
    generate_ligand(data, model, device, bs, outdir=outdir, resample_r=resample_r,
                    project_enabled=project_enabled, d_min_start=d_min_start,
                    d_min_end=d_min_end)
    valid = count_valid(outdir)
    k = n_ligands if mask_k == 'all' else int(mask_k)
    context_ligs = n_ligands - k
    yield_pct = 100.0 * valid / attempts if attempts else 0.0
    print(f'design_test mask_k={mask_k}  context={context_ligs}/{n_ligands}  '
          f'attempts={attempts}  valid={valid}  yield={yield_pct:.2f}%')
    if add_Hs and valid:
        add_H(complex_path, outdir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--complex', required=True, help='reference complex .xyz')
    p.add_argument('--model', required=True, help='checkpoint .ckpt')
    p.add_argument('--outdir', required=True, help='output root directory')
    p.add_argument('--mask_k', default='1',
                   help="ligands to hide at once: an int (1,2,3,...) or 'all'")
    p.add_argument('--n_samples', type=int, default=150,
                   help='independent seeds per masked subset')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--ligand_sizes', type=str, default='random')
    p.add_argument('--resample_r', type=int, default=5,
                   help='RePaint resampling iterations (working point r=5)')
    p.add_argument('--project_enabled', type=eval, default=False)
    p.add_argument('--d_min_start', type=float, default=1.5)
    p.add_argument('--d_min_end', type=float, default=1.3)
    p.add_argument('--add_Hs', type=eval, default=False)
    args = p.parse_args()

    tag = 'all' if str(args.mask_k) == 'all' else str(int(args.mask_k))
    outdir = os.path.join(args.outdir, f'mask{tag}')
    run_mask_level(args.complex, args.model, outdir, args.mask_k,
                   args.n_samples, args.batch_size, args.ligand_sizes,
                   args.resample_r, args.project_enabled, args.d_min_start,
                   args.d_min_end, args.add_Hs)
    print('Done!')


if __name__ == '__main__':
    main()
