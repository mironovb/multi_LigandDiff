"""Total-generation mode for bare metal (CN_c = 0).

When the reference contains only a bare metal ion with no ligands,
the standard generate_mask1.py / generate.py cannot operate because
ligand_breakdown() returns nothing to mask. This script bypasses the
parse/mask pipeline and directly creates Data objects for total
generation with a user-specified target coordination number.

Usage:
    python generate_bare.py \
        --outdir generated/ctx_eu_bare_epoch48 \
        --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
        --complex refs/eu_bare.xyz \
        --target_cn 9 \
        --batch_size 32 --n_samples 500 \
        --resample_r 10 --project_enabled True
"""

import argparse
import os
import json
import numpy as np
import torch
from collections import Counter
from src import const
from src import utils
from src.lightning import DDPM
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter_add
from src.molecule_builder import (
    BasicLigandMetrics, build_mol, sanitycheck, write_xyz_file,
)

atom2idx = const.ATOM2IDX
idx2atom = const.IDX2ATOM
charges = const.CHARGES
num_atom_types = const.NUMBER_OF_ATOM_TYPES
metal_list = const.metals

parser = argparse.ArgumentParser(description="Total generation from bare metal")
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--complex', type=str, required=True,
                    help='Bare metal xyz file (1 atom)')
parser.add_argument('--target_cn', type=int, default=9,
                    help='Target coordination number (default 9 for Eu)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--ligand_sizes', type=str, default='random')
parser.add_argument('--resample_r', type=int, default=1)
parser.add_argument('--project_enabled', type=eval, default=False)
parser.add_argument('--d_min_start', type=float, default=1.5)
parser.add_argument('--d_min_end', type=float, default=1.3)

# Chemically reasonable LD_g partitions for CN=9 lanthanides.
# Limited to <=5 ligands with denticities 1-4.
SELECTED_LD9 = [
    [3, 3, 3],         # three tridentate
    [4, 4, 1],         # two tetradentate + one monodentate
    [4, 3, 2],         # mixed tetra/tri/bidentate
    [3, 3, 2, 1],      # mixed tri/bi/mono
    [3, 2, 2, 2],      # one tridentate + three bidentate
    [2, 2, 2, 2, 1],   # four bidentate + one monodentate
]


def read_bare_metal(filename):
    """Read a bare-metal xyz file (1 atom) and return (element, position)."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Line 0: atom count, Line 1: comment, Line 2: metal coords
    parts = lines[2].split()
    element = parts[0]
    pos = [float(x) for x in parts[1:4]]
    if element not in metal_list:
        raise ValueError(f"{element} not in supported metals list")
    return element, pos


def build_data_objects(metal_elem, metal_pos, target_cn, n_samples,
                       ligand_size='random', device='cpu'):
    """Create Data objects for total generation (CN_c=0).

    For each selected LD_g partition and each sample, creates one Data
    object with:
      - context: just the metal
      - generated: placeholder atoms sized per denticity
    """
    metal_charge = charges[metal_elem]
    metal_pos_t = torch.tensor([metal_pos], dtype=torch.float32)
    label_base = os.path.splitext(os.path.basename(args.complex))[0]

    # Select partitions: use SELECTED_LD9 for CN=9, else compute all
    if target_cn == 9:
        ld_options = SELECTED_LD9
    else:
        all_parts = const.denticity_partitions(target_cn)
        # Filter to <=5 ligands, max denticity 4
        ld_options = [p for p in all_parts
                      if len(p) <= 5 and max(p) <= 4]
        if not ld_options:
            ld_options = all_parts[:10]  # fallback to first 10

    print(f"Target CN={target_cn}, using {len(ld_options)} LD_g partitions:")
    for ld in ld_options:
        print(f"  {ld}")

    data_list = []
    for ld_g in ld_options:
        for _ in range(n_samples):
            # Context: just the metal
            ctx_pos = metal_pos_t.clone()                  # [1, 3]
            ctx_onehot = torch.zeros(1, num_atom_types)     # metal has zero one-hot
            ctx_charges = torch.tensor([metal_charge], dtype=torch.float32)
            ctx_coord_site = torch.zeros(1)                 # metal is not a coord site
            ctx_ligand_group = torch.zeros(1, const.MAX_LIGANDS)

            # Generated placeholders: one block per ligand in LD_g
            gen_groups = []
            gen_coord_sites = []
            for k, dent in enumerate(ld_g):
                if dent < 3:
                    lig_size = np.random.randint(dent, 10)
                else:
                    if ligand_size == 'random':
                        lig_size = np.random.randint(10, 30)
                    else:
                        lig_size = int(ligand_size)
                assert lig_size >= dent

                grp = torch.zeros(lig_size, const.MAX_LIGANDS)
                grp[:, k] = 1
                gen_groups.append(grp)

                cs = torch.zeros(lig_size)
                cs[:dent] = 1
                gen_coord_sites.append(cs)

            gen_ligand_group = torch.cat(gen_groups, dim=0)
            gen_size = gen_ligand_group.shape[0]
            gen_pos = torch.zeros(gen_size, 3)
            gen_coord_site = torch.cat(gen_coord_sites, dim=0)
            gen_onehot = torch.zeros(gen_size, num_atom_types)

            # Combine context + generated
            pos = torch.cat([ctx_pos, gen_pos], dim=0)
            context = torch.cat([torch.ones(1), torch.zeros(gen_size)], dim=0)
            ligand_diff = torch.cat([torch.zeros(1), torch.ones(gen_size)], dim=0)
            ncharges = torch.cat([ctx_charges, torch.zeros(gen_size)], dim=0)
            coord_site = torch.cat([ctx_coord_site, gen_coord_site], dim=0)
            ligand_group = torch.cat([ctx_ligand_group, gen_ligand_group], dim=0)
            onehot = torch.cat([ctx_onehot, gen_onehot], dim=0)
            natoms = pos.shape[0]

            assert torch.sum(coord_site).item() == target_cn

            data = Data(
                pos=pos.to(device),
                label=f"{label_base}_[]_{ld_g}",
                coord_site=coord_site.to(device),
                nuclear_charges=ncharges.to(device),
                context=context.to(device),
                ligand_diff=ligand_diff.to(device),
                ligand_group=ligand_group.to(device),
                one_hot=onehot.to(device),
                num_atoms=natoms,
            )
            data_list.append(data)

    return data_list


# Reuse generate_ligand from generate_mask1.py (identical logic)
def generate_ligand(data, model, device, batch_size=32, outdir='generated',
                    resample_r=1, project_enabled=False,
                    d_min_start=1.5, d_min_end=1.3):
    os.makedirs(f'{outdir}/noH', exist_ok=True)
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ligand_metrics = BasicLigandMetrics()
    num = 0
    reasons = Counter()
    attempts = 0
    for b, data in enumerate(dataloader):
        pos_original = data['pos']
        batch_seg = data.batch
        bs = torch.max(batch_seg) + 1
        context = data['context'].view(-1, 1)
        metals = [data['nuclear_charges'][batch_seg == i][0] for i in range(bs)]
        fixed_mean = (scatter_add(pos_original * context, batch_seg, dim=0)
                      / scatter_add(context, batch_seg, dim=0).view(-1, 1))
        natoms = data['num_atoms']
        labels = data['label']

        try:
            chain_batch = ddpm.sample_chain(
                data, keep_frames=100, resample_r=resample_r,
                project_enabled=project_enabled,
                d_min_start=d_min_start, d_min_end=d_min_end)
        except utils.FoundNaNException:
            batch_count = int(bs)
            attempts += batch_count
            reasons['nan'] += batch_count
            continue

        x = chain_batch[0][:, :3]
        x = x + fixed_mean[batch_seg]
        one_hot = chain_batch[0][:, 3:]
        unique_indices = torch.unique(batch_seg)
        for i in unique_indices:
            attempts += 1
            n_fragment = int(torch.sum(context[batch_seg == i].squeeze()).item())
            positions = x[batch_seg == i]
            atom_types = one_hot[batch_seg == i].argmax(dim=1)
            metal = metals[i]
            overlapping, liglist = sanitycheck(positions, atom_types, metal)
            total_atoms = sum(len(lig) for lig in liglist) + 1
            if overlapping:
                reasons['overlap'] += 1
                continue
            if total_atoms != natoms[i].item():
                reasons['atom_count'] += 1
                continue
            rdmols = [build_mol(positions[lig], atom_types[lig])
                      for lig in liglist
                      if any(item >= n_fragment for item in lig)]
            valid = ligand_metrics.compute_validity(rdmols)
            if len(valid) != len(rdmols):
                reasons['sanitize'] += 1
                continue
            connected = ligand_metrics.compute_connectivity(valid)
            if len(connected) != len(rdmols):
                reasons['disconnected'] += 1
                continue
            reasons['valid'] += 1
            num += 1
            write_xyz_file(positions, atom_types,
                           f'{outdir}/noH/{b}_{i}_{labels[i]}',
                           metal, n_fragment)

    summary = {'attempts': attempts, 'valid': num, **dict(reasons)}
    os.makedirs(outdir, exist_ok=True)
    with open(f'{outdir}/rejection_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'rejection breakdown: {summary}')
    print(f'Totally {num} valid complexes generated in {outdir}/noH')


def main():
    global args
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metal_elem, metal_pos = read_bare_metal(args.complex)
    print(f"Bare metal: {metal_elem} at {metal_pos}")
    print(f"Target CN: {args.target_cn}")

    data = build_data_objects(
        metal_elem, metal_pos, args.target_cn, args.n_samples,
        args.ligand_sizes, device)
    print(f"{len(data)} total Data objects created")

    batch_size = min(args.batch_size, len(data))
    generate_ligand(data, args.model, device, batch_size, args.outdir,
                    args.resample_r, args.project_enabled,
                    args.d_min_start, args.d_min_end)
    print("Done!")


if __name__ == '__main__':
    main()
