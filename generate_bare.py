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
# d_min kept above the bond-perception cutoffs (>= ~1.72 Å, O–O) so projection
# is not a no-op against get_bond_order; see BOND_PERCEPTION_CUTOFFS in src/projection.py.
parser.add_argument('--d_min_start', type=float, default=2.2)
parser.add_argument('--d_min_end', type=float, default=1.9)
parser.add_argument('--valence_guard', type=eval, default=False,
                    help='Valence-aware type masking during sampling (path item 7): '
                         'suppress elements over their const.ALLOWED_BONDS valence for a '
                         "generated atom's heavy-atom neighbour count (e.g. 4-neighbour "
                         'site -> not N), steering toward valence-legal elements.')
parser.add_argument('--max_denticity', type=int, default=const.MAX_DENTICITY,
                    help='Chelate cap: max donors one generated ligand binds through '
                         '(caps the denticity partitions handed to the model)')
parser.add_argument('--denticity_prior', choices=['uniform', 'csd'], default='uniform',
                    help="How to choose LD_g partitions. 'uniform' (default) uses n_samples "
                         "copies of every candidate partition; 'csd' draws n_samples partitions "
                         "proportional to const.DENTICITY_PRIOR to concentrate attempts on "
                         "realistic mono/bidentate-heavy targets.")
parser.add_argument('--seed', type=int, default=None,
                    help='Seed for partition-sampling + global numpy RNG for reproducible runs. '
                         'Default None = nondeterministic (existing behaviour).')

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
                       ligand_size='random', device='cpu',
                       max_denticity=const.MAX_DENTICITY,
                       denticity_prior='uniform', rng=None):
    """Create Data objects for total generation (CN_c=0).

    For each selected LD_g partition and each sample, creates one Data
    object with:
      - context: just the metal
      - generated: placeholder atoms sized per denticity

    With ``denticity_prior='csd'`` the n_samples attempts are drawn across the
    candidate partitions proportional to const.DENTICITY_PRIOR (instead of
    n_samples copies of every partition), concentrating attempts on realistic
    mono/bidentate-heavy targets. ``rng`` is a numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    metal_charge = charges[metal_elem]
    metal_pos_t = torch.tensor([metal_pos], dtype=torch.float32)
    label_base = os.path.splitext(os.path.basename(args.complex))[0]

    # Select partitions: use SELECTED_LD9 for CN=9, else compute all
    if target_cn == 9:
        ld_options = SELECTED_LD9
    else:
        # denticity_partitions already caps each part at max_denticity (chelate cap);
        # here we additionally limit the ligand count to <=5.
        all_parts = const.denticity_partitions(target_cn, max_denticity=max_denticity)
        ld_options = [p for p in all_parts if len(p) <= 5]
        if not ld_options:
            ld_options = all_parts[:10]  # fallback to first 10

    print(f"Target CN={target_cn}, using {len(ld_options)} LD_g partitions:")
    for ld in ld_options:
        print(f"  {ld}")

    # Build the per-partition attempt plan as (partition, n_repeats) pairs.
    if denticity_prior == 'csd' and ld_options:
        # Draw n_samples partitions proportional to the CSD prior (instead of
        # n_samples copies of every partition), concentrating attempts.
        weights = np.array([const.denticity_prior_weight(p) for p in ld_options], dtype=float)
        total = weights.sum()
        probs = (weights / total) if total > 0 else None
        draws = rng.choice(len(ld_options), size=n_samples, p=probs)
        counts = Counter(int(d) for d in draws)
        sampling_plan = [(ld_options[idx], cnt) for idx, cnt in counts.items()]
        summary = {tuple(ld_options[idx]): cnt for idx, cnt in counts.items()}
        print(f"denticity_prior=csd: drew {n_samples} partitions ~ CSD prior -> {summary}")
    else:
        sampling_plan = [(ld_g, n_samples) for ld_g in ld_options]

    data_list = []
    for ld_g, n_rep in sampling_plan:
        for _ in range(n_rep):
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

    # Return the eligible LD_g partition pool alongside the data so the caller can
    # report an honest denominator (attempts_eligible vs. the uncapped attempts_raw).
    return data_list, ld_options


# Reuse generate_ligand from generate_mask1.py (identical logic)
def generate_ligand(data, model, device, batch_size=32, outdir='generated',
                    resample_r=1, project_enabled=False,
                    d_min_start=2.2, d_min_end=1.9, valence_guard=False):
    os.makedirs(f'{outdir}/noH', exist_ok=True)
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ligand_metrics = BasicLigandMetrics()
    num = 0
    reasons = Counter()
    attempts = 0

    # Projection guard (code-review Finding 5): prove the exclusion shell has
    # eligible pairs instead of silently no-op'ing. The bare-metal context is
    # metal-only -- the metal is an all-zero ligand_group row (group -1) and is
    # exempt -- so the ONLY obstacles are generated atoms in a DIFFERENT ligand
    # group (Prompt 12's generated<->generated shell). Eligible pairs therefore
    # exist iff generated atoms span >=2 ligand groups; assert that and log so a
    # run proves the shell ran.
    if project_enabled and len(data):
        n_metal_only = n_eligible = 0
        rep_gen = 0
        for d in data:
            ctx = d['context'].view(-1) == 1
            lg = d['ligand_group']
            gen = ~ctx
            if rep_gen == 0:
                rep_gen = int(gen.sum().item())
            if float(lg[ctx].sum().item()) != 0.0:
                continue                         # fixed (non-metal) ligand obstacles present
            n_metal_only += 1
            if int((lg[gen].sum(dim=0) > 0).sum().item()) >= 2:
                n_eligible += 1
        if n_metal_only > 0:
            assert n_eligible > 0, (
                "project_enabled=True with metal-only context but no complex has "
                "generated atoms spanning >=2 ligand groups: the exclusion shell "
                "has no eligible generated<->generated pairs and would silently "
                "no-op. See src/projection.py / Prompt 12.")
            print(f"projection: active, {rep_gen} gen atoms "
                  f"({n_eligible}/{n_metal_only} metal-only complexes have "
                  f"eligible gen<->gen pairs)")
        else:
            print(f"projection: active, {rep_gen} gen atoms")

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
                d_min_start=d_min_start, d_min_end=d_min_end,
                valence_guard=valence_guard)
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
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        # Make the legacy global-numpy ligand-size draws reproducible too.
        np.random.seed(args.seed)

    metal_elem, metal_pos = read_bare_metal(args.complex)
    print(f"Bare metal: {metal_elem} at {metal_pos}")
    print(f"Target CN: {args.target_cn}")

    data, ld_options = build_data_objects(
        metal_elem, metal_pos, args.target_cn, args.n_samples,
        args.ligand_sizes, device, args.max_denticity,
        args.denticity_prior, rng)
    print(f"{len(data)} total Data objects created")

    batch_size = min(args.batch_size, len(data))
    generate_ligand(data, args.model, device, batch_size, args.outdir,
                    args.resample_r, args.project_enabled,
                    args.d_min_start, args.d_min_end, args.valence_guard)

    # --- Honest denominator accounting (bookkeeping only; does NOT change what is
    # generated). attempts_eligible is ground truth (len(data)): under 'uniform' it
    # is n_samples x len(eligible partitions); under 'csd' reform sampling makes it
    # n_samples (one partition per seed). attempts_raw counts seeds x ALL (uncapped)
    # integer partitions of the target CN -- the inflated denominator that includes
    # chemically impossible chelates / over-long partitions never actually attempted.
    raw_parts = const.denticity_partitions(args.target_cn, max_denticity=args.target_cn)
    n_raw_parts = len(raw_parts)
    n_eligible_parts = len(ld_options)
    attempts_raw = args.n_samples * n_raw_parts
    attempts_eligible = len(data)
    noh_dir = os.path.join(args.outdir, 'noH')
    valid = (len([f for f in os.listdir(noh_dir) if f.endswith('.xyz')])
             if os.path.isdir(noh_dir) else 0)
    yield_pct = 100.0 * valid / attempts_eligible if attempts_eligible else 0.0

    note_parts = []
    excluded = n_raw_parts - n_eligible_parts
    if excluded > 0:
        if args.target_cn == 9:
            reason = (f"curated to {n_eligible_parts} SELECTED_LD9 partitions "
                      f"(denticity<=4, <=5 ligands)")
        else:
            reason = f"denticity>{args.max_denticity} or >5 ligands"
        note_parts.append(
            f"{excluded}/{n_raw_parts} CN={args.target_cn} partitions excluded ({reason})")
    if args.denticity_prior == 'csd' and n_eligible_parts > 0:
        note_parts.append(
            f"denticity_prior=csd: {args.n_samples} partition(s) sampled "
            f"(not all {n_eligible_parts} eligible enumerated)")
    note = '; '.join(note_parts) if note_parts else 'no partitions excluded (raw == eligible)'

    print(f'bare target_cn={args.target_cn}  '
          f'attempts={attempts_eligible} (raw {attempts_raw})  valid={valid}  '
          f'yield={yield_pct:.2f}%')
    if attempts_raw != attempts_eligible:
        print(f'  note: {note}')

    os.makedirs(args.outdir, exist_ok=True)
    accounting = {
        'mode': 'bare',
        'complex': args.complex,
        'target_cn': args.target_cn,
        'n_seeds': args.n_samples,
        'max_denticity': args.max_denticity,
        'denticity_prior': args.denticity_prior,
        'attempts_raw': attempts_raw,
        'attempts_eligible': attempts_eligible,
        'valid': valid,
        'yield_pct': round(yield_pct, 4),
        'note': note,
        'n_partitions_raw': n_raw_parts,
        'n_partitions_eligible': n_eligible_parts,
        'partitions_eligible': [list(p) for p in ld_options],
    }
    with open(os.path.join(args.outdir, 'accounting.json'), 'w') as fh:
        json.dump(accounting, fh, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
