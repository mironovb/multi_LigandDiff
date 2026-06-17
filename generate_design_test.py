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
import json
import os
from itertools import combinations

import numpy as np
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
    parse_ligand_templates,
    templates_max_denticity,
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


def _partition_accounting(remaining_cn, max_denticity):
    """Raw vs. eligible LD_g partition counts for one remaining coordination number.

    Returns ``(n_raw, n_eligible, eligible_partitions)``:

    * ``n_raw`` -- every unrestricted integer partition of ``remaining_cn``. This is
      the *uncapped* denominator the old reporting implicitly divided by: it counts
      chelates of arbitrary denticity, i.e. chemically impossible targets.
    * ``n_eligible`` -- partitions surviving the chelate cap (parts <= max_denticity),
      i.e. exactly what ``reform_data`` enumerates per context copy.
    * ``eligible_partitions`` -- that capped list.
    """
    if remaining_cn <= 0:
        return 0, 0, []
    raw = const.denticity_partitions(remaining_cn, max_denticity=remaining_cn)
    eligible = const.denticity_partitions(remaining_cn, max_denticity=max_denticity)
    return len(raw), len(eligible), eligible


def _accounting_note(subsets, total_raw_parts, total_eligible_parts,
                     max_denticity, denticity_prior, n_seeds):
    """One-line human note explaining why the raw and eligible denominators differ."""
    parts = []
    excluded = total_raw_parts - total_eligible_parts
    if excluded > 0:
        if len(subsets) == 1:
            s = subsets[0]
            parts.append(
                f"{s['n_partitions_raw'] - s['n_partitions_eligible']}/"
                f"{s['n_partitions_raw']} CN={s['remaining_cn']} partitions excluded "
                f"(denticity>{max_denticity})")
        else:
            parts.append(
                f"{excluded}/{total_raw_parts} partitions excluded across "
                f"{len(subsets)} masked subset(s) (denticity>{max_denticity})")
    if denticity_prior == 'csd' and total_eligible_parts > 0:
        parts.append(
            f"denticity_prior=csd: {n_seeds} partition(s) sampled per subset "
            f"(not all {total_eligible_parts} eligible enumerated)")
    return '; '.join(parts) if parts else 'no partitions excluded (raw == eligible)'


def run_mask_level(complex_path, model, outdir, mask_k, n_samples, batch_size,
                   ligand_size, resample_r, project_enabled, d_min_start,
                   d_min_end, add_Hs, max_denticity=const.MAX_DENTICITY,
                   denticity_prior='uniform', seed=None,
                   ligand_templates=None, template_init_coords=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    if seed is not None:
        # Make the legacy global-numpy ligand-size draws reproducible too.
        np.random.seed(seed)
    parsed_templates = parse_ligand_templates(ligand_templates) if ligand_templates else None
    if parsed_templates is not None:
        max_t_dent = templates_max_denticity(parsed_templates)
        if max_t_dent > max_denticity:
            raise ValueError(
                f"--ligand_templates assigns a {max_t_dent}-dentate template but "
                f"--max_denticity={max_denticity}. Raise --max_denticity to >= {max_t_dent}.")
        if denticity_prior == 'csd':
            print("[ligand_templates] --denticity_prior csd is bypassed when --ligand_templates "
                  "is set; generating exactly the template-matching partition(s).")
        print(f"[ligand_templates] seeding skeletons (init_coords={template_init_coords}): "
              f"{parsed_templates['tags']}")
    data_list, n_ligands = read_molecule_maskk(complex_path, mask_k)

    # --- Honest denominator accounting (bookkeeping only; does NOT change what is
    # generated). Each masked subset fixes a remaining coordination number whose
    # integer partitions are the candidate LD_g targets. The reported yield must be
    # valid / attempts_eligible, where:
    #   attempts_raw      = seeds x ALL (uncapped) partitions -- the inflated 6,300;
    #   attempts_eligible = seeds x capped (and, under csd, sampled) partitions.
    # We replicate reform_data's per-context remaining-CN math here so the JSON below
    # records both denominators and the per-subset partition breakdown.
    subsets = []
    total_raw_parts = 0
    total_eligible_parts = 0
    for si, d in enumerate(data_list):
        target_cn = int(torch.sum(d.coord_site).item())
        cn_c = int(torch.sum(d.coord_site[d.context == 1]).item())
        remaining = target_cn - cn_c
        n_raw, n_elig, elig_parts = _partition_accounting(remaining, max_denticity)
        total_raw_parts += n_raw
        total_eligible_parts += n_elig
        subsets.append({
            'subset_index': si,
            'remaining_cn': remaining,
            'n_partitions_raw': n_raw,
            'n_partitions_eligible': n_elig,
            'partitions_eligible': [list(p) for p in elig_parts],
        })

    dataset = data_list * n_samples
    data = reform_data(dataset, device, ligand_size=ligand_size, max_denticity=max_denticity,
                       denticity_prior=denticity_prior, rng=rng,
                       ligand_templates=parsed_templates,
                       template_init_coords=template_init_coords)

    # attempts_eligible is ground truth (len(data)): under 'uniform' it equals
    # n_samples x sum(eligible partitions); under 'csd' reform_data draws ONE
    # partition per context copy, so it equals n_samples x len(subsets).
    attempts_eligible = len(data)
    attempts_raw = n_samples * total_raw_parts
    k = n_ligands if mask_k == 'all' else int(mask_k)
    context_ligs = n_ligands - k
    note = _accounting_note(subsets, total_raw_parts, total_eligible_parts,
                            max_denticity, denticity_prior, n_samples)
    if parsed_templates is not None:
        # Templates further restrict generation to template-matching partitions, so
        # attempts_eligible (= len(data), ground truth) is below n_samples x eligible
        # partitions. Record that so the denominator stays honest.
        tnote = (f"ligand_templates={parsed_templates['tags']}: generation restricted to "
                 f"template-matching partitions (attempts_eligible reflects this)")
        note = tnote if note.startswith('no partitions') else f"{note}; {tnote}"

    print(f'{attempts_eligible} sampling attempts (raw {attempts_raw}) will be '
          f'generated for mask_k={mask_k}')
    os.makedirs(outdir, exist_ok=True)
    valid = 0
    if attempts_eligible > 0:
        bs = min(batch_size, attempts_eligible)
        generate_ligand(data, model, device, bs, outdir=outdir, resample_r=resample_r,
                        project_enabled=project_enabled, d_min_start=d_min_start,
                        d_min_end=d_min_end)
        valid = count_valid(outdir)
    yield_pct = 100.0 * valid / attempts_eligible if attempts_eligible else 0.0
    print(f'design_test mask_k={mask_k}  context={context_ligs}/{n_ligands}  '
          f'attempts={attempts_eligible} (raw {attempts_raw})  valid={valid}  '
          f'yield={yield_pct:.2f}%')
    if attempts_raw != attempts_eligible:
        print(f'  note: {note}')

    accounting = {
        'tag': 'all' if str(mask_k) == 'all' else str(int(mask_k)),
        'mask_k': str(mask_k),
        'complex': complex_path,
        'n_seeds': n_samples,
        'n_subsets': len(subsets),
        'n_ligands': n_ligands,
        'context_ligs': context_ligs,
        'max_denticity': max_denticity,
        'denticity_prior': denticity_prior,
        'ligand_templates': parsed_templates['tags'] if parsed_templates else None,
        'attempts_raw': attempts_raw,
        'attempts_eligible': attempts_eligible,
        'valid': valid,
        'yield_pct': round(yield_pct, 4),
        'note': note,
        'subsets': subsets,
    }
    with open(os.path.join(outdir, 'accounting.json'), 'w') as fh:
        json.dump(accounting, fh, indent=2)

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
    p.add_argument('--max_denticity', type=int, default=const.MAX_DENTICITY,
                   help='Chelate cap: max donors one generated ligand binds through '
                        '(caps the denticity partitions handed to the model)')
    p.add_argument('--denticity_prior', choices=['uniform', 'csd'], default='uniform',
                   help="How to choose denticity partitions per masked subset. "
                        "'uniform' (default) enumerates every capped partition; 'csd' "
                        "samples n_samples partitions proportional to const.DENTICITY_PRIOR "
                        "to concentrate attempts on realistic mono/bidentate-heavy targets.")
    p.add_argument('--seed', type=int, default=None,
                   help='Seed for partition-sampling + global numpy RNG for reproducible '
                        'runs. Default None = nondeterministic (existing behaviour).')
    p.add_argument('--ligand_templates', type=str, default=None,
                   help="Seed whole ligand SKELETONS per generated slot (Prompt 10): inline "
                        "'nitrate;nitrate;nitrate' (built-in tags: nitrate/water/carboxylate) "
                        "or a path to a templates '.json' bundling {'templates':{...}, "
                        "'assign':[...]}. Overrides the per-slot atom-count budget and restricts "
                        "generation to template-matching denticity partitions. A seeding aid, "
                        "not a hard constraint -- see generate.py. Default None (de-novo).")
    p.add_argument('--template_init_coords', type=eval, default=False,
                   help='When --ligand_templates is set, also initialise each templated slot '
                        "from the template geometry (centred at origin) instead of zeros. Still "
                        'fully diffused (a starting hint, not a constraint). Default False.')
    args = p.parse_args()

    tag = 'all' if str(args.mask_k) == 'all' else str(int(args.mask_k))
    outdir = os.path.join(args.outdir, f'mask{tag}')
    run_mask_level(args.complex, args.model, outdir, args.mask_k,
                   args.n_samples, args.batch_size, args.ligand_sizes,
                   args.resample_r, args.project_enabled, args.d_min_start,
                   args.d_min_end, args.add_Hs, args.max_denticity,
                   args.denticity_prior, args.seed,
                   args.ligand_templates, args.template_init_coords)
    print('Done!')


if __name__ == '__main__':
    main()
