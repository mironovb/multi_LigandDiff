import argparse
import os
import json
import numpy as np
import tempfile
import ast
from collections import Counter
from itertools import combinations
import torch
from src import const
from src import utils
from src.lightning import DDPM
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter_add
from src.molecule_builder import BasicLigandMetrics, build_mol,sanitycheck,write_xyz_file,ligand_slice,reset_dative_bonds
from openbabel import openbabel
from rdkit import Chem
import random

from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--complex', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--ligand_sizes', type=str, default='random')
parser.add_argument('--add_Hs', type=eval, default=False)
parser.add_argument('--resample_r', type=int, default=1,
                    help='RePaint resampling iterations per timestep (1=standard, 10=recommended)')
parser.add_argument('--project_enabled', type=eval, default=False,
                    help='Enable hard exclusion-shell projection during sampling')
parser.add_argument('--d_min_start', type=float, default=1.5,
                    help='Exclusion shell d_min at high noise (start of reverse)')
parser.add_argument('--d_min_end', type=float, default=1.3,
                    help='Exclusion shell d_min at low noise (end of reverse)')
parser.add_argument('--max_denticity', type=int, default=const.MAX_DENTICITY,
                    help='Chelate cap: max donors one generated ligand binds through '
                         '(caps the denticity partitions handed to the model)')
parser.add_argument('--denticity_prior', choices=['uniform', 'csd'], default='uniform',
                    help="How to choose denticity partitions per masked context. "
                         "'uniform' (default) enumerates every capped partition; 'csd' "
                         "samples n_samples partitions per context proportional to the "
                         "CSD-observed denticity prior (const.DENTICITY_PRIOR), concentrating "
                         "attempts on realistic mono/bidentate-heavy targets.")
parser.add_argument('--seed', type=int, default=None,
                    help='Seed for the partition-sampling RNG (and the global numpy RNG used '
                         'for ligand sizes) for reproducible runs. Default None = '
                         'nondeterministic (existing behaviour).')
parser.add_argument('--donor_spec', type=str, default=None,
                    help="Optional: fix the ELEMENT of each donor atom (the coord_site==1 atoms "
                         "that touch the metal) instead of letting the diffusion invent them. "
                         "Per-ligand form 'O,O;N,O,O,O' => ligand 1 binds via 2 O, ligand 2 via "
                         "N+3 O (a nitrate); ';' separates generated ligands, ',' their donors. "
                         "Whole-complex form 'N,N,O,O,O' lists every donor in slot order without "
                         "pinning the per-ligand split. Elements must be keys of const.ATOM2IDX. "
                         "When set, only denticity partitions the spec can fill are generated "
                         "(the csd prior is bypassed); geometry and all non-donor atoms are still "
                         "diffused. Omit for the default all-zeros de-novo behaviour.")


atom2idx=const.ATOM2IDX
idx2atom=const.IDX2ATOM 
charges=const.CHARGES
num_atom_types=const.NUMBER_OF_ATOM_TYPES
metal_list=const.metals


def sort_pos(xyz_file):
    metal_index = None
    with open(xyz_file, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(tuple(metal_list)):
            metal_index = i
            break
    if metal_index is not None :
        lines.insert(2, lines.pop(metal_index))
        with open(f'{xyz_file[:-4]}_re.xyz', 'w') as new_file:
            new_file.writelines(lines)
        return True
    else:
        return False
    

def parse_complex(filename):
    label=filename[:-4]
    data_list=[]
    ele=[]
    pos=[]
    nuclear_charges=[]
    H_list=[]# store H atoms, maybe add them back later
    noH_list=[]
    with open(filename, 'r') as f:
        lines=f.readlines()
    
    for i in lines[3:]:
        if i.split()[0] =='H':
            H_list.append(i)
        else:
            noH_list.append(i)
            ele.append(atom2idx[i.split()[0]])
            nuclear_charges.append(charges[i.split()[0]])
            pos.append([float(j) for j in i.split()[1:]])
    noH_list.insert(0,lines[2])
    pos.insert(0,[float(j) for j in lines[2].split()[1:]]) # add  the position of metal
    nuclear_charges.insert(0,charges[lines[2].split()[0]]) # add  the nuclear charge of metal
    one_hot=torch.zeros(len(ele),8)
    one_hot[range(len(ele)),ele]=1
    one_hot=torch.cat([torch.zeros(8).view(1,-1),one_hot],dim=0)
    num_atoms=len(pos)
    pos=torch.tensor(pos)
    nuclear_charges=torch.tensor(nuclear_charges)

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        with open(f'{tmp_file}.xyz', 'w') as file:
            file.write(f"{num_atoms}\n\n")
            file.writelines(noH_list)

    mol=mol3D()
    mol.readfromxyz(f'{tmp_file}.xyz')
    liglist,ligdents,ligcon=ligand_breakdown(mol,silent=True,BondedOct=False,transition_metals_only=False)
    f_group=torch.zeros(num_atoms)
    for i in range(len(liglist)):
        f_group[liglist[i]]=i+1

    ligand_group=torch.zeros((num_atoms, const.MAX_LIGANDS + 1))
    ligand_group[range(len(f_group.long())),f_group.long()]=1
    
    anchor_group=torch.zeros(num_atoms)
    for i in range(len(ligcon)):
        anchor_group[ligcon[i]]=i+1
    anchors_group=torch.zeros((num_atoms, const.MAX_LIGANDS + 1))
    anchors_group[range(len(anchor_group.long())),anchor_group.long()]=1
    coord_site=anchors_group[:,1:].any(dim=1).to(torch.int)
    # list all combinations of ligands
    all_lig=[]
    for i in range(len(liglist)): 
        all_lig.extend(list(list(combinations(liglist, i+1)))) 
    all_anchor=[]
    for i in range(len(ligcon)): 
        all_anchor.extend(list(list(combinations(ligcon, i+1))))
    for k in range(len(all_lig)):
        anchors=torch.zeros(num_atoms)
        ligand=torch.zeros(num_atoms)
        for i in all_anchor[k]:
            anchors[i]=1
        for i in all_lig[k]:
            ligand[i]=1
        
        context = 1-ligand

        data = Data(pos=pos,label=label,  context=context,  nuclear_charges=nuclear_charges,ligand_diff=ligand, num_atoms=num_atoms, one_hot=one_hot,ligand_group=ligand_group[:,1:],coord_site=coord_site)
        data_list.append(data)
    print('The coordination type of the given complex is:',ligdents)
    print('The number of combinations by masking the ligands from partially to totally is:',len(data_list))
    return data_list






def read_molecule(filename):
    if not filename.endswith('.xyz'):
        raise Exception('Unknown file extension, only .xyz file is supported')
    
    with open(filename, 'r') as file:
        metal = file.readlines()[2]
        if metal.split()[0] not in metal_list:
            if sort_pos(filename):
                print(f'Metal is not located at the begining of the coordinates.The {filename} is rearranged and saved to {filename[:-4]}_re.xyz')
                return parse_complex(f'{filename[:-4]}_re.xyz')
            else:
                print('Metal is not found in the supported metals list, please add the metal to the list of metals in const.py')
        else:
            return parse_complex(filename)


def get_ligand_size(ligand_size='random',startnum=1,endnum=10):
    if ligand_size == 'random':
        ligand_size=np.random.randint(startnum,endnum)
    else:
        ligand_size=int(ligand_size)
    return ligand_size


def parse_donor_spec(spec_str):
    """Parse the --donor_spec string into a structured, validated form.

    Two accepted forms:
      per-ligand    'O,O;N,O,O,O'  -> ';' separates generated ligands, ',' the donors of one
                                      ligand. Here: ligand 1 binds via 2 O, ligand 2 via N + 3 O
                                      (a nitrate). Pins the per-ligand denticity structure.
      whole-complex 'N,N,O,O,O'    -> a flat donor list assigned to the metal-touching slots in
                                      build order, without pinning how they split across ligands.

    Returns {'mode': 'per_ligand'|'flat', 'groups': [[el,...],...], 'flat': [el,...]}.
    Raises ValueError on an empty spec/slot or an element outside const.ATOM2IDX.
    """
    spec_str = spec_str.strip()
    if not spec_str:
        raise ValueError("--donor_spec is empty")
    per_ligand = ';' in spec_str
    raw_groups = spec_str.split(';') if per_ligand else [spec_str]
    groups = []
    for raw in raw_groups:
        tokens = [t.strip() for t in raw.split(',')]
        if any(t == '' for t in tokens):
            raise ValueError(
                f"--donor_spec '{spec_str}' has an empty donor slot (stray/duplicate ',' or ';'?)")
        for t in tokens:
            if t not in const.ATOM2IDX:
                raise ValueError(
                    f"--donor_spec element '{t}' is not a supported donor atom; "
                    f"choose from {sorted(const.ATOM2IDX)}")
        groups.append(tokens)
    return {
        'mode': 'per_ligand' if per_ligand else 'flat',
        'groups': groups,
        'flat': [el for g in groups for el in g],
    }


def donor_elements_for_partition(donor_spec, LD_g):
    """Donor element for every donor slot of partition ``LD_g``, in build order, or None.

    Build order follows reform_data's layout: ligand 0's donors, then ligand 1's donors, ...
    (each ligand contributes its first ``LD_g[k]`` atoms as donors). Returns None when the spec
    cannot fill this partition, so the caller skips that partition.

    - flat mode: matches any partition whose donor count equals len(flat); donors fill in order.
    - per_ligand mode: matches iff the multiset of ligand denticities equals the multiset of
      group sizes, then each ligand slot is filled from a spec group of the same denticity.
    """
    if donor_spec['mode'] == 'flat':
        flat = donor_spec['flat']
        return list(flat) if len(flat) == sum(LD_g) else None
    remaining_groups = [list(g) for g in donor_spec['groups']]
    if sorted(len(g) for g in remaining_groups) != sorted(LD_g):
        return None
    elements = []
    for d in LD_g:
        match_idx = next((gi for gi, g in enumerate(remaining_groups) if len(g) == d), None)
        if match_idx is None:
            return None
        elements.extend(remaining_groups.pop(match_idx))
    return elements


def reform_data(dataset,device,ligand_size='random',max_denticity=const.MAX_DENTICITY,
                denticity_prior='uniform',rng=None,donor_spec=None):
    if rng is None:
        rng = np.random.default_rng()
    new_data=[]
    spec_matched = False
    remaining_seen = set()
    for i in dataset:
        #context
        x=i['pos'][i['context']==1]
        one_hot=i['one_hot'][i['context']==1]
        ligand_group=i['ligand_group'][i['context']==1]
        nuclear_charges=i['nuclear_charges'][i['context']==1]
        c_coord_site=i['coord_site'][i['context']==1]

        #all possible ligand index to generate under given context
        index=torch.all(ligand_group == 0, dim=0).nonzero(as_tuple=True)[0]
        #coordination number of context
        cn_c=torch.sum(c_coord_site).item()
        ##Ligand denticity of context
        ligand_slices=ligand_slice(ligand_group[1:])# remove metal in the context
        LD_c=[]
        for item in ligand_slices:
            item=[i+1 for i in item]
            LD_c.append(torch.sum(i['coord_site'][i['context'].squeeze()==1][item]).item())
        assert sum(LD_c)==cn_c 
        #coordination type of generated ligands,i.e,ligand denticity(LD_g)
        # Determine target CN from total coordination sites
        target_cn = int(torch.sum(i['coord_site']).item())
        remaining = target_cn - cn_c
        if remaining <= 0:
            continue
        remaining_seen.add(remaining)
        ld_options = const.denticity_partitions(remaining, max_denticity=max_denticity)
        # A donor spec is authoritative: keep the full (capped) enumeration and filter it to
        # spec-matching partitions in the loop below, so the CSD prior sampling is bypassed (it
        # would otherwise throw away most of the partitions the spec is allowed to fill).
        if denticity_prior == 'csd' and ld_options and donor_spec is None:
            # Sample ONE partition for this context copy proportional to the CSD
            # prior. The dataset already holds n_samples copies of each context, so
            # this draws n_samples partitions ~ prior per context (vs. enumerating
            # all of them uniformly), concentrating attempts on realistic targets.
            weights = np.array([const.denticity_prior_weight(p) for p in ld_options], dtype=float)
            total = weights.sum()
            probs = (weights / total) if total > 0 else None
            ld_options = [ld_options[rng.choice(len(ld_options), p=probs)]]
        for LD_g in ld_options:
            donor_elements = None
            if donor_spec is not None:
                donor_elements = donor_elements_for_partition(donor_spec, LD_g)
                if donor_elements is None:
                    # This partition's denticity structure can't host the requested donor set.
                    continue
                spec_matched = True
            ligand_index=index[:len(LD_g)]
            gen_ligand_groups=[]
            gen_ligand_coord_sites=[]
            for k,num_coord_site in zip(ligand_index,LD_g):
                # Chemistry-derived atom budget: a denticity-indexed floor (enough atoms
                # for the donors + bridging skeleton) plus a modest seeded-random spread,
                # so the model still sees size variety but is never handed too few atoms
                # to build the donor motif (a bidentate slot can now fit a nitrate, N+3O=4).
                floor=const.DENTICITY_MIN_ATOMS.get(num_coord_site,num_coord_site)
                if num_coord_site<3:
                    g_ligand_size=floor+int(rng.integers(0,6))
                else:
                    # Tridentate+ is already chemistry-scaled to a larger range; keep it.
                    g_ligand_size=get_ligand_size(ligand_size,startnum=10,endnum=30)
                g_ligand_size=max(g_ligand_size,floor,num_coord_site)
                assert g_ligand_size>= num_coord_site,"The assigned ligand size is smaller than the denticity of the generated ligand. Please assign a larger ligand size."
                gen_ligand_group=torch.zeros(g_ligand_size, const.MAX_LIGANDS)
                gen_ligand_group[:,k]=1
                gen_ligand_groups.append(gen_ligand_group)
                gen_coord_site=torch.zeros(g_ligand_size)
                gen_coord_site[:num_coord_site]=1
                gen_ligand_coord_sites.append(gen_coord_site)
            gen_ligand_group=torch.cat(gen_ligand_groups,dim=0)
            gen_ligand_size=gen_ligand_group.shape[0]
            gen_ligand_x=torch.zeros(gen_ligand_size,3)
            gen_ligand_coord_site=torch.cat(gen_ligand_coord_sites,dim=0)
            gen_ligand_onehot=torch.zeros(gen_ligand_size,num_atom_types)
            if donor_elements is not None:
                # Seed the ELEMENT of each donor atom (the coord_site==1 rows that touch the
                # metal); the diffusion still fills every other row and all coordinates.
                donor_slots = (gen_ligand_coord_site == 1).nonzero(as_tuple=True)[0].tolist()
                assert len(donor_slots) == len(donor_elements), (
                    f"donor_spec slot mismatch: {len(donor_slots)} donor sites vs "
                    f"{len(donor_elements)} specified for partition {LD_g}")
                for slot, element in zip(donor_slots, donor_elements):
                    gen_ligand_onehot[slot, const.ATOM2IDX[element]] = 1
            new_x=torch.cat([x,gen_ligand_x],dim=0)
            new_context=torch.cat([torch.ones(x.shape[0]),torch.zeros(gen_ligand_size)],dim=0)
            new_ligand_diff=torch.cat([torch.zeros(x.shape[0]),torch.ones(gen_ligand_size)],dim=0)
            new_nuclear_charges=torch.cat([nuclear_charges,torch.zeros(gen_ligand_size)],dim=0)
            new_coord_site=torch.cat([c_coord_site,gen_ligand_coord_site],dim=0)
            assert new_x.shape[0]==new_nuclear_charges.shape[0]
            assert torch.sum(new_coord_site).item() == target_cn, f"CN mismatch: expected {target_cn}, got {torch.sum(new_coord_site).item()}"
            new_ligand_group=torch.cat([ligand_group,gen_ligand_group],dim=0)
            new_onehot=torch.cat([one_hot,gen_ligand_onehot],dim=0)
            natoms=new_x.shape[0]
            data = Data(pos=new_x.to(device),label=f"{i['label']}_{LD_c}_{LD_g}",coord_site=new_coord_site.to(device),nuclear_charges=new_nuclear_charges.to(device), context=new_context.to(device), ligand_diff=new_ligand_diff.to(device), ligand_group=new_ligand_group.to(device), one_hot=new_onehot.to(device), num_atoms=natoms)
            new_data.append(data)
    #new_data=[item for item in new_data for _ in range(2)]
    if donor_spec is not None and not spec_matched:
        wanted = ([len(g) for g in donor_spec['groups']]
                  if donor_spec['mode'] == 'per_ligand'
                  else f"{len(donor_spec['flat'])} donor(s) (any partition)")
        raise ValueError(
            f"--donor_spec could not be matched to any maskable context in this complex. "
            f"Spec needs {wanted}; contexts (re)generate one of these donor counts: "
            f"{sorted(remaining_seen)} (capped at --max_denticity={max_denticity}). "
            f"Adjust the donor count / per-ligand denticities, --max_denticity, or the complex.")
    return new_data

def generate_ligand(data,model,device,batch_size=64,outdir='generated_complexes',resample_r=1,
                    project_enabled=False,d_min_start=1.5,d_min_end=1.3):
    os.makedirs(f'{outdir}/noH', exist_ok=True)
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ligand_metrics=BasicLigandMetrics()
    num=0
    reasons = Counter()
    attempts = 0
    for b, data in enumerate(dataloader):
        pos_orginal=data['pos']
        batch_seg=data.batch
        batch_size=torch.max(batch_seg)+1
        context = data['context'].view(-1,1)
        metals=[data['nuclear_charges'][batch_seg==i][0] for i in range(batch_size)]
        fixed_mean = scatter_add(pos_orginal*context, batch_seg, dim=0)/scatter_add(context, batch_seg, dim=0).view(-1,1)
        natoms=data['num_atoms']
        labels=data['label']
        
        try:
            chain_batch = ddpm.sample_chain(data, keep_frames=100, resample_r=resample_r,
                                              project_enabled=project_enabled,
                                              d_min_start=d_min_start, d_min_end=d_min_end)
        except utils.FoundNaNException as e:
            batch_count = int(batch_size)
            attempts += batch_count
            reasons['nan'] += batch_count
            continue

        x = chain_batch[0][ :, :3]
        x=x+fixed_mean[batch_seg]
        one_hot = chain_batch[0][ :, 3:]
        unique_indices = torch.unique(batch_seg)
        for i in unique_indices:
            attempts += 1
            n_fragment=int(torch.sum(context[batch_seg==i].squeeze()).item())
            positions=x[batch_seg==i]
            atom_types=one_hot[batch_seg==i].argmax(dim=1)
            metal=metals[i]
            overlapping,liglist=sanitycheck(positions, atom_types,metal)
            total_atoms=sum(len(lig) for lig in liglist)+1
            if overlapping:
                reasons['overlap'] += 1
                continue
            if total_atoms != natoms[i].item():
                reasons['atom_count'] += 1
                continue
            rdmols=[build_mol(positions[lig],atom_types[lig]) for lig in liglist if any(item >= n_fragment for item in lig)]
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
            write_xyz_file(positions, atom_types,f'{outdir}/noH/{b}_{i}_{labels[i]}',metal,n_fragment)

    summary = {'attempts': attempts, 'valid': num, **dict(reasons)}
    os.makedirs(outdir, exist_ok=True)
    with open(f'{outdir}/rejection_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'rejection breakdown: {summary}')
    print(f'Totally {num} valid complexes are generated and saved in {outdir}/noH')



def add_H(org_xyz,gen_dir):
    """
    Add H from the original complex to the generated complex. 
    For ligands in context, H atoms are copied from the original complex.
    For generated ligands, H atoms are generated by RDKit.
    Args:
        org_xyz: orginial complex xyz file
        gen_dir: directory of generated complexes
    """
    #If using RDKit to automatically add H atoms for generated ligands, manual check after protonation is highly recommended.
    #one of possible issues:https://github.com/rdkit/rdkit/issues/4667
    #Alternative: ChimeraX --addh
    
    os.makedirs(f'{gen_dir}/add_H', exist_ok=True)
    my_mol=mol3D()
    my_mol.readfromxyz(f'{org_xyz}')
    liglist,ligdents,ligcon=ligand_breakdown(my_mol,silent=True,BondedOct=False,transition_metals_only=False)

    with open(f'{org_xyz}','r+') as f:
        lines=f.readlines()
        atom_hs=[]
        atom_nohs=[]
        for i in liglist:
            ligand_h=[lines[k+2] for k in i]
            atom_h=[]
            atom_noh=[]
            for atom in ligand_h:
                if atom.split()[0]=='H':
                    atom_h.append(atom)
                else:
                    atom_noh.append(atom)
            round_noh = []
            for item in atom_noh:
                elements = item.split('\t')
                new_elements = []
                for elem in elements:
                    try:
                        num = float(elem)
                        new_elements.append(f"{num:.3f}")
                    except ValueError:
                        new_elements.append(elem)
                round_noh.append('\t'.join(new_elements).strip())
            atom_hs.append(atom_h)
            atom_nohs.append(round_noh)  
            
    
    for gen_xyz in os.listdir(f'{gen_dir}/noH'):
        # add H atoms to heavy atoms in context
        my_mol=mol3D()
        my_mol.readfromxyz(f'{gen_dir}/noH/{gen_xyz}')
        liglist,ligdents,ligcon=ligand_breakdown(my_mol,silent=True,BondedOct=False,transition_metals_only=False)
        h_atoms=[]
        with open(f'{gen_dir}/noH/{gen_xyz}','r+') as f:
            lines=f.readlines()
        for i in liglist:
            ligand=[lines[k+2] for k in i]
            round_noh_ligand = []
            for item in ligand:
                elements = item.split('\t')  
                new_elements = []
                for elem in elements:
                    try:
                        num = float(elem)
                        new_elements.append(f"{num:.3f}")
                    except ValueError:
                        new_elements.append(elem)
                round_noh_ligand.append('\t'.join(new_elements).strip())
            if round_noh_ligand in atom_nohs:
                h_atoms.extend(atom_hs[atom_nohs.index(round_noh_ligand)])
                    
        #generate H atoms for the generated ligands
        context=int(lines[1])
        gen_ligands=lines[context+2:]
        gen_ligands.insert(0,lines[2])
        with tempfile.NamedTemporaryFile() as tmp:
            tmp_file = tmp.name
        with open(f'{tmp_file}.xyz','w') as f:
            f.write(f'{len(gen_ligands)}\n')
            f.write('ligand\n')
            f.write(''.join(gen_ligands))

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")     
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, f'{tmp_file}.xyz')
        obConversion.WriteFile(ob_mol, f'{tmp_file}.sdf')
        tmp_mol = Chem.SDMolSupplier(f'{tmp_file}.sdf', sanitize=False)[0]
        mol = Chem.RWMol()
        for atom in tmp_mol.GetAtoms():
            mol.AddAtom(Chem.Atom(atom.GetSymbol()))     
        mol.AddConformer(tmp_mol.GetConformer(0))
        for bond in tmp_mol.GetBonds():
            mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                bond.GetBondType())     
        m2=reset_dative_bonds(mol)
        try:
            Chem.SanitizeMol(m2,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            mh=Chem.AddHs(m2, addCoords=(len(m2.GetConformers()) > 0))
            Chem.SanitizeMol(mh,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            coord=Chem.MolToXYZBlock(mh)
            mol_H=[i for i in coord.split('\n') if i.startswith('H')]
            mol_H=['\t'.join(item.split())+'\n' for item in mol_H]
            lines.extend(h_atoms)
            lines.extend(mol_H)
            lines[0]=f'{len(lines)-2}\n'
            with open(f'{gen_dir}/add_H/{gen_xyz}','w+') as g:
                g.writelines(lines)
        except ValueError:
            continue





def main(outdir,model,complex,batch_size=64,n_samples=1,ligand_size='random',add_Hs=False,resample_r=1,
         project_enabled=False,d_min_start=1.5,d_min_end=1.3,max_denticity=const.MAX_DENTICITY,
         denticity_prior='uniform',seed=None,donor_spec=None):
    """
    Generate multiple new structures for each variation in a given complex
    Args:
        outdir: path to save generated complexes
        model:path to the pretrained model
        complex: path to the reference complex
        ligand_size: number of ligand atoms to generate, default is random
        add_Hs: add H atoms to the generated complexes
        denticity_prior: 'uniform' (all capped partitions) or 'csd' (sample
            n_samples partitions ~ const.DENTICITY_PRIOR per context)
        seed: optional int seed for reproducible sampling (None = nondeterministic)
        donor_spec: optional donor-atom identity spec (e.g. 'O,O;N,O,O,O' or 'N,N,O,O,O');
            seeds the element of each metal-touching donor atom. None = today's all-zeros
            de-novo behaviour. See parse_donor_spec / --donor_spec for the grammar.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    if seed is not None:
        # Make the legacy global-numpy ligand-size draws reproducible too.
        np.random.seed(seed)
    parsed_donor_spec = parse_donor_spec(donor_spec) if donor_spec else None
    if parsed_donor_spec is not None:
        if parsed_donor_spec['mode'] == 'per_ligand':
            max_grp = max(len(g) for g in parsed_donor_spec['groups'])
            if max_grp > max_denticity:
                raise ValueError(
                    f"--donor_spec requests a {max_grp}-dentate ligand but "
                    f"--max_denticity={max_denticity}. Raise --max_denticity to >= {max_grp}.")
        if denticity_prior == 'csd':
            print("[donor_spec] --denticity_prior csd is bypassed when --donor_spec is set; "
                  "generating exactly the spec-matching partition(s).")
        print(f"[donor_spec] conditioning donor identities ({parsed_donor_spec['mode']}): "
              f"{parsed_donor_spec['groups']}")
    dataset=read_molecule(complex)*n_samples
    print(f'{len(dataset)} samples will be generated')
    data=reform_data(dataset,device,ligand_size=ligand_size,max_denticity=max_denticity,
                     denticity_prior=denticity_prior,rng=rng,donor_spec=parsed_donor_spec)
    batch_size=min(batch_size,len(data))
    generate_ligand(data,model,device,batch_size,outdir=outdir,resample_r=resample_r,
                    project_enabled=project_enabled,d_min_start=d_min_start,d_min_end=d_min_end)
    if add_Hs:
        add_H(complex,outdir)
    print('Done!')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.outdir,args.model,args.complex,args.batch_size,args.n_samples,args.ligand_sizes,args.add_Hs,args.resample_r,
         args.project_enabled,args.d_min_start,args.d_min_end,args.max_denticity,
         args.denticity_prior,args.seed,args.donor_spec)

    

    