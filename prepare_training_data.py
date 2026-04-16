#!/usr/bin/env python
"""Convert CIF structures to training data for Ln-adapted multi-LigandDiff.

Input:  CIF files in cif_output/{Element}/{REFCODE}.cif
        CSV listing which structures to use
Output: data/train_ln.pt and data/val_ln.pt
"""

import argparse
import csv
import glob
import logging
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from torch_geometric.data import Data

from src.const import ATOM2IDX, CHARGES, MAX_LIGANDS, metals2idx

# Force unbuffered output for SLURM logs (Python 3.7+).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

# Silence noisy library warnings.
warnings.filterwarnings("ignore")
logging.getLogger("pymatgen").setLevel(logging.ERROR)
logging.getLogger("ase").setLevel(logging.ERROR)

# Lanthanide elements
LN_ELEMENTS = {
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
}

# Donor distance cutoff in Angstrom (Ln–donor bonds)
DONOR_CUTOFF = 3.0

# Organic bond cutoff in Angstrom
ORGANIC_BOND_CUTOFF = 1.9


# ---------------------------------------------------------------------------
# CIF parsing
# ---------------------------------------------------------------------------

def parse_cif(cif_path):
    """Parse a CIF file and return a pymatgen Structure. Falls back to ASE."""
    try:
        from pymatgen.core import Structure
        structure = Structure.from_file(cif_path)
        return structure
    except Exception:
        pass
    try:
        from ase.io import read as ase_read
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = ase_read(cif_path)
        structure = AseAtomsAdaptor.get_structure(atoms)
        return structure
    except Exception as e:
        warnings.warn(f"Failed to parse {cif_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Bond graph & BFS molecular-unit extraction
# ---------------------------------------------------------------------------

def build_bond_graph(structure):
    """Build a bond graph using CrystalNN with distance-based fallback.

    Returns dict[int, list[int]] mapping site index to neighbor indices.
    """
    n = len(structure)
    graph = defaultdict(list)

    try:
        from pymatgen.analysis.local_env import CrystalNN
        cnn = CrystalNN(weighted_cn=False, cation_anion=False)
        for i in range(n):
            try:
                info = cnn.get_nn_info(structure, i)
                for nn_entry in info:
                    j = nn_entry['site_index']
                    if j not in graph[i]:
                        graph[i].append(j)
                    if i not in graph[j]:
                        graph[j].append(i)
            except Exception:
                continue
        if len(graph) > 0:
            return graph
    except ImportError:
        pass

    # Fallback: distance-based
    for i in range(n):
        ei = str(structure[i].specie)
        for j in range(i + 1, n):
            ej = str(structure[j].specie)
            dist = structure.get_distance(i, j)
            if ei in LN_ELEMENTS or ej in LN_ELEMENTS:
                thresh = DONOR_CUTOFF
            else:
                thresh = ORGANIC_BOND_CUTOFF
            if dist < thresh:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def extract_molecular_unit(structure, ln_idx, bond_graph):
    """BFS from the Ln center through the bond graph.

    Returns list of site indices with ln_idx first.
    """
    visited = {ln_idx}
    queue = [ln_idx]
    order = [ln_idx]
    while queue:
        current = queue.pop(0)
        for nb in bond_graph.get(current, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                order.append(nb)
    return order


def find_ln_site(structure):
    """Return the index of the first lanthanide site, or None."""
    for i, site in enumerate(structure):
        if str(site.specie) in LN_ELEMENTS:
            return i
    return None


# ---------------------------------------------------------------------------
# Ligand decomposition
# ---------------------------------------------------------------------------

def decompose_ligands(positions, ln_local_idx=0):
    """Decompose a molecular unit into ligands.

    Removes the Ln centre (assumed at *ln_local_idx*) and finds connected
    components among the remaining atoms using an organic-bond distance
    cutoff.

    Returns
    -------
    ligands : list[list[int]]
        Each inner list holds local atom indices belonging to one ligand.
    donors  : list[int]
        Local indices of atoms within DONOR_CUTOFF of the Ln centre.
    """
    n = len(positions)
    non_ln = [i for i in range(n) if i != ln_local_idx]
    ln_pos = positions[ln_local_idx]

    # Identify donors
    donors = []
    for i in non_ln:
        if np.linalg.norm(positions[i] - ln_pos) < DONOR_CUTOFF:
            donors.append(i)

    # Build organic graph among non-Ln atoms
    organic_adj = defaultdict(set)
    for ii, i in enumerate(non_ln):
        for jj in range(ii + 1, len(non_ln)):
            j = non_ln[jj]
            if np.linalg.norm(positions[i] - positions[j]) < ORGANIC_BOND_CUTOFF:
                organic_adj[i].add(j)
                organic_adj[j].add(i)

    # Connected components via BFS
    visited = set()
    ligands = []
    for start in non_ln:
        if start in visited:
            continue
        component = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            component.append(node)
            for nb in organic_adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        ligands.append(component)

    return ligands, donors


# ---------------------------------------------------------------------------
# Structure → tensors
# ---------------------------------------------------------------------------

def process_structure(structure, ln_idx):
    """Convert a crystal structure into the tensor dict used by the model.

    Returns a dict with pos, one_hot, nuclear_charges, ligand_group,
    coord_site, num_atoms, ligands — or None if the structure is invalid.
    """
    bond_graph = build_bond_graph(structure)
    unit_indices = extract_molecular_unit(structure, ln_idx, bond_graph)

    if len(unit_indices) < 3:
        return None

    elements = [str(structure[i].specie) for i in unit_indices]
    positions = np.array([structure[i].coords for i in unit_indices])

    # Strip hydrogen
    non_h = [i for i, e in enumerate(elements) if e != 'H']
    if len(non_h) < 2:
        return None
    elements = [elements[i] for i in non_h]
    positions = positions[non_h]

    # Put Ln at index 0
    ln_local = None
    for i, e in enumerate(elements):
        if e in LN_ELEMENTS:
            ln_local = i
            break
    if ln_local is None:
        return None
    if ln_local != 0:
        elements[0], elements[ln_local] = elements[ln_local], elements[0]
        positions[[0, ln_local]] = positions[[ln_local, 0]]

    num_atoms = len(elements)

    # Validate non-Ln elements are in ATOM2IDX
    for e in elements[1:]:
        if e not in ATOM2IDX:
            return None

    # one_hot: (N, 8) — Ln row is all zeros
    one_hot = torch.zeros(num_atoms, len(ATOM2IDX))
    for i in range(1, num_atoms):
        one_hot[i, ATOM2IDX[elements[i]]] = 1.0

    # nuclear_charges: (N,)
    nuclear_charges = torch.zeros(num_atoms, dtype=torch.long)
    for i, e in enumerate(elements):
        if e in CHARGES:
            nuclear_charges[i] = CHARGES[e]
        elif e in metals2idx:
            nuclear_charges[i] = metals2idx[e]

    pos = torch.tensor(positions, dtype=torch.float32)

    # Ligand decomposition
    ligands, donor_indices = decompose_ligands(positions, ln_local_idx=0)
    if len(ligands) == 0 or len(ligands) > MAX_LIGANDS:
        return None

    # ligand_group: (N, MAX_LIGANDS+1) then slice to (N, MAX_LIGANDS)
    ligand_group_full = torch.zeros(num_atoms, MAX_LIGANDS + 1)
    ligand_group_full[0, 0] = 1.0  # metal → column 0
    for lig_idx, lig_atoms in enumerate(ligands):
        for atom_idx in lig_atoms:
            ligand_group_full[atom_idx, lig_idx + 1] = 1.0
    ligand_group = ligand_group_full[:, 1:]  # (N, MAX_LIGANDS)

    # coord_site: (N,) binary
    coord_site = torch.zeros(num_atoms, dtype=torch.int)
    for d in donor_indices:
        coord_site[d] = 1

    return {
        'pos': pos,
        'one_hot': one_hot,
        'nuclear_charges': nuclear_charges,
        'ligand_group': ligand_group,
        'coord_site': coord_site,
        'num_atoms': num_atoms,
        'ligands': ligands,
    }


# ---------------------------------------------------------------------------
# Ligand-masking augmentation
# ---------------------------------------------------------------------------

def generate_masking_augmentations(processed, label, max_augment=20):
    """Generate masking combinations for a processed complex.

    For k ligands enumerate all 2^k − 1 non-empty subsets.  If that exceeds
    *max_augment*, randomly sample instead.
    """
    ligands = processed['ligands']
    k = len(ligands)
    num_atoms = processed['num_atoms']

    all_combos = []
    for r in range(1, k + 1):
        all_combos.extend(combinations(range(k), r))

    if len(all_combos) > max_augment:
        all_combos = random.sample(all_combos, max_augment)

    data_list = []
    for combo in all_combos:
        ligand_diff = torch.zeros(num_atoms)
        for lig_idx in combo:
            for atom_idx in ligands[lig_idx]:
                ligand_diff[atom_idx] = 1.0
        context = 1.0 - ligand_diff

        data = Data(
            pos=processed['pos'].clone(),
            one_hot=processed['one_hot'].clone(),
            nuclear_charges=processed['nuclear_charges'].clone(),
            ligand_group=processed['ligand_group'].clone(),
            coord_site=processed['coord_site'].clone(),
            context=context,
            ligand_diff=ligand_diff,
            num_atoms=processed['num_atoms'],
            label=label,
        )
        data_list.append(data)

    return data_list


# ---------------------------------------------------------------------------
# Worker entry point & progress logging
# ---------------------------------------------------------------------------

def process_one_complex(args_tuple):
    """Process a single complex. Returns a status dict.

    args_tuple: (element, refcode, filename, cif_dir, max_augment)
    Module-level so it is picklable for ``multiprocessing.Pool``.

    Returns dict with keys:
      status: "ok" | "empty" | "failed"
      element, refcode
      samples: list[Data] (empty unless status == "ok")
      reason: str (only for non-ok)
    """
    element, refcode, filename, cif_dir, max_augment = args_tuple
    try:
        warnings.filterwarnings("ignore")

        if filename:
            cif_path = os.path.join(cif_dir, element, filename)
        else:
            cif_path = os.path.join(cif_dir, element, f"{refcode}.cif")

        if not os.path.exists(cif_path):
            return {"status": "empty", "element": element, "refcode": refcode,
                    "reason": "cif file not found", "samples": []}

        structure = parse_cif(cif_path)
        if structure is None:
            return {"status": "empty", "element": element, "refcode": refcode,
                    "reason": "cif parse failed", "samples": []}

        ln_idx = find_ln_site(structure)
        if ln_idx is None:
            return {"status": "empty", "element": element, "refcode": refcode,
                    "reason": "no Ln site found", "samples": []}

        processed = process_structure(structure, ln_idx)
        if processed is None:
            return {"status": "empty", "element": element, "refcode": refcode,
                    "reason": "process_structure returned None", "samples": []}

        label = f"{element}_{refcode}"
        augmented_samples = generate_masking_augmentations(
            processed, label, max_augment)

        if not augmented_samples:
            return {"status": "empty", "element": element, "refcode": refcode,
                    "reason": "no samples produced", "samples": []}

        return {"status": "ok", "element": element, "refcode": refcode,
                "samples": augmented_samples,
                "n_samples": len(augmented_samples)}
    except Exception as e:
        return {"status": "failed", "element": element, "refcode": refcode,
                "reason": f"{type(e).__name__}: {str(e)[:200]}",
                "samples": []}


def _log_progress(done, total, n_samples, n_failed, start_time):
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-3)
    remaining = (total - done) / max(rate, 1e-3)
    pct = 100.0 * done / max(total, 1)
    print(
        f"[{done:>6}/{total}] {pct:5.1f}%  "
        f"samples={n_samples:>7}  failed={n_failed:>5}  "
        f"rate={rate:5.1f}/s  ETA={remaining/60:6.1f}min  "
        f"elapsed={elapsed/60:6.1f}min",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def _validate_sample(sample, max_ligands):
    """Check a Data object has the expected multi-LigandDiff format.

    Returns a list of error strings (empty if the sample is valid).
    """
    errors = []

    required_fields = ["pos", "one_hot", "nuclear_charges", "ligand_group",
                       "coord_site", "context", "ligand_diff", "num_atoms",
                       "label"]
    for f in required_fields:
        if not hasattr(sample, f):
            errors.append(f"missing field: {f}")
    if errors:
        return errors

    N = sample.num_atoms

    if sample.pos.shape != (N, 3):
        errors.append(f"pos shape {tuple(sample.pos.shape)} != ({N}, 3)")
    if sample.one_hot.shape != (N, 8):
        errors.append(
            f"one_hot shape {tuple(sample.one_hot.shape)} != ({N}, 8)")
    if sample.ligand_group.shape != (N, max_ligands):
        errors.append(
            f"ligand_group shape {tuple(sample.ligand_group.shape)} "
            f"!= ({N}, {max_ligands})")
    if sample.nuclear_charges.shape != (N,):
        errors.append(
            f"nuclear_charges shape {tuple(sample.nuclear_charges.shape)} "
            f"!= ({N},)")
    if sample.context.shape != (N,):
        errors.append(
            f"context shape {tuple(sample.context.shape)} != ({N},)")
    if sample.ligand_diff.shape != (N,):
        errors.append(
            f"ligand_diff shape {tuple(sample.ligand_diff.shape)} != ({N},)")

    if (sample.context + sample.ligand_diff - 1).abs().max() > 0:
        errors.append("context + ligand_diff != 1 for all atoms")
    if sample.one_hot[0].sum() != 0:
        errors.append("metal (index 0) one_hot should be all zeros")
    metal_z = sample.nuclear_charges[0].item()
    if metal_z < 57 or metal_z > 71:
        errors.append(
            f"metal at index 0 has atomic number {metal_z}, "
            f"expected Ln (57-71)")

    return errors


# ---------------------------------------------------------------------------
# Checkpoint consolidation
# ---------------------------------------------------------------------------

def consolidate_chunks(checkpoint_dir, output_dir, val_fraction, seed,
                       max_ligands):
    """Load all chunk_*.pt files and write final train_ln.pt / val_ln.pt.

    Complexes are split by label (not by sample) so all augmentations of a
    given complex land on the same side of the split.
    """
    chunk_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'chunk_*.pt')))
    print(
        f"\nConsolidating {len(chunk_files)} checkpoint chunks...",
        flush=True,
    )

    all_samples = []
    for cf in chunk_files:
        samples = torch.load(cf, map_location='cpu', weights_only=False)
        all_samples.extend(samples)

    print(f"Total samples loaded: {len(all_samples)}", flush=True)

    if len(all_samples) == 0:
        print("No samples to consolidate; skipping train/val split.",
              flush=True)
        return 0, 0

    rng = random.Random(seed)
    sample_idxs = rng.sample(range(len(all_samples)),
                             min(50, len(all_samples)))

    n_valid, n_invalid = 0, 0
    all_errors = []
    for idx in sample_idxs:
        errors = _validate_sample(all_samples[idx], max_ligands)
        if errors:
            n_invalid += 1
            all_errors.append((idx, errors))
        else:
            n_valid += 1

    print(
        f"\nValidation (random {len(sample_idxs)} samples): "
        f"{n_valid} valid, {n_invalid} invalid",
        flush=True,
    )
    if all_errors:
        print("First 5 invalid samples:", flush=True)
        for idx, errs in all_errors[:5]:
            print(f"  sample {idx}: {errs}", flush=True)

    groups = defaultdict(list)
    for s in all_samples:
        groups[s.label].append(s)

    complex_labels = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(complex_labels)

    n_val = max(1, int(len(complex_labels) * val_fraction))
    val_labels = set(complex_labels[:n_val])

    train_samples, val_samples = [], []
    for label, samples in groups.items():
        if label in val_labels:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

    train_path = os.path.join(output_dir, 'train_ln.pt')
    val_path = os.path.join(output_dir, 'val_ln.pt')
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)

    print(
        f"  train: {len(train_samples)} samples from "
        f"{len(complex_labels) - n_val} complexes",
        flush=True,
    )
    print(
        f"  val:   {len(val_samples)} samples from {n_val} complexes",
        flush=True,
    )
    print(f"Saved {train_path} and {val_path}", flush=True)

    return len(train_samples), len(val_samples)


# ---------------------------------------------------------------------------
# Format comparison against reference data/ppr.pt
# ---------------------------------------------------------------------------

def _compare_against_reference(output_dir):
    """Compare train_ln.pt field shapes against data/ppr.pt."""
    train_path = os.path.join(output_dir, "train_ln.pt")
    if not os.path.exists(train_path):
        print(f"\n(skipping format comparison — {train_path} not found)",
              flush=True)
        return

    ref_path = os.path.join(os.path.dirname(output_dir), "data", "ppr.pt")
    if not os.path.exists(ref_path):
        ref_path = None
        for candidate in ["data/ppr.pt", "../data/ppr.pt", "./ppr.pt"]:
            if os.path.exists(candidate):
                ref_path = candidate
                break
        if ref_path is None:
            print(
                "\n(skipping format comparison — no reference ppr.pt found)",
                flush=True,
            )
            return

    train = torch.load(train_path, map_location="cpu", weights_only=False)
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)

    print(f"\nFormat comparison:", flush=True)
    print(f"  train_ln.pt[0] fields:  {sorted(train[0].keys())}", flush=True)
    print(f"  ppr.pt[0] fields:       {sorted(ref[0].keys())}", flush=True)

    common = set(train[0].keys()) & set(ref[0].keys())
    missing_in_train = set(ref[0].keys()) - set(train[0].keys())
    extra_in_train = set(train[0].keys()) - set(ref[0].keys())

    if missing_in_train:
        print(f"  WARNING: missing from train_ln.pt: {missing_in_train}",
              flush=True)
    if extra_in_train:
        print(
            f"  INFO: extra in train_ln.pt (Ln-specific): {extra_in_train}",
            flush=True,
        )

    print(f"  Shape comparison for common fields:", flush=True)
    for field in sorted(common):
        t_val = getattr(train[0], field)
        r_val = getattr(ref[0], field)
        if hasattr(t_val, 'shape') and hasattr(r_val, 'shape'):
            t_last = (tuple(t_val.shape)[1:]
                      if len(t_val.shape) > 1 else "scalar")
            r_last = (tuple(r_val.shape)[1:]
                      if len(r_val.shape) > 1 else "scalar")
            match = "OK" if t_last == r_last else "MISMATCH"
            print(
                f"    {field:20s}  train={t_last}  ref={r_last}  {match}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for Ln-adapted multi-LigandDiff')
    parser.add_argument('--cif_dir', type=str, default='cif_output',
                        help='Directory of CIF files (Element/REFCODE.cif)')
    parser.add_argument('--candidates', type=str,
                        default='results_v2/training_candidates.csv',
                        help='CSV listing training candidates')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for .pt files')
    parser.add_argument('--max_augment', type=int, default=20,
                        help='Max masking augmentations per complex')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of complexes for validation')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes (1 = sequential)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_every', type=int, default=500,
                        help='Flush a chunk file after this many complexes')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing chunk_*.pt files')
    parser.add_argument('--consolidate_only', action='store_true',
                        help='Skip processing; merge existing chunks into '
                             'final train/val files and exit')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.consolidate_only:
        consolidate_chunks(checkpoint_dir, args.output_dir,
                           args.val_fraction, args.seed, MAX_LIGANDS)
        _compare_against_reference(args.output_dir)
        return

    df = pd.read_csv(args.candidates)
    print(f"Loaded {len(df)} candidates from {args.candidates}", flush=True)

    has_filename = 'filename' in df.columns
    arg_tuples = []
    for _, row in df.iterrows():
        if has_filename and pd.notna(row.get('filename')):
            filename = row['filename']
        else:
            filename = None
        arg_tuples.append((
            row['element'], row['refcode'], filename,
            args.cif_dir, args.max_augment,
        ))

    total = len(arg_tuples)
    CHUNK_SIZE = args.checkpoint_every
    batch_size = 50  # progress logging granularity

    chunk_idx = 0
    chunk_buffer = []
    total_samples_written = 0
    complexes_skipped = 0
    n_failed = 0
    failure_counter = Counter()
    failed_structures = []  # (element, refcode, reason) for CSV

    if args.resume:
        existing = sorted(glob.glob(
            os.path.join(checkpoint_dir, 'chunk_*.pt')))
        if existing:
            last_chunk = int(
                os.path.basename(existing[-1]).split('_')[-1].split('.')[0])
            chunk_idx = last_chunk + 1
            complexes_skipped = chunk_idx * CHUNK_SIZE
            print(
                f"Resuming from chunk {chunk_idx} "
                f"(skipping first {complexes_skipped} complexes)",
                flush=True,
            )
            for cf in existing:
                samples = torch.load(cf, map_location='cpu',
                                     weights_only=False)
                total_samples_written += len(samples)
                del samples
            print(
                f"Existing chunks contain {total_samples_written} samples",
                flush=True,
            )
            arg_tuples = arg_tuples[complexes_skipped:]
        else:
            print("No existing checkpoints found, starting fresh",
                  flush=True)

    def flush_chunk():
        nonlocal chunk_idx, total_samples_written
        if not chunk_buffer:
            return
        path = os.path.join(checkpoint_dir, f'chunk_{chunk_idx:04d}.pt')
        torch.save(chunk_buffer, path)
        total_samples_written += len(chunk_buffer)
        print(
            f"  -> wrote checkpoint {path} "
            f"({len(chunk_buffer)} samples, "
            f"{total_samples_written} total)",
            flush=True,
        )
        chunk_buffer.clear()
        chunk_idx += 1

    start_time = time.time()
    print(
        f"Processing {len(arg_tuples)} complexes "
        f"(of {total} total) with {args.workers} worker(s); "
        f"checkpoint every {CHUNK_SIZE} complexes",
        flush=True,
    )

    def handle_result(i, result):
        nonlocal n_failed
        if result["status"] == "ok":
            chunk_buffer.extend(result["samples"])
        else:
            n_failed += 1
            failure_counter[result["reason"]] += 1
            failed_structures.append(
                (result["element"], result["refcode"], result["reason"]))

        if (i + 1) % CHUNK_SIZE == 0:
            flush_chunk()

        if (i + 1) % batch_size == 0:
            done_total = i + 1 + complexes_skipped
            sample_total = total_samples_written + len(chunk_buffer)
            _log_progress(done_total, total, sample_total, n_failed,
                          start_time)

    if args.workers == 1:
        for i, args_t in enumerate(arg_tuples):
            result = process_one_complex(args_t)
            handle_result(i, result)
    else:
        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(
                    process_one_complex, arg_tuples, chunksize=4)):
                handle_result(i, result)

    flush_chunk()  # save final partial chunk if any
    _log_progress(len(arg_tuples) + complexes_skipped, total,
                  total_samples_written, n_failed, start_time)

    print(
        f"\nProcessing done. {total_samples_written} samples written "
        f"across {chunk_idx} chunk(s). Failed/empty: {n_failed}",
        flush=True,
    )

    if failed_structures:
        failed_path = os.path.join(args.output_dir, "failed_structures.csv")
        with open(failed_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["element", "refcode", "reason"])
            writer.writerows(failed_structures)
        print(
            f"\nWrote {len(failed_structures)} failure records to "
            f"{failed_path}",
            flush=True,
        )

        print("\nTop failure reasons:", flush=True)
        for reason, count in failure_counter.most_common(10):
            print(f"  [{count:>5}] {reason}", flush=True)

    n_train, n_val = consolidate_chunks(
        checkpoint_dir, args.output_dir, args.val_fraction, args.seed,
        MAX_LIGANDS)
    print(
        f"\nDONE. {n_train} train + {n_val} val samples saved.",
        flush=True,
    )

    _compare_against_reference(args.output_dir)


if __name__ == '__main__':
    main()
