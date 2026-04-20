#!/usr/bin/env python
"""Failure-mode taxonomy for generated lanthanide complexes.

Computes per-structure metrics across four failure axes and produces
aggregate statistics for publication in JCIM/JCTC.

Metrics
-------
1. Context-atom perturbation distance  (sampler drift)
2. Inter-ligand bridging rate          (steric clashes across ligand boundaries)
3. Ligand-type mutation rate           (composition / connectivity changes in context)
4. Denticity recovery                  (requested vs actual donor count)

Usage
-----
  python metrics/taxonomy.py \\
      --input-dir generated/eu_tmma_mask1_epoch48/noH \\
      --reference eu_tmma_cis.xyz

  python metrics/taxonomy.py \\
      --input-file some_structure.xyz \\
      --reference eu_tmma_cis.xyz
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────

METALS = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Mo",
    "Ru", "Rh", "Pd", "Cd", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "W", "Re", "Os", "Ir", "Pt",
}

LN_DONOR_RANGE = (2.2, 2.9)  # angstrom
BRIDGE_CUTOFF = 1.5           # angstrom
HASH_BUCKET = 0.1             # angstrom resolution for ligand-distance hashing


# ── xyz I/O ────────────────────────────────────────────────────────────────

def load_xyz(path):
    """Load an xyz file.

    Returns
    -------
    atoms : list of (elem, x, y, z)
    n_context : int or None  (None for standard xyz without context-count line)
    """
    with open(path) as fh:
        lines = fh.readlines()
    if len(lines) < 3:
        raise ValueError(f"Fewer than 3 lines in {path}")
    try:
        n_context = int(lines[1].strip())
    except ValueError:
        n_context = None
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                atoms.append((parts[0], float(parts[1]),
                              float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return atoms, n_context


def parse_denticity(filename):
    """Extract (LD_c, LD_g) from ``..._[c1, c2]_[g1, g2].xyz``."""
    m = re.search(r"\[([^\]]*)\]_\[([^\]]*)\]", filename)
    if not m:
        return None, None

    def _ints(s):
        return [int(x) for x in s.split(",") if x.strip()] if s.strip() else []

    return _ints(m.group(1)), _ints(m.group(2))


def _dist(a, b):
    """Euclidean distance between two atom tuples (elem, x, y, z)."""
    return float(np.sqrt((a[1] - b[1]) ** 2
                         + (a[2] - b[2]) ** 2
                         + (a[3] - b[3]) ** 2))


# ── reference analysis (cached) ───────────────────────────────────────────

_ref_cache: dict = {}


def analyze_reference(ref_path):
    """Run ``ligand_breakdown`` on *ref_path* and cache the result."""
    if ref_path in _ref_cache:
        return _ref_cache[ref_path]

    from molSimplify.Classes.mol3D import mol3D
    from molSimplify.Classes.ligand import ligand_breakdown

    atoms, _ = load_xyz(ref_path)

    with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w",
                                     delete=False) as tmp:
        tmp.write(f"{len(atoms)}\n\n")
        for el, x, y, z in atoms:
            tmp.write(f"{el}\t{x:.5f}\t{y:.5f}\t{z:.5f}\n")
        tmp_path = tmp.name

    mol = mol3D()
    mol.readfromxyz(tmp_path)
    liglist, ligdents, ligcon = ligand_breakdown(
        mol, silent=True, BondedOct=False, transition_metals_only=False,
    )
    os.unlink(tmp_path)

    atom_to_group: dict[int, int] = {}
    for gidx, indices in enumerate(liglist):
        for ai in indices:
            atom_to_group[ai] = gidx

    result = dict(
        atoms=atoms,
        liglist=[list(lg) for lg in liglist],
        ligdents=list(ligdents),
        ligcon=[list(c) for c in ligcon],
        atom_to_group=atom_to_group,
    )
    _ref_cache[ref_path] = result
    return result


# ── context-to-reference matching ──────────────────────────────────────────

def match_context(gen_atoms, n_context, ref_atoms):
    """Greedy nearest-neighbour matching of context atoms to reference atoms.

    Returns list of ``(gen_idx, ref_idx, displacement_angstrom)``.
    """
    used: set[int] = set()
    matches = []
    for gi in range(n_context):
        ga = gen_atoms[gi]
        best_d, best_ri = float("inf"), None
        for ri, ra in enumerate(ref_atoms):
            if ri in used or ra[0] != ga[0]:
                continue
            d = _dist(ga, ra)
            if d < best_d:
                best_d, best_ri = d, ri
        if best_ri is not None:
            used.add(best_ri)
            matches.append((gi, best_ri, best_d))
        else:
            matches.append((gi, None, None))
    return matches


# ── metric 1: context-atom perturbation ────────────────────────────────────

def metric_perturbation(gen_atoms, n_context, ref_atoms):
    matches = match_context(gen_atoms, n_context, ref_atoms)
    disps = [d for _, _, d in matches if d is not None]
    per_atom = [
        dict(gen_idx=gi, ref_idx=ri, element=gen_atoms[gi][0],
             displacement_A=round(d, 4) if d is not None else None)
        for gi, ri, d in matches
    ]
    return dict(
        per_atom=per_atom,
        max_A=round(max(disps), 4) if disps else None,
        mean_A=round(float(np.mean(disps)), 4) if disps else None,
        n_context=n_context,
    )


# ── metric 2: inter-ligand bridging ───────────────────────────────────────

def metric_bridging(gen_atoms, n_context, ref_info):
    """Detect close contacts between generated atoms and non-metal context atoms."""
    ref_atoms = ref_info["atoms"]
    a2g = ref_info["atom_to_group"]

    # assign context atoms to their reference ligand groups
    ctx_matches = match_context(gen_atoms, n_context, ref_atoms)
    ctx_group: dict[int, int] = {}
    for gi, ri, d in ctx_matches:
        if ri is not None and d is not None and d < 1.0:
            ctx_group[gi] = a2g.get(ri, -1)

    bridges = []
    for gi in range(n_context, len(gen_atoms)):
        ga = gen_atoms[gi]
        for ci in range(1, n_context):            # skip metal at 0
            d = _dist(ga, gen_atoms[ci])
            if d < BRIDGE_CUTOFF:
                bridges.append(dict(
                    gen_idx=gi, ctx_idx=ci,
                    gen_elem=ga[0], ctx_elem=gen_atoms[ci][0],
                    distance_A=round(d, 3),
                    ctx_group=ctx_group.get(ci, -1),
                ))

    pairs = Counter()
    for b in bridges:
        key = "-".join(sorted([b["gen_elem"], b["ctx_elem"]]))
        pairs[key] += 1

    return dict(
        n_bridges=len(bridges),
        avg_distance_A=(round(float(np.mean([b["distance_A"] for b in bridges])), 3)
                        if bridges else None),
        element_pairs=dict(pairs),
        bridges=bridges,
    )


# ── metric 3: ligand-type mutation ────────────────────────────────────────

def _ligand_hash(atoms, indices):
    """Hash a ligand by sorted element tuple + bucketed pairwise distances."""
    elems = tuple(sorted(atoms[i][0] for i in indices))
    idx = list(indices)
    dists = sorted(
        round(_dist(atoms[idx[a]], atoms[idx[b]]) / HASH_BUCKET) * HASH_BUCKET
        for a in range(len(idx)) for b in range(a + 1, len(idx))
    )
    return (elems, tuple(round(d, 1) for d in dists))


def metric_mutation(gen_atoms, n_context, ref_info):
    """Check whether each context ligand's composition + connectivity is preserved."""
    ref_atoms = ref_info["atoms"]
    liglist = ref_info["liglist"]

    ctx_matches = match_context(gen_atoms, n_context, ref_atoms)
    ref_to_gen: dict[int, int] = {}
    for gi, ri, d in ctx_matches:
        if ri is not None and d is not None and d < 1.0:
            ref_to_gen[ri] = gi

    ligand_results = []
    n_ctx_lig = 0
    n_mutated = 0

    for lig_idx, lig_atoms in enumerate(liglist):
        mapped = [ai for ai in lig_atoms if ai in ref_to_gen]
        if not mapped:
            continue                       # masked (generated) ligand
        n_ctx_lig += 1

        if len(mapped) < len(lig_atoms):   # partial match — shouldn't happen
            n_mutated += 1
            ligand_results.append(dict(
                ligand_idx=lig_idx,
                denticity=ref_info["ligdents"][lig_idx],
                mutated=True,
                reason=f"partial ({len(mapped)}/{len(lig_atoms)})",
            ))
            continue

        ref_hash = _ligand_hash(ref_atoms, lig_atoms)
        gen_hash = _ligand_hash(gen_atoms, [ref_to_gen[ai] for ai in lig_atoms])
        is_mut = ref_hash != gen_hash
        if is_mut:
            n_mutated += 1
        ligand_results.append(dict(
            ligand_idx=lig_idx,
            denticity=ref_info["ligdents"][lig_idx],
            ref_elements=list(ref_hash[0]),
            gen_elements=list(gen_hash[0]),
            mutated=is_mut,
            reason="hash_mismatch" if is_mut else "preserved",
        ))

    return dict(
        n_context_ligands=n_ctx_lig,
        n_mutated=n_mutated,
        mutation_rate=round(n_mutated / n_ctx_lig, 3) if n_ctx_lig else None,
        ligands=ligand_results,
    )


# ── metric 4: denticity recovery ──────────────────────────────────────────

def metric_denticity(gen_atoms, n_context, filename):
    """Compare requested (from filename) vs actual denticity of generated ligands."""
    _, ld_g = parse_denticity(filename)
    metal = gen_atoms[0]

    donors = []
    for gi in range(n_context, len(gen_atoms)):
        d = _dist(gen_atoms[gi], metal)
        if LN_DONOR_RANGE[0] <= d <= LN_DONOR_RANGE[1]:
            donors.append(dict(idx=gi, element=gen_atoms[gi][0],
                               distance_A=round(d, 3)))

    actual = len(donors)
    requested = sum(ld_g) if ld_g else None
    return dict(
        requested_ld_g=ld_g,
        requested_total=requested,
        actual=actual,
        match=(actual == requested) if requested is not None else None,
        donors=donors,
        donor_elements=dict(Counter(d["element"] for d in donors)),
    )


# ── per-structure driver ──────────────────────────────────────────────────

def _resolve_n_context(path, atoms, project_root):
    """Attempt to recover *n_context* for an xtb-optimised file."""
    parent = Path(path).parent.name
    for subdir in ("generated/eu_tmma_mask1_epoch48/noH",
                    "generated/eu_tmma_real_epoch48/noH"):
        orig = project_root / subdir / f"{parent}.xyz"
        if orig.exists():
            _, nc = load_xyz(str(orig))
            if nc is not None:
                return nc
    return None


def analyze_structure(path, ref_path, project_root):
    """Run all four metrics on a single generated xyz file."""
    fname = os.path.basename(path)
    atoms, n_ctx = load_xyz(path)

    # xtb files lack the n_context line — try to recover it
    if n_ctx is None:
        n_ctx = _resolve_n_context(path, atoms, project_root)
        if n_ctx is None:
            # last resort: match atoms to reference by element + proximity
            ref_atoms, _ = load_xyz(ref_path)
            used: set[int] = set()
            count = 0
            for a in atoms:
                for ri, ra in enumerate(ref_atoms):
                    if ri not in used and ra[0] == a[0] and _dist(a, ra) < 0.5:
                        used.add(ri)
                        count += 1
                        break
            n_ctx = count
            log.warning("%s: n_context inferred as %d (xtb file)", fname, n_ctx)

    if atoms[0][0] not in METALS:
        log.warning("%s: atom 0 is %s, not a recognised metal", fname, atoms[0][0])

    ref_info = analyze_reference(ref_path)

    return dict(
        filename=fname,
        path=str(path),
        reference=os.path.basename(ref_path),
        n_total=len(atoms),
        n_context=n_ctx,
        n_generated=len(atoms) - n_ctx,
        metal=atoms[0][0],
        perturbation=metric_perturbation(atoms, n_ctx, ref_info["atoms"]),
        bridging=metric_bridging(atoms, n_ctx, ref_info),
        mutation=metric_mutation(atoms, n_ctx, ref_info),
        denticity=metric_denticity(atoms, n_ctx, fname),
    )


# ── output writers ─────────────────────────────────────────────────────────

def write_json(result, out_dir):
    name = Path(result["filename"]).stem
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)


def write_csv(results, path):
    fields = [
        "filename", "n_gen", "max_perturb", "mean_perturb",
        "n_bridges", "n_mutated_ligands", "mutation_rate",
        "requested_denticity", "actual_denticity", "denticity_match",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(dict(
                filename=r["filename"],
                n_gen=r["n_generated"],
                max_perturb=r["perturbation"]["max_A"],
                mean_perturb=r["perturbation"]["mean_A"],
                n_bridges=r["bridging"]["n_bridges"],
                n_mutated_ligands=r["mutation"]["n_mutated"],
                mutation_rate=r["mutation"]["mutation_rate"],
                requested_denticity=r["denticity"]["requested_total"],
                actual_denticity=r["denticity"]["actual"],
                denticity_match=r["denticity"]["match"],
            ))


def write_aggregate(results, path):
    """Write human-readable aggregate summary."""
    n = len(results)
    L = []
    L.append("Failure-Mode Taxonomy  --  Aggregate Summary")
    L.append("=" * 60)
    L.append(f"Structures analysed: {n}")
    L.append("")

    # ── 1. Perturbation ────────────────────────────────────────
    maxp = [r["perturbation"]["max_A"] for r in results
            if r["perturbation"]["max_A"] is not None]
    meanp = [r["perturbation"]["mean_A"] for r in results
             if r["perturbation"]["mean_A"] is not None]

    L.append("1. CONTEXT-ATOM PERTURBATION")
    if maxp:
        L.append(f"   Mean of max displacement:  {np.mean(maxp):.4f} A")
        L.append(f"   Std  of max displacement:  {np.std(maxp):.4f} A")
        L.append(f"   Mean of mean displacement: {np.mean(meanp):.4f} A")
        L.append(f"   Range: [{min(maxp):.4f}, {max(maxp):.4f}] A")
        drifted = sum(1 for d in maxp if d > 0.1)
        L.append(f"   Drifted (max > 0.1 A): {drifted}/{len(maxp)}")
    else:
        L.append("   (no data)")
    L.append("")

    # ── 2. Bridging ────────────────────────────────────────────
    bc = [r["bridging"]["n_bridges"] for r in results]
    L.append("2. INTER-LIGAND BRIDGING")
    L.append(f"   Mean bridges/structure: {np.mean(bc):.2f}")
    L.append(f"   Median: {np.median(bc):.0f}   Max: {max(bc)}")
    L.append(f"   Zero bridges: {sum(1 for b in bc if b == 0)}/{n}")
    hist = Counter(bc)
    L.append("   Histogram:")
    for k in sorted(hist):
        L.append(f"     {k:3d}: {hist[k]:3d}  {'#' * hist[k]}")
    all_pairs = Counter()
    for r in results:
        for p, c in r["bridging"]["element_pairs"].items():
            all_pairs[p] += c
    if all_pairs:
        L.append("   Element pairs (total):")
        for p, c in all_pairs.most_common():
            L.append(f"     {p}: {c}")
    L.append("")

    # ── 3. Mutation ────────────────────────────────────────────
    mr = [r["mutation"]["mutation_rate"] for r in results
          if r["mutation"]["mutation_rate"] is not None]
    n_any = sum(1 for r in results if r["mutation"]["n_mutated"] > 0)
    L.append("3. LIGAND-TYPE MUTATION")
    if mr:
        L.append(f"   Mean mutation rate: {np.mean(mr):.3f}")
    else:
        L.append("   (no data)")
    L.append(f"   Structures with any mutation: {n_any}/{n}")
    ld_stats: dict = defaultdict(lambda: dict(total=0, mutated=0))
    for r in results:
        for lig in r["mutation"]["ligands"]:
            d = lig["denticity"]
            ld_stats[d]["total"] += 1
            if lig["mutated"]:
                ld_stats[d]["mutated"] += 1
    if ld_stats:
        L.append(f"   {'Dent':>6} {'Mut':>6} {'Tot':>6} {'Rate':>8}")
        for d in sorted(ld_stats):
            s = ld_stats[d]
            rate = s["mutated"] / s["total"] if s["total"] else 0
            L.append(f"   {d:>6} {s['mutated']:>6} {s['total']:>6} {rate:>8.3f}")
    L.append("")

    # ── 4. Denticity recovery ──────────────────────────────────
    pairs = [(r["denticity"]["requested_total"], r["denticity"]["actual"])
             for r in results if r["denticity"]["requested_total"] is not None]
    n_match = sum(1 for req, act in pairs if req == act)

    L.append("4. DENTICITY RECOVERY")
    if pairs:
        pct = 100 * n_match / len(pairs)
        L.append(f"   Exact match: {n_match}/{len(pairs)} ({pct:.1f}%)")
        vals = sorted(set(v for p in pairs for v in p))
        hdr = "   Req\\Act   " + "  ".join(f"{v:>3}" for v in vals)
        L.append(hdr)
        for rv in vals:
            counts = [sum(1 for req, act in pairs
                          if req == rv and act == av) for av in vals]
            row = "  ".join(f"{c:>3}" if c else "  ." for c in counts)
            L.append(f"   {rv:>7}   {row}")
        diffs = [act - req for req, act in pairs]
        L.append(f"   Mean error (act - req): {np.mean(diffs):+.2f}")
        L.append(f"   Under-coord: {sum(1 for d in diffs if d < 0)}   "
                 f"Over-coord: {sum(1 for d in diffs if d > 0)}")
    else:
        L.append("   (no denticity data in filenames)")
    L.append("")

    text = "\n".join(L)
    with open(path, "w") as f:
        f.write(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────

def collect_files(input_path):
    """Gather xyz files from a file path or directory."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        direct = sorted(p.glob("*.xyz"))
        return [str(f) for f in direct] if direct else [str(f) for f in sorted(p.rglob("*.xyz"))]
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Failure-mode taxonomy for generated Ln complexes")
    ap.add_argument("--input-dir", help="Directory of generated xyz files")
    ap.add_argument("--input-file", help="Single xyz file")
    ap.add_argument("--reference", required=True,
                    help="Reference xyz file")
    ap.add_argument("--output-dir", default="metrics/results",
                    help="Output directory (default: metrics/results)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    ref = Path(args.reference)
    if not ref.is_absolute():
        ref = project_root / ref
    if not ref.exists():
        sys.exit(f"Reference not found: {ref}")

    inp = args.input_dir or args.input_file
    if not inp:
        sys.exit("Provide --input-dir or --input-file")
    files = collect_files(inp)
    if not files:
        sys.exit(f"No xyz files in {inp}")

    out = Path(args.output_dir)
    per_dir = out / "per_structure"
    per_dir.mkdir(parents=True, exist_ok=True)

    ref_info = analyze_reference(str(ref))
    log.info("Reference %s: %d ligands, denticities %s",
             ref.name, len(ref_info["liglist"]), ref_info["ligdents"])

    results = []
    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        try:
            r = analyze_structure(fpath, str(ref), project_root)
            results.append(r)
            write_json(r, per_dir)
            d = r["denticity"]
            print(f"[{i}/{len(files)}] {fname}: "
                  f"{r['bridging']['n_bridges']} bridges, "
                  f"{r['mutation']['n_mutated']} mutations, "
                  f"denticity {d['actual']} (requested {d['requested_total'] or '?'})")
        except Exception as e:
            log.error("[%d/%d] %s: %s", i, len(files), fname, e)

    if not results:
        sys.exit("No structures analysed successfully")

    write_csv(results, out / "summary.csv")
    agg = write_aggregate(results, out / "aggregate.txt")
    print(f"\n{agg}")


if __name__ == "__main__":
    main()
