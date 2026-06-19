"""
DFT validation pipeline for representative generated Ln complexes.

Selects 5-10 structures from different failure categories (from xTB results),
prepares ORCA PBE0-D4/def2-TZVP inputs with Stuttgart ECP for Eu, submits
SLURM jobs, and parses results into a comparison CSV.

Usage:
    # 1. Select structures and prepare ORCA inputs:
    python dft_pipeline.py prepare \
        --xtb-results-dir xtb_results \
        --reference refs/eu_tmma_cis.xyz \
        --output-dir dft_work \
        --nprocs 16 --maxcore 3500

    # 2. Submit all jobs:
    python dft_pipeline.py submit --work-dir dft_work

    # 3. After jobs finish, parse results:
    python dft_pipeline.py parse \
        --work-dir dft_work \
        --reference refs/eu_tmma_cis.xyz \
        --output-csv dft_comparison.csv
"""
from __future__ import annotations  # PEP 604 unions (`list | None`) run on Py3.9 (cluster ligdiff)

import argparse
import csv
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HARTREE_TO_KCAL = 627.509

# Experimental Eu-donor distances from Kravchuk et al. 2024 (VEDTAA01)
REF_EU_O_TMMA = (2.33, 2.40)   # angstrom range for Eu-O(TMMA)
REF_EU_O_NO3 = (2.51, 2.56)    # angstrom range for Eu-O(NO3)
DISSOCIATION_CUTOFF = 3.5       # angstrom — donor beyond this = dissociated

TEMPLATE_PATH = Path(__file__).parent / "orca_templates" / "pbe0_eu.inp"

# ---------------------------------------------------------------------------
# xyz utilities
# ---------------------------------------------------------------------------

def load_xyz(path: str) -> list[tuple[str, float, float, float]]:
    """Load atoms from an xyz file."""
    with open(path) as f:
        lines = f.readlines()
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                atoms.append((parts[0], float(parts[1]),
                              float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return atoms


def write_xyz(atoms: list, path: str, comment: str = ""):
    """Write a standard xyz file."""
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for el, x, y, z in atoms:
            f.write(f"{el:4s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def eu_donor_distances(atoms: list) -> list[dict]:
    """Compute distances from Eu to all potential donor atoms (O, N, S)."""
    eu = None
    for i, (el, x, y, z) in enumerate(atoms):
        if el == "Eu":
            eu = (i, x, y, z)
            break
    if eu is None:
        return []

    donors = []
    for i, (el, x, y, z) in enumerate(atoms):
        if i == eu[0]:
            continue
        if el in ("O", "N", "S"):
            d = np.sqrt((x - eu[1])**2 + (y - eu[2])**2 + (z - eu[3])**2)
            if d < DISSOCIATION_CUTOFF + 1.0:  # generous cutoff for reporting
                donors.append({"idx": i, "element": el,
                               "distance_A": round(float(d), 4)})
    donors.sort(key=lambda r: r["distance_A"])
    return donors


# ---------------------------------------------------------------------------
# Structure selection from xTB results
# ---------------------------------------------------------------------------

def _load_xtb_summary(results_dir: str) -> list[dict]:
    """Load xTB summary CSV(s) from results directory."""
    base = Path(results_dir)
    rows = []

    # Try aggregated first
    agg = base / "summary_all.csv"
    if agg.exists():
        with open(agg) as f:
            rows = list(csv.DictReader(f))
        return rows

    # Individual summaries
    for csv_file in sorted(base.rglob("summary.csv")):
        run_name = csv_file.parent.name
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                row["run"] = run_name
                rows.append(row)
    return rows


def _load_taxonomy_results(metrics_dir: str = "metrics/results") -> dict:
    """Load taxonomy JSON results keyed by structure name."""
    per_struct = Path(metrics_dir) / "per_structure"
    results = {}
    if per_struct.exists():
        for jf in per_struct.glob("*.json"):
            with open(jf) as f:
                data = json.load(f)
            name = Path(data.get("filename", jf.stem)).stem
            results[name] = data
    return results


def select_structures(xtb_dir: str, reference_xyz: str,
                      max_per_category: int = 2) -> list[dict]:
    """Select representative structures from different failure categories.

    Categories:
      - good:     converged, low energy, chemically sensible
      - mutated:  context NO3- extended with side chains
      - bridged:  generated atoms bridge two ligands
      - aqua:     from aqua-seed context ablation (if available)
      - reference: the CCDC VEDTAA01 structure

    Returns list of dicts with keys: name, category, xyz_path, xtb_energy.
    """
    xtb_rows = _load_xtb_summary(xtb_dir)
    taxonomy = _load_taxonomy_results()

    # Partition converged structures
    converged = []
    for row in xtb_rows:
        if row.get("xtb_success", "").lower() != "true":
            continue
        name = row["name"]
        energy = float(row["final_energy_hartree"]) if row.get("final_energy_hartree") else None

        # Find the xyz file. Structure names embed brackets (e.g.
        # "..._[2, 2, 2, 2]_[2]"); glob would read "[...]" as a character class
        # and never match, so escape the name (glob.escape keeps "**/" intact).
        xyz_path = None
        ename = glob.escape(name)
        for pattern in [
            f"{xtb_dir}/**/converged/{ename}.xyz",
            f"{xtb_dir}/**/xtb_work/{ename}/xtbopt.xyz",
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                xyz_path = matches[0]
                break

        if xyz_path is None:
            continue

        tax = taxonomy.get(name, {})
        n_bridges = tax.get("bridging", {}).get("n_bridges", 0)
        n_mutated = tax.get("mutation", {}).get("n_mutated", 0)
        is_aqua = "aqua" in name.lower() or "h2o" in name.lower()

        converged.append({
            "name": name,
            "xyz_path": xyz_path,
            "xtb_energy": energy,
            "n_bridges": n_bridges,
            "n_mutated": n_mutated,
            "is_aqua": is_aqua,
        })

    if not converged:
        log.warning("No converged xTB structures found in %s", xtb_dir)
        log.info("Will only include reference structure.")
        selected = []
    else:
        # Sort by energy (lowest first)
        converged.sort(key=lambda r: r["xtb_energy"] if r["xtb_energy"] else 0)

        selected = []
        cats_picked = {"good": 0, "mutated": 0, "bridged": 0, "aqua": 0}

        # Good structures: lowest energy, no bridges, no mutations
        for s in converged:
            if cats_picked["good"] >= max_per_category:
                break
            if s["n_bridges"] == 0 and s["n_mutated"] == 0 and not s["is_aqua"]:
                selected.append({**s, "category": "good"})
                cats_picked["good"] += 1

        # Mutated: context ligands show mutations
        for s in converged:
            if cats_picked["mutated"] >= max_per_category:
                break
            if s["n_mutated"] > 0 and s["name"] not in {r["name"] for r in selected}:
                selected.append({**s, "category": "mutated_context"})
                cats_picked["mutated"] += 1

        # Bridged: inter-ligand bridges
        for s in converged:
            if cats_picked["bridged"] >= max_per_category:
                break
            if s["n_bridges"] > 0 and s["name"] not in {r["name"] for r in selected}:
                selected.append({**s, "category": "bridged"})
                cats_picked["bridged"] += 1

        # Aqua-seed
        for s in converged:
            if cats_picked["aqua"] >= 1:
                break
            if s["is_aqua"] and s["name"] not in {r["name"] for r in selected}:
                selected.append({**s, "category": "aqua_seed"})
                cats_picked["aqua"] += 1

        # If we still need more, fill from remaining converged
        remaining = [s for s in converged
                     if s["name"] not in {r["name"] for r in selected}]
        for s in remaining:
            if len(selected) >= 8:
                break
            selected.append({**s, "category": "other"})

        log.info("Selected %d generated structures:", len(selected))
        for s in selected:
            log.info("  %-12s  %s  E=%.4f Ha",
                     s["category"], s["name"],
                     s["xtb_energy"] if s["xtb_energy"] else 0)

    # Always include reference
    selected.append({
        "name": "VEDTAA01_reference",
        "category": "reference",
        "xyz_path": str(reference_xyz),
        "xtb_energy": None,
        "n_bridges": 0,
        "n_mutated": 0,
        "is_aqua": False,
    })

    return selected


# ---------------------------------------------------------------------------
# ORCA input preparation
# ---------------------------------------------------------------------------

def prepare_orca_input(xyz_path: str, work_dir: str, name: str,
                       charge: int = 0, mult: int = 7,
                       nprocs: int = 16, maxcore: int = 3500) -> str:
    """Create ORCA input directory with input.inp and input.xyz."""
    struct_dir = Path(work_dir) / name
    struct_dir.mkdir(parents=True, exist_ok=True)

    # Copy and clean xyz
    atoms = load_xyz(xyz_path)
    out_xyz = struct_dir / "input.xyz"
    write_xyz(atoms, str(out_xyz), comment=f"{name} — DFT opt input")

    # Fill template
    template = TEMPLATE_PATH.read_text()
    inp_text = template.format(
        xyzfile="input.xyz",
        charge=charge,
        mult=mult,
        nprocs=nprocs,
        maxcore=maxcore,
    )

    inp_path = struct_dir / "input.inp"
    inp_path.write_text(inp_text)

    log.info("Prepared %s: charge=%d mult=%d nprocs=%d maxcore=%d",
             name, charge, mult, nprocs, maxcore)
    return str(struct_dir)


# ---------------------------------------------------------------------------
# ORCA output parsing
# ---------------------------------------------------------------------------

def parse_orca_output(log_path: str) -> dict:
    """Parse ORCA output log for energy, convergence, and Mulliken charges."""
    result = {
        "converged": False,
        "final_energy_hartree": None,
        "mulliken_eu_charge": None,
        "n_opt_cycles": None,
        "scf_converged": True,
        "error": None,
    }

    if not os.path.exists(log_path):
        result["error"] = "output file not found"
        return result

    with open(log_path) as f:
        text = f.read()

    # Check for errors
    if "ORCA TERMINATED NORMALLY" not in text:
        result["error"] = "ORCA did not terminate normally"
        if "SCF NOT CONVERGED" in text:
            result["scf_converged"] = False
            result["error"] = "SCF not converged"
        elif "Error" in text or "ERROR" in text:
            # Extract last error line
            for line in reversed(text.split("\n")):
                if "error" in line.lower() or "ERROR" in line:
                    result["error"] = line.strip()[:200]
                    break

    # Final single-point energy (last occurrence)
    for match in re.finditer(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", text):
        result["final_energy_hartree"] = float(match.group(1))

    # Optimization convergence
    if "THE OPTIMIZATION HAS CONVERGED" in text:
        result["converged"] = True

    # Count opt cycles
    cycles = re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", text)
    if cycles:
        result["n_opt_cycles"] = int(cycles[-1])

    # Mulliken charges for Eu
    mulliken_block = re.search(
        r"MULLIKEN ATOMIC CHARGES.*?\n(.*?)\n\s*Sum of Mulliken Charges",
        text, re.DOTALL
    )
    if mulliken_block:
        for line in mulliken_block.group(1).split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "Eu":
                try:
                    result["mulliken_eu_charge"] = float(parts[-1])
                except ValueError:
                    pass

    return result


def parse_orca_final_xyz(work_dir: str) -> list | None:
    """Load the final optimised geometry from ORCA output."""
    # ORCA writes the optimised geometry to input.xyz (overwritten) or input_trj.xyz
    opt_xyz = Path(work_dir) / "input.xyz"
    trj_xyz = Path(work_dir) / "input_trj.xyz"

    # Prefer the final frame of the trajectory
    if trj_xyz.exists():
        with open(trj_xyz) as f:
            lines = f.readlines()
        # Trajectory has multiple xyz frames; take the last one
        frames = []
        i = 0
        while i < len(lines):
            try:
                n = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue
            frame_lines = lines[i:i + n + 2]
            if len(frame_lines) == n + 2:
                frames.append(frame_lines)
            i += n + 2
        if frames:
            return _parse_xyz_lines(frames[-1])

    # Fallback: the overwritten input.xyz
    if opt_xyz.exists():
        return load_xyz(str(opt_xyz))

    return None


def _parse_xyz_lines(lines: list[str]) -> list[tuple]:
    """Parse atoms from xyz-format lines."""
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                atoms.append((parts[0], float(parts[1]),
                              float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return atoms


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    """Select structures and prepare ORCA inputs."""
    selected = select_structures(
        args.xtb_results_dir, args.reference,
        max_per_category=args.max_per_category,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for s in selected:
        struct_dir = prepare_orca_input(
            s["xyz_path"], str(out), s["name"],
            charge=args.charge, mult=args.mult,
            nprocs=args.nprocs, maxcore=args.maxcore,
        )
        manifest.append({
            "name": s["name"],
            "category": s["category"],
            "xtb_energy_hartree": s["xtb_energy"],
            "orca_dir": struct_dir,
        })

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest written: %s (%d structures)", manifest_path, len(manifest))

    # Also write a convenience submission script
    submit_script = out / "submit_all.sh"
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated: submit all DFT jobs\n\n")
        for m in manifest:
            f.write(f'STRUCTURE={m["name"]} sbatch sbatches/dft_orca.sbatch\n')
    os.chmod(str(submit_script), 0o755)
    log.info("Submission script: %s", submit_script)


def cmd_submit(args):
    """Submit SLURM jobs for all structures in work directory."""
    work = Path(args.work_dir)
    manifest_path = work / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No manifest.json in {work}. Run 'prepare' first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    job_ids = []
    for m in manifest:
        name = m["name"]
        struct_dir = Path(m["orca_dir"])
        if not (struct_dir / "input.inp").exists():
            log.warning("Skipping %s — no input.inp", name)
            continue

        env = os.environ.copy()
        env["STRUCTURE"] = name

        try:
            result = subprocess.run(
                ["sbatch", "sbatches/dft_orca.sbatch"],
                capture_output=True, text=True, env=env,
            )
            if result.returncode == 0:
                # Parse job ID from "Submitted batch job 12345"
                match = re.search(r"(\d+)", result.stdout)
                jid = match.group(1) if match else "unknown"
                job_ids.append({"name": name, "job_id": jid})
                log.info("Submitted %s -> job %s", name, jid)
            else:
                log.error("Failed to submit %s: %s", name, result.stderr.strip())
        except FileNotFoundError:
            log.error("sbatch not found — are you on a SLURM cluster?")
            log.info("Use the submit_all.sh script on the cluster instead.")
            return

    # Save job IDs
    if job_ids:
        with open(work / "job_ids.json", "w") as f:
            json.dump(job_ids, f, indent=2)
        log.info("Submitted %d jobs. IDs saved to %s/job_ids.json",
                 len(job_ids), work)


def cmd_parse(args):
    """Parse ORCA outputs and compile comparison CSV."""
    work = Path(args.work_dir)
    manifest_path = work / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No manifest.json in {work}. Run 'prepare' first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Parse all results
    results = []
    ref_energy = None
    ref_n_atoms = None            # reference atom count, for the ΔE formula-match guard
    n_atoms_by_name = {}

    for m in manifest:
        name = m["name"]
        category = m["category"]
        struct_dir = Path(m["orca_dir"])
        log_file = struct_dir / "orca_output.log"

        orca = parse_orca_output(str(log_file))

        # Parse final geometry for donor distances
        final_atoms = parse_orca_final_xyz(str(struct_dir))
        n_atoms_by_name[name] = len(final_atoms) if final_atoms else None
        donors = eu_donor_distances(final_atoms) if final_atoms else []
        coord_donors = [d for d in donors if d["distance_A"] <= DISSOCIATION_CUTOFF]
        dissociated_donors = [d for d in donors if d["distance_A"] > DISSOCIATION_CUTOFF]
        max_eu_donor = max((d["distance_A"] for d in coord_donors), default=None)

        rec = {
            "name": name,
            "category": category,
            "xtb_energy_hartree": m.get("xtb_energy_hartree"),
            "dft_energy_hartree": orca["final_energy_hartree"],
            "dft_converged": orca["converged"],
            "dft_opt_cycles": orca["n_opt_cycles"],
            "mulliken_eu_charge": orca["mulliken_eu_charge"],
            "cn_dft": len(coord_donors),
            "max_eu_donor_A": max_eu_donor,
            "dissociated": len(dissociated_donors) > 0,
            "n_dissociated_donors": len(dissociated_donors),
            "dft_error": orca["error"],
            "eu_donor_distances": [d["distance_A"] for d in coord_donors],
        }
        results.append(rec)

        if category == "reference" and orca["final_energy_hartree"] is not None:
            ref_energy = orca["final_energy_hartree"]
            ref_n_atoms = n_atoms_by_name.get(name)

        status = "OK" if orca["converged"] else (orca["error"] or "not converged")
        log.info("  %-25s  %-15s  E=%-15s  %s",
                 name, category,
                 f"{orca['final_energy_hartree']:.6f}" if orca["final_energy_hartree"] else "N/A",
                 status)

    # Compute relative energies vs reference -- ONLY between identical molecular
    # formulae. The VEDTAA01 reference is heavy-atom-only (35 atoms, no H) while the
    # completions are H-complete (43 atoms), so E - E_ref across them is physically
    # meaningless; suppress ΔE (emit None) on any atom-count mismatch rather than
    # reporting a bogus number (see reports/dft_showcase_result.md caveat).
    if ref_energy is not None:
        for rec in results:
            n_at = n_atoms_by_name.get(rec["name"])
            if rec["dft_energy_hartree"] is not None and n_at == ref_n_atoms:
                delta = (rec["dft_energy_hartree"] - ref_energy) * HARTREE_TO_KCAL
                rec["dft_deltaE_kcal_mol"] = round(delta, 2)
            else:
                rec["dft_deltaE_kcal_mol"] = None
                if rec["dft_energy_hartree"] is not None and n_at != ref_n_atoms:
                    log.warning("  %s: deltaE suppressed (formula mismatch vs reference: "
                                "%s vs %s atoms)", rec["name"], n_at, ref_n_atoms)
        log.info("Reference energy: %.6f Ha (%s atoms)", ref_energy, ref_n_atoms)
    else:
        log.warning("No reference energy — cannot compute relative energies.")
        for rec in results:
            rec["dft_deltaE_kcal_mol"] = None

    # Write CSV
    csv_path = Path(args.output_csv)
    fieldnames = [
        "name", "category", "xtb_energy_hartree", "dft_energy_hartree",
        "dft_deltaE_kcal_mol", "dft_converged", "dft_opt_cycles",
        "mulliken_eu_charge", "cn_dft", "max_eu_donor_A",
        "dissociated", "n_dissociated_donors", "dft_error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info("Comparison CSV: %s", csv_path)

    # Also save full JSON (includes donor distance lists)
    json_path = csv_path.with_suffix(".json")
    # Convert numpy types for JSON serialization
    for rec in results:
        for k, v in rec.items():
            if isinstance(v, (np.floating,)):
                rec[k] = float(v)
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 100)
    print("DFT VALIDATION SUMMARY")
    print("=" * 100)
    print(f"{'Name':<30} {'Category':<18} {'DFT E (Ha)':<16} "
          f"{'dE (kcal/mol)':<14} {'CN':<4} {'Dissoc?':<8} {'Max Eu-D (A)':<13}")
    print("-" * 100)
    for rec in results:
        e_str = f"{rec['dft_energy_hartree']:.6f}" if rec["dft_energy_hartree"] else "N/A"
        de_str = f"{rec['dft_deltaE_kcal_mol']:+.1f}" if rec["dft_deltaE_kcal_mol"] is not None else "N/A"
        max_d = f"{rec['max_eu_donor_A']:.3f}" if rec["max_eu_donor_A"] else "N/A"
        print(f"{rec['name']:<30} {rec['category']:<18} {e_str:<16} "
              f"{de_str:<14} {rec['cn_dft']:<4} {str(rec['dissociated']):<8} {max_d:<13}")
    print("=" * 100)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="DFT validation pipeline for generated Ln complexes",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # --- prepare ---
    p_prep = sub.add_parser("prepare",
                            help="Select structures and prepare ORCA inputs")
    p_prep.add_argument("--xtb-results-dir", required=True,
                        help="Directory with xTB summary CSV(s)")
    p_prep.add_argument("--reference", required=True,
                        help="Reference xyz (CCDC VEDTAA01)")
    p_prep.add_argument("--output-dir", default="dft_work",
                        help="Output directory for ORCA work dirs")
    p_prep.add_argument("--charge", type=int, default=0,
                        help="Net charge (0 for neutral Eu(TMMA)2(NO3)3)")
    p_prep.add_argument("--mult", type=int, default=7,
                        help="Spin multiplicity (7 for Eu3+ 4f^6)")
    p_prep.add_argument("--nprocs", type=int, default=16,
                        help="ORCA parallel procs")
    p_prep.add_argument("--maxcore", type=int, default=3500,
                        help="Memory per core in MB")
    p_prep.add_argument("--max-per-category", type=int, default=2,
                        help="Max structures per failure category")

    # --- submit ---
    p_sub = sub.add_parser("submit", help="Submit SLURM jobs")
    p_sub.add_argument("--work-dir", default="dft_work",
                       help="Directory with prepared ORCA inputs")

    # --- parse ---
    p_parse = sub.add_parser("parse",
                             help="Parse ORCA outputs and compile CSV")
    p_parse.add_argument("--work-dir", default="dft_work",
                         help="Directory with ORCA output logs")
    p_parse.add_argument("--reference", default="refs/eu_tmma_cis.xyz",
                         help="Reference xyz for donor distance comparison")
    p_parse.add_argument("--output-csv", default="dft_comparison.csv",
                         help="Output CSV path")

    args = ap.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "parse":
        cmd_parse(args)


if __name__ == "__main__":
    main()
