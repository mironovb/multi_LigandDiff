"""
End-to-end xTB optimization pipeline for generated Ln complexes.

For each structure: add Hs -> run GFN2-xTB -> collect results.
Parallelised with multiprocessing.Pool.

Usage:
    python xtb_pipeline.py \
        --input-dir generated/eu_tmma_mask1_epoch48/noH \
        --output-dir xtb_results/mask1 \
        --org-xyz eu_tmma_cis.xyz \
        --nproc 8 \
        --timeout-minutes 10
"""
import argparse, csv, glob, json, logging, os, shutil, subprocess, tempfile
from multiprocessing import Pool
from pathlib import Path

from add_hydrogens import add_hydrogens
from guess_charge import guess_charge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# xTB runner (per-structure)
# ---------------------------------------------------------------------------

def _clean_xyz(src: str, dst: str):
    """Write a standard xyz (comment on line 2) from our format."""
    with open(src) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    with open(dst, "w") as f:
        f.write(f"{n}\n")
        f.write(f"Ln complex\n")
        f.writelines(lines[2:2 + n])


def _parse_energy(stdout: str) -> float | None:
    """Extract last TOTAL ENERGY from xTB stdout."""
    energy = None
    for line in stdout.split("\n"):
        if "TOTAL ENERGY" in line:
            for tok in line.split():
                try:
                    energy = float(tok)
                    break
                except ValueError:
                    pass
    return energy


def _parse_cycles(stdout: str) -> int | None:
    """Extract number of optimisation cycles from xTB stdout."""
    for line in reversed(stdout.split("\n")):
        if "GEOMETRY OPTIMIZATION CONVERGED" in line:
            return None  # converged, but cycle count parsed below
        if line.strip().startswith("*** GEOMETRY OPTIMIZATION CONVERGED"):
            break
    # look for "converged in N iterations"
    for line in stdout.split("\n"):
        if "converged in" in line.lower():
            for tok in line.split():
                try:
                    return int(tok)
                except ValueError:
                    pass
    return None


def _classify_failure(stderr: str, stdout: str) -> str:
    """Return a short failure-mode label."""
    if "IEEE_INVALID_FLAG" in stderr or "IEEE_INVALID_FLAG" in stdout:
        return "IEEE_error"
    if "FAILED TO CONVERGE" in stdout.upper() or "NOT CONVERGED" in stdout.upper():
        return "not_converged"
    if "SCC not converged" in stdout:
        return "SCC_not_converged"
    if stderr:
        return "xtb_error"
    return "unknown"


def run_xtb_single(args_tuple):
    """Run xTB on one structure.  Designed for Pool.map."""
    xyz_path, work_dir, charge, uhf, timeout_s = args_tuple
    name = Path(xyz_path).stem
    work = Path(work_dir) / name
    work.mkdir(parents=True, exist_ok=True)

    rec = {
        "name": name,
        "h_added_method": None,
        "charge_guess": charge,
        "xtb_success": False,
        "n_cycles": None,
        "final_energy_hartree": None,
        "converged_xyz_path": None,
        "failure_mode": None,
    }

    # Clean xyz for xTB
    clean = work / "input.xyz"
    try:
        _clean_xyz(xyz_path, str(clean))
    except Exception as e:
        rec["failure_mode"] = f"xyz_parse_error: {e}"
        return rec

    cmd = [
        "xtb", "input.xyz",
        "--opt", "normal",
        "--gfn", "2",
        "--chrg", str(charge),
        "--uhf", str(uhf),
        "--cycles", "500",
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(work), capture_output=True, text=True, timeout=timeout_s,
        )
        opt_xyz = work / "xtbopt.xyz"
        if opt_xyz.exists():
            rec["xtb_success"] = True
            rec["final_energy_hartree"] = _parse_energy(result.stdout)
            rec["n_cycles"] = _parse_cycles(result.stdout)
            rec["converged_xyz_path"] = str(opt_xyz)
        else:
            rec["failure_mode"] = _classify_failure(result.stderr, result.stdout)
    except subprocess.TimeoutExpired:
        rec["failure_mode"] = "timeout"
    except Exception as e:
        rec["failure_mode"] = f"exception: {e}"

    # Save per-structure JSON
    with open(work / "result.json", "w") as f:
        json.dump(rec, f, indent=2)

    return rec


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(input_dir: str, output_dir: str, org_xyz: str | None,
                 nproc: int, timeout_min: float, charge: int, uhf: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    converged_dir = out / "converged"
    converged_dir.mkdir(exist_ok=True)

    # --- Step 1: Add hydrogens ---
    log.info("=== Step 1: Adding hydrogens ===")
    h_dir = out / "with_H"
    h_results = add_hydrogens(input_dir, org_xyz, str(h_dir))
    h_method_map = {r["name"]: r["method"] for r in h_results}

    h_xyz_files = sorted(h_dir.glob("*.xyz")) if h_dir.exists() else []
    if not h_xyz_files:
        log.error("No H-added structures produced. Aborting.")
        return []

    log.info("Proceeding with %d H-added structures", len(h_xyz_files))

    # --- Step 2: Guess charges (log only) ---
    log.info("=== Step 2: Charge guesses (informational) ===")
    for xyz in h_xyz_files[:5]:  # sample
        ch, reason = guess_charge(str(xyz), verbose=True)

    # --- Step 3: Run xTB in parallel ---
    log.info("=== Step 3: xTB optimization (nproc=%d, timeout=%d min) ===", nproc, timeout_min)
    timeout_s = int(timeout_min * 60)
    work_dir = str(out / "xtb_work")

    tasks = [
        (str(xyz), work_dir, charge, uhf, timeout_s)
        for xyz in h_xyz_files
    ]

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            results = pool.map(run_xtb_single, tasks)
    else:
        results = [run_xtb_single(t) for t in tasks]

    # Annotate h-addition method
    for rec in results:
        rec["h_added_method"] = h_method_map.get(rec["name"])

    # --- Step 4: Collect results ---
    n_success = sum(r["xtb_success"] for r in results)
    log.info("=== Results: %d/%d converged (%.0f%%) ===",
             n_success, len(results),
             100 * n_success / len(results) if results else 0)

    # Failure breakdown
    from collections import Counter
    failures = Counter(r["failure_mode"] for r in results if not r["xtb_success"])
    for mode, cnt in failures.most_common():
        log.info("  failure: %-25s %d", mode, cnt)

    # --- Step 5: Write CSV ---
    csv_path = out / "summary.csv"
    fieldnames = [
        "name", "h_added_method", "charge_guess", "xtb_success",
        "n_cycles", "final_energy_hartree", "failure_mode",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info("Summary CSV: %s", csv_path)

    # Also write full JSON
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Step 6: Copy converged structures ---
    energies = []
    for rec in results:
        if rec["xtb_success"] and rec["converged_xyz_path"]:
            src = rec["converged_xyz_path"]
            dst = converged_dir / f"{rec['name']}.xyz"
            if os.path.exists(src):
                shutil.copy2(src, str(dst))
            if rec["final_energy_hartree"] is not None:
                energies.append((rec["final_energy_hartree"], rec["name"]))

    log.info("Converged structures copied to %s", converged_dir)

    # --- Step 7: Top 10 by energy ---
    if energies:
        top10_dir = out / "top10_for_figure"
        top10_dir.mkdir(exist_ok=True)
        energies.sort()  # lowest energy first
        for e, name in energies[:10]:
            src = converged_dir / f"{name}.xyz"
            if src.exists():
                shutil.copy2(str(src), str(top10_dir / f"{name}.xyz"))
            log.info("  top10: %s  E=%.6f Ha", name, e)

    return results


# ---------------------------------------------------------------------------
# Aggregate across multiple runs
# ---------------------------------------------------------------------------

def aggregate_summaries(base_dir: str):
    """Merge summary.csv from all subdirectories into summary_all.csv."""
    base = Path(base_dir)
    all_rows = []
    for csv_file in sorted(base.rglob("summary.csv")):
        run_name = csv_file.parent.name
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["run"] = run_name
                all_rows.append(row)

    if not all_rows:
        log.warning("No summary.csv files found under %s", base_dir)
        return

    out_path = base / "summary_all.csv"
    fieldnames = ["run"] + [k for k in all_rows[0] if k != "run"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    log.info("Aggregated %d records -> %s", len(all_rows), out_path)

    # Print convergence table
    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "converged": 0})
    for row in all_rows:
        run = row["run"]
        stats[run]["total"] += 1
        if row.get("xtb_success", "").lower() == "true":
            stats[run]["converged"] += 1

    print("\n=== xTB Convergence Rates ===")
    print(f"{'Run':<40} {'Converged':>10} {'Total':>8} {'Rate':>8}")
    print("-" * 70)
    for run in sorted(stats):
        s = stats[run]
        rate = 100 * s["converged"] / s["total"] if s["total"] else 0
        print(f"{run:<40} {s['converged']:>10} {s['total']:>8} {rate:>7.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end xTB pipeline for generated Ln complexes",
    )
    ap.add_argument("--input-dir", required=True,
                    help="Directory of noH xyz files (or parent containing noH/)")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for results")
    ap.add_argument("--org-xyz", default="eu_tmma_cis.xyz",
                    help="Original complex xyz for RDKit H-addition")
    ap.add_argument("--nproc", type=int, default=4,
                    help="Number of parallel xTB workers")
    ap.add_argument("--timeout-minutes", type=float, default=15,
                    help="Per-structure xTB timeout in minutes")
    ap.add_argument("--charge", type=int, default=0,
                    help="Net charge of complex (default 0)")
    ap.add_argument("--uhf", type=int, default=6,
                    help="Number of unpaired electrons (default 6 for Eu3+ f6)")
    ap.add_argument("--aggregate", action="store_true",
                    help="Instead of running, aggregate existing summary.csv files")
    args = ap.parse_args()

    if args.aggregate:
        aggregate_summaries(args.output_dir)
    else:
        run_pipeline(
            args.input_dir, args.output_dir, args.org_xyz,
            args.nproc, args.timeout_minutes, args.charge, args.uhf,
        )


if __name__ == "__main__":
    main()
