"""Add hydrogens to generated no-H xyz structures.

Wraps the add_H() logic from generate.py with error handling and
an OpenBabel fallback for structures RDKit rejects.

Usage:
    python add_hydrogens.py --input-dir generated/run1/noH --org-xyz eu_tmma_cis.xyz
    # outputs to generated/run1/add_H/
"""
import argparse, os, sys, subprocess, tempfile, shutil, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def add_h_rdkit(org_xyz: str, gen_dir: str) -> list[str]:
    """Run the project's native add_H pipeline (molSimplify + RDKit).

    Returns list of basenames that were successfully protonated.
    """
    from generate import add_H  # project-local
    add_H(org_xyz, gen_dir)
    out = Path(gen_dir) / "add_H"
    return [f.name for f in out.iterdir() if f.suffix == ".xyz"] if out.exists() else []


def add_h_obabel(xyz_in: str, xyz_out: str) -> bool:
    """Fallback: add hydrogens via OpenBabel CLI (obabel -h)."""
    try:
        r = subprocess.run(
            ["obabel", xyz_in, "-O", xyz_out, "-h"],
            capture_output=True, text=True, timeout=60,
        )
        return os.path.exists(xyz_out) and os.path.getsize(xyz_out) > 0
    except Exception as e:
        log.warning("obabel failed for %s: %s", xyz_in, e)
        return False


def clean_xyz_header(xyz_path: str, out_path: str):
    """Rewrite xyz so line 2 is a comment (not the n_context integer)."""
    with open(xyz_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    with open(out_path, "w") as f:
        f.write(f"{n}\n")
        f.write(f"generated complex\n")
        f.writelines(lines[2:])


def add_hydrogens(input_dir: str, org_xyz: str | None, output_dir: str | None = None):
    """Main entry point.  Returns list of dicts with per-structure results."""
    input_dir = str(Path(input_dir).resolve())
    # gen_dir is parent of noH/
    if Path(input_dir).name == "noH":
        gen_dir = str(Path(input_dir).parent)
    else:
        gen_dir = input_dir
        input_dir = str(Path(gen_dir) / "noH")

    noH_dir = Path(input_dir)
    addH_dir = Path(gen_dir) / "add_H"
    addH_dir.mkdir(parents=True, exist_ok=True)

    if output_dir:
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
    else:
        final_dir = addH_dir

    xyz_files = sorted(noH_dir.glob("*.xyz"))
    if not xyz_files:
        log.warning("No .xyz files found in %s", noH_dir)
        return []

    log.info("Found %d structures in %s", len(xyz_files), noH_dir)

    # --- Step 1: try native add_H (batch, requires org_xyz) ---
    rdkit_ok: set[str] = set()
    if org_xyz and os.path.exists(org_xyz):
        try:
            rdkit_ok = set(add_h_rdkit(org_xyz, gen_dir))
            log.info("RDKit/molSimplify succeeded for %d/%d structures",
                     len(rdkit_ok), len(xyz_files))
        except Exception as e:
            log.warning("Native add_H raised %s — falling back to obabel for all", e)

    # --- Step 2: obabel fallback for failures ---
    results = []
    for xyz in xyz_files:
        name = xyz.name
        rec = {"name": xyz.stem, "file": name, "method": None, "success": False,
               "output_path": None}

        if name in rdkit_ok and (addH_dir / name).exists():
            rec["method"] = "rdkit"
            rec["success"] = True
            rec["output_path"] = str(addH_dir / name)
        else:
            # obabel fallback — needs clean xyz (no context-count line)
            with tempfile.TemporaryDirectory() as tmp:
                clean = os.path.join(tmp, name)
                clean_xyz_header(str(xyz), clean)
                ob_out = str(addH_dir / name)
                if add_h_obabel(clean, ob_out):
                    rec["method"] = "obabel"
                    rec["success"] = True
                    rec["output_path"] = ob_out
                else:
                    rec["method"] = "failed"
                    log.warning("H-addition failed for %s (both rdkit and obabel)", name)

        # Copy to final_dir if different from addH_dir
        if rec["success"] and final_dir != addH_dir:
            dst = final_dir / name
            shutil.copy2(rec["output_path"], str(dst))
            rec["output_path"] = str(dst)

        results.append(rec)

    n_ok = sum(r["success"] for r in results)
    log.info("H-addition complete: %d/%d succeeded (rdkit: %d, obabel: %d, failed: %d)",
             n_ok, len(results),
             sum(1 for r in results if r["method"] == "rdkit"),
             sum(1 for r in results if r["method"] == "obabel"),
             sum(1 for r in results if not r["success"]))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add hydrogens to generated structures")
    ap.add_argument("--input-dir", required=True,
                    help="Directory containing noH xyz files (or parent with noH/ subdir)")
    ap.add_argument("--org-xyz", default=None,
                    help="Original complex xyz (for native RDKit add_H)")
    ap.add_argument("--output-dir", default=None,
                    help="Override output directory (default: sibling add_H/)")
    args = ap.parse_args()
    add_hydrogens(args.input_dir, args.org_xyz, args.output_dir)
