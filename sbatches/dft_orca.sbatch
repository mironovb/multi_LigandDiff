#!/bin/bash
#SBATCH -J dft_eu
#SBATCH -A mit_general
#SBATCH -p mit_normal
#SBATCH -N 1 -n 1 -c 16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o dft_%x_%j.out
#SBATCH -e dft_%x_%j.err

# DFT (ORCA PBE0-D4/def2-TZVP) single-structure optimisation for Eu complexes
# Submit with:  STRUCTURE=<name> sbatch sbatches/dft_orca.sbatch
# Or use:       python dft_pipeline.py submit --work-dir dft_work

echo "=== DFT ORCA optimisation ==="
echo "STRUCTURE = ${STRUCTURE:?'Set STRUCTURE env var (structure name in dft_work/)'}"
echo "Start: $(date)"
echo "Node:  $(hostname)"
echo "Cores: $SLURM_CPUS_PER_TASK"

# --- Environment ---
# Adjust module name to match your cluster's ORCA installation.
# Common variants: orca/5.0.4, orca/5.0.3, chem/orca/5.0.4
module load orca/5.0.4 2>/dev/null || \
module load chem/orca/5.0.4 2>/dev/null || \
echo "WARNING: could not load ORCA module — assuming ORCA is in PATH"

# Verify ORCA is available
ORCA_BIN=$(which orca 2>/dev/null)
if [ -z "$ORCA_BIN" ]; then
    echo "ERROR: orca not found in PATH after module load. Check module name."
    echo "Available ORCA modules:"
    module avail orca 2>&1 | head -20
    exit 1
fi
echo "ORCA binary: $ORCA_BIN"
echo "ORCA version: $($ORCA_BIN --version 2>&1 | head -1 || echo 'unknown')"

# --- Run ---
WORKDIR="$HOME/multi_LigandDiff/dft_work/$STRUCTURE"

if [ ! -d "$WORKDIR" ]; then
    echo "ERROR: work directory not found: $WORKDIR"
    echo "Run 'python dft_pipeline.py prepare ...' first."
    exit 1
fi

cd "$WORKDIR"
echo "Working directory: $(pwd)"
echo "Input file: $(ls -la input.inp)"

# ORCA uses its own MPI; set parallel env
export RSH_COMMAND="ssh"

# Run ORCA
$ORCA_BIN input.inp > orca_output.log 2>&1
ORCA_EXIT=$?

echo "ORCA exit code: $ORCA_EXIT"
echo "Output size: $(wc -l < orca_output.log) lines"

# Quick convergence check
if grep -q "THE OPTIMIZATION HAS CONVERGED" orca_output.log; then
    echo "STATUS: Geometry optimisation CONVERGED"
    grep "FINAL SINGLE POINT ENERGY" orca_output.log | tail -1
elif grep -q "SCF NOT CONVERGED" orca_output.log; then
    echo "STATUS: SCF NOT CONVERGED"
elif grep -q "ORCA TERMINATED NORMALLY" orca_output.log; then
    echo "STATUS: ORCA terminated normally (opt may not have converged)"
    grep "FINAL SINGLE POINT ENERGY" orca_output.log | tail -1
else
    echo "STATUS: ORCA did not terminate normally"
    tail -20 orca_output.log
fi

echo "=== Done at $(date) ==="

# --- Example submission ---
# Single structure:
#   STRUCTURE=VEDTAA01_reference sbatch sbatches/dft_orca.sbatch
#
# All structures (via generated script):
#   bash dft_work/submit_all.sh
#
# All structures (via pipeline):
#   python dft_pipeline.py submit --work-dir dft_work
