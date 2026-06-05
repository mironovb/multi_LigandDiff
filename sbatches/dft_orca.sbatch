#!/bin/bash
#SBATCH -J ln_dft
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# DFT geometry optimisation of one prepared structure with ORCA.
# Level of theory (orca_templates/pbe0_eu.inp): PBE0-D4/def2-TZVP, TightSCF SlowConv,
# Eu via SARC-DKH-TZVP basis + SK-MCDHF-RSC ECP; neutral complex, mult 7 (Eu3+ 4f6).
# CPU job. One structure per job keeps each run inside the 4 h preemptible wall;
# ORCA AutoStart reuses the .gbw on resubmission after preemption.
#
# Prepare inputs first (light, run on the login node):
#   python dft_pipeline.py prepare \
#       --xtb-results-dir xtb_results/eu_tmma_mask1_epoch48 \
#       --reference eu_tmma_cis.xyz --output-dir dft_work --mult 7
# Then submit one job per structure, e.g.:
#   for d in dft_work/*/; do STRUCTURE=$(basename "$d") sbatch sbatches/dft_orca.sbatch; done
# Parse when done:
#   python dft_pipeline.py parse --work-dir dft_work --reference eu_tmma_cis.xyz \
#       --output-csv paper/tables/dft_comparison.csv

export OMP_NUM_THREADS=1   # ORCA parallelises via its own MPI (%pal), not OpenMP

module load miniforge/25.11.0-0
source activate ligdiff
module load orca/5.0.4 2>/dev/null || module load chem/orca/5.0.4 2>/dev/null || \
    echo "WARNING: no ORCA module loaded; assuming orca is on PATH"

cd ~/multi_LigandDiff

: "${STRUCTURE:?Set STRUCTURE=<name> (a directory under dft_work/)}"
WORKDIR="$HOME/multi_LigandDiff/dft_work/$STRUCTURE"

echo "Start: $(date)"
echo "Host: $(hostname)"
ORCA_BIN=$(which orca 2>/dev/null)
echo "ORCA: ${ORCA_BIN:-NOT FOUND}"
if [ -z "$ORCA_BIN" ]; then echo "ERROR: orca not on PATH"; exit 1; fi
if [ ! -d "$WORKDIR" ]; then echo "ERROR: $WORKDIR not found (run prepare first)"; exit 1; fi

cd "$WORKDIR"
export RSH_COMMAND="ssh"

# ORCA needs its absolute path for MPI startup.
"$ORCA_BIN" input.inp > orca_output.log 2>&1
echo "ORCA exit: $?"

if grep -q "THE OPTIMIZATION HAS CONVERGED" orca_output.log; then
    echo "STATUS: CONVERGED"
    grep "FINAL SINGLE POINT ENERGY" orca_output.log | tail -1
elif grep -q "ORCA TERMINATED NORMALLY" orca_output.log; then
    echo "STATUS: terminated normally (opt may be incomplete -- resubmit to continue)"
    grep "FINAL SINGLE POINT ENERGY" orca_output.log | tail -1
else
    echo "STATUS: did not terminate normally"; tail -20 orca_output.log
fi
echo "Done: $(date)"
