#!/bin/bash
#SBATCH -J ln_dft_showcase
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Track A, run 2/2 (prompt 25): DFT-validated completion showcase LAUNCHER.
# DFT is too heavy to batch -- one PBE0-D4/def2-TZVP opt of a CN=10 Eu complex does not
# share a 4 h wall. So this is a light job that (1) prepares ORCA inputs for the most
# xTB-stable mask-1 completions (up to --max-per-category per stability category), then
# (2) submits ONE dft_orca.sbatch per structure
# (each resumable via ORCA AutoStart). Parse afterwards with dft_pipeline.py parse.
#
# HONESTY GATE (experiment log): DFT has never run; only the template exists. Do not
# claim "DFT-confirmed" until orca_output.log files actually show convergence.
# nounset trap: do not `set -euo pipefail` + `source ~/.bashrc`.

export OMP_NUM_THREADS=1

module load miniforge/25.11.0-0
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"

XTB_DIR=xtb_results/eu_tmma_mask1_epoch48
if [ ! -d "$XTB_DIR" ]; then
    echo "ERROR: $XTB_DIR not found -- run the xTB stage (sbatches/xtb_batch.sbatch) first."; exit 1
fi

# 1) prepare inputs (up to --max-per-category per category; neutral complex, Eu3+ 4f6 -> mult 7)
python -u dft_pipeline.py prepare \
    --xtb-results-dir "$XTB_DIR" \
    --reference eu_tmma_cis.xyz \
    --output-dir dft_work \
    --mult 7 \
    --max-per-category 2

# 2) submit one ORCA job per prepared structure
echo "=== submitting per-structure ORCA jobs ==="
for d in dft_work/*/; do
    STRUCTURE=$(basename "$d")
    [ -f "$d/input.inp" ] || { echo "skip $STRUCTURE (no input.inp)"; continue; }
    STRUCTURE="$STRUCTURE" sbatch sbatches/dft_orca.sbatch && echo "submitted: $STRUCTURE"
done

echo "After they finish, parse with:"
echo "  python dft_pipeline.py parse --work-dir dft_work --reference eu_tmma_cis.xyz --output-csv paper/tables/dft_comparison.csv"
echo "Done: $(date)"
