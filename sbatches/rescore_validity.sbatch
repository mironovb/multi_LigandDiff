#!/bin/bash
#SBATCH -J ln_rescore
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# The decisive no-GPU experiment (code review Finding 2). Re-score the pristine
# Kravchuk reference AND every existing generated set through the now charge-aware
# validity gate (prompts 01-02). If the reference's three nitrates now pass where they
# previously failed, part of the headline 0/6300 was instrumental, not the model.
#
# Requires: prompt 04 (rescore_validity.py). CPU only.
# nounset trap: do not `set -euo pipefail` + `source ~/.bashrc`.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load miniforge/25.11.0-0
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"

mkdir -p metrics

python -u rescore_validity.py \
    --reference eu_tmma_cis.xyz \
    --inputs generated/eu_tmma_mask1_epoch48/noH \
             generated/repaint_r5_epoch48/noH \
             design_test_runs/*/mask*/noH \
             design_test_runs/maskall_*/maskall/noH \
    --out metrics/rescore.csv

echo "=== reference verdict (the headline) ==="
grep -E "^eu_tmma_cis|reference" metrics/rescore.csv || true
echo "Done: $(date)"
