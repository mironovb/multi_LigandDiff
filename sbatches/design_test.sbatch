#!/bin/bash
#SBATCH -J ln_design
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1 -n 1 -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Design-ability sweep on Eu(TMMA)2(NO3)3: how does validity degrade as the
# scaffold shrinks? Completes the degradation curve between mask1 (completion,
# works) and maskall (0/6300). The dedicated mask2 point is the one the project
# still needs (see RESEARCH_PLAN.md, change #3) so it runs FIRST -- the
# preemptible partition has a hard 4 h wall, so the most valuable experiment
# must come first.
#
# NOTE: do NOT use `set -euo pipefail` together with `source ~/.bashrc` here.
# The cluster's /etc/bashrc references an unset variable (BASHRCSOURCED), and
# `nounset` trips on it, killing the job at startup (this sank job 14289937).
# `source activate` below is the safe activation path.

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt
COMPLEX=eu_tmma_cis.xyz
OUT=design_test_runs/sweep_${SLURM_JOB_ID}
R=5                 # RePaint working point (denticity-match peaks at r=5)
NSAMP=150           # seeds per masked subset

echo "Start: $(date) on $(hostname)"

# Ordered for the wall limit: mask2 (the missing point) first, then mask3,
# then the mask1 control. maskall (0/6300) has its own longer job below.
for K in 2 3 1; do
    echo "=== mask_k=$K at $(date) ==="
    python -u generate_design_test.py \
        --complex "$COMPLEX" \
        --model "$CKPT" \
        --outdir "$OUT" \
        --mask_k $K \
        --n_samples $NSAMP \
        --batch_size 32 \
        --resample_r $R \
        --ligand_sizes random
done

echo "=== Per-level valid counts ==="
for K in 1 2 3; do
    C=$(ls "$OUT/mask${K}/noH/" 2>/dev/null | wc -l)
    echo "mask${K}: $C valid"
done
echo "Done: $(date)"
