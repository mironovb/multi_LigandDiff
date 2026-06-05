#!/bin/bash
#SBATCH -J ln_design
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Design-ability sweep on Eu(TMMA)2(NO3)3: validity vs. how much scaffold is hidden.
# Completes the degradation curve between mask1 (completion, works) and maskall (0).
# mask2 is the point the project still needs (see RESEARCH_PLAN.md change #3), so it
# runs FIRST -- the preemptible wall is 4 h, most valuable experiment first.
#
# NOTE: do NOT combine `set -euo pipefail` with `source ~/.bashrc` on this cluster.
# /etc/bashrc references an unset variable (BASHRCSOURCED) and `nounset` trips on it,
# killing the job at startup (this sank job 14289937). `source activate` is the safe path.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=offline

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
nvidia-smi

CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt
OUT=design_test_runs/sweep_${SLURM_JOB_ID}

# Ordered for the wall: mask2 (missing point) first, then mask3, then mask1 control.
for K in 2 3 1; do
    echo "=== mask_k=$K at $(date) ==="
    python -u generate_design_test.py \
        --complex eu_tmma_cis.xyz \
        --model "$CKPT" \
        --outdir "$OUT" \
        --mask_k $K \
        --n_samples 150 \
        --batch_size 32 \
        --resample_r 5 \
        --ligand_sizes random
done

echo "=== per-level valid counts ==="
for K in 1 2 3; do
    C=$(ls "$OUT/mask${K}/noH/" 2>/dev/null | wc -l)
    echo "mask${K}: $C valid"
done
echo "Done: $(date)"
