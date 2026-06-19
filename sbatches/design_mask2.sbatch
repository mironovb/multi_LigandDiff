#!/bin/bash
#SBATCH -J ln_mask2
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Track A, run 1/2 (prompt 24): the dedicated mask-2 completion run that finishes the
# degradation curve. mask 2 hides 3 of 5 ligands -> C(5,2)=10 masked subsets per seed.
# The old mask2=4 was only what survived a 4 h cutoff in a combined sweep; this runs
# mask 2 ALONE to completion so 126 -> <mask2> -> 0 -> 0 becomes a measured curve.
# Uses the charge-aware gate (prompts 01-02) so the count is the fixed-instrument count.
#
# Sizing: uniform denticity_prior (default) enumerates EVERY eligible partition per masked
# subset -- mask-2 = C(5,2)=10 subsets, each remaining_cn=4 -> 5 partitions = 50 eligible.
# So attempts = n_samples x 50.  n_samples 50 -> 2500 attempts == the mask-1 control's load
# (uniform, 2500, completed), comfortably inside one 4 h wall and count-comparable to mask 1.
# (n_samples scales attempts x50: 250 would be 12,500 and risk the wall cutoff this run avoids.)
# nounset trap: do not `set -euo pipefail` + `source ~/.bashrc`.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=offline

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
nvidia-smi

# !! CHECKPOINT CAVEAT (normalization_factor bug, prompts 19-20) -- read docs/H200_RUNBOOK.md.
# epoch=48.ckpt stores normalization_factor=1 (warm-start scrambled the factor=100 backbone). It
# samples self-consistently, so THIS run is valid and apples-to-apples with the old mask-2 point --
# but it is a BUG-TRAINED model. For the clean degradation curve, first produce a factor=100
# checkpoint on legacy (uniform) masking (sbatches/finetune_fixed.sbatch) and re-point CKPT.
CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt   # bug-trained (factor=1); preliminary number
# CKPT=models/ln_finetuned_fixed/<best>.ckpt          # <- clean factor=100, legacy masking (recommended for the curve)
OUT=design_test_runs/mask2_${SLURM_JOB_ID}

python -u generate_design_test.py \
    --complex eu_tmma_cis.xyz \
    --model "$CKPT" \
    --outdir "$OUT" \
    --mask_k 2 \
    --n_samples 50 \
    --batch_size 32 \
    --resample_r 5 \
    --ligand_sizes random \
    --max_denticity 4

echo "=== mask2 valid count ==="
ls "$OUT"/mask2/noH/ 2>/dev/null | wc -l
cat "$OUT"/mask2/accounting.json 2>/dev/null || true
echo "Done: $(date)"
