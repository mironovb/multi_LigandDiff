#!/bin/bash
#SBATCH -J ln_smoke_fix
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Tiny GPU smoke test after the code fixes: a handful of mask-1 completions to confirm
# the sampler + (charge-aware) gate + projection + valence guard still run end to end
# and write a rejection_summary.json. NOT a result run -- just "did we break sampling?".
#
# Requires: prompts 01-03 at minimum; safe to run with up to 01-17 landed.
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

CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt
OUT=generated/smoke_fixes_${SLURM_JOB_ID}

python -u generate_design_test.py \
    --complex eu_tmma_cis.xyz \
    --model "$CKPT" \
    --outdir "$OUT" \
    --mask_k 1 \
    --n_samples 5 \
    --batch_size 8 \
    --resample_r 1 \
    --ligand_sizes random

echo "=== smoke valid count + rejection breakdown ==="
ls "$OUT"/mask1/noH/ 2>/dev/null | wc -l
cat "$OUT"/mask1/rejection_summary.json 2>/dev/null || echo "(no rejection_summary.json -- prompt 03 not landed?)"
echo "Done: $(date)"
