#!/bin/bash
#SBATCH -J ln_maskall
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1 -n 1 -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# True de-novo test: hide ALL ligands, leaving only the bare Eu center as
# context. reform_data then expands over every legal denticity partition of
# CN=10 (42 partitions); with --n_samples 150 that is 150 x 42 = 6300 attempts.
# This is the headline negative result (0 valid / 6300 on the epoch-48 model).
#
# See the nounset note in design_test.sbatch -- same pitfall applies.

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt
COMPLEX=eu_tmma_cis.xyz
OUT=design_test_runs/maskall_${SLURM_JOB_ID}

echo "Start: $(date) on $(hostname)"

python -u generate_design_test.py \
    --complex "$COMPLEX" \
    --model "$CKPT" \
    --outdir "$OUT" \
    --mask_k all \
    --n_samples 150 \
    --batch_size 32 \
    --resample_r 5 \
    --ligand_sizes random

echo "=== maskall valid count ==="
ls "$OUT/maskall/noH/" 2>/dev/null | wc -l
echo "Done: $(date)"
