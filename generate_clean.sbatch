#!/bin/bash
#SBATCH -J ln_gen_clean
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"

python -u generate.py \
    --outdir generated/eu_tmma_real_epoch48 \
    --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
    --complex eu_tmma_cis.xyz \
    --batch_size 32 \
    --n_samples 50 \
    --ligand_sizes random

echo "=== Results ==="
ls generated/eu_tmma_real_epoch48/noH/ 2>/dev/null | wc -l
ls generated/eu_tmma_real_epoch48/noH/ 2>/dev/null | head -30
echo "Done: $(date)"
