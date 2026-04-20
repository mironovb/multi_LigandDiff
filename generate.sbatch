#!/bin/bash
#SBATCH -J ln_gen
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
nvidia-smi | head -15

python -u generate.py \
    --outdir generated/eu_tbma_epoch48 \
    --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
    --complex eu_tbma_noH.xyz \
    --batch_size 16 \
    --n_samples 20 \
    --ligand_sizes random

echo "=== Results ==="
ls generated/eu_tbma_epoch48/noH/ 2>/dev/null | wc -l
ls generated/eu_tbma_epoch48/noH/ 2>/dev/null | head -20
echo "Done: $(date)"
