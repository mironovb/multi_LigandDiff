#!/bin/bash
#SBATCH -J proj_stack
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1 -n 1 -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load miniforge/25.11.0-0 cuda/12.9.1
source activate ligdiff
cd ~/multi_LigandDiff

# Cell 1: projection ON, r=1
echo "=== proj ON, r=1 at $(date) ==="
python -u generate_mask1.py \
    --outdir generated/proj_r1_epoch48 \
    --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
    --complex eu_tmma_cis.xyz \
    --batch_size 32 \
    --n_samples 500 \
    --ligand_sizes random \
    --resample_r 1 \
    --project_enabled True \
    --d_min_start 1.5 \
    --d_min_end 1.3

# Cell 2: projection ON, r=10 (the stack)
echo "=== proj ON, r=10 at $(date) ==="
python -u generate_mask1.py \
    --outdir generated/proj_r10_epoch48 \
    --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
    --complex eu_tmma_cis.xyz \
    --batch_size 32 \
    --n_samples 500 \
    --ligand_sizes random \
    --resample_r 10 \
    --project_enabled True \
    --d_min_start 1.5 \
    --d_min_end 1.3

echo "=== All complete at $(date) ==="
for CELL in proj_r1_epoch48 proj_r10_epoch48; do
    COUNT=$(ls generated/${CELL}/noH/ 2>/dev/null | wc -l)
    echo "${CELL}: $COUNT valid complexes"
done
