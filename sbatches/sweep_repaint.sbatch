#!/bin/bash
#SBATCH -J repaint_sweep
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1 -n 1 -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load miniforge/25.11.0-0 cuda/12.9.1
source activate ligdiff
cd ~/multi_LigandDiff

for R in 1 5 10 20; do
    echo "=== resample_r=$R at $(date) ==="
    NSAMP=500
    if [ $R -eq 20 ]; then NSAMP=250; fi
    python -u generate_mask1.py \
        --outdir generated/repaint_r${R}_epoch48 \
        --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
        --complex eu_tmma_cis.xyz \
        --batch_size 32 \
        --n_samples $NSAMP \
        --ligand_sizes random \
        --resample_r $R
done

echo "=== All complete at $(date) ==="
for R in 1 5 10 20; do
    COUNT=$(ls generated/repaint_r${R}_epoch48/noH/ 2>/dev/null | wc -l)
    echo "r=$R: $COUNT valid complexes"
done
