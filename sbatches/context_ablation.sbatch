#!/bin/bash
#SBATCH -J context_abl
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1 -n 1 -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module load miniforge/25.11.0-0 cuda/12.9.1
source activate ligdiff
cd ~/multi_LigandDiff

# Best-practice flags from Prompts 4-5: RePaint r=10 + hard projection
FLAGS="--resample_r 10 --project_enabled True --d_min_start 1.5 --d_min_end 1.3"

# ── Context 1: bare Eu (total generation, CN_c=0) ──────────────────────
# Uses generate_bare.py because there are no ligands to mask.
# Enumerates 6 chemically reasonable LD_g partitions of CN=9.
echo "=== Context 1: bare Eu (total generation) ==="
python -u generate_bare.py \
    --outdir generated/ctx_eu_bare_epoch48 \
    --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
    --complex refs/eu_bare.xyz \
    --target_cn 9 \
    --batch_size 32 \
    --n_samples 500 \
    $FLAGS

# ── Contexts 2-4: mask-one-ligand mode ─────────────────────────────────
for REF in eu_aqua9 eu_no3_3_h2o_3 eu_tmma_cis; do
    echo "=== Context: ${REF} ==="
    python -u generate_mask1.py \
        --outdir generated/ctx_${REF}_epoch48 \
        --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
        --complex refs/${REF}.xyz \
        --batch_size 32 \
        --n_samples 500 \
        $FLAGS
done

echo "=== All context ablation runs complete ==="
