#!/bin/bash
#SBATCH -J dblock_cross
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1 -n 1 -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

module load miniforge/25.11.0-0 cuda/12.9.1
source activate ligdiff
cd ~/multi_LigandDiff

# ── Cross-architecture replication (Prompt 7) ─────────────────────────
# Tests the original d-block pretrained checkpoint on CN=10 geometries.
#
# Key finding from code inspection: Eu IS in the metal vocabulary and
# metals get all-zero one-hot encoding, so the d-block model CAN process
# Eu inputs.  The real question is whether a model trained on CN<=6
# d-block complexes produces valid structures at CN=10.
#
# Three experiments:
#   1. d-block pretrained on Eu-TMMA (CN=10, vanilla sampling)
#   2. d-block pretrained on Fe-TMMA (CN=10, rescaled coordination sphere)
#   3. d-block pretrained on Eu-TMMA with best-practice flags (r=10 + projection)

DBLOCK="model/pre_trained.ckpt"
FLAGS_BEST="--resample_r 10 --project_enabled True --d_min_start 1.5 --d_min_end 1.3"

# ── Experiment 1: d-block pretrained, Eu-TMMA, vanilla ────────────────
echo "=== Exp 1: d-block pretrained on Eu-TMMA (vanilla, r=1) ==="
python -u generate_mask1.py \
    --outdir generated/dblock_eu_tmma_r1 \
    --model $DBLOCK \
    --complex refs/eu_tmma_cis.xyz \
    --batch_size 32 \
    --n_samples 200 \
    --resample_r 1 \
    --project_enabled False

# ── Experiment 2: d-block pretrained, Fe-TMMA substituted, vanilla ────
echo "=== Exp 2: d-block pretrained on Fe-TMMA (vanilla, r=1) ==="
python -u generate_mask1.py \
    --outdir generated/dblock_fe_tmma_r1 \
    --model $DBLOCK \
    --complex refs/fe_tmma_substituted.xyz \
    --batch_size 32 \
    --n_samples 200 \
    --resample_r 1 \
    --project_enabled False

# ── Experiment 3: d-block pretrained, Eu-TMMA, best-practice flags ────
echo "=== Exp 3: d-block pretrained on Eu-TMMA (r=10 + projection) ==="
python -u generate_mask1.py \
    --outdir generated/dblock_eu_tmma_r10_proj \
    --model $DBLOCK \
    --complex refs/eu_tmma_cis.xyz \
    --batch_size 32 \
    --n_samples 200 \
    $FLAGS_BEST

echo "=== All d-block cross-architecture runs complete ==="
