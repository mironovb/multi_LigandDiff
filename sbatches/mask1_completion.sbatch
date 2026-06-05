#!/bin/bash
#SBATCH -J ln_mask1
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Mask1 completion baseline on Eu(TMMA)2(NO3)3: fix the full complex, hide ONE
# ligand, regenerate it. Produces the baseline set that feeds the xTB stage.
# Result of record: 41 valid (job 12207365), xTB-stable 38/41 = 92.7%.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=offline

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
nvidia-smi

python -u generate_mask1.py \
    --outdir generated/eu_tmma_mask1_epoch48 \
    --model models/ln_finetuned/ln_finetuned_epoch=48.ckpt \
    --complex eu_tmma_cis.xyz \
    --batch_size 32 \
    --n_samples 500 \
    --ligand_sizes random \
    --resample_r 1

echo "=== valid count ==="
ls generated/eu_tmma_mask1_epoch48/noH/ 2>/dev/null | wc -l
echo "Done: $(date)"
