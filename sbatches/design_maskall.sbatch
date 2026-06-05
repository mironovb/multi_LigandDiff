#!/bin/bash
#SBATCH -J ln_maskall
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# True de-novo test: hide ALL ligands, leaving only the bare Eu center as context.
# reform_data expands over every legal denticity partition of CN=10 (42 partitions);
# with --n_samples 150 that is 150 x 42 = 6300 attempts.
# Headline negative result of record: 0 valid / 6300 on the epoch-48 model.
# See the nounset note in design_test.sbatch -- same pitfall applies.

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
OUT=design_test_runs/maskall_${SLURM_JOB_ID}

python -u generate_design_test.py \
    --complex eu_tmma_cis.xyz \
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
