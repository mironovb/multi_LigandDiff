#!/bin/bash
#SBATCH -J ln_train_h200
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=offline

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
nvidia-smi

python -u finetune.py \
    --train_data data/train_ln.pt \
    --val_data data/val_ln.pt \
    --pretrained model/pre_trained.ckpt \
    --epochs 100 \
    --batch_size 256 \
    --num_gpus 1 \
    --early_stop_patience 15 \
    --output_dir models/ln_finetuned

echo "Done: $(date)"
