#!/bin/bash
#SBATCH -J ln_train_fixed
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# CLEAN re-fine-tune that fixes the normalization_factor bug (prompts 19-20) WITHOUT the
# high-mask curriculum -- the controlled checkpoint for the degradation curve.
#
# WHY THIS EXISTS: the shipped ln_finetuned_epoch=48.ckpt stores normalization_factor=1.
# It was warm-started from model/pre_trained.ckpt (which stores 100) into a factor-1
# fine-tune, so every transferred GVP layer ran at 100x the aggregation scale its weights
# were trained for (conv_layer.py divides summed messages by this factor). It samples
# self-consistently, but it is a BUG-TRAINED model. finetune.py:202 now passes 100, so a
# fresh warm-start from pre_trained.ckpt realizes the intended transfer.
#
# This variant uses the LEGACY (uniform) masking distribution so the resulting checkpoint
# is apples-to-apples with the original mask-1/2/3 degradation curve -- it isolates the
# normalization + bond-cutoff (prompt 14) corrections from the curriculum regime change.
# For the bare-metal/de-novo headline, use sbatches/finetune_curriculum.sbatch instead
# (same factor=100, but trains the high-mask regime de-novo needs).
#
# STEP 0 (CPU, run FIRST -- login node or a CPU job, not here): re-prep with the corrected
# src/bonding.py cutoffs (prompt 14) and LEGACY masking:
#
#   python -u prepare_training_data.py --cif_dir <cifs> \
#       --output_dir data_fixed \
#       --mask_curriculum uniform --no_force_all_masked --max_augment 30 --workers 16
#   (writes data_fixed/train_ln.pt + val_ln.pt; --no_force_all_masked reproduces the
#    legacy uniform draw so this is a clean control vs the old epoch=48 curve.)
#
# Then submit this job. Spans the 4 h wall across resubmissions via --resume_from_checkpoint.
# nounset trap: do not `set -euo pipefail` + `source ~/.bashrc`.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=offline

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
nvidia-smi

TRAIN=data_fixed/train_ln.pt
VAL=data_fixed/val_ln.pt
if [ ! -f "$TRAIN" ]; then
    echo "ERROR: $TRAIN not found -- run STEP 0 (clean re-prep) first."; exit 1
fi

# Resume if a checkpoint from a previous wall-limited session exists.
RESUME=""
LAST=$(ls -t models/ln_finetuned_fixed/*.ckpt 2>/dev/null | head -1)
[ -n "$LAST" ] && RESUME="--resume_from_checkpoint $LAST" && echo "Resuming from $LAST"

python -u finetune.py \
    --train_data "$TRAIN" \
    --val_data "$VAL" \
    --pretrained model/pre_trained.ckpt \
    --epochs 100 \
    --batch_size 256 \
    --num_gpus 1 \
    --early_stop_patience 15 \
    --output_dir models/ln_finetuned_fixed \
    $RESUME

echo "Done: $(date)"
echo "Re-point the design sbatches' CKPT at the best models/ln_finetuned_fixed/*.ckpt"
echo "and confirm it stores normalization_factor=100:"
echo "  python -c \"import torch; print(torch.load('<ckpt>', map_location='cpu', weights_only=False)['hyper_parameters']['normalization_factor'])\""
