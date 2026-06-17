#!/bin/bash
#SBATCH -J ln_train_curric
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Re-finetune with the high-mask curriculum (prompt 18) so the model actually SEES the
# bare-metal / high-mask-count task during training -- the regime de-novo fails in.
#
# STEP 0 (CPU, run FIRST on the login node or a CPU job -- not here): re-prep the
# training data with the curriculum augmentation so the all-masked subset is guaranteed
# and high-mask configs are over-represented:
#
#   python -u prepare_training_data.py --cif_dir <cifs> \
#       --output_dir data_curriculum \
#       --mask_curriculum quadratic --force_all_masked --max_augment 30 --workers 16
#   (writes data_curriculum/train_ln.pt + val_ln.pt; the val split is --val_fraction,
#    default 0.1 -- there is no separate train/val path flag, only --output_dir.
#    --force_all_masked is a store_true flag (default on); --no_force_all_masked ablates it.)
#
# Then submit this job. Like the original fine-tune it spans the 4 h preemptible wall
# across resubmissions; finetune.py supports --resume_from_checkpoint for that.
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

TRAIN=data_curriculum/train_ln.pt
VAL=data_curriculum/val_ln.pt
if [ ! -f "$TRAIN" ]; then
    echo "ERROR: $TRAIN not found -- run STEP 0 (curriculum re-prep) first."; exit 1
fi

# Resume if a checkpoint from a previous wall-limited session exists.
RESUME=""
LAST=$(ls -t models/ln_finetuned_curriculum/*.ckpt 2>/dev/null | head -1)
[ -n "$LAST" ] && RESUME="--resume_from_checkpoint $LAST" && echo "Resuming from $LAST"

python -u finetune.py \
    --train_data "$TRAIN" \
    --val_data "$VAL" \
    --pretrained model/pre_trained.ckpt \
    --epochs 100 \
    --batch_size 256 \
    --num_gpus 1 \
    --early_stop_patience 15 \
    --output_dir models/ln_finetuned_curriculum \
    $RESUME

echo "Done: $(date)"
