#!/bin/bash
#SBATCH -J ln_repaint
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# RePaint resampling sweep (mask1 completion on Eu(TMMA)2(NO3)3).
# Yield rises monotonically with r; denticity-match peaks at r=5 (working point).
# Results of record: r=1 29/2500 (1.16%), r=5 85/2500 (3.40%),
#                    r=10 57/1500 (3.80%), r=20 39/750 (5.20%).
#
# The preemptible wall is 4 h. A full r in {1,5,10,20} at these sample counts
# does not fit one job -- split it:
#     RVALS="1 5"   sbatch sbatches/repaint_sweep.sbatch
#     RVALS="10 20" sbatch sbatches/repaint_sweep.sbatch
# RVALS defaults to "1 5 10 20"; override via the environment as above.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=offline

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
nvidia-smi

RVALS="${RVALS:-1 5 10 20}"
CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt

for R in $RVALS; do
    case $R in
        20) NSAMP=750 ;;
        10) NSAMP=1500 ;;
        *)  NSAMP=2500 ;;
    esac
    echo "=== resample_r=$R  n_samples=$NSAMP  at $(date) ==="
    python -u generate_mask1.py \
        --outdir generated/repaint_r${R}_epoch48 \
        --model "$CKPT" \
        --complex eu_tmma_cis.xyz \
        --batch_size 32 \
        --n_samples $NSAMP \
        --ligand_sizes random \
        --resample_r $R
done

echo "=== valid counts ==="
for R in $RVALS; do
    C=$(ls generated/repaint_r${R}_epoch48/noH/ 2>/dev/null | wc -l)
    echo "r=$R: $C valid"
done
echo "Done: $(date)"
