#!/bin/bash
#SBATCH -J ln_maskall_fixed
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# The de-novo RE-TEST (prompt 23). Same bare-Eu task that gave 0 valid / 6300, but with
# every instrument/denominator/sampler fix turned ON, so the new number separates model
# incapacity from instrument brittleness:
#   - charge-aware validity gate           (prompts 01-02)  -> nitrate can pass
#   - per-reason rejection_summary.json     (prompt 03)      -> see WHERE attempts die
#   - denticity cap <=4 + CSD prior         (prompts 05-07)  -> honest denominator
#   - donor-identity / composition spec     (prompts 08-10)  -> well-posed target
#   - projection above bond cutoff + gen<->gen (prompts 11-13) -> active in bare-metal
#   - valence-aware type masking            (prompt 16)      -> N cannot take 4 neighbours
#
# Requires those prompts landed (the flags below must exist). --ligand_templates seeds the
# reference first shell (2 bidentate-O ligands + 3 nitrates); drop it for unconditioned.
# (generate_design_test.py conditions via --ligand_templates, the seeding route; the hard
#  --donor_spec constraint lives in generate.py, a different entry point.)
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

CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt
OUT=design_test_runs/maskall_fixed_${SLURM_JOB_ID}

python -u generate_design_test.py \
    --complex eu_tmma_cis.xyz \
    --model "$CKPT" \
    --outdir "$OUT" \
    --mask_k all \
    --n_samples 150 \
    --batch_size 32 \
    --resample_r 5 \
    --ligand_sizes random \
    --max_denticity 4 \
    --denticity_prior csd \
    --project_enabled True \
    --d_min_start 2.2 \
    --d_min_end 1.9 \
    --valence_guard True
    # optional conditioning (Prompt 10): add a trailing '\' to the line above, then:
    #   --ligand_templates "carboxylate;carboxylate;nitrate;nitrate;nitrate"

echo "=== maskall_fixed valid count ==="
ls "$OUT"/maskall/noH/ 2>/dev/null | wc -l
echo "=== rejection breakdown (was: silent geometric gate) ==="
cat "$OUT"/maskall/rejection_summary.json 2>/dev/null || echo "(no rejection_summary.json)"
echo "=== honest denominator ==="
cat "$OUT"/maskall/accounting.json 2>/dev/null || echo "(no accounting.json)"
echo "Done: $(date)"
