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
#   - denticity cap <=4, eligible-only enum  (prompts 05-07)  -> honest denominator
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

# !! CHECKPOINT CAVEAT (normalization_factor bug, prompts 19-20) -- read docs/H200_RUNBOOK.md.
# epoch=48.ckpt stores normalization_factor=1: warm-started from a factor=100 backbone into a
# factor=1 fine-tune, so the TM transfer was scrambled. It samples self-consistently, so THIS run
# is valid and apples-to-apples with the old 0/6300 -- but the number is a BUG-TRAINED model's
# capability, not the clean ceiling. For the clean de-novo number, first produce a factor=100
# checkpoint (sbatches/finetune_curriculum.sbatch -- trains the bare-metal regime) and re-point CKPT.
CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt   # bug-trained (factor=1); preliminary number
# CKPT=models/ln_finetuned_curriculum/<best>.ckpt     # <- clean factor=100 + bare-metal curriculum (recommended)
OUT=design_test_runs/maskall_fixed_${SLURM_JOB_ID}

# ATTEMPT COUNT (statistical power) = n_samples x partitions-per-subset; maskall has n_subsets=1:
#   uniform -> n_samples x 23 eligible CN=10 partitions  (274 x 23 = 6302 == the old run's power, POSSIBLE targets only)
#   csd     -> n_samples x 1 sampled partition/seed       (the earlier csd run did only 150 -> 42x under-powered!)
# Why it matters: at a 0.08% true yield, P(observe 0 valid) ~= 89% at n=150 but ~0.6% at n=6300. A low/zero
# yield is only meaningful at thousands of attempts. uniform/274 restores the old run's power on capped targets.
# Realistic-target alternative (same power, concentrates on likely partitions): --n_samples 6000 --denticity_prior csd
# WALL CHECK: 6302 attempts ~= 42x the 150-run; `sacct -j <150-run jid> --format=Elapsed` -- if it took >5 min,
# drop to --n_samples 137 (~3150 attempts, still ~95% power at a 0.1% yield) to stay under the 4 h wall.
python -u generate_design_test.py \
    --complex eu_tmma_cis.xyz \
    --model "$CKPT" \
    --outdir "$OUT" \
    --mask_k all \
    --n_samples 274 \
    --batch_size 32 \
    --resample_r 5 \
    --ligand_sizes random \
    --max_denticity 4 \
    --denticity_prior uniform \
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
