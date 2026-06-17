# Cluster job suite — submission guide

All jobs run on `mit_preemptable` (4 h wall, `--time=03:59:59`), single H200 for the
GPU stages, CPU-only for xTB/DFT. Environment in every script:
`module load miniforge/25.11.0-0 [cuda/12.9.1]; source activate ligdiff`. Submit from
`~/multi_LigandDiff`. Logs land as `<jobname>_<jobid>.out/.err`.

## Order of submission

| # | Job | Script | GPU | Produces |
|---|---|---|---|---|
| 0 | Fine-tune (only if re-training) | `../finetune_h200.sbatch` | yes | `models/ln_finetuned/ln_finetuned_epoch=48.ckpt` |
| 1 | Mask1 completion baseline | `mask1_completion.sbatch` | yes | `generated/eu_tmma_mask1_epoch48/` (xTB input) |
| 2 | RePaint sweep | `repaint_sweep.sbatch` | yes | `generated/repaint_r{1,5,10,20}_epoch48/` |
| 3 | Design-ability sweep (mask 2/3/1) | `design_test.sbatch` | yes | `design_test_runs/sweep_<jid>/mask{1,2,3}/` |
| 4 | De-novo test (mask all) | `design_maskall.sbatch` | yes | `design_test_runs/maskall_<jid>/maskall/` |
| 5 | xTB optimisation | `xtb_batch.sbatch` | no | `xtb_results/<set>/summary.json` |
| 6 | DFT (one job per structure) | `dft_orca.sbatch` | no | `dft_work/<name>/orca_output.log` |

Stages 1–4 are independent and can be submitted in parallel. Stage 5 depends on 1
(and optionally 2); stage 6 depends on 5.

## Fix-validation harness (code-review fixes)

These accompany the `prompts/` worklist (gitignored) that delivers the
[Code Review](../ligandgen/) fixes. CPU jobs gate the no-GPU fixes; GPU/ORCA jobs run the
validation experiments. Submit the CPU gates after each batch of fixes; run the GPU jobs
once their prerequisite prompts have landed (each script's header lists them).

| Job | Script | GPU | Gates / runs |
|---|---|---|---|
| Unit tests | `run_unit_tests.sbatch` | no | pytest + import/compile smoke for fixes 01–21 |
| Re-score validity | `rescore_validity.sbatch` | no | reference + saved outputs through the fixed gate (Finding 2) |
| Post-fix smoke | `smoke_fixes.sbatch` | yes (short) | a few mask-1 completions — "did we break sampling?" |
| De-novo re-test | `design_maskall_fixed.sbatch` | yes | maskall with all fixes on (vs. the old 0/6300) |
| Mask-2 curve | `design_mask2.sbatch` | yes | dedicated mask-2 run to completion (Track A) |
| Curriculum re-finetune | `finetune_curriculum.sbatch` | H200 | retrain on the high-mask curriculum (needs CPU re-prep first) |
| DFT showcase | `dft_showcase.sbatch` | ORCA | prepare + submit per-structure DFT on top mask-1 (Track A) |

```bash
# CPU gates (no GPU) — run after each batch of no-GPU fixes
sbatch sbatches/run_unit_tests.sbatch
sbatch sbatches/rescore_validity.sbatch      # the decisive Finding-2 experiment

# GPU validation — after the relevant prompts land
sbatch sbatches/smoke_fixes.sbatch
sbatch sbatches/design_maskall_fixed.sbatch  # the de-novo re-test
sbatch sbatches/design_mask2.sbatch          # finishes the degradation curve

# DFT showcase (after the xTB stage): launcher prepares + submits one ORCA job per structure
sbatch sbatches/dft_showcase.sbatch
```

## Quick start

```bash
cd ~/multi_LigandDiff

# 1–2: generation (GPU). The RePaint sweep does not fit one 4 h wall; split it.
sbatch sbatches/mask1_completion.sbatch
RVALS="1 5"   sbatch sbatches/repaint_sweep.sbatch
RVALS="10 20" sbatch sbatches/repaint_sweep.sbatch

# 3–4: the design experiments (GPU)
sbatch sbatches/design_test.sbatch          # mask 2/3/1 degradation points
sbatch sbatches/design_maskall.sbatch        # 150 x 42 = 6300 attempts

# 5: xTB validation (CPU)
sbatch sbatches/xtb_batch.sbatch

# 6: DFT (CPU). Prepare on the login node, then one job per structure.
python dft_pipeline.py prepare \
    --xtb-results-dir xtb_results/eu_tmma_mask1_epoch48 \
    --reference eu_tmma_cis.xyz --output-dir dft_work --mult 7
for d in dft_work/*/; do STRUCTURE=$(basename "$d") sbatch sbatches/dft_orca.sbatch; done
```

## Analysis (login node, no GPU)

```bash
# RePaint sweep metrics
python analyze_repaint_sweep.py            # -> metrics/results/aggregate_r*.txt

# Design-ability curve + N-valence rejection mechanism (the headline)
python analyze_design_test.py \
    --runs design_test_runs/sweep_*  design_test_runs/maskall_* \
    --logs ln_design_*.err ln_maskall_*.err ln_design_*.out ln_maskall_*.out \
    --out metrics/design_test
#   -> metrics/design_test/design_degradation.csv, rejection_tally.csv, design_degradation.png

# DFT comparison (after stage 6)
python dft_pipeline.py parse --work-dir dft_work --reference eu_tmma_cis.xyz \
    --output-csv paper/tables/dft_comparison.csv
```

## Notes / gotchas

- **4 h wall:** put the most important experiment first in any multi-step loop. The
  RePaint sweep and the design sweep are ordered with that in mind.
- **`nounset` trap:** never combine `set -euo pipefail` with `source ~/.bashrc` here;
  `/etc/bashrc` references an unset `BASHRCSOURCED` and kills the job at startup
  (this sank job 14289937). Use `source activate ligdiff`.
- **`python -u`** everywhere for unbuffered, live logs.
- **xTB/DFT** need no GPU; they request CPU cores (`-c 16`) on the same partition.
- **Checkpoint:** all generation uses `models/ln_finetuned/ln_finetuned_epoch=48.ckpt`.
