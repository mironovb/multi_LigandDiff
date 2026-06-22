# H200 runbook — execute the pending experiments

Canonical, ordered procedure to finish the worklist on the MIT H200 cluster after the
2026-06-18 audit. Companion docs: `docs/FIXLOG.md` (what changed), `docs/KNOWN_ISSUES.md`
(residual items). All GPU jobs: `mit_preemptable`, `--time=03:59:59`, single H200. Submit from
`~/multi_LigandDiff`.

> **nounset trap:** never combine `set -euo pipefail` with `source ~/.bashrc` — `/etc/bashrc`
> references an unset `BASHRCSOURCED` and kills the job (sank job 14289937). Use
> `source activate ligdiff`. Use `python -u` everywhere.

---

## 0. Pre-flight (on the cluster, once)

```bash
ssh mit
cd ~/multi_LigandDiff
git pull                                   # must include 00a317e … 1bf16bf (the audit fixes)
git log --oneline -6                        # confirm 1bf16bf is present

# checkpoint must exist at the path every sbatch expects
test -f models/ln_finetuned/ln_finetuned_epoch=48.ckpt && echo CKPT_OK || echo CKPT_MISSING
```

If `CKPT_MISSING`: locate the real `epoch=48.ckpt` on the cluster and symlink/copy it to
`models/ln_finetuned/` (locally it lived at repo root + under `ligandgen/.../models/`).

---

## 1. The checkpoint decision — read before running the design jobs

`ln_finetuned_epoch=48.ckpt` stores **`normalization_factor=1`** (confirmed). It was warm-started
from the factor=100 pretrained backbone into a factor=1 fine-tune, so the transfer was scrambled
(prompts 19-20). It samples self-consistently, so:

- **Running on epoch=48 is valid** and apples-to-apples with the old `0/6300` / mask-2 numbers —
  but the result is a **bug-trained model's** capability, *not* the clean ceiling. Label it preliminary.
- **The clean number needs a factor=100 re-finetune.** Two checkpoints, two purposes:

| Checkpoint | Producer | Masking | Use for |
|---|---|---|---|
| `ln_finetuned_epoch=48` | (shipped) | legacy | preliminary numbers, apples-to-apples with old runs |
| `ln_finetuned_fixed` | `finetune_fixed.sbatch` | uniform (legacy) | the **degradation curve** (mask 1/2/3) — isolates the normalization+bonding fix |
| `ln_finetuned_curriculum` | `finetune_curriculum.sbatch` | high-mask curriculum | the **de-novo / maskall** headline — the model actually trained for the bare-metal regime |

Recommended: run the preliminary pass on epoch=48 now (§3), and in parallel start the clean
re-finetunes (§5) so the final numbers use the corrected checkpoints.

---

## 2. No-GPU gates (run first; fast)

```bash
sbatch sbatches/run_unit_tests.sbatch      # pytest (incl. the metal-donor gate test) + import/compile smoke
sbatch sbatches/rescore_validity.sbatch    # decisive Finding-2 test: reference + saved outputs through the fixed gate
```
- **rescore_validity** is the real Finding-2 experiment: the reference passes both gates, so the
  signal (if any) is in **generated** sets that now pass but were rejected pre-fix.

## 3. GPU design re-runs (preliminary, on epoch=48 — submit in parallel)

```bash
sbatch sbatches/smoke_fixes.sbatch          # "did the fixes break sampling?" (short)
sbatch sbatches/design_maskall_fixed.sbatch # prompt 23 — de-novo re-test vs old 0/6300
sbatch sbatches/design_mask2.sbatch         # prompt 24 — mask-2 degradation point (2500 attempts)
```

Outputs land in `design_test_runs/{maskall_fixed,mask2}_<jid>/<mask>/`:
`noH/*.xyz` (valid structures), `rejection_summary.json` (per-reason counts),
`accounting.json` (`attempts_eligible` vs `attempts_raw`).

## 4. Analysis (login node, after §3 finishes)

```bash
# mask2_* LAST: discover_mask_dirs is last-wins; a stale mask2/ in sweep_* would shadow the fresh count
python analyze_design_test.py \
    --runs design_test_runs/sweep_* design_test_runs/maskall_fixed_* design_test_runs/mask2_* \
    --logs ln_maskall_fixed_*.err ln_maskall_fixed_*.out ln_mask2_*.err ln_mask2_*.out \
    --out metrics/design_test_fixed
python rescore_validity.py --reference eu_tmma_cis.xyz \
    --inputs design_test_runs/maskall_fixed_*/maskall/noH design_test_runs/mask2_*/mask2/noH \
    --out metrics/design_test_fixed/rescore.csv
```

### How to read it
- **valid > 0** on maskall → part of the old `0/6300` was the *instrument*. Report `valid/attempts_eligible`
  (≈6302 now that the sbatch is power-matched to the old run) **and** the raw count (`accounting.json` keeps both).
  NOTE: a low/zero maskall yield is only meaningful at thousands of attempts — at a 0.08% true yield, n=150 shows
  0 ~89% of the time. The sbatch's ATTEMPT COUNT note explains the `n_samples × partitions` math; don't read a 0
  from a few-hundred-attempt run as "de-novo fails."
- **valid = 0** → a rigorous zero **for a bug-trained model** — it does not exonerate the model;
  the clean number needs the §5 re-finetune.
- **Rejection breakdown:** the key signal is whether failures moved **off** the silent geometric
  gate (`overlap`/`atom_count`) onto informative chemistry (`sanitize`/`disconnected`/valence).

## 5. Clean checkpoints (the real numbers) — CPU re-prep FIRST, then GPU

```bash
# Control (degradation curve): corrected bonding + factor=100, legacy uniform masking
python -u prepare_training_data.py --cif_dir <cifs> --output_dir data_fixed \
    --mask_curriculum uniform --no_force_all_masked --max_augment 30 --workers 16
sbatch sbatches/finetune_fixed.sbatch

# Bare-metal/de-novo: + high-mask curriculum
python -u prepare_training_data.py --cif_dir <cifs> --output_dir data_curriculum \
    --mask_curriculum quadratic --force_all_masked --max_augment 30 --workers 16
sbatch sbatches/finetune_curriculum.sbatch
```
These span the 4 h wall across resubmissions (`--resume_from_checkpoint`, auto-detected). When a
checkpoint converges, confirm the fix took and re-point the design sbatches' `CKPT`:
```bash
python -c "import torch; print(torch.load('models/ln_finetuned_fixed/<best>.ckpt', map_location='cpu', weights_only=False)['hyper_parameters']['normalization_factor'])"   # -> 100.0
```
Then re-run §3–§4 against the clean checkpoint(s): `ln_finetuned_fixed` for the mask-1/2/3 curve,
`ln_finetuned_curriculum` for the maskall de-novo headline.

## 6. xTB → DFT showcase (CPU; sequential; level-of-theory already reconciled)

```bash
sbatch sbatches/xtb_batch.sbatch            # -> xtb_results/eu_tmma_mask1_epoch48/
sbatch sbatches/dft_showcase.sbatch         # prepares + submits 1 ORCA job/structure (PBE0-D4, gas-phase)
grep -l "THE OPTIMIZATION HAS CONVERGED" dft_work/*/orca_output.log | wc -l
python dft_pipeline.py parse --work-dir dft_work --reference eu_tmma_cis.xyz \
    --output-csv paper/tables/dft_comparison.csv
```
- Level of theory is now consistent across the template + tracked docs: **PBE0-D4 / def2-TZVP /
  Eu: SARC-DKH-TZVP + SK-MCDHF-RSC ECP / gas-phase**. Do **not** report implicit solvent (SMD
  dodecane is an optional, not-yet-implemented upgrade).
- The `dft_deltaE_kcal_mol` column is now **suppressed (None)** for the H-complete completions vs
  the no-H reference (cross-formula). Compare **convergence + Eu–donor distances**, not ΔE.
- Capture provenance at run time: ORCA version (`orca_output.log` header), SLURM job ids, host.
- **Honesty gate:** keep "xTB-stable (38/41 = 92.7%)"; upgrade to "DFT-confirmed" only after
  converged ORCA logs back a real `dft_comparison.csv`.

---

## 7. Commit the results (after each experiment)
The RESULT commits for prompts 23/24/25 are still pending (only the pre-run code fixes landed).
After a run completes and is analysed, commit its metrics/tables + write the result into
`reports/` (the maskall/mask2 result + the DFT comparison), then update the paper table/claims.
