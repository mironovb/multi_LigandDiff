> **Superseded by `docs/H200_RUNBOOK.md`** (canonical, kept current). This file is the dated
> 2026-06-18 analysis snapshot taken *before* the post-audit fixes; `docs/FIXLOG.md` +
> `docs/KNOWN_ISSUES.md` carry the resolved state. Kept for provenance.

# H200 runbook + flagged inconsistencies — post-arun session 20260616_210131

**Session:** 25 prompts (code-review fix worklist 01–25), run locally on the Mac 2026-06-16/17.
**Result:** 22 committed (`8729f4c..2f4b2b9`), 3 left uncommitted (23/24/25 — they need cluster results
before their *result* commit, but they also carry **pre-run code fixes** that must reach the cluster first).

This doc is the handoff: what to push, what to run on the H200, and every inconsistency the session surfaced.

---

## 0. Pre-flight BLOCKERS — do these before you `ssh mit`

### 0a. Three code fixes are uncommitted and unpushed — the cluster will NOT have them after `git pull`
The arun "commit only after results" rule held these back, but they are **pre-run code fixes**, not result
artifacts. Without them the cluster runs are wrong or crash:

| File | Fix | If missing on cluster |
|---|---|---|
| `analyze_design_test.py` | regex tolerates honest-denominator `attempts=N (raw M)` | step-4 analysis **silently** reports blank attempts / no yield for **every** fixed run (only filesystem valid count survives) |
| `dft_pipeline.py` | `from __future__ import annotations` (Py3.9) + `glob.escape` on bracketed names | `prepare`/`parse` **crash on import** under cluster Py3.9.25; even if patched, showcase collapses to **reference-only** (bracketed names never glob-match) |
| `sbatches/design_mask2.sbatch` | `n_samples 250 → 50` | uniform prior makes attempts = `n_samples × 50` → **12,500**, likely **trips the 4 h wall** — reproducing the exact cutoff prompt 24 exists to fix |

```bash
# run LOCALLY (on the Mac), before going to the cluster:
git add analyze_design_test.py dft_pipeline.py sbatches/design_mask2.sbatch
git commit -m "fix: pre-cluster code fixes — analyzer regex tolerates honest-denominator (raw N); dft_pipeline Py3.9 annotations + glob.escape; mask2 n_samples 250->50 (avoid 4h wall)"
git push
```
(`reports/` is untracked — add it too if you want this runbook + the DFT status report in git.)

### 0b. Checkpoint path divergence — verify on the cluster before submitting
All sbatches expect `models/ln_finetuned/ln_finetuned_epoch=48.ckpt`. **That exact path does not exist
locally** (epoch=48 is at repo root; `ligandgen/.../models/ln_finetuned/` has 57/08/04/25, not 48). Confirm the
cluster layout first:
```bash
cd ~/multi_LigandDiff && test -f models/ln_finetuned/ln_finetuned_epoch=48.ckpt && echo CKPT_OK || echo CKPT_MISSING
```

---

## 1. Cluster runbook (H200) — copy-paste, in order

```bash
ssh mit
cd ~/multi_LigandDiff
git pull                                   # MUST include the 0a push
test -f models/ln_finetuned/ln_finetuned_epoch=48.ckpt && echo CKPT_OK   # 0b

# --- no-GPU gates (fast; run first) ---
sbatch sbatches/run_unit_tests.sbatch      # pytest + import/compile smoke for fixes 01-21
sbatch sbatches/rescore_validity.sbatch    # decisive Finding-2 test: reference + saved outputs through fixed gate

# --- GPU validation (independent, submit in parallel) ---
sbatch sbatches/smoke_fixes.sbatch          # "did the fixes break sampling?" (short)
sbatch sbatches/design_maskall_fixed.sbatch # prompt 23: de-novo re-test vs old 0/6300
sbatch sbatches/design_mask2.sbatch         # prompt 24: finishes the degradation curve

# --- analysis (login node, after the design jobs finish) ---
# NOTE mask2_* LAST: discover_mask_dirs is last-wins; a stale mask2/ in sweep_* would shadow the fresh count
python analyze_design_test.py \
    --runs design_test_runs/sweep_* design_test_runs/maskall_fixed_* design_test_runs/mask2_* \
    --logs ln_maskall_fixed_*.err ln_maskall_fixed_*.out ln_mask2_*.err ln_mask2_*.out \
    --out metrics/design_test_fixed

# --- xTB -> DFT showcase (CPU; sequential; AFTER reconciling level-of-theory, see flag #5) ---
sbatch sbatches/xtb_batch.sbatch            # produces xtb_results/eu_tmma_mask1_epoch48/
sbatch sbatches/dft_showcase.sbatch         # prepares + submits 1 ORCA job/structure
# when ORCA jobs finish:
grep -l "THE OPTIMIZATION HAS CONVERGED" dft_work/*/orca_output.log | wc -l
python dft_pipeline.py parse --work-dir dft_work --reference eu_tmma_cis.xyz \
    --output-csv paper/tables/dft_comparison.csv

# --- OPTIONAL but recommended: clean factor=100 model (see flag #2) ---
# CPU re-prep FIRST (also picks up the prompt-14 bond-cutoff change):
python -u prepare_training_data.py --cif_dir <cifs> --output_dir data_curriculum \
    --mask_curriculum quadratic --force_all_masked --max_augment 30 --workers 16
sbatch sbatches/finetune_curriculum.sbatch  # warm-starts from pre_trained.ckpt at the corrected factor=100
```

---

## 2. Flagged inconsistencies (severity-ordered)

### 🔴 #1 — Pre-run code fixes are uncommitted (see §0a). The #1 blocker.

### 🔴 #2 — The design re-runs use a checkpoint trained WITH the normalization_factor bug
Verified directly from the checkpoint hparams:
- `ln_finetuned_epoch=48.ckpt` → `normalization_factor = 1`  (the bug)
- `model/pre_trained.ckpt` → `normalization_factor = 100`

`DDPM.load_from_checkpoint` restores hparams from the ckpt, so epoch=48 **samples self-consistently at scale 1**
— the design re-runs (23/24) will run fine. **But** it was warm-started from a factor=100 backbone into a
factor=1 fine-tune (`conv_layer.py` divides messages by this factor), so the TM transfer was scrambled at init;
the model never got the pretraining benefit. Prompts 19–20 fixed `finetune.py`/`config.yml` to 100 **for future
training only — they do not touch epoch=48.**

**Consequence for reading the result:** a low/zero valid count from `design_maskall_fixed` (23) **cannot cleanly
separate instrument vs model** — the model has a known, unfixed training deficiency. Decide:
- Run 23/24 on epoch=48 = *fixed-instrument, same-model* comparison, apples-to-apples with the old `0/6300`. ✅ do this.
- For a *clean model-capability* number, re-finetune at factor=100 via `finetune_curriculum.sbatch` (which now
  uses the corrected finetune.py) and re-run. ✅ recommended as the real headline.
- Report both; do not present an epoch=48 number as the model's true ceiling.

### 🟠 #3 — Finding 2's premise is refuted FOR THE REFERENCE (prompt 04)
The reference `eu_tmma_cis.xyz` nitrates pass **both** the old and the new gate (OpenBabel perceives all-single
N–O at crystal geometry → formal charge 0 → nothing for the old charge-blind gate to drop). So the original
`0/6300` was **not** a gate-instrument artifact for the reference. The charge fix only bites on **distorted
generated** geometries. → The decisive test is `rescore_validity.sbatch` over **generated** sets, not the
reference. Don't claim the charge fix "rescued the reference."

### 🟠 #4 — Checkpoint path local↔cluster divergence (see §0b). Verify before submitting.

### 🟠 #5 — DFT level-of-theory provenance conflict (prompt 25) — resolve BEFORE `dft_showcase`
- Template `orca_templates/pbe0_eu.inp` + `paper/draft_SI.md`: **PBE0-D4 / SARC-DKH-TZVP + SK-MCDHF-RSC ECP /
  gas-phase** (no `%cpcm`/SMD block).
- `strategy.md` + the prompt text: **PBE0-D3 / ECP28MWB / SMD dodecane (implicit solvent)**.

These disagree. Either add `%cpcm SMD true; SMDsolvent "..." end` to the template (and keep the solvent wording),
or drop the "D3 / ECP28MWB / implicit solvent" wording from `strategy.md`. **Do not report "implicit solvent"
until the template implements it.**

### 🟠 #6 — DFT ΔE-vs-reference is cross-formula (prompt 25)
Reference is **heavy-atom-only (35 atoms, no H)**; completions are **H-complete (43 atoms)**. The
`dft_deltaE_kcal_mol` column compares different molecular formulae → not physically meaningful. Compare
**per-structure convergence** and **Eu–donor distances** vs the Kravchuk/VEDTAA01 reference, not ΔE.

### 🟡 #7 — Residual low projection defaults still in 3 places (prompt 11, deferred)
`1.5/1.3` d_min defaults remain at `src/lightning.py:340`, `src/edm.py:128`, and `generate_mask1.py`
(34/36/228/402). **Not a live risk for 23/24**: the production design path overrides with explicit `2.2/1.9`,
and `edm.py` now warns once if the shell is ever exercised below the 1.72 Å bond-perception floor. But
`generate_mask1.py` is a 4th generate script with the unsafe defaults baked in. Track for a unify pass.

### 🟡 #8 — Four divergent copies of reform_data/generate_ligand (prompts 03, 09)
`generate.py`, `generate_design_test.py`, `generate_bare.py`, `generate_mask1.py` each carry their own copy.
`--donor_spec` (prompt 09) landed **only** in `generate.py`; `generate_design_test.py` conditions via
`--ligand_templates` instead (documented in `design_maskall_fixed.sbatch` header). Known divergence — track for
consolidation. (It's why the maskall re-test conditions with `--ligand_templates`, not `--donor_spec`.)

### 🟡 #9 — Curriculum re-finetune needs a CPU re-prep first (prompts 14, 18)
The bond-cutoff unification (14) and the high-mask curriculum (18) both change training-data prep. The existing
`train_ln.pt` was prepped with the **old flat cutoffs** and **uniform masking**. `finetune_curriculum.sbatch`
STEP-0 (`prepare_training_data.py --output_dir data_curriculum ...`) is a **manual prerequisite** — it also
pulls in the corrected bonding. Don't submit the re-finetune before STEP-0 produces `data_curriculum/train_ln.pt`.

### ℹ️ #10 — Denominator changed: report both eligible and raw
Old maskall headline `0/6300` = seeds × all 42 **uncapped** CN=10 partitions. New honest eligible denominator
≈ **150** (csd draws one partition/seed); `accounting.json` keeps `attempts_raw=6300`. When reporting the
re-test, quote **valid/150** (eligible) **and** valid/6300 (raw, for apples-to-apples with the old number).

---

## 3. How to read the design re-test (prompt 23)
- **valid > 0** → part of the original `0/6300` was the *instrument*. Report valid/150 and valid/6300.
- **valid = 0** → a *rigorous* zero **given a model trained with the factor=1 bug** (flag #2). It indicts the
  instrument-fixed pipeline + this checkpoint together; the clean model number needs the factor=100 re-finetune.
- **Rejection breakdown** (`rejection_summary.json`): the key signal is whether failures moved **off** the silent
  geometric gate (`overlap`/`atom_count`) onto informative chemistry (`sanitize`/`disconnected`/valence). Dying in
  sanitization instead of vanishing pre-gate = the instrument is now doing its job even if valid is still low.
</content>
</invoke>
