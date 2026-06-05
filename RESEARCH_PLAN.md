# Research plan: getting scientific results out of (Ln-adapted) multi-LigandDiff

_Branch: `ln-adaptation`. This document is the working analysis of what the project
verifiably has, what is wrong with the current repo, and the concrete series of
changes to turn it into a defensible scientific result. It is grounded in the
audited reference material kept locally (and git-ignored) under `ligandgen/`
(`PROJECT_OVERVIEW.md`, `VERIFICATION_REPORT.md`, `LITERATURE_AND_STRATEGY.md`,
`SITE_AUDIT.md`). Every number below traces to on-cluster logs / metrics files as
recorded in those reports._

---

## 1. What the model verifiably does

multi-LigandDiff is a 3D equivariant diffusion model (EDM + GVP-GNN score function,
~3.8M params, T=500) for metal coordination complexes. This branch is the **first
adaptation to f-block (lanthanide) chemistry and to high coordination numbers
(CN 7–10)** — the base model was d-block, CN ≤ 6.

**Verified positive results (logs + directories + metrics agree):**

- **Fine-tuning converged.** Validation loss 977.76 (epoch 0) → **49.91 (epoch 48,
  ≈95% reduction)**; clean early-stop at epoch 63 (patience 15); ~10 GPU-h on a
  **single H200**. Convergence sampling: `valid_ligand 0.970`, `connected_ligand
  0.938`, `valid_complex 0.005`.
- **Completion (mask1) works.** On Eu(TMMA)₂(NO₃)₃ (CCDC VEDTAA01; CN 10; 5 ligands,
  35 heavy atoms; cis): hide one ligand, regenerate it. **126 valid** structures in
  the design-test run; the RePaint sweep adds more (below). xTB-stable: **38/41 =
  92.7%** converge under GFN2-xTB.
- **RePaint buys yield (not validity).** Yield rises monotonically with resampling
  `r`: **1.16% (r=1) → 3.40% (r=5) → 3.80% (r=10) → 5.20% (r=20)**; denticity-match
  quality **peaks at r=5 (15.3%)** and falls at r=10. Working point: **r=5**.
  Totals: **171 valid** (r1+r5+r10), 210 including r=20.

**Verified negative result (this is the scientifically valuable finding):**

- **De-novo design fails, with a mechanism.** As the scaffold shrinks, validity
  collapses: **mask1 = 126 → mask2 = 4 → mask3 = 0 → maskall = 0 / 6300 attempts
  (0.00%)**. The rejections are **~100% nitrogen explicit-valence violations**
  ("Explicit valence for atom # N, 4, is greater than permitted"; 151/170 logged
  lines). The model learned *local completion consistent with neighbors* but has no
  notion of composing a valence-correct fragment from scratch. This is a chemical
  failure, not a geometric near-miss.

**One-line story (honest, publishable):** _A 3D diffusion model can be fine-tuned to
**complete** lanthanide coordination spheres but cannot **design** them de novo; the
failure is a specific, diagnosable coordination-validity (valence) gap, and
inference-time resampling buys yield but not validity — motivating coordination-aware
generation._

---

## 2. Shortcomings of the current repo

Ordered by severity.

### S1 — The committed `paper/` artifacts are partly fabricated and contradict the real data (credibility-critical)

`paper/executive_summary.md`, `paper/draft_results.md`, `paper/draft_methods.md`,
`paper/draft_discussion.md`, and `paper/tables/table{1,2,3}*.csv` describe completed
experiments that **have no supporting job logs and disagree with the verified
results**:

| Claim in committed `paper/` | Reality (verified) |
|---|---|
| "context-density ablation" 31/15/9/6% valid (the "key novel Figure 4") | **No job logs.** `sbatches/context_ablation.sbatch` never ran. Fabricated. |
| "RePaint + projection = 6.4% valid", "12.8× improvement" | **No job logs.** `sbatches/projection_stack.sbatch` never ran. |
| cross-architecture (d-block Fe 89%, etc.), Table 3 | `cross_arch_summary.csv` is **all `pending`/empty**; `dblock_cross.sbatch` never ran. |
| xTB "4/7 = 57% converged" | Real xTB: **38/41 = 92.7%** baseline, 81/85 r=5. Direct contradiction. |
| "95,279 CSD complexes", "4× H200, 18 GPU-h", `MAX_LIGANDS` 4→10, early-stop patience 10 | Real: **9,306** training complexes (~85,760 samples/epoch); **single H200, ~10 GPU-h**; `MAX_LIGANDS=10`; patience **15**. |
| valid_ligand "0.45 → 0.82" | Convergence sampling valid_ligand was **0.970**. |
| DFT "submitted, results pending" | **No DFT run.** Only `orca_templates/pbe0_eu.inp` exists. |
| draft **omits** the real headline (maskall = 0/6300, N-valence mechanism) | This is the paper's most valuable content and it is missing. |

These read as placeholder/synthetic numbers from an earlier "Prompts 1–10" drafting
pass. They must not go out under Jiang's name. **Fix: replace with verified-data
artifacts (done in this change set) and delete the fabricated tables.**

### S2 — The script behind the headline result is missing (reproducibility)

`generate_design_test.py` — the mask-size sweep that produced the
126→4→0→0 / maskall-0/6300 result — **was never committed** (it lived only on the
cluster). The single most important experiment is therefore not reproducible from
the repo. **Fix: reconstructed `generate_design_test.py` + `analyze_design_test.py`
(done).**

### S3 — The degradation curve has a hole at mask2/mask3

`mask2 = 4` and `mask3 = 0` came from **time-limit-cut-off** jobs (14292188,
14344725); the dedicated mask2/mask3 run was never completed, so the exact location
of the validity cliff between mask1 (works) and mask3 (0) is not pinned down.
**Fix: `sbatches/design_test.sbatch` runs mask2/mask3 first within the 4 h wall
(done — needs a cluster run).**

### S4 — No DFT validation has actually been run

Only ORCA templates exist; `dft_pipeline.py` and `sbatches/dft_orca.sbatch` are
present but unexercised. Any completion showcase ("the model places a TMMA with
Eu–O ≈ 2.3–2.6 Å") is currently asserted from xTB + visual inspection only.
**Fix: run the DFT pipeline on a small stratified set (reference + a few valid
completions) once a candidate set is trusted (cluster work).**

### S5 — Operational fragility that has already cost runs

- The discriminative-LR **resume path failed 5×** at startup with "loaded state dict
  has a different number of parameter groups" before training finally completed.
- A design-test job died at startup from the **`set -euo pipefail` + `source
  ~/.bashrc`** `nounset` trap (`/etc/bashrc: BASHRCSOURCED: unbound variable`).

**Fix: the nounset pitfall is documented and avoided in the new sbatches; the resume
bug is flagged here for a follow-up hardening pass on `finetune.py`/`src/lightning.py`
(load optimizer state defensively / rebuild param groups on resume).**

### S6 — Metric artifact must stay correctly framed

The graph-level "fragment fusion" / "60.5% cross-ligand bond" numbers are an artifact
of the 1.3× covalent-radii bond-detection cutoff (and xTB locking it in), **not**
evidence of chemical fusion — confirmed by visual inspection. Any write-up must
present these as a **geometry sensitivity check**, never as a failure mode.

---

## 3. The series of changes

### Track A — land an honest multi-LigandDiff paper (the achievable result now)

Code/repo changes in this change set:

1. **Gitignore `ligandgen/`** — the overview website / audited reports are
   reference-only and never tracked here. _(done)_
2. **Replace fabricated `paper/` artifacts with verified-data versions** — rewrite
   the executive summary, methods/results/discussion drafts, and the data tables to
   the verified story; delete the empty/contradictory cross-architecture and
   context-ablation tables; add the **design-degradation table** that carries the
   headline. _(done)_
3. **Restore `generate_design_test.py`** (mask-size sweep) and add
   **`analyze_design_test.py`** (degradation curve + N-valence rejection tally). _(done)_
4. **Clean design-test sbatches** (`sbatches/design_test.sbatch`,
   `sbatches/design_maskall.sbatch`) with the nounset pitfall documented/avoided. _(done)_

Cluster experiments to run (compute, not code):

5. **Dedicated mask2/mask3 run** to close S3 (a few GPU-h). Sharpens the degradation
   curve for the paper.
6. **DFT validation** of the reference + ~3–5 valid mask1 completions (closes S4):
   optionally tighten xTB first (`--opt tight`), then run the prepared ORCA protocol
   (`orca_templates/pbe0_eu.inp`): PBE0-D4/def2-TZVP, Eu via SARC-DKH-TZVP + SK-MCDHF-RSC
   ECP, neutral, mult 7. Driver: `sbatches/dft_orca.sbatch` + `dft_pipeline.py`.
7. **Harden the resume path** in `finetune.py` (S5) so a re-fine-tune is robust.

Scope discipline for the paper: keep it to **what ran** — fine-tune convergence,
mask1 completion (+xTB), the RePaint yield-vs-resampling trade-off, and the de-novo
failure with its valence mechanism. **Leave out** projection-stack "6.4%",
context-density ablation, cross-architecture, and any DFT claim until logs exist.

### Track B — coordination-aware, rare-earth-native platform (next, larger program)

The de-novo failure is the motivation. The fix, drawing on the Kulik/`pydentate` line
of work (predict-then-build) and Ln coordination chemistry, is to **fix chemistry
first, then realize geometry**:

- **Valence-valid by construction:** generate/select the ligand as a chemical graph
  (where N cannot have four bonds) before 3D placement, instead of painting raw 3D
  points and hoping the post-hoc Lewis structure is legal.
- **Denticity/CN/donor identity as explicit inputs** (coordination-aware
  representation), extended past pydentate's CN ≤ 6 to the **CN 7–10** Ln regime.
- **Hard valence/CN/charge constraints inside the sampler** (the repo's
  `src/projection.py` is a starting point), not post-hoc rejection of 99.99% of
  attempts.
- **Rare-earth priors:** hard-donor (O/N) bias, CN 8–10 saturation, lanthanide
  contraction as the selectivity handle, counter-ion/solvent (nitrate, water) in the
  first shell, spec-conditioned generation ("2 N + 3 O, neutral, CN 8, Eu").

Open questions for the advisor meeting (from `LITERATURE_AND_STRATEGY.md`): is the
diagnostic result publishable as-is or does it need a DFT-validated completion
showcase; predict-then-build vs. constrained-3D for Track B; is the objective
*selectivity* (adjacent-Ln differential binding) rather than single-complex
stability; build on `pydentate` or from the Jiang group's ~29,891-complex dataset;
how central is hemilability.

---

## 4. Reproducibility quick reference

- Checkpoint: `models/ln_finetuned/ln_finetuned_epoch=48.ckpt`
- Reference complex: `eu_tmma_cis.xyz` (Eu(TMMA)₂(NO₃)₃, CCDC VEDTAA01)
- Env: `module load miniforge/25.11.0-0 cuda/12.9.1 && source activate ligdiff`
- Completion: `generate_mask1.py`; design sweep: `generate_design_test.py`
  (`--mask_k 1|2|3|all`); analysis: `analyze_design_test.py`
- xTB: `xtb_pipeline.py`; DFT: `dft_pipeline.py` (+ `orca_templates/`)
- Cluster caveats: preemptible partition has a 4 h wall (most important experiment
  first); avoid `set -euo pipefail` with `source ~/.bashrc`; use `python -u`.
