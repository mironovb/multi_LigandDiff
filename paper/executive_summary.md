# Executive summary: Ln-adapted multi-LigandDiff

**Re:** status of the lanthanide multi-LigandDiff paper artifacts.

> This file was rewritten to reflect **only verified, log-traceable results**. An
> earlier "Prompts 1–10" draft asserted experiments (context-density ablation,
> projection-stack "6.4%", cross-architecture, "DFT submitted", xTB "4/7=57%") that
> have **no supporting job logs and contradicted the real data**; those have been
> removed. See `RESEARCH_PLAN.md` (§2 shortcoming S1) for the full discrepancy list.
> All numbers below trace to on-cluster logs / metrics files.

## Headline

First adaptation of a 3D equivariant diffusion model to **f-block (lanthanide)
coordination chemistry and to high coordination numbers (CN 7–10)** — a regime the
base d-block, CN ≤ 6 model (and the comparable d-block coordination-ML tools) never
covered. The honest one-line result:

> A 3D diffusion model can be fine-tuned to **complete** lanthanide coordination
> spheres but cannot **design** them de novo; the failure is a specific, diagnosable
> coordination-validity (valence) gap, and inference-time resampling buys yield but
> not validity.

## Verified results

1. **Fine-tuning converged.** val_loss **977.8 → 49.9 @ epoch 48 (≈95%)**, clean
   early-stop @ epoch 63 (patience 15), ~10 GPU-h on a **single H200**. Convergence
   sampling: valid_ligand 0.970 / connected_ligand 0.938 / valid_complex 0.005.
2. **Completion (mask1) works.** Eu(TMMA)₂(NO₃)₃ (CCDC VEDTAA01, CN 10): **126
   valid** structures; xTB-stable (**38/41 = 92.7%** GFN2-xTB convergence). Table 3.
3. **RePaint yield engineering.** Yield rises monotonically with resampling r:
   **1.16% → 3.40% → 3.80% → 5.20%** (r=1/5/10/20); denticity-match peaks at **r=5
   (15.3%)**. Yield, not validity — r=5 is the working point. Table 1. Totals: 171
   valid (r1+r5+r10), 210 incl. r=20.
4. **De-novo design fails, with a mechanism (the valuable finding).** Scaffold
   gradient **mask1 126 → mask2 4 → mask3 0 → maskall 0 / 6300 (0.00%)**; rejections
   **~100% nitrogen explicit-valence violations** (151/170 logged). Table 2. A
   generic 3D diffusion model has no valence/denticity constraint, so without
   scaffolding it produces atom clouds that resolve to impossible Lewis structures.

## What is not done (and is not claimed)

| Item | Status |
|---|---|
| DFT validation | **Not run.** Only `orca_templates/pbe0_eu.inp` exists. Do not claim DFT as done. |
| Dedicated mask2/mask3 run | mask2/mask3 are from cut-off jobs; `sbatches/design_test.sbatch` will firm them up. |
| Context-density ablation, projection-stack, cross-architecture | **Never ran** (no logs). Removed from artifacts. |

## Suggested paper scope (Track A)

Keep to what ran: fine-tune convergence; mask1 completion + xTB; RePaint
yield-vs-resampling trade-off; the de-novo failure + valence mechanism. The negative
result is the contribution and it motivates a coordination-aware platform
(`RESEARCH_PLAN.md` Track B). A DFT-validated completion showcase is optional polish,
not a prerequisite — confirm scope with the advisor.
