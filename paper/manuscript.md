# Manuscript — assembly and front/back matter

This is the submission-ready assembly. Section prose lives in the modular `draft_*.md`
files (single source of truth); assemble in the order below. All numbers are verified
and log-traceable (see `../RESEARCH_PLAN.md` and `draft_SI.md` §S9).

---

## Title (working)

**Completing but not designing lanthanide coordination spheres: adapting a 3D
equivariant diffusion model to the f-block reveals a coordination-validity gap**

_Alternative:_ "Adapting multi-LigandDiff to lanthanides (CN 7–10): completion works,
de-novo design fails on valence."

## Authors (confirm before submission)

Bogdan Mironov¹, De-en Jiang² (corresponding)
¹ Berea College  · ² Vanderbilt University

> Confirm the full author list, ordering, affiliations, and corresponding author with
> the PI before submission.

## Assembly order

| § | Section | Source file |
|---|---|---|
| — | Abstract | `draft_abstract.md` |
| 1 | Introduction | `draft_introduction.md` |
| 2 | Methods | `draft_methods.md` |
| 3 | Results | `draft_results.md` |
| 4 | Discussion | `draft_discussion.md` |
| 5 | Conclusion | `draft_conclusion.md` |
| — | Supporting Information | `draft_SI.md` |

> The Results file is numbered "4.x" for historical reasons; renumber to "3.x" (and
> Discussion to 4, Conclusion to 5) when flattening into the journal template.

## Figures (see `figures/README.md`)

- **Fig 1** Method schematic (CSD → fine-tune → samplers → validation).
- **Fig 2** Training curve (977.8 → 49.9 @ ep48; early-stop ep63). _Regenerate from the
  real log._
- **Fig 3 (key)** Scaffold-degradation / de-novo failure (126 → 4 → 0 → 0; maskall
  0/6300), produced by `analyze_design_test.py`.
- **Fig 4** RePaint yield-vs-resampling trade-off (yield ↑ monotonically; denticity-
  match peaks at r = 5), from `table1_repaint_sweep.csv`.
- **Fig 5** Representative valid mask1 completion vs. crystallographic reference
  (Eu–O ≈ 2.3–2.6 Å). _Render from real xyz._
- _Dropped:_ the earlier "validity vs. crowding" (Fig 4) and "DFT validation" (Fig 6)
  figures — those experiments were never run (see `figures/README.md`).

## Tables

- `tables/table1_repaint_sweep.csv` — RePaint sweep.
- `tables/table2_design_degradation.csv` — scaffold-degradation (headline).
- `tables/table3_xtb_validation.csv` — GFN2-xTB convergence.

## Data and code availability

Code: `github.com/mironovb/multi_LigandDiff`, branch `ln-adaptation`, including the
fine-tuned checkpoint, the reference complex `eu_tmma_cis.xyz`, all generation/analysis
scripts, and the cluster job suite (`sbatches/`). CSD-derived structures are subject to
CCDC licensing; the curation pipeline and per-element summary are provided.

## Author contributions / acknowledgements (template)

B.M. designed and ran the experiments and wrote the manuscript; D.J. conceived and
supervised the project. _Fill in funding/compute acknowledgements (HPC cluster, etc.)._

## Target journal

- **J. Chem. Inf. Model. (JCIM)** or **J. Chem. Theory Comput. (JCTC)** — methods
  adaptation + informative negative result. JCTC is the natural home given the
  multi-LigandDiff lineage; a DFT-validated completion showcase strengthens but is not
  required for the JCIM framing.

## Pre-submission checklist

- [ ] Confirm author list, affiliations, corresponding author, funding.
- [ ] Run `sbatches/design_test.sbatch` to complete the mask 2/3 degradation points;
      update `table2_design_degradation.csv` and Fig 3.
- [ ] Regenerate Fig 2 from the real training log; render Fig 5 from real xyz.
- [ ] (Optional, for JCTC) run the DFT protocol on the reference + 3–5 completions;
      add a DFT comparison table/figure.
- [ ] Verify every in-text number against `draft_SI.md` §S9 (job-ID index).
- [ ] Ensure **no** claims of context-density ablation, projection-stack "6.4%",
      cross-architecture, or completed DFT remain anywhere (these were never run).
- [ ] Draft cover letter; format to the target journal template; assemble SI.
