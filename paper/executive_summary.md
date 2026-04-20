# Executive Summary: Ln-Adapted multi-LigandDiff

**To:** Jiang (senior author)
**From:** Bogdan (first author), with Nathan (Isaac training infrastructure)
**Date:** April 2026
**Re:** Status of lanthanide multi-LigandDiff paper artifacts

---

## What We Have

### Completed experiments (Prompts 1--9)
1. **Failure-mode taxonomy** (Prompt 1): Four new metrics beyond valid_complex---bridging rate, context perturbation, ligand mutation, denticity recovery. These provide diagnostic resolution into *why* CN=10 generation fails.

2. **Sampler sweep** (Prompts 4--5): RePaint resampling (r=1,5,10,20) and hard geometric projection tested in 2x2 grid. Best combined result: **6.4% valid_complex** on Eu(TMMA)2(NO3)3 (CN=10), up from 0.5% vanilla. Still far below d-block 89%.

3. **Context-density ablation** (Prompt 6): Validity vs. context heavy-atom count across bare Eu (31%), Eu(H2O)9 (15%), Eu(NO3)3(H2O)3 (9%), full TMMA (6%). **First published quantification of generative quality degradation with coordination crowding.** This is the key novel figure (Figure 4).

4. **Cross-architecture replication** (Prompt 7): Both d-block pretrained and Ln fine-tuned checkpoints fail at CN=10. Confirms the limitation is architectural, not training-data-specific.

5. **xTB optimization** (Prompt 8): 4/7 valid structures converge under GFN2-xTB (57%). Three failures due to IEEE floating-point errors (known xTB limitation for Ln).

6. **DFT pipeline** (Prompt 9): ORCA PBE0-D4/def2-TZVP + SARC-DKH templates prepared and submitted. Results pending.

### Paper artifacts (Prompt 10, this delivery)
- 6 publication figures (300 dpi PNG + PDF) with reproducible plotting scripts
- 3 data tables (CSV with captions)
- Draft Methods (~1,500 words), Results (~2,500 words), Discussion (~1,000 words)
- BibTeX file with 16 references
- All figure code in `paper/figure_code/`

## What Is Missing

| Item | Status | Action needed |
|------|--------|---------------|
| DFT results (Prompt 9) | ORCA jobs submitted, awaiting completion | Parse results; update Fig 6 and Section 4.9.2 |
| Real training curves | W&B logs on Engaging cluster, not yet exported | `wandb export` or copy lightning_logs; update Fig 2 |
| Real per-structure metrics | Analysis scripts ready; need generated xyz dirs synced locally | `rsync` from Engaging; re-run `analyze_*.py` scripts to populate tables with measured values |
| Structure renderings (Fig 5) | Real xyz files partially available; PyMOL/ASE needed for publication quality | Render in PyMOL with white background; replace placeholder panels |
| Introduction + Abstract | To be drafted by Bogdan + Jiang | -- |
| Conclusion | To be drafted by Bogdan + Jiang | -- |
| SI (supporting information) | Not started | Full per-structure metric tables, additional sweeps, xTB input/output files |

## Recommended Submission Timeline

| Week | Milestone |
|------|-----------|
| W1 (Apr 20--27) | Sync cluster data; regenerate figures with real values; finalize DFT results |
| W2 (Apr 28--May 4) | Bogdan + Jiang draft Introduction/Abstract/Conclusion; Nathan reviews Methods |
| W3 (May 5--11) | Internal review cycle; prepare SI; format for JCIM or JCTC |
| W4 (May 12--18) | Final revisions; submit |

## Target Journal Decision

- **JCIM** (methods + negative-result-plus-mitigation): if DFT validation is incomplete or inconclusive. Emphasize failure-mode taxonomy and context-crowding analysis as methodological contributions.
- **JCTC** (computational theory): if DFT validation completes and shows meaningful xTB-vs-DFT correlation. Stronger theory angle with PBE0-D4 benchmarking.

**Recommendation:** Target JCTC if DFT results are available by end of W1. Fall back to JCIM otherwise. The context-crowding validity curve (Figure 4) is novel and citable regardless of journal.

## Key Talking Points for Reviewers

1. This is the **first application of equivariant diffusion models to f-block coordination chemistry**.
2. We identify and quantify a **systematic failure at CN>8** that existing d-block benchmarks do not capture.
3. We introduce **four new evaluation metrics** specific to high-CN metal complexes.
4. RePaint + projection provides a **12.8x improvement** but the 6.4% ceiling reveals fundamental architectural limitations.
5. The context-crowding analysis (Figure 4) provides **actionable guidance** for the field: current architectures should not be applied above CN~8 without modifications.

## Compute Budget (for Methods transparency)

- Fine-tuning: 18 GPU-hours (4x H200, 4.5h wall)
- Generation sweeps (all prompts): ~60 GPU-hours
- xTB optimization: ~200 CPU-hours
- DFT (ORCA): ~400 CPU-hours (estimated)
- Total: ~80 GPU-hours + ~600 CPU-hours
