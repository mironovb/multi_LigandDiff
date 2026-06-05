# Figure status

Cross-checked against the verified results (see `../../RESEARCH_PLAN.md`). The earlier
"Prompts 1–10" drafting pass generated some figures from experiments that **never ran**;
those are flagged here and should be dropped or regenerated before any submission.

| Figure | Status | Action |
|---|---|---|
| `fig1_method_schematic` | OK (schematic) | keep |
| `fig2_training_curves` | OK, but uses representative curve | regenerate from the real log (977.8 → 49.9 @ ep48; early-stop ep63) |
| `fig3_failure_taxonomy` | **Unsupported** — bridging/mutation/perturbation metrics never ran | drop; the real failure mode is the N-valence rejection (Results §4.5) |
| `fig4_validity_vs_crowding` | **Fabricated** — the context-density ablation never ran (no logs) | **delete** |
| `fig5_structure_showcase` | placeholder panels | regenerate from real `eu_tmma_cis` + valid mask1 xyz |
| `fig6_dft_validation` | **No DFT was run** — only ORCA templates exist | drop until DFT is actually run |

## Verified key figures to use instead

- **RePaint sweep** (yield 1.16 → 3.40 → 3.80 → 5.20%; denticity-match peaks at r=5):
  from `paper/tables/table1_repaint_sweep.csv` / `analyze_repaint_sweep.py`.
- **Design degradation** (the headline: 126 → 4 → 0 → 0; maskall 0/6300):
  produced by `analyze_design_test.py` (`design_degradation.png`) from
  `paper/tables/table2_design_degradation.csv`.
- **xTB convergence** (38/41 = 92.7%): `paper/tables/table3_xtb_validation.csv`.
