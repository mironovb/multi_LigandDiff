# DFT-Validated Completion Showcase — Track A, run 2/2 (Prompt 25)

> **STATUS: PREPARED & RUNBOOK-VERIFIED — DFT HAS NOT RUN.**
> No ORCA outputs exist; no comparison table exists; **no "DFT-confirmed" claim has been
> made anywhere in the paper.** This file records the verified prepared state, two blocking
> bugs that were found and fixed, and the exact steps that remain on the cluster. The real
> results table will overwrite this file once `dft_pipeline.py parse` runs against converged
> ORCA logs.
>
> _Run on `MacBook-Pro-Bogdan-700.local` (local, off-cluster), 2026-06-17._

## Honesty gate (held)

- `grep -l "THE OPTIMIZATION HAS CONVERGED" dft_work/*/orca_output.log | wc -l` → **0**
- `paper/tables/dft_comparison.csv` → **absent**
- Completion claim left **unchanged**: *"Eu(TMMA)₂(NO₃)₃ in the fixed pocket, **xTB-stable
  (38/41 = 92.7%)**"* (`strategy.md`, `executive_summary.md`, `manuscript.md`,
  `draft_SI.md` — all still read "DFT prepared, not yet run"). It will be upgraded to
  "DFT-confirmed" **only** after converged ORCA logs back a real `dft_comparison.csv`.

## Why DFT did not run here

This session is on a local macOS machine: `orca` not on PATH, `sbatch` not present, and the
selection input `xtb_results/eu_tmma_mask1_epoch48/` (cluster-only) is absent. A single
PBE0-D4/def2-TZVP optimisation of a CN≈10 Eu complex needs the cluster's 4 h preemptible
wall (`sbatches/dft_showcase.sbatch` → `sbatches/dft_orca.sbatch`). Everything that does
**not** require ORCA was executed and verified here.

## Two blocking bugs found & fixed in `dft_pipeline.py`

Both were in committed code (commit `8542bb7`) and would have silently broken the cluster run.

1. **Module crashed on import under the ligdiff Python (3.9.25).**
   `parse_orca_final_xyz(...) -> list | None` is a PEP 604 union, evaluated at definition
   time → `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`. Both
   `prepare` and `parse` aborted before doing any work.
   **Fix:** `from __future__ import annotations` (defers all annotations to strings; no
   runtime behaviour change — the module never introspects annotations). Verified: module
   now imports and `dft_pipeline.py parse --help` works.

2. **`glob` bracket bug would have emptied the showcase to the reference alone.**
   Converged xTB geometries are named with literal brackets, e.g.
   `0_22_eu_tmma_cis_[2, 2, 2, 2]_[2].xyz`. `select_structures` resolved them with
   `glob.glob(f"{xtb_dir}/**/converged/{name}.xyz")`, and `glob` reads `[...]` as a
   character class → **0 matches for every structure** → all generated candidates skipped
   → `prepare` would write only the `VEDTAA01_reference` job. Demonstrated:
   `glob.glob(".../converged/0_22_..._[2, 2, 2, 2]_[2].xyz")` → `[]`, whereas
   `glob.escape(name)` matches the file.
   **Fix:** escape the structure name (`glob.escape(name)`) before building the pattern
   (`**/` left intact).

## Local verification (real code, real data — no fabrication)

- `bash -n sbatches/dft_showcase.sbatch` → OK; `bash -n sbatches/dft_orca.sbatch` → OK.
- `orca_templates/pbe0_eu.inp` present; renders with full token substitution.
- Ran the **real** `prepare` against a same-format xTB result set
  (`ligandgen/multi_LigandDiff/xtb_results/mask1_baseline`, produced by the identical xTB
  pipeline, same `converged/` + `summary.csv` layout as `eu_tmma_mask1_epoch48`). It
  selected **9 dirs (8 generated completions + VEDTAA01 reference)**, each with:
  - `input.inp` fully substituted — `! PBE0 def2-TZVP D4 TightSCF SlowConv`, `%pal nprocs 16`,
    `%maxcore 3500`, `* xyzfile 0 7 input.xyz` — **no stray `{tokens}`**;
  - `input.xyz` with exactly one `Eu` and H-complete ligands (43 atoms).
  This proves the `prepare → ORCA-input` path is correct end to end. The actual epoch-48
  showcase set is selected on the cluster from `xtb_results/eu_tmma_mask1_epoch48` (38/41
  converged), ranked lowest-xTB-energy first, ≤2 per category, capped at 8 + reference.

## Level of theory (ground truth = `orca_templates/pbe0_eu.inp`)

**PBE0-D4 / def2-TZVP**, TightSCF SlowConv; **Eu** via **SARC-DKH-TZVP** basis +
**SK-MCDHF-RSC** ECP; charge **0**, multiplicity **7** (Eu³⁺ 4f⁶, S = 3); SCF MaxIter 300,
geom MaxIter 200. **Gas-phase — the template contains no `%cpcm`/SMD implicit-solvent block.**

> ⚠ **Provenance discrepancy to resolve before running.** `strategy.md` and the Prompt-25
> text describe "PBE0-**D3**, **ECP28MWB**, **SMD dodecane** (implicit solvent)". The actual
> template — and `paper/draft_SI.md` — are **D4, SARC-DKH-TZVP + SK-MCDHF-RSC, gas-phase**.
> The paper SI is consistent with the template; `strategy.md` is stale. Decide which is
> canonical: either add a `%cpcm SMD true; SMDsolvent "..." end` block to the template (and
> keep the solvent wording), or drop the "implicit solvent / D3 / ECP28MWB" wording from
> `strategy.md`. Do **not** report "implicit solvent" until the template implements it.

## Caveat for the eventual comparison table

The reference `eu_tmma_cis.xyz` (VEDTAA01) is **heavy-atom only (35 atoms, no H)**, whereas
the generated completions are **H-complete (43 atoms)**. Total-energy differences across
different molecular formulae are not physically meaningful, so the `dft_deltaE_kcal_mol`
column (computed as `E − E_ref`) should **not** be over-interpreted. The sound comparisons
are **per-structure SCF/geometry convergence** and **Eu–donor distances vs. the Kravchuk
reference**, not ΔE-vs-reference.

## Remaining steps (cluster) to finish the showcase

```bash
# 1. prepare + submit one ORCA job per structure (AutoStart resumes from .gbw on preemption)
sbatch sbatches/dft_showcase.sbatch

# 2. when jobs finish, count convergence
grep -l "THE OPTIMIZATION HAS CONVERGED" dft_work/*/orca_output.log | wc -l

# 3. build the comparison table from converged logs
python dft_pipeline.py parse --work-dir dft_work --reference eu_tmma_cis.xyz \
    --output-csv paper/tables/dft_comparison.csv

# 4. ONLY if structures converged: upgrade "xTB-stable" -> "DFT-confirmed" in
#    strategy.md / manuscript.md / executive_summary.md / draft_SI.md, then commit per Prompt 25.
```

## Results (to be filled by `parse` against converged ORCA logs)

| structure | category | DFT converged | opt cycles | final E (Ha) | CN | Eu–O(TMMA) (Å) | Eu–O(NO₃) (Å) | dissociated? |
|---|---|---|---|---|---|---|---|---|
| _pending — no ORCA runs yet_ | | | | | | | | |

Kravchuk/VEDTAA01 reference donor ranges (from `dft_pipeline.py`):
Eu–O(TMMA) **2.33–2.40 Å**, Eu–O(NO₃) **2.51–2.56 Å**, dissociation cutoff 3.5 Å.

**Provenance still to capture at run time:** ORCA version (header of `orca_output.log`),
SLURM job ids (`dft_work/job_ids.json` or the launcher `.out`), and compute host.
