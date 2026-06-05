# Supporting Information

_All numbers below are log-traceable; job IDs refer to the cluster runs of record._

## S1. CSD data pipeline

Custom pipeline (pymatgen + ASE; no CSD API license available). Stages: parse 53,333
Ln-containing CIFs (CSD v5.46) → remove disorder/polymeric → **31,979 mononuclear
molecular complexes** → first-shell donors within 2.8 Å of Ln → covalent-graph ligand
decomposition (1.3× sum of covalent radii) → filter (mononuclear, molecular,
CN ∈ {7,8,9,10}, non-Ln elements ∈ {C,N,O,F,P,S,Cl,Br}) → **9,306 training complexes**
(6,563 O/N-donor only). Ligand inventory: 216,509 coordinating instances, 54,946 unique
SMILES.

Consistency check vs. the Jiang group's published CSD analyses: donor distribution
≈ 65% O / 18% N / 14% C; CN distribution peaks at CN 8; Ln–donor distance contracts
2.61 Å (La) → 2.43 Å (Lu) (lanthanide contraction). A per-element breakdown for all 14
lanthanides (parse rates, CN distribution, mean Ln–donor distance, training-candidate
and ligand counts) accompanies this SI (`summary_by_element.csv`).

> These pipeline counts derive from the source CIF curation and are reproduced from the
> project records; the source CIFs are large and not distributed with the code.

## S2. Model and fine-tuning

EDM + GVP-GNN score function, ~3.8M parameters; 5 message-passing layers; 192 scalar +
32 vector features; fully connected graph; variance-preserving diffusion, T = 500.
Adaptation: lanthanides La–Lu (Z = 57–71, except Pm) added to the metal registry;
`MAX_LIGANDS = 10` with a dynamic `denticity_partitions()` generator replacing the
hard-coded CN-6 tables; input projection reshaped 7 → 11 input channels (**zero-padded**,
not re-initialized). Fine-tuning: single NVIDIA H200; AdamW + cosine annealing;
discriminative LR 1e-4 (input projection) / 1e-5 (GVP backbone); batch 256; EMA 0.999;
early-stop patience 15. val-loss 977.76 (ep0) → **49.91 (ep48)**, next-lowest ep57
(60.78), early-stop ep63 (106.37); ~10 GPU-h across three sessions (jobs 12124078,
12138834, 12151181). Convergence sampling: valid_ligand 0.970, connected_ligand 0.938,
valid_complex 0.005.

## S3. RePaint sweep (mask1 completion, Eu(TMMA)₂(NO₃)₃)

See `tables/table1_repaint_sweep.csv`.

| r | attempts | valid | valid_complex | denticity-match | jobs |
|---|---|---|---|---|---|
| 1 | 2,500 | 29 | 1.16% | 13.8% (4/29) | 12329152 |
| 5 | 2,500 | 85 | 3.40% | 15.3% (13/85) | 12329152 |
| 10 | 1,500 | 57 | 3.80% | 10.5% (2/19, partial) | 12340606 |
| 20 | 750 | 39 | 5.20% | — | 12340606 |

Totals: 171 valid (r1+r5+r10), 210 incl. r = 20. Yield rises monotonically; denticity-
match peaks at r = 5 (working point).

## S4. Design-ability sweep

See `tables/table2_design_degradation.csv`. mask_k ligands hidden at once on
Eu(TMMA)₂(NO₃)₃ (5 ligands); RePaint r = 5.

| mask level | context | attempts | valid | jobs |
|---|---|---|---|---|
| mask 1 | 4 of 5 | 2,500 | 126 | 14292188 |
| mask 2 | 3 of 5 | (cut-off) | 4 | 14292188 |
| mask 3 | 2 of 5 | ≥1,500 (cut-off) | 0 | 14344725 |
| mask all | 0 of 5 (bare Eu) | 6,300 (150 × 42 partitions) | 0 | 14344725 |

mask 2/3 are from time-limit-cut-off jobs; `sbatches/design_test.sbatch` re-runs them.
Reproduce with `generate_design_test.py --mask_k {1,2,3,all}` and tabulate with
`analyze_design_test.py`.

## S5. Rejection-mechanism tally

From the design-test logs (`analyze_design_test.py` over the `.err` files): the
dominant rejection is the RDKit message *"Explicit valence for atom # N, 4, is greater
than permitted"* — 151 of 170 logged rejection lines on job 14292188, all nitrogen; the
maskall logged rejections are likewise all nitrogen. Failures are chemical (impossible
Lewis structures), not geometric near-misses.

## S6. Bond-detection artifact (geometry sensitivity check)

A connected-component fragment count returned the expected 5 fragments in 9/38 (24%) of
xTB-converged mask1 structures and 3–4 in the rest; 23/38 (60.5%) showed "new"
cross-ligand bonds after xTB, vs. 0% for the pristine crystallographic reference under
the same protocol. Visual inspection (Avogadro) confirms these are artifacts of the
1.3× covalent-radii bond-detection cutoff merging atoms at threshold-ambiguous
distances — a **geometry sensitivity check**, not chemical fusion. `metrics/` holds the
per-structure `bond_classification.csv` and `pristine_reference.json`.

## S7. xTB and DFT protocols

**GFN2-xTB** [bannwarth2019gfn2xtb] `--opt normal --uhf 6 --cycles 500`, gas phase,
charge 0 (Eu³⁺ f⁶). Convergence of record: mask1 baseline 38/41 (92.7%), RePaint r = 5
81/85. A minority of structures raise IEEE floating-point exceptions, a known GFN2-xTB
limitation for f-block elements. See `tables/table3_xtb_validation.csv`.

**DFT** (prepared, not yet run): ORCA [neese2020orca] PBE0 [adamo1999pbe0]-D4
[caldeweyher2019d4]/def2-TZVP [weigend2005def2], TightSCF SlowConv; Eu via SARC-DKH-TZVP
basis + SK-MCDHF-RSC ECP; neutral, multiplicity 7. Template `orca_templates/pbe0_eu.inp`;
driver `dft_pipeline.py` (prepare/submit/parse) + `sbatches/dft_orca.sbatch`.

## S8. Reproducibility

- Code: `github.com/mironovb/multi_LigandDiff`, branch `ln-adaptation`.
- Checkpoint: `models/ln_finetuned/ln_finetuned_epoch=48.ckpt`.
- Reference complex: `eu_tmma_cis.xyz` (Eu(TMMA)₂(NO₃)₃, CCDC VEDTAA01).
- Environment: `module load miniforge/25.11.0-0 cuda/12.9.1; conda activate ligdiff`.
- Cluster job suite and submission order: `sbatches/README.md`.
- Key scripts: `finetune.py`, `generate_mask1.py`, `generate_design_test.py`,
  `analyze_design_test.py`, `analyze_repaint_sweep.py`, `xtb_pipeline.py`,
  `dft_pipeline.py`.

## S9. Job-ID index

| Stage | Job ID(s) | Outcome |
|---|---|---|
| Fine-tune (3 sessions) | 12124078, 12138834, 12151181 | best val-loss 49.9 @ ep48 |
| Mask1 baseline gen | 12207365 | 41 valid |
| RePaint r=1, r=5 | 12329152 | 29, 85 valid |
| RePaint r=10, r=20 | 12340606 | 57, 39 valid |
| xTB (baseline, r=5) | 12329159, 12340608 | 38/41, 81/85 |
| Design test (mask1/2) | 14292188 | 126, 4 valid |
| De-novo (mask3/all) | 14344725 | 0, 0/6300 |
