# Methods

> Rewritten to verified, log-traceable values (see `RESEARCH_PLAN.md`). The previous
> draft contained unsupported figures (e.g. "95,279 complexes", "4× H200, 18 GPU-h",
> a context-density ablation and a cross-architecture experiment that never ran);
> those have been corrected or removed.

## 2.1 Data preparation

Training data were derived from lanthanide (Ln) coordination complexes in the
Cambridge Structural Database (CSD v5.46). Because no CSD Python-API license was
available, a custom pipeline using pymatgen and ASE was built in place of the CSD
API. Starting from **53,333 Ln-containing CIFs**, disordered and polymeric
structures were removed, leaving **31,979 mononuclear molecular complexes**. First-
shell donors were identified as atoms within 2.8 Å of the Ln center (in code this is
the molSimplify covalent-radii rule shared with the validity gate, `src/bonding.py`,
which gives 2.78–2.81 Å for the dominant Ln–O/N donors — matching this figure; an
earlier flat 3.0 Å prep cutoff was retired as it inflated the CN 9/10 tail, Finding 6);
ligands were obtained by building a covalent-bond graph (1.3× the sum of covalent radii),
removing the Ln node, and taking connected components containing at least one donor.

Filtering to training candidates (mononuclear; molecular; CN ∈ {7,8,9,10}; all
non-Ln elements in {C,N,O,F,P,S,Cl,Br}) yielded **9,306 training complexes** (6,563
O/N-donor only). The pipeline reproduces the Jiang group's published CSD trends
(donor distribution ≈ 65% O / 18% N / 14% C; CN peak at 8; Ln–donor distance
contracting 2.61 Å (La) → 2.43 Å (Lu)), which validated the home-built decomposition.

Each complex was tensorized (positions, one-hot atom types, nuclear charges, ligand-
group membership, coordination-site index) and augmented by ligand masking. The
validation split was performed **by complex** (no leakage between mask configurations
of the same molecule).

## 2.2 Model architecture

The model inherits the multi-LigandDiff architecture: an equivariant diffusion model
(EDM) with a geometric vector perceptron graph neural network (GVP-GNN) as the score
function (~3.8M parameters; 5 message-passing layers; 192 scalar + 32 vector features;
fully connected graph; variance-preserving diffusion, T = 500). The DDPM jointly
diffuses atom coordinates (continuous) and atom types (one-hot), generating the atoms
of one or more ligands conditioned on the metal center and any fixed context.

Two changes adapt it to the f block. First, the metal vocabulary in `src/const.py`
was extended to include the lanthanides La–Lu (Z = 57–71, except Pm). Second, the
hard-coded octahedral (CN = 6) denticity partition tables were replaced by a dynamic
`denticity_partitions()` generator and `MAX_LIGANDS = 10`, so CN 7–10 (with far more
denticity partitions than CN 6) is supported. Donor detection in
`src/molecule_builder.py` was extended for Ln complexes (`BondedOct=False`; F/Cl/Br
donors permitted). Architecture hyperparameters were otherwise held identical to the
pretrained d-block model to avoid confounding architecture changes with training-data
effects.

The adaptation requires reshaping a single layer: the input projection grows from
in-dim 7 to in-dim 11 (output stays 192) to carry the wider ligand-group tensor. The
new columns were **zero-padded** (not randomly re-initialized) so the transferred
d-block weights stay aligned with the original 7 ligand-group slots.

## 2.3 Fine-tuning

The model was initialized from the pretrained d-block checkpoint of Jin and Merz and
fine-tuned on a **single NVIDIA H200**. AdamW with cosine annealing was used with
**discriminative learning rates**: 1e-4 for the new input projection, 1e-5 for the
pretrained GVP backbone (10× lower, to avoid overwriting transferred features). Batch
size 256; EMA 0.999 at validation; early stopping on validation loss with **patience
15**.

Validation loss fell from **977.8** (pretrained weights on Ln data, epoch 0) to a
minimum of **49.9 at epoch 48** (≈95% reduction); early stopping fired at epoch 63.
The epoch-48 checkpoint was selected. Total fine-tuning was **≈10 GPU-hours** across
three preemption-interrupted sessions, resumed via PyTorch Lightning checkpoints
(335 steps/epoch × batch 256 ≈ 85,760 samples/epoch).

## 2.4 Sampler variants

Two reverse-sampling strategies were used during DDPM denoising from x_T to x_0.

**Baseline (vanilla DDPM).** Standard ancestral sampling, identical to Jin and Merz.

**RePaint resampling.** At each timestep t, after computing x_{t-1}, the sample is
re-noised back to x_t and re-denoised, repeating r times, so the generated region
re-aligns with the fixed context ("boundary harmonization"). r ∈ {1,5,10,20}; r = 1
recovers the baseline. Wall time scales linearly with r. No retraining is required.

A hard exclusion-shell projection (`src/projection.py`) is also implemented but was
not used for the results reported here; it is retained for the constrained-sampling
direction discussed in Track B.

## 2.5 Generation experiments

All generation used the Eu(TMMA)₂(NO₃)₃ reference (TMMA =
N,N,N′,N′-tetramethylmalonamide; CCDC VEDTAA01; CN 10; 5 ligands — 2 bidentate TMMA +
3 bidentate nitrate — 35 heavy atoms; cis).

**Completion (mask1).** Fix the full complex as context, hide one ligand, regenerate
it (`generate_mask1.py`). The RePaint sweep was run in this mode.

**Design-ability sweep.** Hide `k` ligands at once and ask the model to regenerate
them (`generate_design_test.py --mask_k 1|2|3|all`). For each k, every size-k subset
of ligands is masked, with independent seeds per subset, expanded over all legal
denticity partitions of the remaining CN. `mask all` leaves only the bare Eu center
as context — true de-novo generation of the whole coordination sphere (150 seeds ×
42 CN-10 partitions = 6300 attempts).

## 2.6 Validity and evaluation

Generated heavy-atom structures were assessed with the validity metrics of Jin and
Merz (valid_ligand, connected_ligand, valid_complex): bond perception
(molSimplify/OpenBabel) assigns bonds and RDKit sanitizes. Denticity-match is the
fraction of valid structures whose generated-ligand denticity matches the reference
partition. Rejection reasons were tallied from the RDKit sanitization messages in the
job logs (`analyze_design_test.py`).

A note on bond perception: the 1.3× covalent-radii cutoff can merge a generated atom
sitting at a threshold-ambiguous distance from a context ligand into that ligand,
producing spurious "cross-ligand bonds." This is a **geometry sensitivity check**,
not chemical fusion (confirmed by visual inspection; 0% for the pristine reference
under the same protocol), and is reported as such.

## 2.7 Geometry optimization

**GFN2-xTB.** Valid structures were optimized with GFN2-xTB (`--opt normal --uhf 6
--cycles 500`, gas phase, charge 0 for Eu³⁺ f⁶). Structures failing to converge in
500 cycles or raising IEEE floating-point exceptions were counted as xTB failures.

**DFT.** ORCA templates are prepared (`orca_templates/pbe0_eu.inp`, `dft_pipeline.py`)
at the **PBE0-D4/def2-TZVP** level (TightSCF, SlowConv), with europium described by the
**SARC-DKH-TZVP basis and the SK-MCDHF-RSC effective core potential**; the complex is
treated as neutral with spin multiplicity 7 (Eu³⁺, 4f⁶, S = 3), gas phase. **DFT
calculations have not yet been run** and no DFT results are reported here; the protocol
and the submission/parse pipeline are provided for the planned validation of the
reference plus a small stratified set of valid completions.
