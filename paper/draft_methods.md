# Methods

## 2.1 Data Preparation

We extracted 95,279 lanthanide (Ln) coordination complexes from the Cambridge Structural Database (CSD),^[ccdc2024csd] retaining structures with coordination numbers (CN) 7--10.
This CN range captures the dominant coordination geometries of trivalent Ln ions, which exhibit higher CN than d-block metals due to larger ionic radii and the non-directional character of 4f orbitals.
The dataset spans all 15 lanthanides (La--Lu, excluding Pm) with the most represented metals being Dy (17,842), Eu (12,631), Gd (11,203), and Tb (9,817).

Curation followed the protocol of Jin and Merz^[jin2024multiligand] with modifications for f-block chemistry.
Structures were filtered to remove: (i) entries with disorder flags or missing coordinates, (ii) complexes with >200 atoms (computational tractability), (iii) structures containing elements outside the supported vocabulary (C, N, O, S, P, F, Cl, Br plus the 33 metals in our extended vocabulary), and (iv) structures where the metal--donor distance exceeded 3.5 Angstroms for any coordinated atom, indicating possible dissociation.

Each complex was decomposed into metal center, context atoms (coordinated ligands to remain fixed during generation), and target atoms (ligands to be generated).
Denticity partitioning was generalized from the octahedral (CN=6) scheme of Jin and Merz to arbitrary CN using an integer-partition function that enumerates all possible ligand denticity combinations summing to a given remaining CN.
For example, a CN=10 complex with two bidentate context ligands (contributing CN=4) has remaining CN=6, yielding partition options [6], [5,1], [4,2], ..., [1,1,1,1,1,1].
The training split was 80/10/10 (train/val/test) stratified by metal identity.

## 2.2 Model Architecture

Our model inherits the multi-LigandDiff architecture:^[jin2024multiligand] an equivariant graph neural network (EGNN)^[hoogeboom2022edm] augmented with geometric vector perceptron (GVP) layers for processing 3D molecular graphs.
The denoising diffusion probabilistic model (DDPM) operates on atom coordinates and types jointly, generating all atoms of one or more ligands conditioned on the metal center and any context atoms.

Two modifications were required for f-block adaptation.
First, `MAX_LIGANDS` was increased from 4 to 10 to accommodate the higher CN of Ln complexes.
At CN=10, a complex may contain up to 10 monodentate ligands, each requiring an independent ligand-group channel.
Second, the element vocabulary (`metals2idx`) was extended to include all 14 stable lanthanides (La--Lu, Z=57--71), adding 14 entries to the original 19 d-block metals.
The atom-type one-hot encoding (C, N, O, S, Br, Cl, P, F; 8 types) was unchanged, as Ln donor atoms are overwhelmingly O and N.

The architecture hyperparameters were kept identical to the pretrained d-block model: hidden_nf=192, n_layers=5, drop_rate=0.2.
This choice enables direct comparison between the pretrained and fine-tuned checkpoints and avoids confounding architecture changes with training-data effects.

## 2.3 Fine-Tuning

The model was initialized from the pretrained d-block checkpoint of Jin and Merz,^[jin2024multiligand] which was trained on ~370,000 d-block complexes with CN=2--6.
Fine-tuning was performed for 63 epochs on 4 NVIDIA H200 GPUs on the MIT Engaging cluster using PyTorch Lightning with the Adam optimizer (lr=1e-4), batch size 256 (64 per GPU), and early stopping on validation loss with patience 10.

Validation loss decreased from 977 (pretrained weights evaluated on Ln data) to 50 over the training run, with the majority of the drop occurring in the first 20 epochs (Figure 2a).
The large initial loss reflects the domain gap between d-block (CN 2--6) and f-block (CN 7--10) coordination chemistry.
The final checkpoint at epoch 48 was selected by minimum validation loss (early stopping triggered at epoch 63).

Total fine-tuning wall time was approximately 18 GPU-hours (4.5 hours wall clock on 4x H200).
Including generation sweeps (Prompts 4--7), xTB optimization (Prompt 8), and DFT validation (Prompt 9), the total compute budget for this study was approximately 120 GPU-hours and 800 CPU-hours.

## 2.4 Sampler Variants

We tested three reverse-sampling strategies, applied during the DDPM denoising trajectory from x_T to x_0.

**Baseline (vanilla DDPM).** Standard ancestral sampling identical to Jin and Merz,^[jin2024multiligand] where each denoising step applies the learned score function without modification.

**RePaint resampling.**^[lugmayr2022repaint] At each timestep t, after computing the denoised estimate x_{t-1}, we re-noise x_{t-1} back to x_t and re-denoise, repeating this cycle r times.
This resampling strategy, originally developed for image inpainting, provides the denoising network with multiple opportunities to harmonize the generated atoms with the fixed context.
We swept r in {1, 5, 10, 20}, where r=1 recovers the vanilla baseline.
Higher r incurs proportionally higher compute cost (wall time scales linearly with r).

**Hard geometric projection.**^[christopher2024projected] After each denoising step, generated atoms are projected onto the surface of a minimum-distance exclusion shell around context atoms from different ligand groups.
The exclusion distance d_min is annealed linearly from 1.5 Angstroms (at the noisiest timestep) to 1.3 Angstroms (at the final step).
This projection is applied only between atoms belonging to different ligand groups; same-group and metal--atom distances are unconstrained.
The 1.3 Angstrom final threshold was chosen to be below the shortest physically reasonable O--O single bond (1.48 Angstroms) but above the numerical noise floor, thereby preventing inter-ligand bridging artefacts without distorting legitimate intra-ligand geometry.

**Combined (RePaint + projection).** The recommended configuration applies both resampling (r=10) and projection at each timestep.
RePaint reduces context mutations by providing repeated context-aware denoising, while projection eliminates residual inter-ligand bridges.

## 2.5 Evaluation Metrics

Generated structures were assessed using the validity metrics of Jin and Merz^[jin2024multiligand] (valid_ligand, connected_ligand, valid_complex) supplemented by four new metrics designed for high-CN metal complexes.

**Context-atom perturbation distance.** The mean Euclidean displacement of context atoms from their reference positions, measured in Angstroms.
Although context atoms are nominally fixed during generation, diffusion noise at early timesteps can perturb them, and the denoising trajectory may not fully recover their original positions.
Values above 0.5 Angstroms indicate significant context drift.

**Inter-ligand bridging rate.** The number of atom pairs from different ligand groups with interatomic distance below 1.5 Angstroms.
Such bridges are chemically implausible (they would require covalent bonding between independent ligands) and represent the most common failure mode at high CN.

**Ligand-type mutation rate.** The number of context ligands whose heavy-atom composition (element hash) differs from the reference after generation.
A mutation indicates that the denoising process has altered the chemical identity of a context ligand---for example, converting a nitrate (NO3) into a water (H2O) or an organic fragment.

**Denticity recovery.** The fraction of generated ligands whose denticity (number of metal-coordinating atoms) matches the requested denticity from the reference partition.
We report this as a confusion matrix of requested vs. actual denticity (Figure 3d).

These metrics complement the validity/invalidity binary of PoseBusters-style evaluation^[buttenschoen2024posebusters] by providing diagnostic resolution into *why* structures fail, enabling targeted mitigation (Sections 4.4--4.6).

## 2.6 Geometry Optimization

Structures passing validity checks were optimized at two levels of theory.

**GFN2-xTB.** Semiempirical tight-binding optimization using GFN2-xTB^[bannwarth2019gfn2xtb] (version 6.7.1, conda-forge).
Gas-phase calculations used charge 0 and UHF=6 (consistent with Eu^3+ f^6 configuration).
Geometry optimization ran for up to 500 cycles with the `--opt normal` convergence criterion.
Structures not converging within 500 cycles or producing floating-point exceptions were classified as xTB failures.

**DFT (JCTC submission only).** Selected structures (6--10, stratified by failure category) were further optimized at the PBE0-D4/def2-TZVP level^[adamo1999pbe0,caldeweyher2019d4,weigend2005def2] using ORCA 5.0.4.^[neese2020orca]
Europium was treated with the SARC-DKH-TZVP basis set and second-order Douglas--Kroll--Hess scalar relativistic Hamiltonian.
Mulliken population analysis provided Eu partial charges for assessing charge-transfer accuracy.
Eu--donor distances were compared to experimental crystallographic values from Kravchuk et al.^[kravchuk2024ejic] (CSD refcode VEDTAA01): Eu--O(TMMA) = 2.33--2.40 Angstroms, Eu--O(NO3) = 2.51--2.56 Angstroms.
