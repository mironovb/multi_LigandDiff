# Discussion

## Why Does f-Block Fail Where d-Block Succeeds?

The central finding of this study is that the multi-LigandDiff architecture, which achieves 89% valid_complex on d-block octahedral complexes (CN=6),^[jin2024multiligand] produces only 0.5--6.4% valid structures for lanthanide complexes at CN=10.
Three factors contribute to this gap.

**Coordination crowding.**
The context-density ablation (Figure 4) demonstrates that validity degrades monotonically with the number of fixed context atoms, independent of chemical identity.
At CN=10, the coordination sphere contains 35 heavy atoms within ~3 Angstroms of the metal center, compared to ~15 for a typical CN=6 octahedral complex.
The denoising network must resolve steric clashes among atoms packed at approximately twice the spatial density, and the DDPM noise schedule---calibrated for d-block geometries---does not provide sufficient signal for the network to resolve these clashes at intermediate timesteps.

**Fluxional bonding and geometric diversity.**
Lanthanide coordination geometry is significantly more variable than d-block geometry.
While d-block complexes strongly prefer octahedral, tetrahedral, or square-planar arrangements (determined by crystal-field splitting), Ln complexes adopt geometries ranging from capped trigonal prism to bicapped square antiprism with low energy barriers between isomers.^[kravchuk2024ejic]
This conformational flexibility means the training data contains a broader distribution of metal--donor angles and distances, making it harder for the model to learn a sharp posterior over valid geometries.

**Training data sparsity.**
Although the CSD contains 95,279 Ln complexes, these are distributed across 14 metals and CN 7--10, yielding an average of ~1,700 structures per (metal, CN) pair.
The pretrained d-block model was trained on ~370,000 structures concentrated in CN 2--6, providing ~20x more training examples per CN bin.
Fine-tuning partially compensates (val_loss 977 to 50), but the remaining loss gap indicates incomplete convergence.

## What Does This Tell Us About Generative Models for Metal Complexes?

Our results suggest that current equivariant diffusion architectures face a scalability wall at high coordination numbers.
The valid_complex metric compounds per-ligand error rates across ligand groups, and this compounding becomes catastrophic at CN>8 even when individual ligand validity is reasonable (82%).

The failure-mode taxonomy (Section 4.2) reveals that the dominant errors---bridging, mutation, and perturbation---are all *geometric* rather than *chemical* in nature.
The model generates plausible atom types but places them in physically impossible positions.
This suggests that architectural improvements should focus on geometric conditioning rather than chemical diversity: for example, explicit distance-matrix constraints or equivariant attention mechanisms that can resolve multi-body steric interactions.

The partial success of RePaint resampling and hard projection (Section 4.6) supports this interpretation.
RePaint provides repeated opportunities for the network to resolve geometric clashes, while projection enforces hard geometric constraints post-hoc.
Neither modifies the network itself, and their combined 12.8-fold improvement over vanilla sampling indicates substantial untapped geometric reasoning in the existing model weights.

## Limitations

Several limitations should be considered when interpreting these results.

First, all experiments were conducted on a single metal (Eu) and a single reference structure (Eu(TMMA)_2(NO_3)_3).
While Eu was chosen for its experimental relevance and well-characterized crystallography,^[kravchuk2024ejic] the generalizability to other lanthanides (particularly early Ln with larger ionic radii) and to different ligand systems has not been demonstrated.

Second, the GFN2-xTB level of theory is known to be unreliable for lanthanides: 43% of our valid structures failed to converge, and the parametrization of 4f electrons in tight-binding methods remains an active area of development.^[bannwarth2019gfn2xtb]
The DFT validation (PBE0-D4 with SARC-DKH relativistic treatment) provides a more reliable assessment but was limited to a small number of structures.

Third, the computational cost of RePaint resampling scales linearly with r, making r=10 approximately 10x more expensive than vanilla sampling.
At the current 6.4% validity rate, generating 100 valid structures requires ~1,500 samples and ~75 minutes of H200 GPU time, which may limit high-throughput applications.

Fourth, our evaluation metrics, while more detailed than prior work, still do not capture all aspects of chemical plausibility.
For example, we do not assess metal oxidation state consistency, spin-state compatibility, or thermodynamic stability relative to alternative coordination isomers.

## Future Directions

Several approaches may improve generative quality for high-CN complexes.

**Classifier-free guidance.** Retraining with conditional and unconditional denoising (as in classifier-free guidance for images) could allow the model to increase the weight of context-conditioning at inference time, potentially reducing mutations without post-hoc fixes.

**DPS-style Tweedie-decoded penalties.** Diffusion posterior sampling (DPS) enables differentiable loss functions (e.g., Lennard-Jones repulsion between ligand groups) to be applied through the Tweedie denoised estimate x_0|t at each timestep.
Unlike hard projection, this approach is smooth and could better preserve coordination geometry while penalizing steric clashes.

**Architector-augmented training data.** The CSD provides only experimentally characterized structures, biasing the training distribution toward crystallizable complexes.
Architector^[taylor2023architector] can generate physically reasonable Ln geometries from SMILES input, potentially expanding the training set by 10--100x and covering CN=10 geometries more densely.

**Multi-scale denoising.**
A hierarchical approach---first generating the coordination polyhedron (metal + donor atoms), then generating each ligand scaffold conditioned on its donor atom positions---could decompose the CN=10 problem into a manageable metal-geometry step (10 atoms) followed by several independent ligand-generation steps (5--10 atoms each).

These directions suggest that the CN=10 barrier is not fundamental but rather reflects the mismatch between current architectures' implicit assumptions (moderate CN, well-separated ligand groups) and the geometric reality of f-block coordination chemistry.
