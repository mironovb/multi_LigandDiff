# Introduction

## 1.1 Motivation: ligand design for rare-earth separations

Separating individual rare-earth elements from one another is a major industrial and
strategic problem, and the selectivity that drives it lives in the **ligand** — the
organic molecule that wraps around the metal ion. Designing ligands that bind one
lanthanide more tightly than its neighbors is slow, expensive, and largely guided by
chemical intuition. A model that could *propose* new ligand structures for a specified
lanthanide coordination environment — for example "two nitrogen and three oxygen
donors, neutral charge, around Eu" — would turn that intuition-driven loop into a
screenable pool of candidates.

## 1.2 Generative models of 3D metal complexes

Equivariant diffusion models have made 3D molecular generation practical: EDM
[hoogeboom2022edm] diffuses atomic coordinates and types under exact SE(3)
equivariance, and structure-conditioned variants (DiffLinker [igashov2024difflinker],
DiffSBDD [schneuing2024diffsbdd]) generate fragments inside a fixed context. For metal
complexes specifically, LigandDiff [jin2024liganddiff] and its multi-ligand successor
**multi-LigandDiff** [jin2024multiligand] generate the 3D coordinates and element
identities of one or more ligands conditioned on a metal center and any fixed context
ligands, using a geometric vector perceptron graph neural network as the diffusion
score function.

These models were developed and benchmarked on **transition-metal** complexes at
**coordination number ≤ 6** — multi-LigandDiff reports ≈89% whole-complex validity on
octahedral (CN = 6) d-block complexes. They have never been tested on **f-block
(lanthanide)** elements or at the **high coordination numbers (CN 7–10)** that
trivalent lanthanides adopt because of their large ionic radii and the non-directional
character of 4f bonding. Whether a 3D diffusion model trained on d-block chemistry can
be carried into this regime — and whether it can *design* rather than merely *complete*
coordination spheres — is open.

## 1.3 This work

We adapt multi-LigandDiff to lanthanide coordination chemistry and high CN, and we
characterize what the adapted model can and cannot do. Our contributions:

1. **A minimal, transferable adaptation.** Extending the metal vocabulary to the
   lanthanides and replacing the hard-coded octahedral denticity tables with a dynamic
   partition generator (`MAX_LIGANDS = 10`) requires reshaping only the input projection
   layer (7 → 11 input channels, zero-padded). Fine-tuning from the d-block checkpoint on
   9,306 CSD lanthanide complexes converges in ≈10 GPU-hours (val-loss −95%).

2. **Completion works, and resampling is a yield knob.** On Eu(TMMA)₂(NO₃)₃ (CN 10),
   the model regenerates a missing ligand reliably and xTB-stably; RePaint resampling
   [lugmayr2022repaint] raises the completion yield monotonically but does not improve
   donor-placement quality past r = 5 — an honest yield-vs-validity trade-off.

3. **De-novo design fails, with a mechanism.** As the fixed scaffold shrinks, validity
   collapses to zero (0 valid / 6,300 attempts from the bare metal), and the rejections
   are ≈100% nitrogen explicit-valence violations. A generic 3D diffusion model that
   paints atom positions and infers chemistry post hoc learns local completion but not
   valence-correct composition.

The result is a clean, reproducible characterization of a generative model on a new
chemical regime, including an informative negative result with a specific chemical
cause. We use it to argue that high-CN, hard-donor rare-earth ligand design needs
**coordination-aware** generation — methods that fix chemical (valence/denticity)
validity first and realize geometry second — rather than generic geometric diffusion.
