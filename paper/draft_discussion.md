# Discussion

> Rewritten to match the verified results (see `RESEARCH_PLAN.md`). The previous
> draft attributed the failure to "coordination crowding" on the basis of a context-
> density ablation that never ran, and reported a "12.8× improvement" to 6.4% that
> has no logs. The verified mechanism is different and more specific: a valence /
> coordination-validity gap exposed when the scaffold is removed.

## Completion vs. design: what the model actually learned

The fine-tuned model is a competent **completion** tool and an incompetent
**designer**, and the boundary between the two is sharp. Given the metal plus four of
five ligands it regenerates the fifth reliably (126 valid mask1 structures, xTB-
stable at 92.7%); given the metal alone it produces nothing valid (0/6300). The
degradation is monotonic and steep (126 → 4 → 0 → 0 as 1, 2, 3, then all ligands are
hidden), and the cliff arrives as early as two hidden ligands.

The mechanism is the key point. The de-novo rejections are **~100% nitrogen explicit-
valence violations**, not geometric near-misses. multi-LigandDiff diffuses atom
positions and types jointly and then *infers* chemistry post hoc (bond perception +
RDKit sanitization). Its only coordination knowledge is implicit — the geometric
distribution of training complexes — plus a procedural denticity partition table; it
has **no explicit notion of atomic valence**. With a scaffold, the fixed context
constrains the pocket enough that local valence falls out correctly. Without one, the
model paints an atom cloud into a fully connected graph and hopes the resulting Lewis
structure is legal; for a ~35-heavy-atom CN-10 sphere it essentially never is.

This reframes the result away from "high coordination number is intrinsically too
crowded." The completions *are* CN-10 and *do* succeed. What fails is **composition
from scratch**: building a valence-correct ligand framework without a template to
copy. The inpainting objective, as trained, teaches local consistency, not global
composition.

## Why resampling helps with yield but not validity

RePaint raises the valid-complex yield monotonically (1.16 → 3.40 → 3.80 → 5.20% from
r = 1 to 20) by giving the denoiser repeated chances to re-harmonize generated atoms
with the fixed context. But it does not install a notion of valence, so the donor-
placement quality (denticity-match) peaks at r = 5 and then declines: beyond the
working point, extra resampling yields more structures that pass the coarse filter
without better fine-grained placement. Inference-time tricks buy throughput on the
*completion* task; they do not address the *design* failure, which is chemical.

## Implications for generative models of metal complexes

The honest contribution here is a clean, diagnosable **negative result** on a regime
no prior generative model had touched (f-block, CN 7–10): a generic 3D diffusion
model adapts cheaply and completes well but cannot design de novo, and the failure has
a specific, citable cause (valence). This directly motivates **coordination-aware
generation**: fix chemistry first, then realize geometry, rather than generating
geometry and inferring chemistry. Concretely — generate or select the ligand as a
valence-valid chemical graph (where nitrogen cannot have four bonds) before 3D
placement; make denticity / CN / donor identity explicit inputs; and, if a 3D
generative stage is retained, enforce valence/CN/charge constraints *inside* the
sampler (e.g. via the exclusion-shell projection already implemented) instead of
rejecting 99.99% of free samples.

## Limitations

- All generation experiments used a single metal (Eu) and a single reference
  (Eu(TMMA)₂(NO₃)₃). Generalization to other lanthanides and ligand systems is not
  demonstrated.
- The mask 2 and mask 3 points come from time-limit-cut-off jobs; a dedicated run is
  needed to pin the exact validity cliff. maskall = 0/6300 is conclusive on its own.
- Validity here is GFN2-xTB + bond-perception/RDKit. **No DFT validation has been
  run**; the prepared ORCA PBE0-D4/def2-TZVP protocol (Eu via SARC-DKH-TZVP + SK-MCDHF-RSC
  ECP, gas-phase) is future work. (Implicit solvent (SMD dodecane) is an OPTIONAL upgrade,
  not yet implemented in the template.)
- GFN2-xTB is known to be imperfect for f-block elements (a minority of structures
  raise IEEE exceptions); reported convergence rates should be read in that light.

## Future directions

The de-novo gap is the entry point to a rare-earth-native, coordination-aware
platform (see `RESEARCH_PLAN.md`, Track B): predict-then-build with a valence-valid
ligand graph, extended past the d-block CN ≤ 6 regime to CN 7–10; hard donor (O/N)
priors and CN 8–10 saturation for trivalent lanthanides; the lanthanide contraction
as a selectivity handle for differential binding of adjacent Ln; and first-shell
counter-ions/solvent (nitrate, water) modeled explicitly. Whether to keep diffusion
with hard constraints or move to a predict-then-build architecture is the central
open design decision.
