# Conclusion

We adapted multi-LigandDiff, a 3D equivariant diffusion model built for d-block metal
complexes at CN ≤ 6, to f-block lanthanide coordination chemistry and to coordination
numbers 7–10 — a regime no prior generative model of metal complexes had addressed. The
adaptation is deliberately minimal: only the input-projection layer is reshaped, and
fine-tuning from the pretrained checkpoint converges in ≈10 GPU-hours with a ≈95%
reduction in validation loss.

The adapted model is a competent **completion** tool and an incompetent **designer**,
and the boundary between the two is sharp. Given a lanthanide center plus most of its
ligands, it regenerates a missing ligand reliably and xTB-stably, and inference-time
RePaint resampling buys completion yield (1.16% → 5.20%) without improving placement
quality beyond r = 5. Given only the bare metal, it produces **0 valid structures out
of 6,300 attempts**, and the degradation as the scaffold shrinks (126 → 4 → 0 → 0) is
steep. The failure is chemical, not geometric: ≈100% of rejections are nitrogen
explicit-valence violations. A model that generates 3D geometry and infers a Lewis
structure post hoc has no notion of valence-correct composition and cannot build a
coordination sphere from scratch.

This is an informative negative result with a concrete mechanism, and it points
directly at the fix: **coordination-aware generation** that fixes chemistry first
(valence-valid ligand graphs, explicit denticity/CN/donor identity, hard constraints
inside the sampler) and realizes geometry second — extended to the high-CN, hard-donor
rare-earth regime that the present d-block-derived tools, and this model, do not yet
master. Pursuing that direction, with selectivity between adjacent lanthanides as the
ultimate objective, is the subject of ongoing work.
