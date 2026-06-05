# Abstract

Designing ligands that bind one lanthanide more tightly than its neighbors is central
to rare-earth separations, yet it remains slow and intuition-driven. Generative models
of 3D metal complexes are a promising shortcut, but existing models (LigandDiff,
multi-LigandDiff) were developed and validated only on d-block metals at coordination
numbers (CN) ≤ 6. We present the first adaptation of a 3D equivariant diffusion model
to f-block (lanthanide) coordination chemistry and to the high coordination numbers
(CN 7–10) that trivalent lanthanides adopt. Starting from a d-block-pretrained
multi-LigandDiff checkpoint, a single input-projection layer is reshaped (zero-padded)
and the metal vocabulary and denticity machinery are generalized; fine-tuning on 9,306
CSD lanthanide complexes in ≈10 GPU-hours reduces validation loss by ≈95% (977.8 →
49.9). On the model europium complex Eu(TMMA)₂(NO₃)₃ (CN 10), the fine-tuned model
**completes** a missing ligand reliably — 126 valid structures, 38/41 (92.7%) stable
under GFN2-xTB — and inference-time RePaint resampling raises the completion yield
monotonically (1.16% → 5.20% from r = 1 to 20), though donor-placement quality peaks at
r = 5: resampling buys yield, not validity. The model **cannot design** de novo,
however: as the fixed scaffold is shrunk, valid output collapses (126 → 4 → 0 → 0 valid
as one, two, three, then all ligands are hidden), reaching **0 valid out of 6,300
attempts** when only the bare metal is given. The failures are chemically diagnosable —
≈100% are nitrogen explicit-valence violations — showing that a generic 3D diffusion
model, which generates geometry and infers chemistry post hoc, learns local completion
but not valence-correct composition. We argue this motivates coordination-aware
generation (fix chemistry first, then realize geometry) for high-CN, hard-donor
rare-earth design.
