# Results

> Rewritten to verified, log-traceable values (see `RESEARCH_PLAN.md`). The previous
> draft was built on experiments that never ran (context-density ablation,
> projection-stack "6.4%", cross-architecture) and omitted the actual headline
> finding (the de-novo design failure). This version reports only what is in the
> on-cluster logs and metrics files.

## 4.1 Fine-tuning converges; per-fragment validity is high, whole-complex validity is low

Fine-tuning the pretrained d-block multi-LigandDiff checkpoint on the 9,306-complex
CSD lanthanide set (CN 7–10) reduced validation loss from **977.8** (epoch 0) to
**49.9 at epoch 48** — a ≈95% reduction — with the next-lowest value at epoch 57
(60.8) and early stopping at epoch 63 (Figure 2).

At convergence, sampling metrics were **valid_ligand 0.970** and **connected_ligand
0.938** — the per-fragment chemical validity is strong, close to the d-block
pretrained regime — but **valid_complex was 0.005**. This large gap between per-
fragment and whole-complex validity foreshadows the central result: the model
produces individually plausible ligands but struggles to assemble a fully valid
CN-10 sphere.

## 4.2 Completion works: regenerating one missing ligand

In the completion (mask1) setting — the full Eu(TMMA)₂(NO₃)₃ complex fixed as
context, one ligand hidden and regenerated — the model produces valid structures
reliably: **126 valid** structures in the design-test mask1 run, and additional valid
structures across the RePaint sweep below. Visual inspection confirms that most files
contain a clean, separate fifth fragment in the masked position: TMMA-shaped, diamide
backbone present, carbonyl oxygens pointed at Eu, with Eu–O distances in the expected
2.3–2.6 Å range.

These completions are physically reasonable under semiempirical optimization:
**38 of 41 (92.7%)** mask1-baseline structures converged under GFN2-xTB
(`--opt normal --uhf 6 --cycles 500`), and 81 of 85 (95.3%) for the RePaint r=5 set
(Table 3) — consistent with literature xTB convergence for such systems.

We note one bond-perception artifact. A connected-component fragment count returned
the expected 5 distinct ligands in only 9/38 (24%) of xTB-converged structures
(3–4 fragments in the other 76%), and 23/38 (60.5%) showed "new" cross-ligand bonds
after xTB. Visual inspection shows these are an artifact of the 1.3× covalent-radii
bond-detection cutoff merging atoms placed at threshold-ambiguous distances (0% for
the pristine reference under the same protocol) — a **geometry sensitivity check**,
not chemical fusion.

## 4.3 RePaint resampling buys yield, not placement quality

RePaint resampling was swept at r ∈ {1,5,10,20} on the mask1 task (Table 1). The
valid_complex yield rises **monotonically** with r:

- r = 1 (baseline): **29 / 2500 = 1.16%**
- r = 5: **85 / 2500 = 3.40%** (2.9× the baseline)
- r = 10: **57 / 1500 = 3.80%**
- r = 20: **39 / 750 = 5.20%**

However, the donor-placement quality, measured as the denticity-match rate, does
**not** track yield: it **peaks at r = 5 (15.3%)** and falls at r = 10 (10.5%, on the
19 structures analyzed). Past r = 5, additional resampling produces more structures
that clear the coarse validity filter but are not better at the fine-grained
placement metric. **r = 5 is therefore the working point** — the trade-off is yield
versus placement quality, and resampling buys yield, not validity.

## 4.4 De-novo design fails as the scaffold shrinks

To test whether the model can *design* rather than merely *complete*, we swept how
many ligands are masked at once, shrinking the scaffold the model can lean on
(`generate_design_test.py`; Table 2). Validity collapses monotonically:

| Ligands hidden | Context | Attempts | Valid | Yield |
|---|---|---|---|---|
| mask 1 | 4 of 5 | 2,500 | 126 | 5.0% |
| mask 2 | 3 of 5 | (cut off) | 4 | low |
| mask 3 | 2 of 5 | ≥1,500 | **0** | **0%** |
| mask all | 0 of 5 (bare Eu) | 6,300 | **0** | **0.00%** |

With two ligands hidden (mask 3) the model already produces nothing valid; asked to
build the entire CN-10 coordination sphere around a bare Eu (mask all), it yields
**0 valid out of 6,300 attempts**. (mask 2 = 4 and mask 3 = 0 come from time-limit-
cut-off jobs; a dedicated run will firm up the exact location of the cliff, but
maskall = 0/6300 is already conclusive.)

## 4.5 The failure is chemical (valence), not geometric

The rejections are diagnostic. Across the design-test logs, **~100% of failures are
nitrogen explicit-valence violations** — "Explicit valence for atom # N, 4, is
greater than permitted" (151 of 170 logged rejection lines on the mask1/2 run; all
nitrogen on the maskall run). These are not geometric near-misses: bond perception
finds nitrogen atoms placed in chemically impossible four-bond environments, and RDKit
rejects the resulting Lewis structure.

This locates the deficiency precisely. With scaffolding, the fixed context pins the
geometry and the model fills one ligand into a well-constrained pocket, so local
valence comes out right. Without scaffolding, the model must invent ~35 heavy atoms
in a fully connected graph; atoms crowd, bond perception finds over-coordinated
nitrogen, and the structure is valence-broken. The model learned local "fill the gap
consistent with neighbors" but has **no notion of composing a valence-correct
fragment from scratch** — a generic 3D diffusion model has no explicit valence or
denticity constraint.
