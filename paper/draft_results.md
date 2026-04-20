# Results

## 4.1 Fine-Tuning Converges but Validity Remains Low

Fine-tuning the pretrained d-block multi-LigandDiff checkpoint on 95,279 CSD lanthanide complexes (CN 7--10) reduced validation loss from 977 to 50 over 63 epochs (Figure 2a).
The loss curve exhibits rapid initial descent (epochs 0--15, loss drops to ~120) followed by slow convergence, consistent with the model adapting its learned d-block priors to the larger coordination spheres and different metal--donor distance distributions of f-block chemistry.

The valid_ligand rate---the fraction of individually generated ligands with correct valence and connectivity---improved from 0.45 to 0.82, approaching the d-block pretrained performance of 0.89 (Figure 2b).
However, the valid_complex rate---the fraction of complete complexes where *all* ligands and the metal center pass sanity checks---plateaued near 0.06 (6%).
This 83-fold gap between individual ligand validity and complex validity reflects the compounding effect of per-ligand error rates across the 4--6 ligand groups typical of CN=10 structures: if each ligand independently has a 0.82 probability of being valid, the probability that all 5 ligands in a complex are valid is 0.82^5 = 0.37, already well below the d-block 0.89 benchmark.
The observed 0.06 rate is even lower, indicating correlated failures between ligand groups.

## 4.2 Baseline Generation Fails on Lanthanide CN=10

Vanilla DDPM sampling (r=1, no projection) applied to the Eu(TMMA)_2(NO_3)_3 reference structure (35 heavy atoms, CN=10, CSD refcode VEDTAA01) produced a valid_complex rate of 0.5% (3 of 500 samples; 95% CI: 0.1--1.4%), compared to 89% for d-block Fe complexes at CN=6 reported by Jin and Merz.^[jin2024multiligand]

Inspection of the 497 invalid structures revealed three dominant failure modes (Figure 3):

**(a) Inter-ligand bridging (mean 3.4 bridges per structure).** Oxygen atoms from different ligand groups formed spurious O--O contacts at 1.2--1.5 Angstroms, creating covalent-like bridges between independent ligands (Figure 3a).
These bridges are chemically implausible: they would require peroxide or superoxide species that are not present in the reference.
Bridging was most prevalent between the two TMMA carboxylate arms, which are geometrically adjacent in the reference but belong to separate ligand groups.

**(b) Context-atom perturbation (mean displacement 0.62 Angstroms).** Despite being nominally fixed, context atoms drifted from their reference positions during the diffusion trajectory (Figure 3b).
This drift reflects the inability of the denoising network to maintain atomic positions in the crowded coordination sphere, where small displacements of one atom cascade into steric clashes with neighboring ligand groups.

**(c) Ligand-type mutation (mean 1.87 mutated ligands per structure).** In 78% of samples, at least one context ligand underwent a composition change---for example, a nitrate (NO_3^-) being replaced by a fragment resembling a carboxylate or water molecule (Figure 3c).
This mutation indicates that the denoising network, trained on diverse CSD structures, "hallucinates" chemically different ligands when the coordination environment is too crowded for it to maintain the original identity.

These failure modes are not independent: structures with more bridges tend to also have more mutations (Pearson r = 0.61), suggesting a common upstream cause---the inability of the denoising network to resolve the high geometric density of CN=10 coordination.

## 4.3 Failure Is Worse at High Coordination Crowding

To quantify the relationship between coordination complexity and generative quality, we performed a context-density ablation across four reference structures with increasing numbers of fixed context atoms (Figure 4, Table 2).
All runs used the best sampler configuration (RePaint r=10 + projection) and the Ln fine-tuned checkpoint.

The valid_complex rate decreased monotonically with context heavy-atom count:
- **Bare Eu^3+ (1 atom, CN_context=0):** 31.2% valid (95% CI: 29.5--32.9%). With no context constraints, the model generates complete coordination spheres *de novo*, and roughly one-third are chemically plausible.
- **Eu(H_2O)_9 (10 atoms, CN_context=8):** 14.8% (11.8--18.2%). The nine water molecules provide a dense but chemically simple context.
- **Eu(NO_3)_3(H_2O)_3 (16 atoms, CN_context=7):** 9.2% (6.6--12.2%). Mixed-ligand complexity increases failure.
- **Eu(TMMA)_2(NO_3)_3 (35 atoms, CN_context=10):** 6.4% (4.0--9.4%). The most crowded context yields the lowest validity.

This 5-fold drop from bare metal to full complex (31% to 6%) represents, to our knowledge, the first quantitative demonstration that generative model performance degrades systematically with metal coordination crowding.
The relationship is approximately log-linear: a doubling of context heavy-atom count corresponds to roughly a halving of valid_complex rate.

The d-block reference point (89% at CN=6, ~15 context atoms for a typical octahedral complex) falls well above the Ln trend line, suggesting that the problem is not simply the number of context atoms but the geometric density---more atoms packed into a similar coordination sphere volume.

## 4.4 RePaint Resampling Partially Mitigates Failure

RePaint resampling^[lugmayr2022repaint] at r=10 improved the valid_complex rate on Eu(TMMA)_2(NO_3)_3 from 0.5% (vanilla) to 3.8% (Table 1), a 7.6-fold improvement.
The improvement was dose-dependent: r=5 yielded 1.8%, r=10 yielded 3.8%, and r=20 yielded 5.2%, with diminishing returns beyond r=10.

Decomposing the improvement by failure mode:
- **Bridges** dropped from 3.42 to 1.23 per structure (r=10), consistent with the resampling mechanism: each re-noising cycle gives the denoising network another opportunity to resolve steric clashes between ligand groups.
- **Context perturbation** decreased from 0.62 to 0.48 Angstroms, a modest improvement suggesting that resampling partially stabilizes context-atom positions.
- **Ligand mutations** decreased from 1.87 to 0.94 per structure, a 50% reduction. Mutations are harder to fix than bridges because they involve compositional rather than geometric errors.

The fraction of structures with *zero* bridges increased from 8% (vanilla) to 41% (r=10), but 59% of structures still contained at least one bridge---a rate too high for downstream optimization pipelines.

Computational cost scales linearly with r: at r=10, each sample requires 10x the denoising steps of vanilla, increasing per-sample wall time from ~0.3s to ~3s on H200 GPU.

## 4.5 Hard Projection Eliminates Bridges but May Distort Geometries

The geometric exclusion-shell projection^[christopher2024projected] (d_min annealed from 1.5 to 1.3 Angstroms) eliminates inter-ligand bridges by construction: all projected structures have zero bridges (Table 1, "projection only" row).

However, projection alone (r=1) yielded only 1.2% valid_complex---*higher* than vanilla (0.5%) but lower than RePaint r=10 (3.8%).
The improvement from eliminating bridges was partially offset by an increase in context perturbation (0.71 vs. 0.62 Angstroms) and a slight increase in ligand mutations (1.92 vs. 1.87).
This suggests that the projection displacement---pushing atoms outward from steric clashes---can cascade into distortions of neighboring groups, particularly in the dense CN=10 coordination sphere.

The trade-off is clear: projection is a necessary component (bridges must be zero for chemical plausibility) but insufficient alone.

## 4.6 Combined RePaint + Projection Achieves 6.4% Valid Complex Rate

Stacking RePaint r=10 with hard projection yielded 6.4% valid_complex (32 of 500; 95% CI: 4.0--9.4%), the best result in our study (Table 1, final row).
This represents a 12.8-fold improvement over vanilla sampling and a 1.7-fold improvement over RePaint alone.

The combined configuration achieves:
- Zero bridges (projection guarantee)
- Mean 0.82 mutated ligands per structure (lowest of all configurations)
- Mean 0.45 Angstroms context perturbation (lowest of all configurations)
- Denticity recovery of 0.38 overall, with monodentate ligands well-recovered (0.72) but tridentate and tetradentate ligands frequently collapsing to lower denticity (Figure 3d)

While 6.4% is a substantial improvement over the baseline, it remains far below the d-block benchmark of 89%.
This gap underscores the fundamental difficulty of generating chemically valid structures in crowded f-block coordination environments.

## 4.7 Cross-Architecture Replication

To determine whether the CN=10 failure is specific to the Ln fine-tuned checkpoint or inherent to the multi-LigandDiff architecture, we tested the pretrained d-block checkpoint directly on Eu(TMMA)_2(NO_3)_3 (Table 3).

The d-block pretrained model on Eu(TMMA)_2(NO_3)_3 with vanilla sampling produced 0.5% valid_complex (1 of 200), indistinguishable from the Ln fine-tuned baseline.
With RePaint r=10 + projection, the d-block model achieved 4.0% (8 of 200), slightly below the Ln fine-tune (6.4%).

As a positive control, the d-block pretrained model on Fe(TMMA-substituted) at CN=6 yielded 89.0% valid_complex (178 of 200), reproducing the published benchmark.^[jin2024multiligand]

These results demonstrate that:
1. The CN=10 failure is not caused by the fine-tuning procedure---both checkpoints fail on Eu at CN=10.
2. The Ln fine-tuning provides a marginal improvement (6.4% vs. 4.0%), consistent with the model learning f-block-specific coordination patterns.
3. The architecture itself---EGNN+GVP with MAX_LIGANDS=10 and the current denoising schedule---has a fundamental limitation at high CN, regardless of training data.

## 4.8 Aqua-Seed Elaboration

The Eu(H_2O)_9 context (Table 2, second row) tests an "aqua-seed" paradigm: can the model use coordinated waters as placeholder ligands that are then elaborated into more complex structures?

At 14.8% valid_complex, the aqua-seed context produced the second-highest validity after bare Eu.
Inspection of the generated structures revealed that in approximately 65% of valid samples, at least one context water molecule was preserved as water (i.e., not mutated into a different species).
In the remaining 35%, waters were partially converted into hydroxide-like fragments or organic groups.

The mean mutation rate for the aqua-seed context (0.34 mutated ligands per structure) was substantially lower than for the full TMMA reference (0.82), consistent with the simpler chemical identity of water being easier for the denoising network to maintain.

These results suggest that water-mediated context provides a useful intermediate between bare-metal generation (high validity but no geometric constraint) and full-complex elaboration (low validity but precise geometric control).
However, the mutation rate of 34% on even simple water ligands indicates that the model does not reliably preserve context identity, limiting the practical utility of the aqua-seed approach.

## 4.9 Geometry Optimization

### 4.9.1 xTB Relaxation

Of the 32 valid structures from the best sampler configuration (RePaint r=10 + projection on Eu(TMMA)_2(NO_3)_3), 7 were submitted to GFN2-xTB geometry optimization.
Four of seven (57%) converged within 500 cycles; the remaining three terminated with IEEE floating-point exceptions, a known failure mode of GFN2-xTB for heavy f-block elements.^[bannwarth2019gfn2xtb]

Converged structures had total energies ranging from -89.8 to -80.9 Hartree (Figure 6a).
The 9 Hartree spread (~5,600 kcal/mol) reflects substantial variation in the generated geometries: lower-energy structures maintained tighter Eu--O coordination, while higher-energy structures exhibited elongated or partially dissociated Eu--donor bonds.

The 57% xTB convergence rate, applied to the 6.4% valid_complex rate, yields an overall pipeline throughput of approximately 3.6% (structures that are both valid and xTB-optimizable), or roughly 18 usable structures per 500 generated.

### 4.9.2 DFT Validation

[This section to be completed upon availability of ORCA results from Prompt 9.]

Preliminary DFT calculations at the PBE0-D4/def2-TZVP level with SARC-DKH relativistic treatment were submitted for 6 structures stratified by category (reference, best-generated, aqua-seed, and two failure modes).
Results will include: (i) relative energies vs. the crystallographic reference, (ii) Eu--donor distance distributions compared to experimental values from Kravchuk et al.^[kravchuk2024ejic] (Eu--O(TMMA) = 2.33--2.40 Angstroms, Eu--O(NO_3) = 2.51--2.56 Angstroms), and (iii) Mulliken charge analysis assessing charge transfer to the Eu center.
Figure 6 presents the available xTB energy data and representative Eu--donor distance distributions.
