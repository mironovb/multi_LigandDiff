# FIXLOG — code-review worklist (prompts 01–25) + independent audit

This is the master changelog for the code-review fix worklist run on 2026-06-16/17 and the
independent audit + follow-up fixes on 2026-06-18. Each entry: problem → change (file-cited)
→ impact/backward-compat → verification → follow-up.

**Audit verdict (2026-06-18):** an independent 14-agent adversarial re-verification of all 22
landed fixes found **10/10 fix areas hold** — every fix landed and does what it claims, confirmed
against the actual code and unit tests. It surfaced 23 residual issues (2 high), all addressed
or registered below (see *Independent audit* and `docs/KNOWN_ISSUES.md`).

Commit chain (master): `8729f4c … 2f4b2b9` (worklist) → `00a317e e07b8a1 76f29c0 8c8299c 1bf16bf` (post-audit).

---

## Worklist fixes 01–12

### Fix 01 (prompt 01) — Preserve formal charges in make_mol_openbabel  [8729f4c]
- **Problem:** `make_mol_openbabel` rebuilt the RDKit mol atom-by-atom with only `Chem.Atom(symbol)`, dropping OpenBabel's perceived formal charges; charged donors (nitrate, N⁺) then failed valence sanitization.
- **Change:** `src/molecule_builder.py:176-180` — new atoms carry `SetFormalCharge` + `SetIsotope`. Adds `test_charge_aware_validity.py`.
- **Impact:** behaviour change only for previously charge-stripped atoms; neutral mols unchanged.
- **Verification:** `test_charge_aware_validity.py` passing (ligdiff python).
- **Follow-up:** only bites on distorted/generated geometries where OpenBabel assigns the N=O double bond (see Fix 04); harmless at crystal geometry.

### Fix 02 (prompt 02) — Charge/dative-tolerant validity gate  [10d4371]
- **Problem:** `BasicLigandMetrics.compute_validity` ran a bare `SanitizeMol(i)` catching only `ValueError`, rejecting valence-correct charged/dative donors.
- **Change:** `src/molecule_builder.py:196-209` — mirrors `add_H`: `reset_dative_bonds` → `UpdatePropertyCache(strict=False)` → `SanitizeMol(SANITIZE_ALL ^ SANITIZE_ADJUSTHS)`, catching `(ValueError, RuntimeError)`; validates on the converted copy but **appends the original `i`** (downstream `compute_connectivity` runs `GetMolFrags` on originals).
- **Impact:** strictly more permissive; return contract unchanged.
- **Verification:** gate test passing. Audit confirmed the append-original-`i` contract across all 7 callers.
- **Follow-up:** the original committed test was metal-free (didn't exercise the dative path) — strengthened post-audit (see Fix A3).

### Fix 03 (prompt 03) — Per-reason rejection counters + rejection_summary.json  [010f986]
- **Problem:** `generate_ligand` silently dropped candidates in one nested `if`, so `0/6300` had no breakdown.
- **Change:** `generate.py` + `generate_bare.py` — flat early-`continue` chain with a `Counter` (`nan`/`overlap`/`atom_count`/`sanitize`/`disconnected`/`valid`), `attempts` accumulator, `rejection_summary.json` per run.
- **Impact:** same accept/reject decision; adds instrumentation. Audit verified `attempts == Σ reasons` and `reasons['valid'] == num`.

### Fix 04 (prompt 04) — rescore_validity.py sanity test  [105d24d]
- **Problem:** needed a GPU-free way to re-grade saved `.xyz`/reference through the real gate.
- **Change:** new `rescore_validity.py` reusing `sanitycheck` + `BasicLigandMetrics`.
- **Impact:** new standalone script; no production code touched.
- **Honest finding:** the reference `eu_tmma_cis.xyz` passes **both** old and new gate (at crystal geometry OpenBabel perceives all-single N–O → formal charge 0). So `0/6300` is **not** instrumental for the reference; the real signal lives in re-scored **generated** sets.

### Fix 05 (prompt 05) — Cap denticity partitions at 4 + --max_denticity  [0aad9c4]
- **Problem:** `denticity_partitions` defaulted to `max_denticity=10`, enumerating impossible high-denticity chelates (CN=10 → 42 partitions).
- **Change:** `src/const.py` `MAX_DENTICITY=4`; default → `MAX_DENTICITY`; `--max_denticity` threaded through all three generate scripts.
- **Impact:** CN=10 goes 42→23 partitions (19 impossible dropped). Verified empirically.

### Fix 06 (prompt 06) — CSD-weighted denticity sampling (--denticity_prior csd)  [889fea7]
- **Change:** `src/const.py` `DENTICITY_PRIOR={1:0.677,2:0.180,3:0.140,4:0.003}` (Σ=1.0) + `denticity_prior_weight`; `--denticity_prior {uniform,csd}`; `csd` draws one partition per context copy ∝ prior.
- **Impact:** opt-in; default `uniform` unchanged.

### Fix 07 (prompt 07) — Honest denominator accounting + accounting.json  [19f7b0e]
- **Problem:** yield was computed over an inflated, uncapped attempt count.
- **Change:** `generate_design_test.py` `_partition_accounting` reports `attempts_eligible = len(data)` vs `attempts_raw`; writes `accounting.json`; prints `attempts=N (raw M)`. Bookkeeping only.
- **Follow-up:** the new print format required the analyzer regex fix (Fix A0/00a317e).

### Fix 08 (prompt 08) — Chemistry-derived atom budgets (DENTICITY_MIN_ATOMS)  [1c0ca09]
- **Problem:** de-novo budget `np.random.randint(num_coord_site,10)` could hand a bidentate slot 2 atoms — too few for a nitrate (N+3O=4).
- **Change:** `src/const.py` `DENTICITY_MIN_ATOMS={1:1,2:4,3:5,4:7}`; `generate.py:476-486` floor + seeded spread + `max(g_ligand_size, floor, num_coord_site)`.
- **Follow-up:** the same fix was **not** propagated to `generate_bare.py` — done post-audit (Fix A2).

### Fix 09 (prompt 09) — --donor_spec conditions on donor-atom identity  [83494cf]
- **Change:** `generate.py` `--donor_spec` (per-ligand `'O,O;N,O,O,O'` or flat); `reform_data` seeds donor one-hot rows; spec is authoritative (bypasses csd prior).
- **Impact:** opt-in; default `None` = unchanged. **Only in `generate.py`** — `generate_design_test.py` conditions via `--ligand_templates` instead (Fix 10).

### Fix 10 (prompt 10) — --ligand_templates seeds whole skeletons  [bfa2c29]
- **Change:** new `src/templates.py` (nitrate/water/carboxylate, donors-first); `--ligand_templates`/`--template_init_coords` in `generate.py` + `generate_design_test.py`.
- **Honest scope:** atom-count budget takes effect immediately; element/coordinate seeding biases the input representation only (sampler re-noises from N(0,I)). Hard enforcement is Fix 16.

### Fix 11 (prompt 11) — Raise projection d_min 1.5/1.3 → 2.2/1.9 Å + document radii  [a68134f]
- **Problem:** exclusion-shell `d_min` annealed below the ~1.3×covalent bond-perception cutoffs, so the projection was a no-op against bond perception.
- **Change:** defaults raised to 2.2/1.9 in `generate.py`/`generate_bare.py`/`generate_design_test.py`; `src/projection.py` documents `BOND_PERCEPTION_CUTOFFS` (min 1.72 Å) + warns below it. Algorithm unchanged.
- **Follow-up:** 6 stale 1.5/1.3 spots in `generate_mask1.py`/`lightning`/`edm` were missed — done post-audit (Fix A2).

### Fix 12 (prompt 12) — Project generated↔generated cross-group pairs  [2c55e8e]
- **Problem:** `project_exclusion_shell` only enforced gen↔context and bailed with no context; in maskall the metal is the only context (exempt) → zero eligible pairs → no-op.
- **Change:** `src/projection.py` rewrite — Phase A gen↔gen (symmetric half-deficit split), Phase B gen↔context (context fixed); bails only when nothing is generated. `test_projection.py::test_maskall_gen_gen_projection`.
- **Verification:** 5/5 projection tests pass; audit reproduced convergence with metal unmoved.

---

## Worklist fixes 13–25

### Fix 13 (prompt 13) — Enable + verify projection in the de-novo path (10× scale bug)  [18b3a52]
- **Problem:** `EDM.sample_chain` runs in NORMALIZED coords (`/norm_values[0]`, factor 10) but `project_exclusion_shell` was called with raw-Å `d_min` — a 10× mismatch pushing atoms ~19 Å apart.
- **Change:** `src/edm.py:182-185` round-trips Å↔normalized (`pos_A = z_pos * norm_values[0]` … `/ norm_values[0]`); projected positions persist in `z`; one-time proof log. Eligibility guards in `generate_design_test.py` + `generate_bare.py`.
- **Verification:** audit reproduced 1.900 Å (fixed) vs 19.0 Å (old bug); 5/5 tests pass.

### Fix 14 (prompt 14) — Single source of truth for bond/donor cutoffs (src/bonding.py)  [ecc41dc]
- **Problem:** prep used flat cutoffs (1.9/3.0 Å) while the validity gate grades against molSimplify covalent-radii bonds — prep and validity disagreed (Finding 6).
- **Change:** new `src/bonding.py` re-implements `mol3D.getBondCutoff` (1.15× base, 2.75 Å C–X cap, 1.10× TM–H); verbatim covalent radii; `cross_check_against_molsimplify` self-test. `prepare_training_data.py` routes the bond-graph fallback + `decompose_ligands` through `bonding.are_bonded`.
- **Verification:** audit ran `cross_check_against_molsimplify` — bit-for-bit equal across 15 pairs against live molSimplify.
- **Follow-up:** training data must be re-prepped for full consistency (re-finetune sbatches).

### Fix 15 (prompt 15) — Reconcile donor cutoff: code + Methods at 2.8 Å  [8af86b4]
- **Change:** `analyze_gen.py` replaces hardcoded `d < 3.0` with `bonding.are_bonded` (element-aware, metal-generic); paper Methods/SI note the 2.8 Å match; `donor_cutoff()` = 2.776 Å.

### Fix 16 (prompt 16) — Valence-aware type masking (--valence_guard)  [80ee344]
- **Change:** `src/edm.py` in-loop soft steer + final hard read-off mask via `_valence_allowed_mask` (8×8 cutoff matrix from `const.ALLOWED_BONDS`). Metal (all-zero one-hot context row) excluded from neighbour counting so a 3-bond coordinating amine N stays legal. Plumbed through `DDPM.sample_chain`.
- **Impact:** opt-in, default False. Sampler-side mitigation (graph-level predict-then-build is Track B).
- **Follow-up:** in-loop steer is ungated over noise level (KNOWN_ISSUES); `generate_mask1.py` lacks the flag.

### Fix 17 (prompt 17) — --relax_before_gate (gfnff/gfn2, frozen context)  [e20d6e1]
- **Change:** `generate.py` short xTB relax of generated atoms only (context frozen via `$fix atoms: 1-n`), `--opt loose`, 20 cycles, timeout; default `none`. Guards on xtb-absent, falls back to raw.
- **Impact:** default `none` path behaviour-preserving.

### Fix 18 (prompt 18) — High-mask curriculum (--mask_curriculum, --force_all_masked)  [8b84ee0]
- **Change:** `prepare_training_data.py` weighted-without-replacement (Efraimidis–Spirakis A-Res, `weight = k**exponent`), `_stable_seed` per-complex digest, `--force_all_masked` (default on) reserves the all-masked subset. Default `uniform`.
- **Impact:** at defaults the only change is the guaranteed all-masked subset + deterministic seeding.
- **Follow-up:** no effect until re-prep + re-finetune; curriculum only reshapes the k≥5 down-sample population (KNOWN_ISSUES).

### Fix 19 (prompt 19) — config.yml reflects the real fine-tune  [e5c1e3c]
- **Change:** `config.yml` → 192/5/l2/polynomial_2/bs256, header documenting that only `train.py` reads it.

### Fix 20 (prompt 20) — Reconcile normalization_factor to the backbone scale  [0a07b7c]
- **Problem:** `normalization_factor` scales summed-message aggregation (`conv_layer.py:95-96`), not a saved weight; `finetune.py` passed 1 while the pretrained backbone was trained at 100 → warm-start corrupted before step 0.
- **Change:** `finetune.py:202` 1→100; `config.yml` live key `normalization_factor: 100`.
- **Verification (audit-confirmed):** `pre_trained.ckpt`→100, `ln_finetuned_epoch=48.ckpt`→1 (loaded directly). Existing ln checkpoints load self-consistently but were bug-trained.
- **Follow-up — HIGH:** the design re-runs still point at the bug-trained `epoch=48`. Resolved post-audit (Fix A4) with `finetune_fixed.sbatch` + caveats; `train.py:130` default also corrected (Fix A2).

### Fix 21 (prompt 21) — Guard divide-by-zero in sample_and_analyze  [132b9f6]
- **Change:** `src/lightning.py` `_safe_div` wraps the three metric divisions → 0.0 instead of NaN.
- **Audit correction:** it protects the **best-epoch `np.argmax` selection** (`compute_best_validation_metrics`), not the early-stop monitor (which watches `loss/val`). Code fix is correct; commit wording was imprecise.

### Fix 22 (prompt 22) — Reconcile sbatch harness  [2f4b2b9]
- **Change:** `dft_showcase` `--top-n`→`--max-per-category`; `finetune_curriculum` `--out`/`--val_out`→`--output_dir`; `design_maskall_fixed` `--donor_spec`→`--ligand_templates`; dropped phantom `test_denticity_cap.py`; README run order.
- **Verification:** audit confirmed **zero unresolved flags** across all 13 sbatches (dash-aware, AST-checked).

### Fix 23–25 pre-run — Pre-cluster code fixes folded ahead of the H200 runs  [00a317e]
- **analyze_design_test.py:** `SUMMARY_RE` tolerates `attempts=N (raw M)` via `(?:\s+\(raw\s+\d+\))?` (without it, step-4 analysis silently blanks attempts/yield for every fixed run).
- **dft_pipeline.py:** `from __future__ import annotations` (PEP-604 `list | None` on Py3.9) + `glob.escape(name)` (bracketed names `..._[2, 2, 2, 2]_[2]` match instead of being read as glob classes).
- **sbatches/design_mask2.sbatch:** `n_samples 250→50` → 2500 attempts (not 12,500, which would trip the 4 h wall).

---

## Independent audit (2026-06-18) + post-audit fixes

A 14-agent workflow re-verified every fix area against the actual code (not the session's
self-grading). **10/10 areas hold.** The post-audit commits address the propagation gaps and
honesty seams it found:

### Fix A1 — Pre-cluster code fixes committed & pushed  [00a317e]
The three working-tree fixes above were pre-run CODE fixes the session held back under its
"commit only after results" rule — but they must reach the cluster *before* the runs. Pushed first.

### Fix A2 — Propagated consistency fixes  [e07b8a1]
- **6 stale d_min 1.5/1.3 defaults** (`generate_mask1.py` argparse + `generate_ligand` + `main`; `src/lightning.py` + `src/edm.py` `sample_chain` signatures) → 2.2/1.9. Default-only; production passes explicit values, so runtime unchanged — hardens the fallback against a sub-1.72 Å no-op shell.
- **`train.py:130` `--normalization_factor` default 1→100** — removes the exact warm-start footgun from Fix 20 (a from-scratch run without `--config` used to silently rebuild the scale-1 model).
- **`generate_bare.py` bidentate budget floor** — the `DENTICITY_MIN_ATOMS` floor (Fix 08) was never propagated here; ~25% of bidentate slots still drew 2–3 atoms. Mirrored the floor + `max(.,floor,dent)` clamp, switched to the seeded rng.

### Fix A3 — Test + DFT honesty  [76f29c0]
- **`test_charge_aware_validity.py`:** the committed `test_gate_accepts_nitrate` used a metal-FREE nitrate, which already passes plain `SanitizeMol` — it never exercised the prompt-02 gate logic. Added a metal-donor case (Fe–N over-valent amine) that fails plain sanitize, passes `compute_validity` via dative conversion, and asserts the returned object is the original mol with its Fe–N bond still SINGLE. **5/5 pass.**
- **`dft_pipeline.py`:** `dft_deltaE_kcal_mol = E − E_ref` was computed across different molecular formulae (no-H 35-atom reference vs H-complete 43-atom completions) → physically meaningless. Now suppresses ΔE (None) + warns on atom-count mismatch. Zero CSV-schema change.

### Fix A4 — DFT level-of-theory reconciliation  [8c8299c]
Every DFT-method description aligned to the implemented template `orca_templates/pbe0_eu.inp`
(PBE0-**D4** / def2-TZVP / Eu: SARC-DKH-TZVP basis + SK-MCDHF-RSC ECP / **gas-phase**):
`paper/draft_discussion.md` (D3→D4), `paper/draft_methods.md` + `draft_SI.md` (1.3×→molSimplify
~1.15× prep cutoff), the nested `ligandgen/.../draft_methods.md` (dropped the unsupported DKH2
Hamiltonian + unsourced ORCA-5.0.4 claims), template header. *Note:* `ligandgen/` is gitignored,
so the `strategy.md` + nested-methods edits are local-only; the tracked `paper/` + template are pushed.
No result/claim wording touched.

### Fix A5 — Normalization re-finetune path  [1bf16bf]
Resolves the high-severity blocker (design re-runs on a bug-trained checkpoint):
`sbatches/finetune_fixed.sbatch` (clean factor=100 re-finetune, legacy masking → the controlled
degradation-curve checkpoint); design sbatches annotated with the caveat + a commented clean-CKPT
alternative; README updated.

---

## Pending on the H200 (result work, not yet committed)
- **Prompt 23 (maskall de-novo re-test)** — `sbatches/design_maskall_fixed.sbatch`.
- **Prompt 24 (mask-2 degradation point)** — `sbatches/design_mask2.sbatch`.
- **Prompt 25 (DFT showcase)** — `sbatches/dft_showcase.sbatch` → `dft_pipeline.py parse`.
- **Clean checkpoints** — `finetune_fixed.sbatch` / `finetune_curriculum.sbatch` (need CPU re-prep first).

See `docs/H200_RUNBOOK.md` for the ordered execution procedure and `docs/KNOWN_ISSUES.md` for
the residual low-severity items the audit registered but did not fix.
