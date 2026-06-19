# KNOWN ISSUES — residual items from the 2026-06-18 independent audit

The audit confirmed all 22 worklist fixes hold and surfaced 23 residual issues. The
high/medium-severity ones and clear honesty seams were **fixed** (see `docs/FIXLOG.md`,
Fix A1–A5). The items below are the **low-severity / deferred** remainder — registered here
with location, why it was deferred, and the recommended fix. None blocks the H200 runs.

## Deferred — needs GPU testing or judgment

| # | Issue | Severity | Location | Why deferred / recommendation |
|---|---|---|---|---|
| K1 | `--valence_guard` in-loop soft steer is **not gated over noise level** — it runs at every reverse step, including high-noise steps where neighbour counts on near-Gaussian coords are meaningless (and adds an O(N²) cdist/step). | medium | `src/edm.py:223` | Self-correcting (final hard mask guarantees a legal type), so not a correctness bug. Gating it to low-noise steps (`s < timesteps*0.3`, mirroring the projection schedule) changes the sampling trajectory — needs a GPU A/B before changing blind. |
| K2 | `--valence_guard` can over-correct element identity at low noise: a transiently-close non-bonded atom (<1.15× covalent) is counted as a valence-consuming bond, so a real N near a 4th atom can be forced to C. | low | `src/edm.py:394` | Document as a known over-correction; optionally count neighbours below the tighter `BOND_PERCEPTION_CUTOFFS` instead of the 1.15× covalent cutoff. |
| K3 | `generate_bare.py` uses `SELECTED_LD9` unconditionally for CN=9, so `--max_denticity < 4` is silently ignored for the most common Ln CN. | low | `generate_bare.py:122-123` | Default-safe (SELECTED_LD9 max part is 4 = default cap). Filtering needs an empty-list fallback; deferred to avoid a silent empty plan. |
| K4 | `generate.py` tridentate+ budget (`get_ligand_size`) uses global `np.random`, made reproducible only by the `np.random.seed(seed)` side-channel at `generate.py:902`. | low | `generate.py:254,484` | Thread the seeded `rng` into `get_ligand_size` so all budget draws share one source. |

## Deferred — out of the fixed path (eval/legacy scripts, by-design)

| # | Issue | Severity | Location | Note |
|---|---|---|---|---|
| K5 | `test.py` / `non_oct.py` have the same unguarded metric divisions that Fix 21 fixed in `lightning.py` — still NaN on empty batches. | low | `test.py:111-113`, `non_oct.py:175-177` | Eval/inference scripts, not the training monitor. Reuse `src.lightning._safe_div` if desired. |
| K6 | `generate_mask1.py` exposes no `--valence_guard` and calls `sample_chain` without it. | low | `generate_mask1.py:245` | Legacy entrypoint; the guard is available via the `generate.py` family. Defaults to off (no crash). |
| K7 | The high-mask curriculum only reshapes complexes with **k≥5** ligands (the down-sample population); for k≤4 all subsets are already kept, so `--mask_curriculum`/`--force_all_masked` are no-ops there. | low | `prepare_training_data.py:374` | Effective coverage is narrower than the commit message implies. If most Ln complexes have ≤4 decomposed ligands, consider up-weighting the all-masked subset for low-k too. (Per-complex CN histogram needed to decide.) |
| K8 | `analyze_design_test.py` headline `valid`/`yield` use the **file count** (`count_valid`) over the logged gate count; they can diverge if `noH/` is cleaned or filenames collide. | low | `analyze_design_test.py:143-147` | The CSV emits both side-by-side. Optionally warn-and-prefer the logged gate count for `yield_pct`. |
| K9 | `attempts_raw` (the inflation factor) is parsed but discarded — it reaches `accounting.json` but never the analyzer CSV. | low | `analyze_design_test.py:40` | By design. Optionally add an `attempts_raw`/`raw_inflation` column so the eligible-vs-raw story survives into the analysis artifact. |

## Documentation / provenance (low)

| # | Issue | Location | Note |
|---|---|---|---|
| K10 | `analyze_gen.py` prints "~2.8 Å for O/N" as the donor headline, but per-element Ln cutoffs span ~2.70–2.86 Å (e.g. Nd–O 2.84). | `analyze_gen.py:42` | 2.8 Å is the representative figure; the binding decision is always the exact per-pair `are_bonded` value. |
| K11 | The `[edm] projection active` / valence proof logs latch on the instance and print once per process, not per run/batch. | `src/edm.py:199` | Intentional ("one-time proof"), but a later batch where projection silently no-ops won't re-warn. |
| K12 | ORCA version is not stated in the template/sbatch; the nested methods draft previously hard-coded "5.0.4" (now "recorded per run"). | `orca_templates/`, methods | Capture the real version from the `orca_output.log` header at run time. |
| K13 | `ligandgen/` is **gitignored** — edits to `ligandgen/site/strategy.md` and `ligandgen/multi_LigandDiff/paper/draft_methods.md` (DFT level-of-theory) are local-only. | — | The tracked canonical docs (`paper/`, `orca_templates/`) are fixed and pushed. The local working copies were also corrected for consistency. |

## Resolved post-audit (for reference — see docs/FIXLOG.md)
d_min default propagation (A2) · `train.py` normalization default (A2) · `generate_bare`
bidentate budget floor (A2) · metal-donor gate test (A3) · DFT cross-formula ΔE suppression
(A3) · DFT level-of-theory reconciliation (A4) · clean-checkpoint re-finetune path + design
caveats (A5).
