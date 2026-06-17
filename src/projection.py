"""
Hard geometric exclusion-shell projection for diffusion sampling.

Projects generated atoms out of a minimum-distance shell around atoms from
*different ligand groups*, preventing inter-ligand bridging artefacts (O-O
contacts at ~1.3 Å across ligand boundaries). Two kinds of obstacle are
enforced:

  * generated <-> context    — the context atom is fixed, so the generated
    atom absorbs the whole correction;
  * generated <-> generated  — both atoms move apart by half the deficit, so
    the shell still acts in the bare-metal de-novo (maskall) regime where
    every ligand is generated and the metal is the only context atom.

The metal (an all-zero ``ligand_group`` row, group -1) is never an obstacle —
donors are supposed to approach it — and atoms within one ligand group are
left alone (their short contacts are real bonds).

For the projection to actually prevent these artefacts its ``d_min`` must
clear the bond-perception cutoffs (see ``BOND_PERCEPTION_CUTOFFS`` below):
pushing two atoms to a distance that is still inside the ~1.3 × covalent-radii
bond radius leaves them perceived as bonded, so ``get_bond_order`` re-fuses
them and the projection is a no-op against bond perception. ``d_min`` must
therefore sit at or above ~1.72 Å (the O–O cutoff, the smallest of them).

Reference: Christopher, Baek & Fioretto, "Projected Diffusion Models",
NeurIPS 2024 (arXiv:2402.03559).
"""

import torch


# ---------------------------------------------------------------------------
# Bond-perception cutoffs the exclusion-shell ``d_min`` must clear.
#
# Bond perception (``get_bond_order`` in src/molecule_builder.py, via the
# OpenBabel/RDKit covalent-radii rule) declares a bond whenever the
# interatomic distance is below roughly 1.3 × (sum of covalent radii). For the
# donor elements LigandDiff places across ligand boundaries (C, N, O) those
# thresholds are:
#
#       pair    ~1.3 × (r_cov_i + r_cov_j)   [Å]
#       -----   --------------------------
#       C–C                1.98
#       C–N                1.91
#       C–O                1.85
#       N–O                1.78
#       O–O                1.72    <-- smallest, the binding constraint
#
# The projection only prevents cross-ligand fusion if it pushes atoms PAST the
# distance at which they would be perceived as bonded. A ``d_min`` below the
# O–O cutoff (~1.72 Å) separates atoms but leaves them inside the bond radius,
# so ``get_bond_order`` still bonds them and the shell is a no-op against bond
# perception. ``d_min`` must therefore be >= ~1.72 Å; the generate scripts
# default to 2.2 -> 1.9 Å (annealed) to keep headroom above every cutoff here.
#
# DO NOT lower these defaults below ~1.72 Å without re-reading this table:
# anything smaller cannot stop two donors from being perceived as bonded.
# ---------------------------------------------------------------------------
BOND_PERCEPTION_CUTOFFS = {
    ("C", "C"): 1.98,
    ("C", "N"): 1.91,
    ("C", "O"): 1.85,
    ("N", "O"): 1.78,
    ("O", "O"): 1.72,
}

# Smallest cutoff above: the floor ``d_min`` must clear to affect bond
# perception at all. Derived from the table so the two never drift apart.
MIN_BOND_PERCEPTION_CUTOFF = min(BOND_PERCEPTION_CUTOFFS.values())  # 1.72 Å (O–O)

# Module-level latch so the "d_min too low" diagnostic prints at most once,
# rather than once per timestep/batch during sampling.
_warned_d_min_below_cutoff = False


def project_exclusion_shell(
    pos: torch.Tensor,
    ligand_group: torch.Tensor,
    context_mask: torch.Tensor,
    d_min: float,
    same_group_allowed: bool = True,
) -> torch.Tensor:
    """Project generated atoms ``d_min`` away from every cross-group atom.

    Enforces two exclusion shells (see the module docstring):
      * generated <-> context   — context is fixed; the generated atom moves;
      * generated <-> generated — both move apart by half (so the shell still
        acts when every ligand is generated and only the metal is context).
    Same-group pairs (intra-ligand bonds) and the metal (group -1) are exempt.

    Args:
        pos: (N, 3) atom positions.
        ligand_group: (N, MAX_LIGANDS) one-hot ligand group assignment.
        context_mask: (N, 1) binary mask — 1 for context (fixed), 0 for generated.
        d_min: minimum allowed distance (Angstroms).
        same_group_allowed: if True, skip projection for same-group pairs.

    Returns:
        (N, 3) projected positions (only generated atoms may be modified).
    """
    # Tripwire: a d_min below the smallest bond-perception cutoff (~1.72 Å,
    # O–O) cannot stop atoms from being perceived as bonded — the projection
    # becomes a no-op against bond perception. Warn once so a too-low value
    # does not silently neuter the exclusion shell. See BOND_PERCEPTION_CUTOFFS.
    global _warned_d_min_below_cutoff
    if d_min < MIN_BOND_PERCEPTION_CUTOFF and not _warned_d_min_below_cutoff:
        _warned_d_min_below_cutoff = True
        print(
            f"[projection] WARNING: d_min={d_min:.3f} Å is below the smallest "
            f"bond-perception cutoff (~{MIN_BOND_PERCEPTION_CUTOFF:.2f} Å, O–O). "
            f"Atoms separated to this distance are still perceived as bonded by "
            f"get_bond_order, so the exclusion-shell projection is a no-op "
            f"against bond perception. Raise d_min (>= ~1.9 Å recommended) — "
            f"see BOND_PERCEPTION_CUTOFFS in src/projection.py."
        )

    ctx = context_mask.squeeze(-1).bool()      # (N,)
    gen = ~ctx                                  # generated atoms

    gen_idx = gen.nonzero(as_tuple=True)[0]     # indices of generated atoms
    ctx_idx = ctx.nonzero(as_tuple=True)[0]     # indices of context atoms

    # Only bail when there is nothing generated to move. We deliberately do
    # NOT bail when there is no context: in the bare-metal de-novo (maskall)
    # regime the metal is the only context atom and it is exempt, so the only
    # obstacles are *other generated ligand atoms* — projection must still run.
    if gen_idx.numel() == 0:
        return pos

    # Group id per atom (argmax of one-hot). Metal / unassigned atoms have
    # all-zero rows; mark them with -1 so they never match any real group and
    # are never treated as obstacles (donors should be free to approach the
    # metal; atoms within one group keep their real bonds).
    group_sum = ligand_group.sum(dim=-1)        # (N,)
    group_id = ligand_group.argmax(dim=-1)      # (N,)
    group_id[group_sum == 0] = -1               # metal / unassigned

    pos_out = pos.clone()
    ctx_pos = pos[ctx_idx]                      # (C, 3) — fixed, never modified

    gen_groups = group_id[gen_idx]              # (G,)
    ctx_groups = group_id[ctx_idx]              # (C,)

    G = gen_idx.size(0)
    eps = 1e-8
    max_iters = 20  # converges in 2-3 for typical geometries

    # --- generated <-> context obstacle mask (fixed obstacles) -------------
    # Skip same-group pairs (intra-ligand bonds) and the metal (group -1).
    has_ctx = ctx_idx.numel() > 0
    if has_ctx:
        if same_group_allowed:
            gc_same = gen_groups.unsqueeze(1) == ctx_groups.unsqueeze(0)    # (G, C)
        else:
            gc_same = torch.zeros(G, ctx_idx.size(0), dtype=torch.bool,
                                  device=pos.device)
        gc_metal = (ctx_groups == -1).unsqueeze(0).expand_as(gc_same)      # (G, C)
        gc_skip = gc_same | gc_metal            # don't project these pairs

    # --- generated <-> generated obstacle mask (movable obstacles) ---------
    # Same exemptions, but both atoms are generated so BOTH get pushed (below).
    # Skip self, same-group pairs, and any (defensive) metal-row generated atom.
    self_mask = torch.eye(G, dtype=torch.bool, device=pos.device)          # (G, G)
    if same_group_allowed:
        gg_same = gen_groups.unsqueeze(1) == gen_groups.unsqueeze(0)       # (G, G)
    else:
        gg_same = self_mask                     # only self counts as "same"
    gg_metal = gen_groups == -1                                            # (G,)
    gg_skip = gg_same | self_mask | gg_metal.unsqueeze(1) | gg_metal.unsqueeze(0)

    for _ in range(max_iters):
        gen_pos = pos_out[gen_idx]              # (G, 3) snapshot this iteration

        # generated <-> generated violations (symmetric (G, G))
        gdiff = gen_pos.unsqueeze(1) - gen_pos.unsqueeze(0)  # (G, G, 3): i - j
        gdist = gdiff.norm(dim=-1)              # (G, G)
        gg_violates = (gdist < d_min) & ~gg_skip

        # generated <-> context violations (rectangular (G, C))
        if has_ctx:
            diff = gen_pos.unsqueeze(1) - ctx_pos.unsqueeze(0)  # (G, C, 3)
            dist = diff.norm(dim=-1)            # (G, C)
            gc_violates = (dist < d_min) & ~gc_skip
        else:
            gc_violates = None

        if not gg_violates.any() and (gc_violates is None or not gc_violates.any()):
            break

        # --- Phase A: generated <-> generated, symmetric split push --------
        # Both atoms in a violating cross-group pair share the correction, each
        # moving apart by half the deficit. Accumulate every pair's push per
        # atom and apply the MEAN: an atom squeezed by several neighbours moves
        # along the average direction. This is order-independent (no atom is
        # privileged by loop order) and stable under the outer iteration.
        if gg_violates.any():
            u = gdiff / (gdist.unsqueeze(-1) + eps)         # (G, G, 3): j -> i
            half = 0.5 * (d_min - gdist).clamp(min=0.0)     # (G, G)
            contrib = half.unsqueeze(-1) * u                # (G, G, 3)
            contrib = contrib * gg_violates.unsqueeze(-1)   # zero non-violating
            n_obst = gg_violates.sum(dim=1)                 # (G,) obstacles per atom
            mean_corr = contrib.sum(dim=1) / n_obst.clamp(min=1).unsqueeze(-1)
            pos_out[gen_idx] = gen_pos + mean_corr

        # --- Phase B: generated <-> context, fixed obstacles ---------------
        # Context never moves, so the generated atom absorbs the full
        # correction. Project sequentially against each violating context atom
        # (the running update lets an atom tunnel past a line of obstacles).
        # Re-read pos_out so any Phase A shift is included; the inner
        # ``d < d_min`` guard ignores contacts Phase A already resolved.
        if gc_violates is not None and gc_violates.any():
            for gi in range(G):
                viol_j = gc_violates[gi].nonzero(as_tuple=True)[0]
                if viol_j.numel() == 0:
                    continue
                cur_pos = pos_out[gen_idx[gi]]
                for cj in viol_j:
                    c_pos = ctx_pos[cj]
                    delta = cur_pos - c_pos
                    d = delta.norm() + eps
                    if d < d_min:
                        direction = delta / d
                        cur_pos = c_pos + direction * d_min
                pos_out[gen_idx[gi]] = cur_pos

    return pos_out


def d_min_schedule(s: int, T: int, d_min_start: float, d_min_end: float) -> float:
    """Piecewise-linear annealing of d_min from start (high t) to end (low t).

    At timestep s (0 = final, T-1 = noisiest):
        d_min(s) = d_min_start - (d_min_start - d_min_end) * (1 - s / T)

    So at s ≈ T (early reverse, high noise): d_min ≈ d_min_start (looser).
       at s = 0 (final step):                d_min = d_min_end   (tighter).

    Both endpoints should sit at or above the bond-perception cutoffs in
    BOND_PERCEPTION_CUTOFFS (>= ~1.72 Å, the O–O floor); a d_min below that
    cannot stop two atoms from being perceived as bonded. See the module
    header for the cutoff table.
    """
    frac = s / T  # 0 at end, ~1 at start of reverse
    return d_min_start - (d_min_start - d_min_end) * (1.0 - frac)
