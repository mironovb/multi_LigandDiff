"""
Hard geometric exclusion-shell projection for diffusion sampling.

Projects generated atoms onto the surface of a minimum-distance shell
around context atoms from different ligand groups, preventing inter-ligand
bridging artefacts (O-O contacts at ~1.3 Å across ligand boundaries).

Reference: Christopher, Baek & Fioretto, "Projected Diffusion Models",
NeurIPS 2024 (arXiv:2402.03559).
"""

import torch


def project_exclusion_shell(
    pos: torch.Tensor,
    ligand_group: torch.Tensor,
    context_mask: torch.Tensor,
    d_min: float,
    same_group_allowed: bool = True,
) -> torch.Tensor:
    """Project generated atoms so they are at least d_min from cross-group context atoms.

    Args:
        pos: (N, 3) atom positions.
        ligand_group: (N, MAX_LIGANDS) one-hot ligand group assignment.
        context_mask: (N, 1) binary mask — 1 for context (fixed), 0 for generated.
        d_min: minimum allowed distance (Angstroms).
        same_group_allowed: if True, skip projection for same-group pairs.

    Returns:
        (N, 3) projected positions (only generated atoms may be modified).
    """
    ctx = context_mask.squeeze(-1).bool()      # (N,)
    gen = ~ctx                                  # generated atoms

    gen_idx = gen.nonzero(as_tuple=True)[0]     # indices of generated atoms
    ctx_idx = ctx.nonzero(as_tuple=True)[0]     # indices of context atoms

    if gen_idx.numel() == 0 or ctx_idx.numel() == 0:
        return pos

    # Group id per atom (argmax of one-hot). Metal / unassigned atoms have
    # all-zero rows; mark them with -1 so they never match any real group.
    group_sum = ligand_group.sum(dim=-1)        # (N,)
    group_id = ligand_group.argmax(dim=-1)      # (N,)
    group_id[group_sum == 0] = -1               # metal / unassigned

    pos_out = pos.clone()
    ctx_pos = pos[ctx_idx]                      # (C, 3) — fixed, never modified

    gen_groups = group_id[gen_idx]              # (G,)
    ctx_groups = group_id[ctx_idx]              # (C,)

    # Mask: which (gen, ctx) pairs should be checked?
    # Skip same-group pairs and skip metal context atoms (group_id == -1).
    if same_group_allowed:
        same_group = gen_groups.unsqueeze(1) == ctx_groups.unsqueeze(0)  # (G, C)
    else:
        same_group = torch.zeros(gen_idx.size(0), ctx_idx.size(0),
                                 dtype=torch.bool, device=pos.device)
    metal_ctx = (ctx_groups == -1).unsqueeze(0).expand_as(same_group)  # (G, C)
    skip = same_group | metal_ctx               # don't project these pairs

    eps = 1e-8
    max_iters = 20  # converges in 2-3 for typical geometries

    for _ in range(max_iters):
        gen_pos = pos_out[gen_idx]              # (G, 3)
        diff = gen_pos.unsqueeze(1) - ctx_pos.unsqueeze(0)  # (G, C, 3)
        dist = diff.norm(dim=-1)                # (G, C)
        violates = (dist < d_min) & ~skip       # (G, C)

        if not violates.any():
            break

        # Apply projections atom-by-atom
        for gi in range(gen_idx.size(0)):
            viol_j = violates[gi].nonzero(as_tuple=True)[0]
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
    """
    frac = s / T  # 0 at end, ~1 at start of reverse
    return d_min_start - (d_min_start - d_min_end) * (1.0 - frac)
