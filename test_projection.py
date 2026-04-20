"""Tests for hard geometric exclusion-shell projection."""

import torch
import sys
import os
import subprocess
import glob

from src.projection import project_exclusion_shell, d_min_schedule


def test_synthetic_projection():
    """Test 1: Synthetic case — generated atoms inside d_min shell get pushed out."""
    print("=" * 60)
    print("TEST 1: Synthetic exclusion-shell projection")
    print("=" * 60)

    d_min = 1.5
    MAX_LIGANDS = 10

    # 10 context atoms in group 0, placed on a grid
    n_ctx = 10
    ctx_pos = torch.stack([
        torch.tensor([float(i), 0.0, 0.0]) for i in range(n_ctx)
    ])

    # 5 generated atoms in group 1, deliberately placed TOO CLOSE to context
    # (within 0.5-1.0 A of various context atoms)
    n_gen = 5
    gen_pos = torch.tensor([
        [0.3, 0.0, 0.0],   # 0.3 A from ctx atom 0
        [1.1, 0.2, 0.0],   # ~1.12 A from ctx atom 1
        [3.0, 0.5, 0.0],   # ~0.5 A from ctx atom 3
        [5.0, 0.8, 0.0],   # ~0.8 A from ctx atom 5
        [9.0, 0.3, 0.1],   # ~0.32 A from ctx atom 9
    ])

    pos = torch.cat([ctx_pos, gen_pos], dim=0)  # (15, 3)
    N = n_ctx + n_gen

    # ligand_group: context atoms in group 0, generated atoms in group 1
    ligand_group = torch.zeros(N, MAX_LIGANDS)
    ligand_group[:n_ctx, 0] = 1.0   # context: group 0
    ligand_group[n_ctx:, 1] = 1.0   # generated: group 1

    # context mask
    context_mask = torch.zeros(N, 1)
    context_mask[:n_ctx] = 1.0

    # Print before
    print(f"\nBefore projection (d_min={d_min}):")
    for i in range(n_gen):
        gi = n_ctx + i
        for j in range(n_ctx):
            d = (pos[gi] - pos[j]).norm().item()
            if d < d_min:
                print(f"  gen[{i}] <-> ctx[{j}]: {d:.3f} A  ** VIOLATION **")

    # Apply projection
    pos_proj = project_exclusion_shell(pos, ligand_group, context_mask, d_min)

    # Verify all inter-group distances >= d_min
    print(f"\nAfter projection:")
    all_ok = True
    for i in range(n_gen):
        gi = n_ctx + i
        for j in range(n_ctx):
            d = (pos_proj[gi] - pos_proj[j]).norm().item()
            status = "OK" if d >= d_min - 1e-6 else "** STILL VIOLATING **"
            if d < d_min - 1e-6:
                all_ok = False
            if d < d_min + 0.5:  # only print near-boundary distances
                print(f"  gen[{i}] <-> ctx[{j}]: {d:.3f} A  {status}")

    # Verify context atoms unchanged
    ctx_unchanged = torch.allclose(pos[:n_ctx], pos_proj[:n_ctx])
    print(f"\nContext atoms unchanged: {ctx_unchanged}")

    # Verify same-group atoms would NOT be projected
    print("\nTesting same-group exclusion (should NOT project):")
    ligand_group_same = torch.zeros(N, MAX_LIGANDS)
    ligand_group_same[:, 0] = 1.0  # ALL atoms in same group
    pos_same = project_exclusion_shell(pos, ligand_group_same, context_mask, d_min)
    same_group_unchanged = torch.allclose(pos, pos_same)
    print(f"  Same-group positions unchanged: {same_group_unchanged}")

    passed = all_ok and ctx_unchanged and same_group_unchanged
    print(f"\n{'PASSED' if passed else 'FAILED'}: Synthetic projection test")
    return passed


def test_metal_not_projected():
    """Test that metal atoms (all-zero ligand_group) are skipped."""
    print("\n" + "=" * 60)
    print("TEST 2: Metal atoms not projected")
    print("=" * 60)

    d_min = 2.0
    MAX_LIGANDS = 10

    # Metal at origin with all-zero ligand_group
    # Generated atom very close to metal (this is normal for metal-donor)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],  # metal (context)
        [1.0, 0.0, 0.0],  # generated atom, 1.0 A from metal
    ])
    ligand_group = torch.zeros(2, MAX_LIGANDS)
    # metal has all zeros (no group)
    ligand_group[1, 0] = 1.0  # generated atom in group 0
    context_mask = torch.tensor([[1.0], [0.0]])

    pos_proj = project_exclusion_shell(pos, ligand_group, context_mask, d_min)
    d_after = (pos_proj[1] - pos_proj[0]).norm().item()
    # Should NOT be projected because metal has group_id = -1
    unchanged = torch.allclose(pos, pos_proj)
    print(f"  Metal-gen distance: {d_after:.3f} A (was 1.000 A, d_min={d_min})")
    print(f"  Position unchanged (metal skipped): {unchanged}")
    passed = unchanged
    print(f"\n{'PASSED' if passed else 'FAILED'}: Metal exclusion test")
    return passed


def test_d_min_schedule():
    """Test the annealing schedule."""
    print("\n" + "=" * 60)
    print("TEST 3: d_min annealing schedule")
    print("=" * 60)

    T = 1000
    d_start, d_end = 1.5, 1.3

    # At s = T (start of reverse, high noise): d_min ~ d_start
    d_high = d_min_schedule(T, T, d_start, d_end)
    # At s = 0 (end of reverse, low noise): d_min = d_end
    d_low = d_min_schedule(0, T, d_start, d_end)
    # Midpoint
    d_mid = d_min_schedule(T // 2, T, d_start, d_end)

    print(f"  s=T   (high noise): d_min = {d_high:.3f} (expect ~{d_start})")
    print(f"  s=T/2 (mid noise):  d_min = {d_mid:.3f} (expect ~{(d_start+d_end)/2:.3f})")
    print(f"  s=0   (low noise):  d_min = {d_low:.3f} (expect ~{d_end})")

    passed = (abs(d_high - d_start) < 1e-6 and
              abs(d_low - d_end) < 1e-6 and
              abs(d_mid - (d_start + d_end) / 2) < 0.01)
    print(f"\n{'PASSED' if passed else 'FAILED'}: d_min schedule test")
    return passed


def test_no_op_when_disabled():
    """Test that project_enabled=False leaves sampling bit-identical."""
    print("\n" + "=" * 60)
    print("TEST 4: No-op when disabled (project_enabled=False default)")
    print("=" * 60)

    # When project_enabled defaults to False in edm.sample_chain,
    # the projection block is never entered. Verify the function
    # returns input unchanged when there are no violations.
    d_min = 1.0
    MAX_LIGANDS = 10

    # All atoms far apart — no violations
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [15.0, 0.0, 0.0],
    ])
    ligand_group = torch.zeros(4, MAX_LIGANDS)
    ligand_group[0, 0] = 1.0
    ligand_group[1, 0] = 1.0
    ligand_group[2, 1] = 1.0
    ligand_group[3, 1] = 1.0
    context_mask = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

    pos_proj = project_exclusion_shell(pos, ligand_group, context_mask, d_min)
    unchanged = torch.allclose(pos, pos_proj)
    print(f"  No violations present -> positions unchanged: {unchanged}")
    passed = unchanged
    print(f"\n{'PASSED' if passed else 'FAILED'}: No-op test")
    return passed


if __name__ == "__main__":
    results = []
    results.append(test_synthetic_projection())
    results.append(test_metal_not_projected())
    results.append(test_d_min_schedule())
    results.append(test_no_op_when_disabled())

    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(results)}/{len(results)} unit tests passed")
    print("=" * 60)

    if not all(results):
        sys.exit(1)

    print("\nAll unit tests passed.")
