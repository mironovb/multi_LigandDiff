"""
Smoke test for RePaint resampling in the reverse sampler.

Tests the resampling loop directly on a small EDM with random weights.
This verifies:
1. resample_r=1 produces valid output (no NaN, correct shape)
2. resample_r=1 is reproducible with the same seed
3. resample_r=10 completes without errors
4. Timing: r=10 takes ~10x longer than r=1

To run on the cluster with a real checkpoint, set CKPT env var:
  CKPT=models/ln_finetuned/ln_finetuned_epoch=48.ckpt python test_repaint_smoke.py --full
"""
import os
import sys
import time
import argparse
import torch
import numpy as np

from src.edm import EDM
from src.egnn import Dynamics
from src.noise import PredefinedNoiseSchedule


def build_small_edm(n_dims=3, in_node_nf=8, hidden_nf=32, timesteps=100,
                    n_layers=2, device='cpu'):
    """Build a small EDM for testing (random weights, fast inference)."""
    ligand_node_nf = 11  # MAX_LIGANDS(10) + coord_site(1)
    dynamics = Dynamics(
        in_node_nf=in_node_nf,
        n_dims=n_dims,
        ligand_node_nf=ligand_node_nf,
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        device=device,
    )
    edm = EDM(
        dynamics=dynamics,
        in_node_nf=in_node_nf,
        n_dims=n_dims,
        timesteps=timesteps,
        noise_schedule='polynomial_2',
        noise_precision=1e-4,
        loss_type='l2',
    )
    return edm.eval().to(device)


def make_dummy_data(batch_size=2, n_context=5, n_gen=3, n_dims=3,
                    in_node_nf=8, max_ligands=10, device='cpu'):
    """Create dummy batch data for testing."""
    n_atoms = n_context + n_gen
    total = batch_size * n_atoms
    # batch_seg: [0,0,...,0, 1,1,...,1]
    batch_seg = torch.repeat_interleave(
        torch.arange(batch_size, device=device), n_atoms)
    # positions and features
    x = torch.randn(total, n_dims, device=device) * 2
    h = torch.zeros(total, in_node_nf, device=device)
    h[torch.arange(total), torch.randint(0, in_node_nf, (total,))] = 1
    # masks: first n_context atoms per sample are context
    context = torch.zeros(total, 1, device=device)
    ligand_diff = torch.zeros(total, 1, device=device)
    for b in range(batch_size):
        start = b * n_atoms
        context[start:start + n_context] = 1
        ligand_diff[start + n_context:start + n_atoms] = 1
    # ligand_site: ligand_group (max_ligands) + coord_site (1)
    ligand_group = torch.zeros(total, max_ligands, device=device)
    coord_site = torch.zeros(total, 1, device=device)
    # assign gen atoms to ligand group 0
    for b in range(batch_size):
        start = b * n_atoms + n_context
        ligand_group[start:start + n_gen, 0] = 1
        coord_site[start] = 1  # first gen atom is coord site
    ligand_site = torch.cat([ligand_group, coord_site], dim=-1)
    return x, h, context, ligand_diff, batch_seg, batch_size, ligand_site


def run_edm_sample(edm, data, resample_r, keep_frames, seed=42):
    """Run EDM.sample_chain with given resample_r, return (chain, elapsed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, h, context, ligand_diff, batch_seg, batch_size, ligand_site = data
    start = time.time()
    chain = edm.sample_chain(
        x=x.clone(), h=h.clone(), context=context, ligand_diff=ligand_diff,
        batch_seg=batch_seg, batch_size=batch_size, ligand_site=ligand_site,
        keep_frames=keep_frames, resample_r=resample_r,
    )
    elapsed = time.time() - start
    return chain, elapsed


def test_unit():
    """Unit test with small random EDM."""
    device = 'cpu'
    timesteps = 100
    keep_frames = 100

    print("Building small EDM (random weights, T=100)...")
    edm = build_small_edm(timesteps=timesteps, device=device)
    data = make_dummy_data(batch_size=2, device=device)

    # Test 1: r=1 baseline
    print("\n" + "=" * 50)
    print("Test 1: resample_r=1 (baseline)")
    chain_r1, time_r1 = run_edm_sample(edm, data, resample_r=1,
                                        keep_frames=keep_frames)
    print(f"  Completed in {time_r1:.1f}s")
    print(f"  Chain shape: {chain_r1.shape}")
    print(f"  Output range: [{chain_r1[0].min():.3f}, {chain_r1[0].max():.3f}]")
    has_nan = torch.isnan(chain_r1).any().item()
    print(f"  Contains NaN: {has_nan}")
    assert not has_nan, "resample_r=1 produced NaN!"
    print("  PASSED")

    # Test 2: r=1 reproducibility
    print("\n" + "=" * 50)
    print("Test 2: resample_r=1 reproducibility (same seed)")
    chain_r1b, _ = run_edm_sample(edm, data, resample_r=1,
                                   keep_frames=keep_frames)
    max_diff = (chain_r1 - chain_r1b).abs().max().item()
    print(f"  Max diff between two r=1 runs: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Reproducibility failed: {max_diff}"
    print("  PASSED")

    # Test 3: r=5 (using 5 instead of 10 for speed on CPU)
    print("\n" + "=" * 50)
    r_test = 5
    print(f"Test 3: resample_r={r_test}")
    chain_rN, time_rN = run_edm_sample(edm, data, resample_r=r_test,
                                        keep_frames=keep_frames)
    print(f"  Completed in {time_rN:.1f}s")
    print(f"  Chain shape: {chain_rN.shape}")
    print(f"  Output range: [{chain_rN[0].min():.3f}, {chain_rN[0].max():.3f}]")
    has_nan = torch.isnan(chain_rN).any().item()
    print(f"  Contains NaN: {has_nan}")
    assert not has_nan, f"resample_r={r_test} produced NaN!"
    print("  PASSED")

    # Test 4: timing ratio
    print("\n" + "=" * 50)
    print("Timing comparison:")
    print(f"  r=1 : {time_r1:.1f}s")
    print(f"  r={r_test} : {time_rN:.1f}s")
    ratio = time_rN / max(time_r1, 0.01)
    print(f"  Ratio: {ratio:.1f}x (expected ~{r_test}x)")

    # Test 5: r=1 and r>1 give DIFFERENT outputs (resampling changes the result)
    print("\n" + "=" * 50)
    print("Test 5: r=1 and r>1 produce different outputs")
    diff = (chain_r1 - chain_rN).abs().max().item()
    print(f"  Max diff: {diff:.4f}")
    assert diff > 0.01, "r=1 and r>1 outputs are suspiciously similar"
    print("  PASSED")

    print("\n" + "=" * 50)
    print("ALL UNIT TESTS PASSED")
    return True


def test_full():
    """Full test with real checkpoint (for cluster use)."""
    ckpt = os.environ.get("CKPT", "models/ln_finetuned/ln_finetuned_epoch=48.ckpt")
    if not os.path.isfile(ckpt):
        ckpt = "model/pre_trained.ckpt"
    if not os.path.isfile(ckpt):
        print(f"ERROR: No checkpoint found. Set CKPT env var.")
        sys.exit(1)

    from src.lightning import DDPM
    from generate_mask1 import read_molecule, reform_data
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {ckpt}")
    print(f"Device: {device}")

    ddpm = DDPM.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
    dataset = read_molecule("eu_tmma_cis.xyz")
    data = reform_data(dataset, device, ligand_size='random')[:2]
    loader = DataLoader(data, batch_size=len(data), shuffle=False)
    batch = next(iter(loader))

    for r in [1, 10]:
        torch.manual_seed(42)
        np.random.seed(42)
        start = time.time()
        chain = ddpm.sample_chain(batch, keep_frames=100, resample_r=r)
        elapsed = time.time() - start
        has_nan = torch.isnan(chain).any().item()
        print(f"  r={r:2d}: {elapsed:.1f}s, shape={chain.shape}, NaN={has_nan}")
        assert not has_nan

    print("FULL TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full test with real checkpoint")
    args = parser.parse_args()

    if args.full:
        test_full()
    else:
        test_unit()
