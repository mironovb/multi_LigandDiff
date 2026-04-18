#!/usr/bin/env python
"""Fine-tune multi-LigandDiff on lanthanide coordination data.

Loads pretrained weights (originally trained with ligand_node_nf=7) into the
new architecture (ligand_node_nf=11) by zero-padding weight matrices where
the input dimension grew, and uses discriminative learning rates so that
embedding/input layers and padded weights train faster than the frozen-ish
GVP backbone.
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, callbacks, loggers

from src import const
from src.const import NUMBER_OF_ATOM_TYPES, MAX_LIGANDS
from src.lightning import DDPM
from src.utils import disable_rdkit_logging


# ---------------------------------------------------------------------------
# Pretrained weight loading with dimension-mismatch handling
# ---------------------------------------------------------------------------

def load_pretrained_weights(model, checkpoint_path):
    """Load pretrained weights into *model*, handling shape mismatches.

    Strategy:
      - Exact shape match  → copy directly.
      - 2-D weight with more columns (in_features grew) → zero-pad extra cols.
      - 2-D weight with more rows (out_features grew) → zero-pad extra rows.
      - 1-D bias that grew → zero-pad extra entries.
      - Anything else       → leave at default init.

    Returns the set of parameter names that were zero-padded.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained = checkpoint.get('state_dict', checkpoint)

    model_state = model.state_dict()
    loaded, padded, skipped = [], [], []

    for name, param in model_state.items():
        if name not in pretrained:
            skipped.append(name)
            continue

        pre = pretrained[name]

        if param.shape == pre.shape:
            param.copy_(pre)
            loaded.append(name)

        elif param.dim() == 2 and pre.dim() == 2:
            out_ok = param.shape[0] >= pre.shape[0]
            in_ok = param.shape[1] >= pre.shape[1]
            if out_ok and in_ok:
                param.zero_()
                param[:pre.shape[0], :pre.shape[1]].copy_(pre)
                padded.append(f"{name} [{list(pre.shape)} -> {list(param.shape)}]")
            else:
                skipped.append(f"{name} [{list(pre.shape)} vs {list(param.shape)}]")

        elif param.dim() == 1 and pre.dim() == 1 and param.shape[0] >= pre.shape[0]:
            param.zero_()
            param[:pre.shape[0]].copy_(pre)
            padded.append(f"{name} [{list(pre.shape)} -> {list(param.shape)}]")

        else:
            skipped.append(f"{name} [{list(pre.shape)} vs {list(param.shape)}]")

    model.load_state_dict(model_state)

    # Print summary
    print("=" * 60)
    print("Pretrained weight loading summary")
    print("=" * 60)
    print(f"  Loaded exactly : {len(loaded)} layers")
    print(f"  Zero-padded    : {len(padded)} layers")
    for p in padded:
        print(f"    {p}")
    print(f"  Reinitialized  : {len(skipped)} layers")
    for s in skipped:
        print(f"    {s}")
    print("=" * 60)

    return {p.split(' [')[0] for p in padded}


# Layer names that load_pretrained_weights() pads when adapting the
# pretrained CN=6 checkpoint (ligand_node_nf=7) to Ln (ligand_node_nf=11).
# Stable across runs because they are determined by the architecture diff,
# not by checkpoint contents — used on resume when we don't rerun loading.
KNOWN_PADDED_LAYERS = {
    'edm.dynamics.ligand_site_embedding.weight',
    'edm.dynamics.h_embedding_out.weight',
    'edm.dynamics.h_embedding_out.bias',
}


# ---------------------------------------------------------------------------
# Discriminative learning-rate groups
# ---------------------------------------------------------------------------

def build_param_groups(model, padded_names, lr_head=1e-4, lr_backbone=1e-5):
    """Split parameters into two groups with different learning rates.

    Head group (lr_head):
      - Parameters whose name contains 'embed', 'input', or 'embedding'.
      - Parameters that were zero-padded during loading.

    Backbone group (lr_backbone):
      - Everything else (pretrained GVP layers).
    """
    head_params, backbone_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = (
            any(kw in name for kw in ('embed', 'input', 'embedding'))
            or name in padded_names
        )
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"  Head params     (lr={lr_head}): {len(head_params)} tensors")
    print(f"  Backbone params (lr={lr_backbone}): {len(backbone_params)} tensors")

    return [
        {'params': head_params, 'lr': lr_head},
        {'params': backbone_params, 'lr': lr_backbone},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune multi-LigandDiff for lanthanides')
    parser.add_argument('--train_data', type=str, default='data/train_ln.pt')
    parser.add_argument('--val_data', type=str, default='data/val_ln.pt')
    parser.add_argument('--pretrained', type=str,
                        default='model/pre_trained.ckpt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str,
                        default='models/ln_finetuned')
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--hidden_nf', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--attention', type=eval, default=True)
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--wandb_entity', type=str, default='geometric')
    parser.add_argument('--exp_name', type=str, default='ln_finetune')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use (1 or 2 for Isaac V100S)')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Stop if val loss does not improve for N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to Lightning checkpoint to resume training from. '
                             'When set, skips pretrained weight loading entirely and '
                             'continues from the saved epoch/optimizer state.')
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    disable_rdkit_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    # DDPM expects data_path / train_data .pt — split our paths accordingly
    data_dir = os.path.dirname(args.train_data) or 'data'
    train_name = os.path.splitext(os.path.basename(args.train_data))[0]
    val_name = os.path.splitext(os.path.basename(args.val_data))[0]

    in_node_nf = NUMBER_OF_ATOM_TYPES
    ligand_node_nf = MAX_LIGANDS + 1  # ligand_group(10) + coord_site(1)
    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'

    # ---- Build model with the new architecture --------------------------
    ddpm = DDPM(
        data_path=data_dir,
        train_data=train_name,
        val_data=val_name,
        in_node_nf=in_node_nf,
        n_dims=3,
        ligand_node_nf=ligand_node_nf,
        hidden_nf=args.hidden_nf,
        activation='silu',
        n_layers=args.n_layers,
        attention=args.attention,
        normalization_factor=1,
        normalize_factors=[10, 4, 1],
        drop_rate=0.2,
        diffusion_steps=args.diffusion_steps,
        diffusion_noise_schedule='polynomial_2',
        diffusion_noise_precision=1e-5,
        diffusion_loss_type='l2',
        lr=args.lr_head,
        batch_size=args.batch_size,
        torch_device=torch_device,
        model='gvp_dynamics',
        test_epochs=10,
        n_stability_samples=1,
        center_of_mass='context',
        clip_grad=True,
    )

    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        print("Skipping pretrained weight loading (Lightning will restore from checkpoint).")
        # Lightning restores model weights, optimizer state, scheduler state,
        # and epoch count from the checkpoint. We still need to rebuild the
        # 2-group optimizer structure so its shape matches the saved state.
        padded_names = KNOWN_PADDED_LAYERS
    else:
        # ---- Load pretrained weights ------------------------------------
        padded_names = load_pretrained_weights(ddpm, args.pretrained)

    # ---- Discriminative learning rates ----------------------------------
    # Always build 2 param groups so the optimizer structure matches what
    # the checkpoint expects on resume.
    param_groups = build_param_groups(
        ddpm, padded_names, args.lr_head, args.lr_backbone)

    # Override the default optimizer so we can use per-group LRs.
    def _configure_optimizers(self_ignored=None):
        optimizer = torch.optim.AdamW(param_groups, amsgrad=True, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-7,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }

    ddpm.configure_optimizers = _configure_optimizers

    # ---- Trainer --------------------------------------------------------
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename='ln_finetuned_{epoch:02d}',
        monitor='loss/val',
        save_top_k=3,
        mode='min',
    )

    early_stop_callback = callbacks.EarlyStopping(
        monitor='loss/val',
        patience=args.early_stop_patience,
        mode='min',
        verbose=True,
    )

    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    wandb_logger = loggers.WandbLogger(
        save_dir=args.output_dir,
        project='Multi_LigandDiff_Ln',
        name=args.exp_name,
        entity=args.wandb_entity,
        offline=True,  # always offline — avoid auth issues on cluster
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator=args.device,
        devices=args.num_gpus,
        strategy='ddp' if args.num_gpus > 1 else 'auto',
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    print(f"\nStarting fine-tuning for {args.epochs} epochs")
    print(f"  Train: {args.train_data}")
    print(f"  Val:   {args.val_data}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Output:     {args.output_dir}")

    if args.resume_from_checkpoint:
        trainer.fit(model=ddpm, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model=ddpm)


if __name__ == '__main__':
    main()
