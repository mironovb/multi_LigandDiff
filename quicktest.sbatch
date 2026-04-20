#!/bin/bash
#SBATCH -J ln_quick
#SBATCH -A mit_general
#SBATCH -p mit_quicktest
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load miniforge/25.11.0-0
module load cuda/12.9.1
source activate ligdiff

cd ~/multi_LigandDiff

echo "=== Imports ==="
python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
from src.lightning import DDPM
from src.const import MAX_LIGANDS
print(f'MAX_LIGANDS: {MAX_LIGANDS}')
"

echo ""
echo "=== Data files ==="
ls -lh data/train_ln.pt data/val_ln.pt model/pre_trained.ckpt

echo ""
echo "=== Load data sample ==="
python -c "
import torch
t = torch.load('data/train_ln.pt', map_location='cpu', weights_only=False)
v = torch.load('data/val_ln.pt', map_location='cpu', weights_only=False)
print(f'train: {len(t)} samples')
print(f'val: {len(v)} samples')
print(f'fields: {list(t[0].keys())}')
print(f'ligand_group shape: {t[0].ligand_group.shape}')
"

echo "Done: $(date)"
