#!/bin/bash
#SBATCH -J xtb_opt
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load miniforge/25.11.0-0
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
which xtb

python xtb_opt.py xtb_opt

echo "Done: $(date)"
echo "=== Summary ==="
cat xtb_opt/summary.json 2>/dev/null
