#!/bin/bash
#SBATCH -J ln_xtb
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=03:59:59
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# GFN2-xTB optimization of valid generated structures (Jin/Merz validity protocol).
# CPU job (no GPU). Adds Hs, runs GFN2-xTB (charge 0, uhf 6 for Eu3+ f6), aggregates.
# Result of record: mask1 baseline 38/41 = 92.7%; RePaint r=5 81/85 converged.
# Set SET=<dir-tag> to point at a different generated set.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load miniforge/25.11.0-0
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"
which xtb && xtb --version 2>&1 | head -3

# Default: optimise the mask1 baseline and the RePaint r=5 set.
for SET in eu_tmma_mask1_epoch48 repaint_r5_epoch48; do
    IN="generated/${SET}/noH"
    OUT="xtb_results/${SET}"
    if [ ! -d "$IN" ]; then
        echo "skip: $IN not found"
        continue
    fi
    echo "=== xTB on $IN -> $OUT at $(date) ==="
    python -u xtb_pipeline.py \
        --input-dir "$IN" \
        --output-dir "$OUT" \
        --org-xyz eu_tmma_cis.xyz \
        --nproc ${SLURM_CPUS_PER_TASK} \
        --timeout-minutes 10 \
        --charge 0 \
        --uhf 6
    python -u xtb_pipeline.py --input-dir "$IN" --output-dir "$OUT" --aggregate
    echo "--- $SET summary ---"
    [ -f "$OUT/summary.json" ] && grep -c '"xtb_success": true' "$OUT/summary.json"
done

echo "Done: $(date)"
