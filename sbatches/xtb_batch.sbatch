#!/bin/bash
#SBATCH -J xtb_batch
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load miniforge/25.11.0-0
source activate ligdiff
cd ~/multi_LigandDiff

# xTB itself is single-threaded; we parallelise at the pipeline level
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== xTB batch pipeline ==="
echo "INPUT_DIR = ${INPUT_DIR:?'Set INPUT_DIR env var'}"
echo "OUTPUT_DIR = ${OUTPUT_DIR:?'Set OUTPUT_DIR env var'}"
echo "Start: $(date)"

python -u xtb_pipeline.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --org-xyz eu_tmma_cis.xyz \
    --nproc 8 \
    --timeout-minutes 10 \
    --charge 0 \
    --uhf 6

echo "=== Pipeline complete at $(date) ==="

# Run analysis plots if there are results
if [ -f "$OUTPUT_DIR/summary.csv" ]; then
    echo "Generating plots..."
    python -u xtb_summary_plot.py --results-dir "$(dirname "$OUTPUT_DIR")"
fi

echo "Done at $(date)"

# --- Example submission ---
# Single run:
#   INPUT_DIR=generated/repaint_r10_epoch48/noH OUTPUT_DIR=xtb_results/r10 sbatch sbatches/xtb_batch.sbatch
#
# All runs:
#   for d in generated/*/noH; do
#       run=$(basename $(dirname $d))
#       INPUT_DIR=$d OUTPUT_DIR=xtb_results/$run sbatch sbatches/xtb_batch.sbatch
#   done
#
# Aggregate after all jobs finish:
#   python xtb_pipeline.py --aggregate --output-dir xtb_results --input-dir dummy
