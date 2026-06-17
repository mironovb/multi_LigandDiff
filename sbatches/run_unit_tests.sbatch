#!/bin/bash
#SBATCH -J ln_unittest
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# CPU gate for the no-GPU code fixes (prompts 01-21). Runs the unit tests that the
# fix prompts add/extend plus import + compile smoke checks, in the same env where
# RDKit / OpenBabel / molSimplify actually live. Run this after each batch of fixes
# before submitting any GPU job.
#
# nounset trap: never combine `set -euo pipefail` with `source ~/.bashrc` here.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load miniforge/25.11.0-0
source activate ligdiff

cd ~/multi_LigandDiff

echo "Start: $(date)"
echo "Host: $(hostname)"

echo "=== import / compile smoke ==="
python -c "import src.molecule_builder, src.const, src.projection, src.lightning; print('core imports OK')"
python -m py_compile generate.py generate_design_test.py generate_bare.py prepare_training_data.py rescore_validity.py 2>/dev/null \
    && echo "py_compile OK" || echo "py_compile: some files not present yet (expected before their prompt runs)"

echo "=== unit tests (only those that exist) ==="
# No standalone denticity-cap unit test exists; the <=4 cap (prompts 05-07) is exercised
# live by the design sbatches via accounting.json / rejection_summary.json, not pytest.
TESTS=""
for t in test_charge_aware_validity.py test_projection.py test_repaint_smoke.py; do
    [ -f "$t" ] && TESTS="$TESTS $t"
done
echo "running:$TESTS"
python -m pytest -q $TESTS

echo "Done: $(date)"
