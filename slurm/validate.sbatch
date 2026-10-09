#!/bin/bash
#SBATCH --job-name=atomworld-check
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validate-%j.out
set -euo pipefail
cd "${ATOMWORLD_PROJECT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/results/mplcache"
python="${ATOMWORLD_PYTHON:-python}"
mkdir -p results
"$python" -m pytest -q
"$python" -m atomworld.cli verify --samples 5000
