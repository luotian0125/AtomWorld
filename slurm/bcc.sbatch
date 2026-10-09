#!/bin/bash
#SBATCH --job-name=atomworld-bcc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training-%j.out
set -euo pipefail
cd "${ATOMWORLD_RUN_ROOT:?}"
export PYTHONPATH="${ATOMWORLD_PROJECT:?}${PYTHONPATH:+:$PYTHONPATH}"
export ATOMWORLD_RESULTS="$ATOMWORLD_RUN_ROOT/models"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$ATOMWORLD_RUN_ROOT/mpl-$SLURM_JOB_ID"
python="${ATOMWORLD_PYTHON:-python}"
"$python" -m pip freeze > logs/environment.txt
"$python" -m atomworld.train_bcc "$@"
