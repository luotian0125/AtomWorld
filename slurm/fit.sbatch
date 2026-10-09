#!/bin/bash
#SBATCH --job-name=atomworld-poisson
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fit-%j.out
set -euo pipefail
cd "${ATOMWORLD_PROJECT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python="${ATOMWORLD_PYTHON:-python}"
"$python" -m atomworld.cli fit --train "${TRAIN_DATA:?}" --validation "${VALIDATION_DATA:?}" \
  --output "${MODEL_OUTPUT:?}" --device cuda --iterations 500
