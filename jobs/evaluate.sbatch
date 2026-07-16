#!/usr/bin/env bash
#SBATCH --job-name=reground-eval
#SBATCH --partition=normal
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:0
#SBATCH --output=reground-eval-%j.log
#SBATCH --error=reground-eval-%j.err

set -eo pipefail

REPO_ROOT="${REGROUND_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${REGROUND_CONDA_ENV:-reground}"
set -u
exec "${REPO_ROOT}/scripts/evaluate.sh"
