#!/usr/bin/env bash
#SBATCH --job-name=reground-vllm
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=reground-vllm-%j.log
#SBATCH --error=reground-vllm-%j.err

set -eo pipefail

REPO_ROOT="${REGROUND_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${REGROUND_CONDA_ENV:-reground}"
set -u
exec "${REPO_ROOT}/scripts/serve_vllm.sh"
