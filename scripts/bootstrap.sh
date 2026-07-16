#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${REGROUND_CONDA_ENV:-reground}"
VLM_EVAL_ROOT="${VLM_EVAL_ROOT:-${REPO_ROOT}/third_party/VLMEvalKit}"
VLM_EVAL_COMMIT="2c25371d602909ae3d6d395185aff1bc9493262d"
WITH_SERVER=0

if [[ "${1:-}" == "--with-server" ]]; then
  WITH_SERVER=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--with-server]" >&2
  exit 2
fi

command -v conda >/dev/null || {
  echo "Conda is required but was not found in PATH." >&2
  exit 1
}
command -v git >/dev/null || {
  echo "Git is required but was not found in PATH." >&2
  exit 1
}

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update --name "${ENV_NAME}" --file "${REPO_ROOT}/environment.yml"
else
  conda env create --name "${ENV_NAME}" --file "${REPO_ROOT}/environment.yml"
fi

if [[ ! -d "${VLM_EVAL_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${VLM_EVAL_ROOT}")"
  git clone https://github.com/open-compass/VLMEvalKit.git "${VLM_EVAL_ROOT}"
fi

git -C "${VLM_EVAL_ROOT}" cat-file -e "${VLM_EVAL_COMMIT}^{commit}" 2>/dev/null || \
  git -C "${VLM_EVAL_ROOT}" fetch origin "${VLM_EVAL_COMMIT}"
git -C "${VLM_EVAL_ROOT}" checkout --detach "${VLM_EVAL_COMMIT}"

conda run --no-capture-output --name "${ENV_NAME}" \
  python -m pip install -e "${VLM_EVAL_ROOT}"

if [[ "${WITH_SERVER}" == "1" ]]; then
  conda run --no-capture-output --name "${ENV_NAME}" \
    python -m pip install -r "${REPO_ROOT}/requirements-server.txt"
fi

"${REPO_ROOT}/scripts/install_vlmevalkit_adapter.sh" "${VLM_EVAL_ROOT}"

if [[ ! -e "${REPO_ROOT}/.env" ]]; then
  install -m 0600 "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  echo "Created private configuration: ${REPO_ROOT}/.env"
fi

echo "Setup complete. Activate with: conda activate ${ENV_NAME}"
echo "Then edit ${REPO_ROOT}/.env before serving or evaluating."
