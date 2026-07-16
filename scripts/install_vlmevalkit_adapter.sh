#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLM_EVAL_ROOT="${1:-${VLM_EVAL_ROOT:-}}"

if [[ -z "${VLM_EVAL_ROOT}" ]]; then
  echo "Usage: $0 /path/to/VLMEvalKit" >&2
  exit 2
fi
if [[ ! -d "${VLM_EVAL_ROOT}/vlmeval/api" ]]; then
  echo "Not a VLMEvalKit checkout: ${VLM_EVAL_ROOT}" >&2
  exit 2
fi

EXPECTED_COMMIT="2c25371d602909ae3d6d395185aff1bc9493262d"
CURRENT_COMMIT="$(git -C "${VLM_EVAL_ROOT}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Warning: adapter was validated at ${EXPECTED_COMMIT}; found ${CURRENT_COMMIT}" >&2
fi

TARGET="${VLM_EVAL_ROOT}/vlmeval/api/qwen_vl_reground_api.py"
if [[ -e "${TARGET}" ]]; then
  if cmp -s "${REPO_ROOT}/src/qwen_vl_reground_api.py" "${TARGET}"; then
    python "${REPO_ROOT}/scripts/register_vlmevalkit.py" "${VLM_EVAL_ROOT}"
    echo "QwenVLRegroundAPI is already installed in ${VLM_EVAL_ROOT}"
    exit 0
  fi
  echo "Refusing to overwrite existing ${TARGET}" >&2
  exit 1
fi

python "${REPO_ROOT}/scripts/register_vlmevalkit.py" \
  --check "${VLM_EVAL_ROOT}"
install -m 0644 "${REPO_ROOT}/src/qwen_vl_reground_api.py" "${TARGET}"
python "${REPO_ROOT}/scripts/register_vlmevalkit.py" "${VLM_EVAL_ROOT}"

echo "Installed QwenVLRegroundAPI into ${VLM_EVAL_ROOT}"
