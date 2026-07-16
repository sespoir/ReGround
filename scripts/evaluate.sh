#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

VLM_EVAL_ROOT="${VLM_EVAL_ROOT:-${REPO_ROOT}/third_party/VLMEvalKit}"
if [[ "${VLM_EVAL_ROOT}" != /* ]]; then
  VLM_EVAL_ROOT="${REPO_ROOT}/${VLM_EVAL_ROOT#./}"
fi
if [[ ! -f "${VLM_EVAL_ROOT}/run.py" ]]; then
  echo "VLMEvalKit not found. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

OUTPUT_DIR="${REGROUND_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR#./}"
fi
mkdir -p "${OUTPUT_DIR}"

REGROUND_LOG_DIR="${REGROUND_LOG_DIR:-${REPO_ROOT}/token_logs}"
if [[ "${REGROUND_LOG_DIR}" != /* ]]; then
  REGROUND_LOG_DIR="${REPO_ROOT}/${REGROUND_LOG_DIR#./}"
fi
export REGROUND_LOG_DIR
mkdir -p "${REGROUND_LOG_DIR}"

cd "${VLM_EVAL_ROOT}"
PYTHON_BIN="${REGROUND_PYTHON:-python}"
"${PYTHON_BIN}" run.py \
  --data "${REGROUND_DATASET:-HallusionBench}" \
  --model "${REGROUND_ADAPTER_NAME:-qwen-vl-reground}" \
  --judge "${REGROUND_JUDGE:-exact_matching}" \
  --api-nproc "${REGROUND_API_WORKERS:-32}" \
  --work-dir "${OUTPUT_DIR}" \
  --verbose
