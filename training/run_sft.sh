#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${REPO_ROOT}/training/sft_qwen2_5_vl_7b.yaml}"
MODEL_DIR="/tmp/reground/models/Qwen2.5-VL-7B-Instruct"
DATA_FILE="/tmp/reground/data/sft.jsonl"
DATA_INFO="/tmp/reground/data/dataset_info.json"

command -v llamafactory-cli >/dev/null || {
  echo "llamafactory-cli was not found. Install LLaMA-Factory first." >&2
  exit 1
}

for required_path in "${CONFIG}" "${MODEL_DIR}" "${DATA_FILE}" "${DATA_INFO}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing ${required_path}. Replace or prepare the /tmp/reground placeholders before training." >&2
    exit 1
  fi
done

cd "${REPO_ROOT}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
exec llamafactory-cli train "${CONFIG}"
