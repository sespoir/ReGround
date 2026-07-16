#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

MODEL_PATH="${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH in the private .env file}"
PORT="${VLLM_PORT:-8011}"
MODEL_NAME="${REGROUND_MODEL_NAME:-qwen2_5-vl-reground}"

# V1 enables hash-based multimodal encoder-output reuse across requests while
# the cache entry is resident. Round 2 sends byte-identical image content.
export VLLM_USE_V1=1
export VLLM_VIT_ATTN_BACKEND="${VLLM_VIT_ATTN_BACKEND:-SDPA}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --mm-encoder-tp-mode weights \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB:-4}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --enable-prefix-caching \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager
