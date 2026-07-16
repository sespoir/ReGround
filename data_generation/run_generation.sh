#!/usr/bin/env bash
#SBATCH --job-name=reground-sft
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --output=reground-sft-%j.log
#SBATCH --error=reground-sft-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATOR="${GENERATOR:-${SCRIPT_DIR}/generate_sft.py}"
VALIDATE_ONLY="${VALIDATE_ONLY:-false}"

: "${INPUT_PATH:?Set INPUT_PATH to a JSON, JSONL, Parquet file, directory, or glob}"
if [[ "${VALIDATE_ONLY}" != "true" ]]; then
    : "${STUDENT_URL:?Set STUDENT_URL to the policy model's OpenAI-compatible endpoint}"
    : "${STUDENT_MODEL:?Set STUDENT_MODEL to the served policy model name}"
    : "${TEACHER_URL:?Set TEACHER_URL to the teacher model's OpenAI-compatible endpoint}"
    : "${TEACHER_MODEL:?Set TEACHER_MODEL to the served teacher model name}"
fi

STUDENT_URL="${STUDENT_URL:-http://localhost:8000/v1}"
STUDENT_MODEL="${STUDENT_MODEL:-policy-model}"
TEACHER_URL="${TEACHER_URL:-http://localhost:8001/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-teacher-model}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_reground_sft}"
INPUT_PATTERN="${INPUT_PATTERN:-*}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SAMPLE_CONCURRENT="${SAMPLE_CONCURRENT:-16}"
STUDENT_CONCURRENT="${STUDENT_CONCURRENT:-16}"
TEACHER_CONCURRENT="${TEACHER_CONCURRENT:-8}"
JUDGE_CONCURRENT="${JUDGE_CONCURRENT:-8}"
EPSILON="${EPSILON:-0.15}"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-300}"
MAX_TOKENS_ROUND1="${MAX_TOKENS_ROUND1:-8192}"
MAX_TOKENS_ROUND2="${MAX_TOKENS_ROUND2:-8192}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-512}"
ROUND1_ATTEMPTS="${ROUND1_ATTEMPTS:-2}"
ROUND2_ATTEMPTS="${ROUND2_ATTEMPTS:-2}"
TEACHER_ATTEMPTS="${TEACHER_ATTEMPTS:-3}"
REQUEST_RETRIES="${REQUEST_RETRIES:-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"

ID_FIELD="${ID_FIELD:-id}"
QUESTION_FIELD="${QUESTION_FIELD:-question}"
ANSWER_FIELD="${ANSWER_FIELD:-answer}"
CHOICES_FIELD="${CHOICES_FIELD:-choices}"
IMAGE_FIELD="${IMAGE_FIELD:-image}"
KNOWLEDGE_FIELD="${KNOWLEDGE_FIELD:-knowledge}"
SOURCE_FIELD="${SOURCE_FIELD:-source}"
METADATA_FIELDS="${METADATA_FIELDS:-}"
ANSWER_INDEX_BASE="${ANSWER_INDEX_BASE:-zero}"
IMAGE_MODE="${IMAGE_MODE:-copy}"

JUDGE_URL="${JUDGE_URL:-${TEACHER_URL}}"
JUDGE_MODEL="${JUDGE_MODEL:-${TEACHER_MODEL}}"

if [[ ! -f "${GENERATOR}" ]]; then
    echo "Generator not found: ${GENERATOR}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

args=(
    "${PYTHON_BIN}" "${GENERATOR}"
    --input "${INPUT_PATH}"
    --input-pattern "${INPUT_PATTERN}"
    --output-dir "${OUTPUT_DIR}"
    --num-samples "${NUM_SAMPLES}"
    --id-field "${ID_FIELD}"
    --question-field "${QUESTION_FIELD}"
    --answer-field "${ANSWER_FIELD}"
    --choices-field "${CHOICES_FIELD}"
    --image-field "${IMAGE_FIELD}"
    --knowledge-field "${KNOWLEDGE_FIELD}"
    --source-field "${SOURCE_FIELD}"
    --metadata-fields "${METADATA_FIELDS}"
    --answer-index-base "${ANSWER_INDEX_BASE}"
    --image-mode "${IMAGE_MODE}"
    --student-url "${STUDENT_URL}"
    --student-model "${STUDENT_MODEL}"
    --teacher-url "${TEACHER_URL}"
    --teacher-model "${TEACHER_MODEL}"
    --judge-url "${JUDGE_URL}"
    --judge-model "${JUDGE_MODEL}"
    --epsilon "${EPSILON}"
    --seed "${SEED}"
    --max-tokens-round1 "${MAX_TOKENS_ROUND1}"
    --max-tokens-round2 "${MAX_TOKENS_ROUND2}"
    --teacher-max-tokens "${TEACHER_MAX_TOKENS}"
    --round1-attempts "${ROUND1_ATTEMPTS}"
    --round2-attempts "${ROUND2_ATTEMPTS}"
    --teacher-attempts "${TEACHER_ATTEMPTS}"
    --sample-concurrent "${SAMPLE_CONCURRENT}"
    --student-concurrent "${STUDENT_CONCURRENT}"
    --teacher-concurrent "${TEACHER_CONCURRENT}"
    --judge-concurrent "${JUDGE_CONCURRENT}"
    --timeout "${TIMEOUT}"
    --request-retries "${REQUEST_RETRIES}"
    --checkpoint-every "${CHECKPOINT_EVERY}"
)

if [[ -n "${IMAGE_ROOT:-}" ]]; then
    args+=(--image-root "${IMAGE_ROOT}")
fi
if [[ "${SHUFFLE:-false}" == "true" ]]; then
    args+=(--shuffle)
fi
if [[ "${RESUME:-true}" == "true" ]]; then
    args+=(--resume)
fi
if [[ "${RETRY_FAILURES:-false}" == "true" ]]; then
    args+=(--retry-failures)
fi
if [[ "${OVERWRITE:-false}" == "true" ]]; then
    args+=(--overwrite)
fi
if [[ "${REQUIRE_ROUND2_JUDGE:-true}" != "true" ]]; then
    args+=(--no-require-round2-judge)
fi
if [[ "${VALIDATE_ONLY}" == "true" ]]; then
    args+=(--validate-only)
fi

echo "ReGround SFT generation"
echo "  Input:          ${INPUT_PATH}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Student:        ${STUDENT_MODEL} @ ${STUDENT_URL}"
echo "  Teacher:        ${TEACHER_MODEL} @ ${TEACHER_URL}"
echo "  Judge:          ${JUDGE_MODEL} @ ${JUDGE_URL}"
echo "  Samples:        ${NUM_SAMPLES} (0 = all)"
echo "  Epsilon:        ${EPSILON}"
echo "  Concurrency:    ${SAMPLE_CONCURRENT}"
echo "  Validate only:  ${VALIDATE_ONLY}"

"${args[@]}"

if [[ "${VALIDATE_ONLY}" == "true" ]]; then
    echo "Validation report: ${OUTPUT_DIR}/input_report.json"
elif [[ -f "${OUTPUT_DIR}/sft.jsonl" ]]; then
    echo "Generated samples: $(wc -l < "${OUTPUT_DIR}/sft.jsonl")"
    echo "Finished. Summary: ${OUTPUT_DIR}/summary.json"
fi
