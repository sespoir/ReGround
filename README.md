# Reground VLM Inference

Reference inference and evaluation code for vision-language models that can
request a second look at the input image by emitting `<reground>`.

> **Paper:** *ReGround: Restoring Visual Grounding in Multi-Step Reasoning
> through Self-Diagnosis and Visual Re-Examination* — accepted at the 34th ACM
> International Conference on Multimedia (ACM MM 2026).
>
> **Release scope:** this initial release contains the inference and evaluation
> artifact. Training and data-construction scripts will be added in a subsequent
> release.

The repository provides a two-round OpenAI-compatible client, a pinned
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) integration, vLLM
serving scripts, Slurm entry points, offline contract tests, and a reproducible
HallusionBench evaluation path. Model weights, datasets, generated predictions,
token logs, and credentials are deliberately kept outside the Git repository.

## Overview

Round 1 sends the image and question to the model. A normal answer ends the
request. If the model emits `<reground>`, the adapter creates Round 2 by
appending the first response, reinserting byte-identical image content at the
new decoder position, and asking for the final answer.

```text
Round 1: image + question -> reasoning + <reground>
Round 2: history + same image + final-answer instruction -> answer
```

With vLLM V1, the repeated image content can hit the server-side multimodal
encoder cache. The client does not transfer or manipulate internal vision
tensors. A cache miss is safe: vLLM simply recomputes the visual features. See
[the design note](docs/design.md) for the exact guarantees and boundaries.

## Verified result

The current adapter and checkpoint were evaluated on the complete 951-example
HallusionBench set with one NVIDIA A100 80 GB GPU, 32 API workers, and the
VLMEvalKit `exact_matching` judge.

| Split | aAcc | fAcc | qAcc |
| --- | ---: | ---: | ---: |
| Overall | 72.8707 | 51.4451 | 52.7473 |
| VD | 68.6971 | 49.1304 | 44.0433 |
| VS | 79.7222 | 56.0345 | 66.2921 |

The model naturally emitted `<reground>` for 154 of 951 examples (16.19%), and
all 154 second-round requests returned non-empty outputs. Inference completed
in 6 minutes 4 seconds. vLLM reported a final multimodal cache hit rate of
73.8%. These numbers are an artifact validation run rather than a replacement
for the evaluation protocol or main results of the associated paper. Full
settings and category-level scores are recorded in
[docs/validation.md](docs/validation.md).

## Repository layout

```text
.
├── src/
│   └── qwen_vl_reground_api.py       # Two-round VLMEvalKit API adapter
├── scripts/
│   ├── bootstrap.sh                  # Create the environment and pin VLMEvalKit
│   ├── install_vlmevalkit_adapter.sh # Install the adapter into VLMEvalKit
│   ├── register_vlmevalkit.py        # Register the model at validated anchors
│   ├── serve_vllm.sh                 # Start the OpenAI-compatible vLLM server
│   ├── evaluate.sh                   # Run a VLMEvalKit evaluation
│   ├── smoke_test_reground.py        # Exercise the natural two-round path
│   └── secret_scan.sh                # Scan tracked content before publication
├── jobs/                              # Slurm serving and evaluation entry points
├── tests/                             # Offline payload and trigger contracts
├── docs/                              # Design, environment, and validation notes
├── CONTRIBUTING.md                    # Development and reporting guidelines
├── environment.yml                   # Portable Conda base environment
├── requirements-server.txt           # Validated vLLM serving stack
└── .env.example                      # Safe configuration template
```

## Requirements

The evaluation client requires Linux, Conda, Git, and a Python 3.10-compatible
environment. Serving additionally requires a CUDA-capable GPU and a compatible
vLLM installation. The validated stack is listed under
[Reproducibility](#reproducibility).

The bootstrap script pins VLMEvalKit to commit
`2c25371d602909ae3d6d395185aff1bc9493262d`. The adapter registration uses
strict source anchors, so an incompatible VLMEvalKit revision fails explicitly
instead of silently installing a broken integration.

## Installation

From the repository root, install the evaluation client with:

```bash
bash scripts/bootstrap.sh
```

To install the validated serving dependencies in the same environment:

```bash
bash scripts/bootstrap.sh --with-server
```

The script creates or updates the `reground` Conda environment, installs the
pinned VLMEvalKit checkout under `third_party/`, registers
`QwenVLRegroundAPI`, and creates a private `.env` file with mode `0600` if one
does not already exist.

Use `REGROUND_CONDA_ENV` to select another environment name or
`VLM_EVAL_ROOT` to select another VLMEvalKit checkout.

## Configuration

Copying is normally handled by `bootstrap.sh`. For a manual setup:

```bash
cp .env.example .env
chmod 600 .env
```

At minimum, configure the OpenAI-compatible endpoint and served model name:

```dotenv
VLM_EVAL_ROOT=./third_party/VLMEvalKit
REGROUND_BASE_URL=http://localhost:8011/v1
REGROUND_MODEL_NAME=qwen2_5-vl-reground
REGROUND_API_KEY=EMPTY
```

For local serving, also set the checkpoint path:

```dotenv
QWEN_MODEL_PATH=/path/to/checkpoint
VLLM_PORT=8011
TENSOR_PARALLEL_SIZE=1
```

Local vLLM deployments commonly accept `EMPTY`. If an endpoint requires a real
credential, store it only in the ignored `.env`, a scheduler secret, or a
dedicated secret manager. Never add it to `.env.example`, shell scripts, job
logs, or Git history.

## Serving

Start a local server:

```bash
conda activate reground
bash scripts/serve_vllm.sh
```

Or submit the included one-GPU Slurm job:

```bash
sbatch jobs/serve_vllm.sbatch
```

The default configuration uses tensor parallel size 1. If a model needs
multiple GPUs, update both the Slurm GPU request and `TENSOR_PARALLEL_SIZE`.
The server enables vLLM V1, multimodal processor caching, and prefix caching.

## Evaluation

The default evaluation is HallusionBench with exact matching:

```bash
conda activate reground
bash scripts/evaluate.sh
```

The equivalent Slurm entry point is:

```bash
sbatch jobs/evaluate.sbatch
```

The main evaluation settings are:

```dotenv
REGROUND_DATASET=HallusionBench
REGROUND_JUDGE=exact_matching
REGROUND_API_WORKERS=32
REGROUND_OUTPUT_DIR=./outputs
REGROUND_LOG_DIR=./token_logs
```

If VLMEvalKit is installed in a separate Python environment, set
`REGROUND_PYTHON=/absolute/path/to/python`; otherwise `python` from the active
environment is used. Use a fresh `REGROUND_OUTPUT_DIR` for each experimental
configuration because VLMEvalKit can reuse existing predictions.

Evaluation workbooks are written under `outputs/`. Per-request token statistics
and first-/second-round model text are written under `token_logs/`. Both
directories are ignored by Git.

## Two-round smoke test

Use a difficult, image-dependent example that is expected to trigger the model:

```bash
conda activate reground
PYTHONPATH="${VLM_EVAL_ROOT:-./third_party/VLMEvalKit}" \
python scripts/smoke_test_reground.py \
  --image /path/to/image.jpg \
  --question "Your image-dependent question" \
  --max-tokens 2048
```

By default the command fails unless Round 1 naturally emits `<reground>` and a
real second HTTP request succeeds. `--force-reground` is available only for
diagnosing the transport and cache path; a forced result is not evidence that
the model learned the trigger.

Run the offline contract tests without a model server:

```bash
python -m pip install requests pillow
python tests/test_reground_payload.py
```

The tests verify that the trigger creates a second request and that Round 2
receives a byte-identical copy of the Round-1 image payload.

## Reproducibility

The end-to-end validation used:

| Component | Version |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.9.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| Transformers | 4.57.6 |
| vLLM | 0.11.2 (V1) |
| VLMEvalKit | `2c25371d602909ae3d6d395185aff1bc9493262d` |

See [docs/environment-audit.md](docs/environment-audit.md) for dependency
provenance and [docs/validation.md](docs/validation.md) for the test protocol,
request counts, token usage, trigger rate, and complete category scores.

## Scope and limitations

- This repository implements inference and evaluation. It does not contain the
  training pipeline or training dataset.
- Encoder-output reuse is a vLLM server-side optimization, not an API-level
  guarantee. Cache entries can be evicted.
- The adapter performs at most one reground step. A second `<reground>` marker
  in Round 2 is not followed recursively.
- The answer extractor supports `<answer>...</answer>`, post-`</think>` text,
  multiple-choice letters, and numeric fallbacks. Dataset-specific evaluation
  may still classify free-form outputs as unknown.
- Benchmark speed and cache hit rate depend on hardware, concurrency, image
  sizes, model output lengths, and server cache pressure.

## Security and publication hygiene

Before publishing changes, run:

```bash
bash scripts/secret_scan.sh
git diff --check
git status --short
```

The ignore rules exclude credentials, checkpoints, safetensors, predictions,
token logs, Slurm logs, bytecode, and the local VLMEvalKit checkout. The adapter
does not write authorization headers or image bytes to its logs, but it does
record questions and model-generated text. Treat those logs as potentially
sensitive research artifacts.

If a credential has ever appeared in a command, log, chat, or earlier commit,
remove it from history and rotate it before making the repository public.

## Model weights

Model weights are intentionally distributed separately from this codebase.
Large checkpoints should be published through a model registry such as
Hugging Face with an explicit model card, base-model attribution, license,
training-data description, intended-use statement, evaluation results, and
known limitations. Do not commit checkpoint shards to this Git repository.

## License

The inference and evaluation code in this repository is released under the
[Apache License 2.0](LICENSE). Model weights are distributed separately and
must include their own license and model card.

## Paper and citation

The paper has been accepted at ACM MM 2026 but is not yet represented by a
final proceedings record. The camera-ready paper link, author list, DOI, and
BibTeX entry should be added here and to `CITATION.cff` when the final public
metadata is available. The anonymous submission PDF is intentionally excluded
from this repository.
