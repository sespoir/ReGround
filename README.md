## ✨ Overview

Vision-language models can gradually lose visual grounding during long reasoning chains. **ReGround** teaches a model to diagnose when its reasoning needs fresh visual evidence and to selectively re-examine the image.

At inference time, the model emits `<reground>` when re-examination is needed. The same image is then re-injected into the conversation, allowing the model to revise its reasoning before producing the final answer. ReGround requires no external visual tools or architecture changes.

This repository provides data construction, inference, and evaluation code for Qwen2.5-VL, built with [vLLM](https://github.com/vllm-project/vllm) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## 🛠️ Setup

```bash
git clone https://github.com/sespoir/ReGround.git
cd ReGround
bash scripts/bootstrap.sh --with-server
conda activate reground
```

The setup script creates a private `.env` file. Add the local checkpoint path and adjust the GPU settings if needed:

```bash
QWEN_MODEL_PATH=/absolute/path/to/checkpoint
TENSOR_PARALLEL_SIZE=1
```

## 🧩 Data Construction

The dataset-agnostic generator in `data_generation/generate_sft.py` accepts JSON, JSONL, or Parquet records and uses OpenAI-compatible policy and teacher models to construct ReGround supervision. It routes samples according to answer correctness, visual grounding quality, and stochastic verification into Correction, Grounding, Verification, or No-ReGround trajectories.

The resulting single- and two-round conversations follow the `<think>`, `<reground>`, and `<answer>` format used for SFT. Run the pipeline with `data_generation/run_generation.sh`; field mappings, concurrency, checkpointing, and resume behavior can be configured through command-line options or environment variables.

## 🚀 Inference

Start the OpenAI-compatible vLLM server:

```bash
bash scripts/serve_vllm.sh
```

Run a two-round smoke test in another terminal:

```bash
conda activate reground
python scripts/smoke_test_reground.py \
  --image /path/to/image.jpg \
  --question "What is shown in this image?"
```

The adapter preserves the first response, detects `<reground>`, re-injects the original image, and returns the final `<answer>` content.

## 📊 Evaluation

Set the dataset and endpoint in `.env`, then run:

```bash
bash scripts/evaluate.sh
```

The default configuration evaluates `HallusionBench` with exact matching. For Slurm clusters, equivalent jobs are available in `jobs/serve_vllm.sbatch` and `jobs/evaluate.sbatch`.

## 🧪 Tests

```bash
ruff check src scripts tests data_generation
python tests/test_reground_payload.py
bash scripts/secret_scan.sh
```

## 🙏 Acknowledgements

This project is built on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [vLLM](https://github.com/vllm-project/vllm), and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank their authors for making their work publicly available.

## 📄 License

Released under the [Apache License 2.0](LICENSE).
