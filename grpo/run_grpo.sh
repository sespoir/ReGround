#!/usr/bin/env bash
# Stage-2 ReGround GRPO recipe for veRL v0.7.1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-/tmp/verl-v0.7.1}"
MODEL_PATH="${REGROUND_SFT_MODEL_PATH:-/tmp/reground/models/qwen2.5-vl-7b-sft}"
TRAIN_FILE="${REGROUND_GRPO_TRAIN_FILE:-/tmp/reground/data/grpo/train.parquet}"
VAL_FILE="${REGROUND_GRPO_VAL_FILE:-/tmp/reground/data/grpo/val.parquet}"
OUTPUT_DIR="${REGROUND_GRPO_OUTPUT_DIR:-/tmp/reground/outputs/qwen2.5-vl-7b-grpo}"
CONFIG_ONLY="${REGROUND_GRPO_CONFIG_ONLY:-false}"

required_paths=("${VERL_ROOT}")
if [[ "${CONFIG_ONLY}" != "true" ]]; then
  required_paths+=("${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}")
fi
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing ${required_path}. Prepare or replace the /tmp/reground placeholder." >&2
    exit 1
  fi
done

if [[ "$(git -C "${VERL_ROOT}" rev-parse HEAD)" != "bec9ef74768dd201881cd4e54cd0385e87caae27" ]]; then
  echo "Warning: this recipe is validated against veRL v0.7.1 (bec9ef7)." >&2
fi

export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1

hydra_flags=()
if [[ "${CONFIG_ONLY}" == "true" ]]; then
  hydra_flags+=(--cfg job --resolve)
fi

cd "${VERL_ROOT}"
python3 -m verl.trainer.main_ppo "${hydra_flags[@]}" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
  data.max_prompt_length=8192 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.image_key=images \
  data.return_raw_chat=true \
  data.return_multi_modal_inputs=false \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1.0e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.freeze_vision_tower=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-2}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_model_len=12288 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.free_cache_engine=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=true \
  actor_rollout_ref.rollout.agent.default_agent_loop=reground_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${REPO_ROOT}/grpo/agent_loop.yaml" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  reward.custom_reward_function.path="${REPO_ROOT}/grpo/reward.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.lambda_reg=0.5 \
  +reward.custom_reward_function.reward_kwargs.lambda_acc=0.7 \
  +reward.custom_reward_function.reward_kwargs.lambda_form=0.01 \
  +reward.custom_reward_function.reward_kwargs.gamma=0.14 \
  +reward.custom_reward_function.reward_kwargs.beta=0.14 \
  +reward.custom_reward_function.reward_kwargs.max_answer_chars=256 \
  +reward.custom_reward_function.reward_kwargs.min_diagnosis_words=5 \
  reward.reward_manager.name=naive \
  reward.num_workers="${REWARD_WORKERS:-32}" \
  trainer.project_name=reground \
  trainer.experiment_name=qwen2_5_vl_7b_grpo \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="${GPUS_PER_NODE:-8}" \
  trainer.nnodes="${NNODES:-4}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=584 \
  trainer.critic_warmup=0 \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-50}" \
  trainer.val_before_train=true \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.max_actor_ckpt_to_keep=3 \
  "$@"
