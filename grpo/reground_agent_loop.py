"""Two-round image re-injection agent loop for veRL v0.7.1."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from grpo.protocol import REGROUND_USER_PROMPT, has_reground_marker


class ReGroundAgentLoop(AgentLoopBase):
    """Generate once, re-inject the same image on ``<reground>``, then finish.

    Generated policy tokens receive a response mask of 1. The environment turn
    containing the repeated image receives a response mask of 0, following
    veRL's official multi-turn agent-loop convention.
    """

    def __init__(
        self,
        *args: Any,
        max_generated_tokens: int = 1024,
        reexamination_prompt: str = REGROUND_USER_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.max_generated_tokens = min(max_generated_tokens, self.response_length)
        self.reexamination_prompt = reexamination_prompt

    @staticmethod
    def _add_preemptions(metrics: dict[str, Any], value: int | None) -> None:
        if value is None:
            value = 0
        previous = metrics.get("num_preempted")
        metrics["num_preempted"] = value if previous is None or previous < 0 else previous + value

    @staticmethod
    def _merge_extra_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
        if not target:
            target.update(source)
            return
        max_global_steps = source.get("max_global_steps")
        if max_global_steps is not None:
            target["max_global_steps"] = max_global_steps

    async def _decode(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        original_images = list(multi_modal_data.get("images") or [])
        videos = multi_modal_data.get("videos")

        initial_prompt_ids = await self.apply_chat_template(
            messages,
            images=original_images or None,
            videos=videos,
        )
        runtime_ids = list(initial_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] | None = None
        all_images = list(original_images)
        metrics: dict[str, Any] = {}
        extra_fields: dict[str, Any] = {}
        request_id = uuid4().hex
        generated_tokens = 0

        started = time.perf_counter()
        first_sampling = dict(sampling_params)
        first_sampling["max_tokens"] = self.max_generated_tokens
        first = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=runtime_ids,
            sampling_params=first_sampling,
            image_data=all_images or None,
            video_data=videos,
        )
        self._add_preemptions(metrics, first.num_preempted)
        self._merge_extra_fields(extra_fields, first.extra_fields or {})

        first_ids = first.token_ids[: self.max_generated_tokens]
        runtime_ids.extend(first_ids)
        response_mask.extend([1] * len(first_ids))
        generated_tokens += len(first_ids)
        if first.log_probs is not None:
            response_logprobs = list(first.log_probs[: len(first_ids)])

        first_text = await self._decode(first_ids)
        triggered = has_reground_marker(first_text) and bool(original_images)
        completed_round2 = False
        capacity_exhausted = False

        if triggered and generated_tokens < self.max_generated_tokens:
            reinjected_images = list(original_images)
            second_user = {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in reinjected_images),
                    {"type": "text", "text": self.reexamination_prompt},
                ],
            }
            environment_ids = await self.apply_chat_template(
                [second_user],
                images=reinjected_images,
                videos=None,
                remove_system_prompt=True,
            )
            remaining_buffer = self.response_length - len(response_mask)
            if len(environment_ids) < remaining_buffer:
                runtime_ids.extend(environment_ids)
                response_mask.extend([0] * len(environment_ids))
                if response_logprobs is not None:
                    response_logprobs.extend([0.0] * len(environment_ids))
                all_images.extend(reinjected_images)

                remaining_generated = self.max_generated_tokens - generated_tokens
                remaining_buffer = self.response_length - len(response_mask)
                second_sampling = dict(sampling_params)
                second_sampling["max_tokens"] = min(remaining_generated, remaining_buffer)
                second = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=runtime_ids,
                    sampling_params=second_sampling,
                    image_data=all_images,
                    video_data=videos,
                )
                self._add_preemptions(metrics, second.num_preempted)
                self._merge_extra_fields(extra_fields, second.extra_fields or {})

                second_ids = second.token_ids[: second_sampling["max_tokens"]]
                runtime_ids.extend(second_ids)
                response_mask.extend([1] * len(second_ids))
                generated_tokens += len(second_ids)
                if response_logprobs is not None:
                    if second.log_probs is None:
                        response_logprobs.extend([0.0] * len(second_ids))
                    else:
                        response_logprobs.extend(second.log_probs[: len(second_ids)])
                completed_round2 = True
            else:
                capacity_exhausted = True

        metrics["generate_sequences"] = time.perf_counter() - started
        response_ids = runtime_ids[len(initial_prompt_ids) :]
        extra_fields.update(
            {
                "reground_triggered": triggered,
                "reground_round2_completed": completed_round2,
                "reground_capacity_exhausted": capacity_exhausted,
                "reground_generated_tokens": generated_tokens,
                "turn_scores": [],
                "tool_rewards": [],
            }
        )

        output_multi_modal = dict(multi_modal_data)
        if all_images:
            output_multi_modal["images"] = all_images

        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                response_logprobs[: self.response_length] if response_logprobs is not None else None
            ),
            multi_modal_data=output_multi_modal,
            num_turns=4 if completed_round2 else 2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
