"""VLMEvalKit API adapter for two-round visual reground inference.

Install this file as ``vlmeval/api/qwen_vl_reground_api.py``.  The adapter sends
OpenAI-compatible chat-completion requests and never logs authentication data.
"""

from __future__ import annotations

import atexit
import copy
from datetime import datetime
import json
import os
import re
import threading
from typing import Any

import requests
from PIL import Image

from .base import BaseAPI
from ..smp import encode_image_to_base64


FAIL_MESSAGE = "Failed to obtain answer via API."


class TokenUsageLogger:
    """Thread-safe request and output logger without credentials or images."""

    def __init__(self, model_name: str, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        self.token_file = os.path.join(
            log_dir, f"token_stats_{safe_name}_{timestamp}.jsonl"
        )
        self.output_file = os.path.join(
            log_dir, f"full_outputs_{safe_name}_{timestamp}.jsonl"
        )
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.round1_count = 0
        self.round2_count = 0

    def log_request(self, stage: str, question: str, usage: dict[str, int],
                    variant: str | None = None) -> None:
        record: dict[str, Any] = {
            "stage": stage,
            "question": question[:200],
            **usage,
        }
        if variant is not None:
            record["variant"] = variant
        with self._lock:
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]
            if stage == "round1":
                self.round1_count += 1
            elif stage == "round2":
                self.round2_count += 1
            with open(self.token_file, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_output(self, record: dict[str, Any]) -> None:
        with self._lock:
            with open(self.output_file, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def print_summary(self) -> None:
        total_requests = self.round1_count + self.round2_count
        total_tokens = self.prompt_tokens + self.completion_tokens
        print("\n" + "=" * 60)
        print("REGROUND TOKEN USAGE")
        print(f"API requests: {total_requests}")
        print(f"Round 1: {self.round1_count}; Round 2: {self.round2_count}")
        print(f"Prompt tokens: {self.prompt_tokens:,}")
        print(f"Completion tokens: {self.completion_tokens:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Token log: {self.token_file}")
        print(f"Output log: {self.output_file}")
        print("=" * 60)


def _usage(response: requests.Response) -> dict[str, int]:
    try:
        usage = response.json().get("usage", {})
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class QwenVLRegroundAPI(BaseAPI):
    """Two-round Qwen-VL client with byte-identical visual reuse."""

    is_api = True
    INTERLEAVE = True
    INSTALL_REQ = False

    def __init__(
        self,
        model: str = "qwen2_5-vl-reground",
        base_url: str | None = None,
        api_base: str | None = None,
        key: str | None = None,
        retry: int = 5,
        wait: int = 3,
        timeout: int = 600,
        verbose: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.01,
        top_p: float = 0.001,
        max_tokens: int = 2048,
        max_new_tokens: int | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        img_size: int = -1,
        img_detail: str = "high",
        custom_prompt: bool = False,
        reground_trigger_regex: str = r"<\s*reground\s*>",
        extract_answer_tag: bool = True,
        inject_train_instruction: bool = False,
        clean_hint: bool = True,
        clean_choices_format: bool = False,
        output_log_dir: str = "./token_logs",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.fail_msg = FAIL_MESSAGE
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_new_tokens or max_tokens
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.timeout = timeout
        self.img_size = img_size
        self.img_detail = img_detail
        self._custom_prompt = custom_prompt
        self.inject_train_instruction = inject_train_instruction
        self.clean_hint = clean_hint
        self.clean_choices_format = clean_choices_format
        self.train_instruction = (
            "\nProvide your final answer as a specific value "
            "(number, expression, or ratio)."
        )
        self.reground_prompt = (
            "Based on your self-reflection, re-examine the image and provide "
            "the final answer. Do not output <reground> again."
        )

        endpoint = base_url or api_base or os.getenv(
            "REGROUND_BASE_URL", "http://localhost:8011/v1"
        )
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            self.api_url = endpoint
        elif endpoint.endswith("/v1"):
            self.api_url = endpoint + "/chat/completions"
        else:
            self.api_url = endpoint + "/v1/chat/completions"

        # Local vLLM normally accepts EMPTY. A real key must only come from
        # the process environment and is never written to logs.
        self.key = key or os.getenv("REGROUND_API_KEY") or "EMPTY"
        self.reground_pattern = re.compile(reground_trigger_regex, re.IGNORECASE)
        self.answer_pattern = re.compile(
            r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>",
            re.DOTALL | re.IGNORECASE,
        )
        self.extract_answer_tag = extract_answer_tag

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )
        self._token_logger = TokenUsageLogger(model, output_log_dir)
        atexit.register(self._token_logger.print_summary)
        self.logger.info("QwenVLRegroundAPI initialized")
        self.logger.info("Model: %s", self.model)
        self.logger.info("API endpoint: %s", self.api_url)

    def _extract_final_answer(self, text: str) -> str:
        if not text:
            return ""
        if self.extract_answer_tag:
            match = self.answer_pattern.search(text)
            if match:
                return match.group(1).strip()
        after_think = re.split(r"<\s*/\s*think\s*>", text, flags=re.IGNORECASE)
        if len(after_think) > 1 and after_think[-1].strip():
            return after_think[-1].strip()
        clean = re.sub(
            r"<\s*think\s*>.*?(?:<\s*/\s*think\s*>|$)",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean = re.sub(
            r"<\s*reground\s*>.*?(?:<\s*/\s*reground\s*>|$)",
            "",
            clean,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if clean:
            return clean
        options = re.findall(r"\b([A-E])\b", text[-200:])
        if options:
            return options[-1]
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text[-500:])
        return numbers[-1] if numbers else ""

    def _encode_image(self, image_path: str) -> dict[str, Any]:
        with Image.open(image_path) as image:
            encoded = encode_image_to_base64(image, target_size=self.img_size)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}",
                "detail": self.img_detail,
            },
        }

    def prepare_itlist(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content = []
        for item in inputs:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            elif item["type"] == "image":
                content.append(self._encode_image(item["value"]))
        return content

    def _clean_prompt(self, text: str) -> str:
        if not self.clean_hint:
            return text
        text = re.sub(r"^Hint:.*?\n", "", text, flags=re.I | re.M)
        text = re.sub(r"^Question:\s*", "", text, flags=re.I | re.M)
        text = re.sub(r"\n?Choices:\s*\n", "\n", text, flags=re.I)
        if self.clean_choices_format:
            text = re.sub(r"^\(([A-Z])\)\s*", r"\1. ", text, flags=re.M)
        return text.strip()

    def prepare_inputs(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        if "role" in inputs[0]:
            for item in inputs:
                messages.append({
                    "role": item["role"],
                    "content": self.prepare_itlist(item["content"]),
                })
            return messages

        content = self.prepare_itlist(inputs)
        for item in reversed(content):
            if item.get("type") == "text":
                item["text"] = self._clean_prompt(item["text"])
                if self.inject_train_instruction:
                    item["text"] += self.train_instruction
                break
        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _prepared_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract the encoded image payloads already sent in Round 1."""
        images = []
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    images.append(copy.deepcopy(item))
        return images

    def _request(self, messages: list[dict[str, Any]], **kwargs: Any) -> requests.Response:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": self.top_p,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }
        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.presence_penalty != 0.0:
            payload["presence_penalty"] = self.presence_penalty
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        return requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

    @staticmethod
    def _parse(response: requests.Response) -> tuple[bool, str, str, str]:
        try:
            if not 200 <= response.status_code < 300:
                return False, "", "", f"HTTP {response.status_code}: {response.text[:500]}"
            body = response.json()
            if "error" in body:
                return False, "", "", f"API error: {body['error']}"
            choice = body["choices"][0]
            return (
                True,
                choice["message"]["content"].strip(),
                choice.get("finish_reason", "stop"),
                "",
            )
        except Exception as error:
            return False, "", "", f"Parse error: {type(error).__name__}: {error}"

    def _round2(
        self,
        round1_messages: list[dict[str, Any]],
        round1_output: str,
        images: list[dict[str, Any]],
        question: str,
        **kwargs: Any,
    ) -> tuple[bool, str, str, requests.Response]:
        messages = copy.deepcopy(round1_messages)
        messages.append({"role": "assistant", "content": round1_output})
        # Reinsert byte-identical Round-1 image data at a new decoder position.
        # vLLM V1 can reuse the encoder tensor because its content hash matches.
        content = copy.deepcopy(images)
        content.append({"type": "text", "text": self.reground_prompt})
        messages.append({"role": "user", "content": content})
        response = self._request(messages, **kwargs)
        ok, output, _, error = self._parse(response)
        self._token_logger.log_request(
            "round2", question, _usage(response), "reused_image_new_position"
        )
        if ok:
            answer = self._extract_final_answer(output)
            if answer:
                return True, answer, output, response
        elif self.verbose:
            self.logger.warning("Reground request failed: %s", error)
        return False, "", output, response

    def generate_inner(self, inputs: list[dict[str, Any]], dataset: str | None = None,
                       **kwargs: Any) -> tuple[int, str, requests.Response]:
        del dataset
        question = "\n".join(
            item.get("value", "")
            for item in inputs
            if isinstance(item, dict) and item.get("type") == "text"
        )
        messages = self.prepare_inputs(inputs)
        images = self._prepared_images(messages)
        response = self._request(messages, **kwargs)
        ok, round1, finish_reason, error = self._parse(response)
        usage = _usage(response)
        self._token_logger.log_request("round1", question, usage)
        record: dict[str, Any] = {
            "question": question[:500],
            "round1_output": round1,
            "round2_output": "",
            "triggered_reground": False,
            "final_answer": "",
            "round1_prompt_tokens": usage["prompt_tokens"],
            "round1_completion_tokens": usage["completion_tokens"],
        }
        if not ok:
            if self.verbose:
                self.logger.error("Round 1 failed: %s", error)
            record["final_answer"] = self.fail_msg
            self._token_logger.log_output(record)
            return -1, self.fail_msg, response

        if finish_reason == "length":
            answer = self._extract_final_answer(round1)
        elif self.reground_pattern.search(round1) and images:
            record["triggered_reground"] = True
            success, answer, round2, round2_response = self._round2(
                messages, round1, images, question, **kwargs
            )
            record["round2_output"] = round2
            if success:
                record["final_answer"] = answer
                self._token_logger.log_output(record)
                return 0, answer, round2_response
            answer = self._extract_final_answer(round1) or self.fail_msg
        else:
            answer = self._extract_final_answer(round1)

        record["final_answer"] = answer
        self._token_logger.log_output(record)
        return 0, answer, response

    def use_custom_prompt(self, dataset: str) -> bool:
        del dataset
        return self._custom_prompt
