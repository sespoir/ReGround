#!/usr/bin/env python3
"""Run one two-round request through the VLMEvalKit reground adapter."""

import argparse
import base64
from io import BytesIO
import importlib.util
import logging
import os
from pathlib import Path
import sys
import types

from dotenv import load_dotenv


def _load_adapter() -> type:
    """Load the installed adapter, or the repository source for smoke tests."""
    try:
        from vlmeval.api.qwen_vl_reground_api import QwenVLRegroundAPI

        return QwenVLRegroundAPI
    except ModuleNotFoundError:
        # A server-only environment may not contain VLMEvalKit's many optional
        # evaluation dependencies. Load the exact repository adapter with the
        # small BaseAPI surface and image encoder needed by this smoke test.
        vlmeval = types.ModuleType("vlmeval")
        vlmeval.__path__ = []
        api = types.ModuleType("vlmeval.api")
        api.__path__ = []
        base = types.ModuleType("vlmeval.api.base")
        smp = types.ModuleType("vlmeval.smp")

        class BaseAPI:
            def __init__(
                self,
                retry=10,
                wait=1,
                system_prompt=None,
                verbose=True,
                fail_msg="Failed to obtain answer via API.",
                **kwargs,
            ) -> None:
                self.retry = retry
                self.wait = wait
                self.system_prompt = system_prompt
                self.verbose = verbose
                self.fail_msg = fail_msg
                self.default_kwargs = kwargs
                self.logger = logging.getLogger("RegroundSmoke")

        def encode_image_to_base64(image, target_size=-1) -> str:
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")
            if target_size > 0:
                image.thumbnail((target_size, target_size))
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        base.BaseAPI = BaseAPI
        smp.encode_image_to_base64 = encode_image_to_base64
        sys.modules.update({
            "vlmeval": vlmeval,
            "vlmeval.api": api,
            "vlmeval.api.base": base,
            "vlmeval.smp": smp,
        })

        module_name = "vlmeval.api.qwen_vl_reground_api"
        source = Path(__file__).resolve().parents[1] / "src" / "qwen_vl_reground_api.py"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load adapter from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.QwenVLRegroundAPI


QwenVLRegroundAPI = _load_adapter()


def _response_usage(response) -> dict[str, int]:
    usage = response.json().get("usage", {})
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("REGROUND_BASE_URL", "http://localhost:8011/v1"),
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--question",
        default=(
            "Inspect the image. Is this a natural photograph? "
            "For this cache smoke test, first output <reground>, then wait "
            "for the follow-up before giving Yes or No."
        ),
        help="Round-1 question; use a hard in-distribution example for natural triggers",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("REGROUND_MODEL_NAME", "qwen2_5-vl-reground"),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("REGROUND_MAX_TOKENS", "2048")),
        help="maximum completion tokens for each round",
    )
    parser.add_argument(
        "--force-reground",
        action="store_true",
        help=(
            "synthesize <reground> when Round 1 does not emit it; disabled by "
            "default so the smoke test verifies the model's real trigger"
        ),
    )
    args = parser.parse_args()

    adapter = QwenVLRegroundAPI(
        model=args.model,
        base_url=args.base_url,
        temperature=0.01,
        top_p=0.001,
        max_tokens=args.max_tokens,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        reground_trigger_regex=r"<\s*reground\s*>",
        extract_answer_tag=True,
        inject_train_instruction=False,
        output_log_dir="/tmp/reground-smoke-token-logs",
        verbose=True,
    )

    message = [
        {"type": "image", "value": args.image},
        {
            "type": "text",
            "value": args.question,
        },
    ]
    # Exercise the same internals as generate_inner while preserving the real
    # Round-1 trigger by default. Synthetic triggering is an explicit opt-in.
    input_messages = adapter.prepare_inputs(message)
    prepared_images = adapter._prepared_images(input_messages)

    response = adapter._request(input_messages)
    ok, round1, finish_reason, error = adapter._parse(response)
    if not ok:
        raise RuntimeError(error)
    adapter._token_logger.log_request(
        "round1", args.question, _response_usage(response)
    )
    print(f"ROUND1:\n{round1}\n")

    assistant_context = round1
    if not adapter.reground_pattern.search(assistant_context):
        if not args.force_reground:
            raise RuntimeError(
                "Round 1 did not emit <reground>; no synthetic trigger was added "
                f"(finish_reason={finish_reason})"
            )
        assistant_context += "\n<reground>"
        print("FORCED_REGROUND: model did not naturally trigger on smoke image")
    else:
        print("NATURAL_REGROUND: model emitted the trigger")

    success, answer, round2, _ = adapter._round2(
        input_messages,
        assistant_context,
        prepared_images,
        question=message[1]["value"],
    )
    if not success:
        raise RuntimeError("Round 2 did not return an extractable answer")
    print(f"ROUND2:\n{round2}\n")
    print(f"FINAL: {answer}")


if __name__ == "__main__":
    main()
