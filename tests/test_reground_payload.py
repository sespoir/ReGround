"""Offline contract tests for the second-round image payload."""

import json
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import requests


def _load_adapter() -> type:
    """Load the repository adapter even when VLMEvalKit is not installed."""
    try:
        from vlmeval.api.qwen_vl_reground_api import QwenVLRegroundAPI

        return QwenVLRegroundAPI
    except ModuleNotFoundError as error:
        if error.name != "vlmeval":
            raise

    vlmeval = types.ModuleType("vlmeval")
    vlmeval.__path__ = []
    api = types.ModuleType("vlmeval.api")
    api.__path__ = []
    base = types.ModuleType("vlmeval.api.base")
    smp = types.ModuleType("vlmeval.smp")

    class BaseAPI:
        def __init__(self, system_prompt=None, verbose=False, **kwargs) -> None:
            del kwargs
            self.system_prompt = system_prompt
            self.verbose = verbose
            self.logger = logging.getLogger("reground-test")

    def encode_image_to_base64(*args, **kwargs) -> str:
        del args, kwargs
        return ""

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


def _response(content: str, finish_reason: str = "stop") -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
    ).encode()
    return response


class RegroundPayloadTest(unittest.TestCase):
    def test_direct_answer_uses_one_request(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            adapter = QwenVLRegroundAPI(output_log_dir=log_dir, verbose=False)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,aW1hZ2UtYnl0ZXM=",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "question"},
                    ],
                }
            ]

            with (
                patch.object(adapter, "prepare_inputs", return_value=messages),
                patch.object(
                    adapter,
                    "_request",
                    return_value=_response("<answer>Yes</answer>"),
                ) as request,
            ):
                status, answer, _ = adapter.generate_inner(
                    [{"type": "text", "value": "question"}]
                )

            self.assertEqual(status, 0)
            self.assertEqual(answer, "Yes")
            self.assertEqual(request.call_count, 1)

    def test_round2_reuses_byte_identical_image_content(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            adapter = QwenVLRegroundAPI(output_log_dir=log_dir, verbose=False)
            image = {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,aW1hZ2UtYnl0ZXM=",
                    "detail": "high",
                },
            }
            round1_messages = [
                {
                    "role": "user",
                    "content": [image, {"type": "text", "text": "question"}],
                }
            ]
            prepared = adapter._prepared_images(round1_messages)

            response = _response("<answer>Yes</answer>")

            with patch.object(adapter, "_request", return_value=response) as request:
                success, answer, _, _ = adapter._round2(
                    round1_messages,
                    "<reground>",
                    prepared,
                    "question",
                )

            sent_messages = request.call_args.args[0]
            round2_image = sent_messages[-1]["content"][0]
            self.assertTrue(success)
            self.assertEqual(answer, "Yes")
            self.assertEqual(round2_image, image)
            self.assertEqual(round1_messages[0]["content"][0], image)

    def test_reground_marker_triggers_a_second_request(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            adapter = QwenVLRegroundAPI(output_log_dir=log_dir, verbose=False)
            image = {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,aW1hZ2UtYnl0ZXM=",
                    "detail": "high",
                },
            }
            messages = [
                {
                    "role": "user",
                    "content": [image, {"type": "text", "text": "question"}],
                }
            ]

            with (
                patch.object(adapter, "prepare_inputs", return_value=messages),
                patch.object(
                    adapter,
                    "_request",
                    side_effect=[
                        _response("< REGROUND >"),
                        _response("<answer>No</answer>"),
                    ],
                ) as request,
            ):
                status, answer, _ = adapter.generate_inner(
                    [{"type": "text", "value": "question"}]
                )

            self.assertEqual(status, 0)
            self.assertEqual(answer, "No")
            self.assertEqual(request.call_count, 2)
            round2_messages = request.call_args_list[1].args[0]
            self.assertEqual(round2_messages[-1]["content"][0], image)

    def test_length_limited_response_is_not_regrounded(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            adapter = QwenVLRegroundAPI(output_log_dir=log_dir, verbose=False)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,aW1hZ2UtYnl0ZXM=",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "question"},
                    ],
                }
            ]

            truncated = _response(
                "<think>unfinished</think><reground><answer>Yes</answer>",
                finish_reason="length",
            )
            with (
                patch.object(adapter, "prepare_inputs", return_value=messages),
                patch.object(adapter, "_request", return_value=truncated) as request,
            ):
                status, answer, _ = adapter.generate_inner(
                    [{"type": "text", "value": "question"}]
                )

            self.assertEqual(status, 0)
            self.assertEqual(answer, "Yes")
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
