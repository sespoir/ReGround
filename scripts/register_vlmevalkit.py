#!/usr/bin/env python3
"""Register the reground adapter in a validated VLMEvalKit checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_ANCHOR = (
    "from .rbdashmm_chat3_5_api import "
    "RBdashMMChat3_78B_API, RBdashMMChat3_5_38B_API\n"
)
IMPORT_LINE = "from .qwen_vl_reground_api import QwenVLRegroundAPI\n"

EXPORT_ANCHOR = (
    "    'VideoChatOnlineV2API', 'TeleMM2_API', 'TeleMM2Thinking_API'\n"
)
EXPORT_LINE = (
    "    'VideoChatOnlineV2API', 'TeleMM2_API', 'TeleMM2Thinking_API', "
    "'QwenVLRegroundAPI'\n"
)

CONFIG_ANCHOR = "}\n\nimport copy as cp\n"
CONFIG_BLOCK = """}

api_models['qwen-vl-reground'] = partial(
    QwenVLRegroundAPI,
    model=os.getenv('REGROUND_MODEL_NAME', 'qwen2_5-vl-reground'),
    base_url=os.getenv('REGROUND_BASE_URL', 'http://localhost:8011/v1'),
    key=os.getenv('REGROUND_API_KEY', 'EMPTY'),
    temperature=float(os.getenv('REGROUND_TEMPERATURE', '0.01')),
    top_p=float(os.getenv('REGROUND_TOP_P', '0.001')),
    max_tokens=int(os.getenv('REGROUND_MAX_TOKENS', '2048')),
    presence_penalty=0.0,
    repetition_penalty=1.0,
    verbose=os.getenv('REGROUND_VERBOSE', '1') == '1',
    reground_trigger_regex=r'<\\s*reground\\s*>',
    extract_answer_tag=True,
    inject_train_instruction=False,
    output_log_dir=os.getenv('REGROUND_LOG_DIR', './token_logs'),
)

import copy as cp
"""


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1), True


def register(root: Path, check_only: bool) -> bool:
    api_init = root / "vlmeval" / "api" / "__init__.py"
    config = root / "vlmeval" / "config.py"
    if not api_init.is_file() or not config.is_file():
        raise RuntimeError(f"Not a VLMEvalKit checkout: {root}")

    api_text = api_init.read_text(encoding="utf-8")
    config_text = config.read_text(encoding="utf-8")
    api_text, import_changed = replace_once(
        api_text, IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, "API import"
    )
    api_text, export_changed = replace_once(
        api_text, EXPORT_ANCHOR, EXPORT_LINE, "API export"
    )
    config_text, config_changed = replace_once(
        config_text, CONFIG_ANCHOR, CONFIG_BLOCK, "model config"
    )

    changed = import_changed or export_changed or config_changed
    states = {import_changed, export_changed, config_changed}
    if len(states) != 1:
        raise RuntimeError("Partial reground registration detected; refusing to continue")
    if changed and not check_only:
        api_init.write_text(api_text, encoding="utf-8")
        config.write_text(config_text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vlmeval_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = register(args.vlmeval_root.resolve(), args.check)
    state = "ready to register" if changed else "already registered"
    print(f"VLMEvalKit reground adapter: {state}")


if __name__ == "__main__":
    main()
