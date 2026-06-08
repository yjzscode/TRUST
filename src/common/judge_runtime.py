from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from common.config import load_config


DEFAULT_CONFIG = Path(
    "/mnt/shared-storage-user/zhouyijin/workspace/MyProj/UQ/formal_version_open/configs/train/v3_train_cm2_aug_balanced_no_neg.yaml"
)
DEFAULT_SERVE_JSON = Path(
    "/mnt/shared-storage-user/zhouyijin/workspace/MyProj/UQ/formal_version_open/src/v2/logs/serve.json"
)

# Central mapping for local judge model names used in YAML configs.
# Add new aliases here when a config introduces a new LLM_AS_A_JUDGE_NAME.
MODEL_PATH_BY_SERVED_NAME: dict[str, str] = {
    "Qwen/Qwen3-4B-Thinking": "/mnt/shared-storage-user/ai4good1-share/zhouyijin/models/Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Thinking-2507": "/mnt/shared-storage-user/ai4good1-share/zhouyijin/models/Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-235B-A22B-Instruct-2507-hf",
    "Qwen/CM2SFT": "/mnt/shared-storage-user/ai4good1-share/zhouyijin/models/CM2SFT",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
}

DEFAULT_VLLM_GPUS_BY_SERVED_NAME: dict[str, int] = {
    "Qwen/Qwen3-4B-Thinking": 1,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 4,
    "Qwen/CM2SFT": 1,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 1,
}


def _train_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    train = cfg.get("train", {})
    if not isinstance(train, dict):
        raise ValueError(f"'train' section must be a mapping in {config_path}")
    return train


def get_judge_name(config_path: str | Path = DEFAULT_CONFIG) -> str:
    judge_name = str(_train_config(config_path).get("LLM_AS_A_JUDGE_NAME", "")).strip()
    if not judge_name:
        raise ValueError(f"LLM_AS_A_JUDGE_NAME is missing in {config_path}")
    return judge_name


def get_model_path(judge_name: str) -> str:
    override_json = os.environ.get("JUDGE_MODEL_PATH_MAP_JSON", "").strip()
    model_map = dict(MODEL_PATH_BY_SERVED_NAME)
    if override_json:
        loaded = json.loads(override_json)
        if not isinstance(loaded, dict):
            raise ValueError("JUDGE_MODEL_PATH_MAP_JSON must be a JSON object")
        model_map.update({str(k): str(v) for k, v in loaded.items()})

    if os.path.isabs(judge_name):
        return judge_name
    if judge_name in model_map:
        return model_map[judge_name]
    known = ", ".join(sorted(model_map))
    raise KeyError(
        f"No local MODEL path mapping for LLM_AS_A_JUDGE_NAME={judge_name!r}. "
        f"Known names: {known}. Add it to common.judge_runtime.MODEL_PATH_BY_SERVED_NAME "
        "or set JUDGE_MODEL_PATH_MAP_JSON."
    )


def get_vllm_num_gpus(judge_name: str, config_path: str | Path = DEFAULT_CONFIG) -> int:
    env_value = os.environ.get("VLLM_NUM_GPUS") or os.environ.get("JUDGE_VLLM_NUM_GPUS")
    if env_value:
        return int(env_value)
    train = _train_config(config_path)
    for key in ("VLLM_NUM_GPUS", "JUDGE_VLLM_NUM_GPUS"):
        if key in train and train[key] is not None:
            return int(train[key])
    return int(DEFAULT_VLLM_GPUS_BY_SERVED_NAME.get(judge_name, 1))


def _load_serve_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"serve.json top-level value must be an object: {path}")
    return data


def resolve_latest_serve_url(
    judge_name: str,
    serve_json: str | Path = DEFAULT_SERVE_JSON,
) -> str | None:
    path = Path(serve_json)
    if not path.is_file():
        return None

    data = _load_serve_json(path)
    instances = data.get(judge_name)
    if not isinstance(instances, dict) or not instances:
        return None

    def sort_key(item: tuple[str, Any]) -> str:
        value = item[1]
        if isinstance(value, dict):
            return str(value.get("create_time", ""))
        return ""

    for _, item in reversed(sorted(instances.items(), key=sort_key)):
        if not isinstance(item, dict):
            continue
        ip = str(item.get("ip", "")).strip()
        port = str(item.get("port", "")).strip()
        if ip and port:
            return f"http://{ip}:{port}/v1/chat/completions"
    return None


def wait_for_serve_url(
    judge_name: str,
    serve_json: str | Path = DEFAULT_SERVE_JSON,
    timeout: float = 1800.0,
    poll_interval: float = 10.0,
) -> str:
    deadline = time.time() + timeout
    path = Path(serve_json)
    while True:
        url = resolve_latest_serve_url(judge_name, path)
        if url:
            return url
        if time.time() >= deadline:
            raise TimeoutError(
                f"No live serve entry for {judge_name!r} in {path} within {timeout:.0f}s"
            )
        time.sleep(poll_interval)


def _cmd_judge_name(args: argparse.Namespace) -> int:
    print(get_judge_name(args.config))
    return 0


def _cmd_model_path(args: argparse.Namespace) -> int:
    judge_name = args.judge_name or get_judge_name(args.config)
    print(get_model_path(judge_name))
    return 0


def _cmd_vllm_gpus(args: argparse.Namespace) -> int:
    judge_name = args.judge_name or get_judge_name(args.config)
    print(get_vllm_num_gpus(judge_name, args.config))
    return 0


def _cmd_serve_url(args: argparse.Namespace) -> int:
    judge_name = args.judge_name or get_judge_name(args.config)
    if args.wait:
        url = wait_for_serve_url(judge_name, args.serve_json, args.timeout, args.poll_interval)
    else:
        url = resolve_latest_serve_url(judge_name, args.serve_json)
        if not url:
            return 1
    print(url)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge/vLLM runtime helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("judge-name")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.set_defaults(func=_cmd_judge_name)

    p = sub.add_parser("model-path")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--judge-name", default="")
    p.set_defaults(func=_cmd_model_path)

    p = sub.add_parser("vllm-gpus")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--judge-name", default="")
    p.set_defaults(func=_cmd_vllm_gpus)

    p = sub.add_parser("serve-url")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--serve-json", default=DEFAULT_SERVE_JSON)
    p.add_argument("--judge-name", default="")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--poll-interval", type=float, default=10.0)
    p.set_defaults(func=_cmd_serve_url)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
