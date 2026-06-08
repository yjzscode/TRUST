from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from common.paths import get_formal_paths

_FORMAL = get_formal_paths()
_V2_ROOT = _FORMAL.src / "v2"


@dataclass(frozen=True)
class V2Config:
    # vLLM / OpenAI-compatible endpoint
    base_url: str = os.getenv("VLLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
    api_key: str = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
    model: str = os.getenv("VLLM_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    # SSL verify (set VERIFY_SSL=0 to disable for LibreSSL / proxy issues)
    verify_ssl: bool = os.getenv("VERIFY_SSL", "1").lower() not in ("0", "false", "no")

    # Generation (max_tokens: ReAct 单步需 Thought+Action+Input，512 易截断，默认 2048)
    temperature: float = float(os.getenv("V2_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("V2_MAX_TOKENS", "4096"))
    top_p: float = float(os.getenv("V2_TOP_P", "1.0"))

    # Agent
    max_steps: int = int(os.getenv("V2_MAX_STEPS", "8"))

    # When2Call UQ prompt test data (default: v2/data/when2call; run_uq_prompt_test 使用)
    when2call_data_dir: str = os.getenv(
        "WHEN2CALL_DATA_DIR_V2",
        str(_FORMAL.root / "data" / "when2call"),
    )

    # Output (default: v2/results, writable)
    results_dir: str = os.getenv(
        "V2_RESULTS_DIR",
        str(_FORMAL.outputs / "results"),
    )

    # ToolSandbox real benchmark: use apple/ToolSandbox CLI when True (requires pip install tool_sandbox)
    toolsandbox_use_real: bool = False
    toolsandbox_output_dir: str = ""  # when real: -o dir; empty = use results_dir


def get_config() -> V2Config:
    return V2Config()


if __name__ == "__main__":
    cfg = get_config()
    print(cfg)
    assert cfg.base_url.startswith("http"), "base_url should be http(s) URL"
    assert cfg.model, "model is required"
