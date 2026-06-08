from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from v3.config import get_paths
from v3.uq.answer_format import (
    extract_action,
    extract_answer,
    extract_label,
    extract_think,
    normalize_unified_output,
    parse_tool_call_blocks,
    render_tool_call_blocks,
)

ACTION_TO_LABEL = {
    "direct_answer": "A",
    "tool_call": "B",
    "request_for_info": "C",
    "cannot_answer": "D",
}


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def build_canonical_when2call_output(
    *,
    action: str | None,
    label: str | None,
    answer: Any,
    think: str | None = None,
    source: str = "trajectory_or_when2call",
) -> str:
    normalized_action = str(action or "").strip().lower()
    normalized_label = str(label or "").strip().upper() or ACTION_TO_LABEL.get(normalized_action, "")
    if not normalized_action or not normalized_label:
        return str(answer or "").strip()

    think_inner = str(think or "").strip()
    if "suggested_action" not in think_inner:
        think_inner = f"source={source} suggested_action={normalized_action} <{normalized_label}>"

    if normalized_action == "tool_call":
        if isinstance(answer, str):
            parsed_blocks = parse_tool_call_blocks(answer)
            answer_text = render_tool_call_blocks(parsed_blocks) if parsed_blocks else answer.strip()
        else:
            answer_text = render_tool_call_blocks(answer)
    else:
        if isinstance(answer, str):
            answer_text = answer.strip()
        else:
            answer_text = json.dumps(answer, ensure_ascii=False, sort_keys=True)

    return (
        f"<think>{think_inner}</think>\n"
        f"{normalized_action}<{normalized_label}>\n"
        f"<answer>{answer_text}</answer>"
    )


@lru_cache(maxsize=1)
def _load_v2_compute_score():
    paths = get_paths()
    v2_root = paths.v2_root
    if str(v2_root) not in sys.path:
        sys.path.insert(0, str(v2_root))
    reward_path = Path(v2_root) / "train" / "reward_when2call.py"
    spec = importlib.util.spec_from_file_location("v2_reward_when2call", reward_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.compute_score


def canonicalize_to_v2_when2call_format(solution_str: str) -> str:
    raw = (solution_str or "").strip()
    if not raw:
        return raw
    action = extract_action(raw)
    label = extract_label(raw) or ACTION_TO_LABEL.get(action or "", "")
    think = extract_think(raw) or ""
    answer = extract_answer(raw)
    if action and label and answer is not None:
        return build_canonical_when2call_output(
            action=action,
            label=label,
            answer=answer,
            think=think,
            source="trajectory_or_when2call",
        )
    return raw


def normalize_when2call_reward_input(solution_str: str) -> str:
    raw = (solution_str or "").strip()
    if not raw:
        return raw
    normalized = normalize_unified_output(raw)
    return canonicalize_to_v2_when2call_format(normalized)


def score_with_v2_when2call_reward(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None,
) -> dict[str, Any]:
    compute_score = _load_v2_compute_score()
    normalized_extra = _ensure_dict(extra_info)
    normalized_solution = normalize_when2call_reward_input(solution_str)
    return compute_score(
        data_source=data_source,
        solution_str=normalized_solution,
        ground_truth=ground_truth,
        extra_info=normalized_extra,
    )
