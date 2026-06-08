from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from v3.config import get_paths
from v3.uq.answer_format import parse_tool_call_blocks, render_tool_call_blocks

ACTION_TO_LABEL = {
    "direct_answer": "A",
    "tool_call": "B",
    "request_for_info": "C",
    "cannot_answer": "D",
}
LABEL_TO_ACTION = {value: key for key, value in ACTION_TO_LABEL.items()}


def ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _normalize_tool_call_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "") or "").strip()
    arguments = item.get("arguments", {})
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            parsed = {}
        arguments = parsed
    if not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def _parse_tool_answer_surface(text: str) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return []

    parsed_blocks = parse_tool_call_blocks(raw)
    if parsed_blocks:
        return parsed_blocks

    try:
        payload = json.loads(raw)
    except Exception:
        return []

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    out: list[dict[str, Any]] = []
    for item in payload:
        normalized = _normalize_tool_call_item(item)
        if normalized:
            out.append(normalized)
    return out


def _normalize_tool_answer_surface(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    tool_calls = _parse_tool_answer_surface(raw)
    if not tool_calls:
        return raw
    return render_tool_call_blocks(tool_calls)


@lru_cache(maxsize=1)
def _load_v2_prepare_module():
    paths = get_paths()
    v2_root = paths.v2_root
    if str(v2_root) not in sys.path:
        sys.path.insert(0, str(v2_root))
    module_path = Path(v2_root) / "train" / "prepare_when2call_pref.py"
    spec = importlib.util.spec_from_file_location("v2_prepare_when2call_pref", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_when2call_rows(src: Path) -> list[dict[str, Any]]:
    prepare_module = _load_v2_prepare_module()
    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            result = prepare_module.build_pref_prompt(item)
            if result is None:
                continue
            system_msg, user_content, gt_letter = result
            chosen = item.get("chosen_response") or {}
            rejected = item.get("rejected_response") or {}
            chosen_content = chosen.get("content", "") if isinstance(chosen, dict) else str(chosen)
            rejected_content = rejected.get("content", "") if isinstance(rejected, dict) else str(rejected)
            if gt_letter == "B":
                chosen_content = _normalize_tool_answer_surface(chosen_content)
                rejected_content = _normalize_tool_answer_surface(rejected_content)
            tools_for_extra = prepare_module.normalize_tools_for_extra_info(item.get("tools"))
            extra_info = {
                "idx": idx,
                "chosen_response": chosen_content,
                "rejected_response": rejected_content,
                "tools": tools_for_extra,
                "gt_letter": gt_letter,
                "rollout_key": f"when2call:{idx}",
            }
            rows.append(
                {
                    "data_source": "when2call_pref",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content},
                    ],
                    "prompt": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content},
                    ],
                    "tools": tools_for_extra,
                    "ability": "when2call",
                    "reward_model": {"style": "mixed", "ground_truth": gt_letter},
                    "extra_info": extra_info,
                }
            )
    return rows


def _row_tools(row: pd.Series) -> list[Any]:
    extra_info = ensure_dict(row.get("extra_info"))
    tools = extra_info.get("tools")
    if tools:
        return ensure_list(tools)
    return ensure_list(row.get("tools"))


def enrich_cm2_annotations_with_v2_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    if "extra_info" not in out.columns:
        out["extra_info"] = pd.Series([{} for _ in range(len(out))], dtype=object)
    else:
        out["extra_info"] = out["extra_info"].astype(object)
    if "tools" not in out.columns:
        out["tools"] = pd.Series([None for _ in range(len(out))], dtype=object)
    else:
        out["tools"] = out["tools"].astype(object)

    extra_info_col_idx = out.columns.get_loc("extra_info")
    tools_col_idx = out.columns.get_loc("tools")

    for row_pos in range(len(out)):
        row = out.iloc[row_pos]
        row_tools = _row_tools(row)
        extra_info = ensure_dict(row.get("extra_info"))
        if row_tools:
            out.iat[row_pos, tools_col_idx] = row_tools
            extra_info["tools"] = row_tools
        annotations = ensure_list(extra_info.get("when2call_annotations"))
        updated_annotations: list[dict[str, Any]] = []
        for ann_idx, raw_annotation in enumerate(annotations):
            annotation = ensure_dict(raw_annotation)
            gt_letter = str(annotation.get("gt_action") or annotation.get("label") or "").strip().upper()
            if gt_letter not in LABEL_TO_ACTION:
                updated_annotations.append(annotation)
                continue
            target_answer = str(
                annotation.get("chosen_response")
                or annotation.get("target_answer")
                or annotation.get("reference_answer")
                or ""
            )
            if gt_letter == "B":
                target_answer = _normalize_tool_answer_surface(target_answer)
            parsed_tool_calls = _parse_tool_answer_surface(target_answer)
            gt_tool = str(parsed_tool_calls[0].get("name") or "").strip() if parsed_tool_calls else ""
            merged = dict(annotation)
            merged["gt_action"] = gt_letter
            merged["gt_action_name"] = LABEL_TO_ACTION[gt_letter]
            merged["chosen_response"] = target_answer
            merged.setdefault("target_answer", target_answer)
            merged["called_tool_name"] = gt_tool
            merged["rollout_key"] = f"cm2:{row_pos}:turn:{merged.get('turn_idx', ann_idx)}"
            merged["use_uq_reward"] = bool(merged.get("use_uq_reward", True))
            updated_annotations.append(merged)
        if updated_annotations:
            extra_info["when2call_annotations"] = updated_annotations
            extra_info["use_uq_reward"] = True
        out.iat[row_pos, extra_info_col_idx] = extra_info
    return out
