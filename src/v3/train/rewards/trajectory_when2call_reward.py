from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any

import torch

from v3.train.judges import get_model_judge
from v3.train.rewards.v2_when2call_adapter import build_canonical_when2call_output, score_with_v2_when2call_reward
from v3.train.rewards.v2_when2call_runtime import V2When2CallRuntime, get_v2_when2call_runtime
from v3.uq.answer_format import extract_action, extract_answer, extract_label, extract_think, render_tool_call_blocks

LABEL_TO_ACTION = {
    "A": "direct_answer",
    "B": "tool_call",
    "C": "request_for_info",
    "D": "cannot_answer",
}


TURN_PARSE_SYSTEM_PROMPT = """You are a structured judge for CM2 trajectory turn parsing.

Return strict json. The word json appears here intentionally.

Given one completed assistant turn inside a multi-turn tool-use trajectory:
1. Infer the assistant's realized next-action decision.
2. Map it to exactly one label:
   A = direct_answer
   B = tool_call
   C = request_for_info
   D = cannot_answer
3. Return the realized answer content in v2-compatible form:
   - For B: tool-call JSON content only, without extra prose.
   - For A/C/D: the actual assistant text.

Output schema:
{
  "pred_label": "C",
  "pred_action": "request_for_info",
  "pred_answer": "Could you provide the account id?",
  "think": "source=judge suggested_action=request_for_info <C>",
  "confidence": 0.81,
  "rationale": "One short sentence."
}
"""


def _msg_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", "") or "")
    return str(getattr(msg, "role", "") or "")


def _msg_content(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", "") or "")
    return str(getattr(msg, "content", "") or "")


def _msg_tool_calls(msg: Any) -> list[Any]:
    if isinstance(msg, dict):
        val = msg.get("tool_calls")
    else:
        val = getattr(msg, "tool_calls", None)
    return val if isinstance(val, list) else []


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _normalize_message(msg: Any) -> dict[str, Any]:
    return {
        "role": _msg_role(msg),
        "content": _json_safe(_msg_content(msg)),
        "tool_calls": _json_safe(_msg_tool_calls(msg)),
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


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _extract_message_list(container: Any) -> list[Any]:
    if isinstance(container, dict) and "messages" in container:
        return _ensure_list(container.get("messages"))
    if isinstance(container, list):
        return container
    return _ensure_list(container)


def _group_messages_by_user_turn(messages: list[Any]) -> list[list[Any]]:
    turns: list[list[Any]] = []
    current: list[Any] = []
    seen_user = False
    for msg in messages:
        role = _msg_role(msg)
        if role == "user":
            if current and seen_user:
                turns.append(current)
                current = []
            seen_user = True
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def _prompt_messages_for_turn(messages: list[Any], turn_idx: int) -> list[Any]:
    turns = _group_messages_by_user_turn(messages)
    if turn_idx < 0 or turn_idx >= len(turns):
        return []
    prompt_messages: list[Any] = []
    for idx in range(turn_idx):
        prompt_messages.extend(turns[idx])
    current_turn = turns[turn_idx]
    for msg in current_turn:
        prompt_messages.append(msg)
        if _msg_role(msg) == "user":
            break
    return prompt_messages


def _future_messages_for_turn(messages: list[Any], turn_idx: int) -> list[Any]:
    turns = _group_messages_by_user_turn(messages)
    if turn_idx < 0 or turn_idx + 1 >= len(turns):
        return []
    future: list[Any] = []
    for idx in range(turn_idx + 1, len(turns)):
        future.extend(turns[idx])
    return future


def _tool_call_to_json(tool_calls: list[Any]) -> str:
    normalized = []
    for tool_call in tool_calls:
        tool_call = _json_safe(tool_call)
        if not isinstance(tool_call, dict):
            continue
        if "function" in tool_call and isinstance(tool_call["function"], dict):
            fn = tool_call["function"]
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass
            normalized.append(
                {
                    "name": fn.get("name", ""),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )
        elif "name" in tool_call:
            normalized.append(
                {
                    "name": tool_call.get("name", ""),
                    "arguments": tool_call.get("arguments", {}),
                }
            )
    if not normalized:
        return ""
    payload: Any = normalized[0] if len(normalized) == 1 else normalized
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _tool_call_to_blocks(tool_calls: list[Any]) -> str:
    normalized = []
    for tool_call in tool_calls:
        tool_call = _json_safe(tool_call)
        if not isinstance(tool_call, dict):
            continue
        if "function" in tool_call and isinstance(tool_call["function"], dict):
            fn = tool_call["function"]
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            name = str(fn.get("name", "") or "").strip()
            if not name:
                continue
            normalized.append({"name": name, "arguments": arguments})
        elif "name" in tool_call:
            name = str(tool_call.get("name", "") or "").strip()
            arguments = tool_call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if not name or not isinstance(arguments, dict):
                continue
            normalized.append({"name": name, "arguments": arguments})
    return render_tool_call_blocks(normalized)


def _structured_turn_prediction(turn_messages: list[Any]) -> dict[str, str]:
    assistant_messages = [msg for msg in turn_messages if _msg_role(msg) == "assistant"]
    if not assistant_messages:
        return {"pred_label": "", "pred_action": "", "pred_answer": "", "think": ""}
    for msg in assistant_messages:
        tool_calls = _msg_tool_calls(msg)
        if tool_calls:
            return {
                "pred_label": "B",
                "pred_action": "tool_call",
                "pred_answer": _tool_call_to_blocks(tool_calls),
                "think": "source=structured suggested_action=tool_call <B>",
            }
    final_text = _msg_content(assistant_messages[-1]).strip()
    action = extract_action(final_text) or ""
    label = extract_label(final_text) or ""
    answer = extract_answer(final_text)
    think = extract_think(final_text) or ""
    if action and label and answer is not None:
        return {
            "pred_label": label,
            "pred_action": action,
            "pred_answer": answer,
            "think": think,
        }
    return {"pred_label": "", "pred_action": "", "pred_answer": "", "think": ""}


def parse_turn_prediction_with_judge(turn_messages: list[Any]) -> dict[str, str]:
    structured = _structured_turn_prediction(turn_messages)
    if structured.get("pred_label") and structured.get("pred_action"):
        return structured
    judge = get_model_judge()
    if not judge.enabled:
        return structured
    assistant_messages = [msg for msg in turn_messages if _msg_role(msg) == "assistant"]
    if not assistant_messages:
        return structured
    payload = {
        "turn_messages": [_normalize_message(msg) for msg in turn_messages],
        "assistant_messages": [
            {
                "content": _msg_content(msg),
                "tool_calls": _json_safe(_msg_tool_calls(msg)),
            }
            for msg in assistant_messages
        ],
    }
    judged = judge.generate_json(system_prompt=TURN_PARSE_SYSTEM_PROMPT, user_payload=payload)
    pred_label = str(judged.get("pred_label") or "").strip().upper()
    pred_action = str(judged.get("pred_action") or "").strip()
    pred_answer = str(judged.get("pred_answer") or "").strip()
    think = str(judged.get("think") or "").strip()
    if pred_label in LABEL_TO_ACTION and pred_action and pred_answer:
        return {
            "pred_label": pred_label,
            "pred_action": pred_action,
            "pred_answer": pred_answer,
            "think": think,
        }
    return structured


def build_v2_style_turn_output(turn_messages: list[Any]) -> str:
    prediction = parse_turn_prediction_with_judge(turn_messages)
    action = prediction["pred_action"]
    label = prediction["pred_label"]
    answer = prediction["pred_answer"]
    if not action or not label:
        assistant_messages = [msg for msg in turn_messages if _msg_role(msg) == "assistant"]
        return _msg_content(assistant_messages[-1]).strip() if assistant_messages else ""
    return build_canonical_when2call_output(
        action=action,
        label=label,
        answer=answer,
        think=prediction["think"],
        source="trajectory_or_when2call",
    )


def compute_trajectory_when2call_reward(
    messages_container: Any,
    when2call_annotations: Any,
    turn_end_mask: torch.Tensor,
    *,
    runtime: V2When2CallRuntime | None = None,
    default_use_uq_reward: bool = False,
    reward_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime = runtime or get_v2_when2call_runtime()
    messages = _extract_message_list(messages_container)
    annotations = _ensure_list(when2call_annotations)
    sparse_reward = torch.zeros_like(turn_end_mask, dtype=torch.float32)
    diagnostics: dict[str, Any] = {
        "num_annotations": len(annotations),
        "used_annotations": 0,
        "trajectory_when2call_raw_sum": 0.0,
        "trajectory_when2call_turns": [],
        "trajectory_when2call_scores": [],
        "trajectory_when2call_pred_labels": [],
        "trajectory_when2call_gt_labels": [],
    }
    if not messages or not annotations or turn_end_mask.numel() == 0:
        return sparse_reward, diagnostics

    turns = _group_messages_by_user_turn(messages)
    end_positions = torch.nonzero(turn_end_mask, as_tuple=False).squeeze(-1).tolist()
    scores_by_turn: dict[int, list[float]] = defaultdict(list)
    pred_by_turn: dict[int, str] = {}
    gt_by_turn: dict[int, str] = {}
    ann_items: list[dict[str, Any]] = []

    for ann in annotations:
        ann_dict = _ensure_dict(ann)
        try:
            turn_idx = int(ann_dict.get("turn_idx"))
        except Exception:
            continue
        if turn_idx < 0 or turn_idx >= len(turns) or turn_idx >= len(end_positions):
            continue

        gt_label = str(ann_dict.get("gt_action") or ann_dict.get("label") or "").strip().upper()
        if gt_label not in LABEL_TO_ACTION:
            continue

        ann_item = {
            "ann_dict": ann_dict,
            "turn_idx": turn_idx,
            "gt_label": gt_label,
            "rollout_key": str(ann_dict.get("rollout_key") or f"turn:{turn_idx}:{gt_label}"),
        }
        ann_items.append(ann_item)

    for ann_item in ann_items:
        ann_dict = ann_item["ann_dict"]
        turn_idx = ann_item["turn_idx"]
        gt_label = ann_item["gt_label"]
        rollout_key = ann_item["rollout_key"]
        target_answer = str(
            ann_dict.get("target_answer")
            or ann_dict.get("reference_answer")
            or ann_dict.get("chosen_response")
            or ""
        )
        use_uq_reward = bool(ann_dict.get("use_uq_reward", default_use_uq_reward))
        solution_str = build_v2_style_turn_output(turns[turn_idx])
        prompt_messages = _prompt_messages_for_turn(messages, turn_idx)
        runtime_extra = runtime.inject_runtime_fields(
            prompt_messages=prompt_messages,
            response_text=solution_str,
            ground_truth=gt_label,
            extra_info={
                "chosen_response": target_answer,
                "target_answer": target_answer,
                "uq_target_type": ann_dict.get("uq_target_type"),
                "use_uq_reward": use_uq_reward,
                "tools": ann_dict.get("tools"),
                "runtime_uq_value": ann_dict.get("runtime_uq_value"),
            },
            rollout_key=rollout_key,
        )
        result = score_with_v2_when2call_reward(
            data_source="cm2_when2call_augmented",
            solution_str=solution_str,
            ground_truth=gt_label,
            extra_info=runtime_extra,
        )
        score = float(result.get("score", 0.0)) * float(reward_weight)
        scores_by_turn[turn_idx].append(score)
        pred_by_turn[turn_idx] = str(result.get("letter", ""))
        gt_by_turn[turn_idx] = gt_label

    for turn_idx, scores in scores_by_turn.items():
        if not scores:
            continue
        end_pos = end_positions[turn_idx]
        score = float(sum(scores) / len(scores))
        sparse_reward[end_pos] += score
        diagnostics["used_annotations"] += len(scores)
        diagnostics["trajectory_when2call_raw_sum"] += score
        diagnostics["trajectory_when2call_turns"].append(turn_idx)
        diagnostics["trajectory_when2call_scores"].append(round(score, 4))
        diagnostics["trajectory_when2call_pred_labels"].append(pred_by_turn.get(turn_idx, ""))
        diagnostics["trajectory_when2call_gt_labels"].append(gt_by_turn.get(turn_idx, ""))

    diagnostics["trajectory_when2call_raw_sum"] = round(diagnostics["trajectory_when2call_raw_sum"], 4)
    return sparse_reward, diagnostics
