from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from v3.config import get_paths
from v3.train.datasets.build_mixed_dataset import _ensure_jsonable_columns

SYSTEM_PROMPT = """You are labeling multi-turn tool-use trajectories for when2call training, i.e. labeling training data for next-action strategy of when (not) to call tools and ask users for information.

Your job:
1. Read the full trajectory, tool list, and checklist metadata.
2. Identify concrete decision turns where a when2call-style supervision signal is useful and supported by the observed trajectory.
3. For each critical turn, assign exactly one ground-truth action:
   A = direct_answer
   B = tool_call
   C = request_for_info
   D = cannot_answer
4. Prefer mature judgement, not shallow lexical heuristics.
5. Maintain broad coverage across A/B/C/D. Do not collapse to B merely because this is a tool-use dataset.

Important labeling rules:
- Label A when the request can be correctly answered directly from the conversation context or general knowledge without asking for more info and without tools.
- Label B when the correct next step is to use tools and the required inputs are already available.
- Label C only when the assistant truly lacks information from the user and clarification is the right immediate next action.
- Label D when clarification would still not solve the problem, or the assistant should refuse / cannot answer / lacks capability or access.
- Prefer the provided `candidate_decision_points` when present. They are extracted from the observed assistant next action at each user turn and already contain candidate labels and target answers.
- Do not return an empty annotations list when a valid candidate decision point exists. Return at least one high-confidence annotation.
- Focus on turns that are decision-critical: missing parameters, tool hallucination, invalid tool usage, tool failure requiring clarification, unjustified tool use, direct-answerable turns, or genuine inability.
- If multiple different turn_idx values are reasonable supervision points in the same trajectory, prefer the turn whose correct action best improves A/B/C/D coverage under the balance guidance.
- Avoid returning many same-class annotations from the same trajectory, especially many C turns from one trajectory.
- Do not choose C merely because it is the safest option. If the assistant already has enough information to answer directly, choose A. If the assistant already has enough information to call a tool, choose B. If no realistic clarification would fix the capability or access gap, choose D.
- Use the full trajectory, including previous tool results and prior assistant actions. Many later turns should be A or B rather than C because required information has already been gathered earlier.
- The `target_answer` must be v2-reward-compatible:
  - For `B`, return one or more concrete tool-call XML blocks in the form `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`.
  - For `C`, return the actual clarification question.
  - For `A`, return the direct answer text.
  - For `D`, return the refusal / cannot-answer text.

Balanced examples:
- A example: the user asks for an explanation or a simple factual transformation that can already be answered without tools.
- B example: the user has already provided all required slots and the next correct action is to call a tool.
- C example: a required slot for a decisive answer or tool call is missing from the user.
- D example: the toolset cannot solve the task, or the assistant lacks authority/access and asking follow-up questions would not fix that.

Return strict JSON:
{
  "annotations": [
    {
      "turn_idx": 0,
      "gt_action": "B",
      "gt_action_name": "tool_call",
      "uq_target_type": "medium",
      "reason_type": "ready_to_execute",
      "rationale": "All required parameters are already available, so the assistant should call the tool now.",
      "target_answer": "<tool_call>{\\"name\\": \\"tool_a\\", \\"arguments\\": {\\"account_id\\": \\"123\\"}}</tool_call>",
      "relevant_tools": ["tool_a"],
      "missing_slots": [],
      "timing_tag": "execute_now",
      "recovery_hint": "call_tool_now"
    }
  ]
}
"""

ACTION_MAP = {
    "A": "direct_answer",
    "B": "tool_call",
    "C": "request_for_info",
    "D": "cannot_answer",
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
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
        except Exception:
            converted = None
        if isinstance(converted, list):
            return converted
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_json_column(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
        except Exception:
            converted = None
        if converted is not None:
            return _to_jsonable(converted)
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for column in ("messages", "tools", "reward_model", "extra_info"):
        if column in df.columns:
            df[column] = df[column].apply(_parse_json_column)
    return df


def _has_existing_annotations(row: pd.Series) -> bool:
    extra_info = _ensure_dict(row.get("extra_info"))
    annotations = _ensure_list(extra_info.get("when2call_annotations"))
    return len(annotations) > 0


def _merge_resume_dataframe(input_df: pd.DataFrame, resume_df: pd.DataFrame | None) -> pd.DataFrame:
    if resume_df is None or resume_df.empty:
        return input_df.copy()
    if len(resume_df) != len(input_df):
        raise ValueError(f"Resume output rows ({len(resume_df)}) != input rows ({len(input_df)})")
    out = input_df.copy()
    if "extra_info" not in out.columns:
        out["extra_info"] = [{} for _ in range(len(out))]
    for idx in out.index:
        resumed_extra = _ensure_dict(resume_df.at[idx, "extra_info"]) if "extra_info" in resume_df.columns else {}
        resumed_annotations = _ensure_list(resumed_extra.get("when2call_annotations"))
        if resumed_annotations:
            current_extra = _ensure_dict(out.at[idx, "extra_info"])
            current_extra["when2call_annotations"] = resumed_annotations
            current_extra["use_uq_reward"] = bool(resumed_extra.get("use_uq_reward", True))
            out.at[idx, "extra_info"] = current_extra
    return out


def _write_output(df: pd.DataFrame, output_path: Path) -> None:
    _ensure_jsonable_columns(df).to_parquet(output_path, index=False)


def _resume_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".resume.jsonl") if output_path.suffix else output_path.with_name(output_path.name + ".resume.jsonl")


def _merge_resume_log(input_df: pd.DataFrame, resume_log_path: Path | None) -> pd.DataFrame:
    if resume_log_path is None or not resume_log_path.exists():
        return input_df.copy()
    out = input_df.copy()
    if "extra_info" not in out.columns:
        out["extra_info"] = [{} for _ in range(len(out))]
    with resume_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                idx = int(record.get("idx"))
            except Exception:
                continue
            if idx not in out.index:
                continue
            annotations = _ensure_list(record.get("when2call_annotations"))
            if not annotations:
                continue
            extra_info = _ensure_dict(out.at[idx, "extra_info"])
            extra_info["when2call_annotations"] = annotations
            extra_info["use_uq_reward"] = bool(record.get("use_uq_reward", True))
            out.at[idx, "extra_info"] = extra_info
    return out


def _append_resume_record(resume_log_path: Path | None, idx: int, annotations: list[dict[str, Any]], use_uq_reward: bool) -> None:
    if resume_log_path is None or not annotations:
        return
    resume_log_path.parent.mkdir(parents=True, exist_ok=True)
    with resume_log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "idx": idx,
                    "when2call_annotations": annotations,
                    "use_uq_reward": use_uq_reward,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _extract_full_messages(row: pd.Series) -> list[Any]:
    extra_info = _ensure_dict(row.get("extra_info"))
    full_messages = extra_info.get("messages")
    if full_messages is not None:
        return _ensure_list(full_messages)
    messages = row.get("messages")
    if isinstance(messages, dict) and "messages" in messages:
        return _ensure_list(messages.get("messages"))
    return _ensure_list(messages)


def _extract_tools(row: pd.Series) -> list[Any]:
    extra_info = _ensure_dict(row.get("extra_info"))
    tools = extra_info.get("tools")
    if tools is not None:
        return _ensure_list(tools)
    return _ensure_list(row.get("tools"))


def _extract_checklists(row: pd.Series) -> list[Any]:
    extra_info = _ensure_dict(row.get("extra_info"))
    interaction_kwargs = _ensure_dict(extra_info.get("interaction_kwargs"))
    return _ensure_list(interaction_kwargs.get("checklist_list"))


def _msg_role(msg: Any) -> str:
    return str(msg.get("role", "") or "") if isinstance(msg, dict) else str(getattr(msg, "role", "") or "")


def _msg_content(msg: Any) -> str:
    return str(msg.get("content", "") or "") if isinstance(msg, dict) else str(getattr(msg, "content", "") or "")


def _msg_tool_calls(msg: Any) -> list[Any]:
    if isinstance(msg, dict):
        val = msg.get("tool_calls")
    else:
        val = getattr(msg, "tool_calls", None)
    return _ensure_list(val)


def _turns_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: list[Any] = []
    seen_user = False
    for msg in messages:
        if _msg_role(msg) == "user":
            if current and seen_user:
                turns.append({"turn_idx": len(turns), "messages": current})
                current = []
            seen_user = True
        current.append(msg)
    if current:
        turns.append({"turn_idx": len(turns), "messages": current})
    return turns


def _summarize_turns(messages: list[Any]) -> list[dict[str, Any]]:
    summaries = []
    for turn in _turns_from_messages(messages):
        turn_msgs = turn["messages"]
        user_text = ""
        assistant_texts: list[str] = []
        tool_results: list[str] = []
        tool_calls: list[Any] = []
        for msg in turn_msgs:
            role = _msg_role(msg)
            if role == "user" and not user_text:
                user_text = _msg_content(msg)
            elif role == "assistant":
                content = _msg_content(msg)
                if content:
                    assistant_texts.append(content)
                if _msg_tool_calls(msg):
                    tool_calls.extend(_msg_tool_calls(msg))
            elif role == "tool":
                content = _msg_content(msg)
                if content:
                    tool_results.append(content)
        summaries.append(
            {
                "turn_idx": turn["turn_idx"],
                "user": user_text,
                "assistant_messages": assistant_texts,
                "assistant_tool_calls": tool_calls,
                "tool_messages": tool_results,
            }
        )
    return summaries


def _build_user_prompt(row: pd.Series) -> str:
    return _build_user_prompt_with_balance(row, None)


def _strip_think(text: str) -> str:
    raw = str(text or "").strip()
    while True:
        lowered = raw.lower()
        start = lowered.find("<think>")
        end = lowered.find("</think>")
        if start == -1 or end == -1 or end < start:
            break
        raw = (raw[:start] + raw[end + len("</think>") :]).strip()
    return raw


def _tool_call_to_block(tool_call: Any) -> str:
    call = _ensure_dict(tool_call)
    if not call:
        return ""
    if isinstance(call.get("function"), dict):
        fn = call["function"]
        name = str(fn.get("name") or "").strip()
        arguments = fn.get("arguments", {})
    else:
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            parsed = {}
        arguments = parsed
    if not name or not isinstance(arguments, dict):
        return ""
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, sort_keys=True)}</tool_call>"


def _tool_calls_to_blocks(tool_calls: list[Any]) -> str:
    return "".join(block for block in (_tool_call_to_block(call) for call in tool_calls) if block)


def _classify_non_tool_response(text: str) -> tuple[str, str, str, str]:
    answer = _strip_think(text)
    lowered = answer.lower()
    cannot_markers = (
        "cannot",
        "can't",
        "unable to",
        "not able to",
        "do not have access",
        "don't have access",
        "i'm sorry",
        "sorry,",
        "i cannot",
        "i can't",
    )
    clarify_markers = (
        "could you",
        "can you provide",
        "please provide",
        "please clarify",
        "clarify",
        "which ",
        "what is",
        "what are",
        "need more information",
        "need a bit more",
        "missing",
        "specify",
    )
    if any(marker in lowered for marker in cannot_markers):
        return ("D", "cannot_answer", "trajectory_refusal_or_capability_gap", "cannot_answer")
    if "?" in answer and any(marker in lowered for marker in clarify_markers):
        return ("C", "request_for_info", "observed_clarification_request", "ask_user_for_missing_info")
    return ("A", "direct_answer", "observed_direct_answer", "answer_directly")


def _observed_action_candidates(row: pd.Series) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    tools = _extract_tools(row)
    tool_names = {
        str(_ensure_dict(_ensure_dict(tool).get("function")).get("name") or _ensure_dict(tool).get("name") or "").strip()
        for tool in tools
    }
    tool_names.discard("")
    for turn in _turns_from_messages(_extract_full_messages(row)):
        assistant = next((msg for msg in turn["messages"] if _msg_role(msg) == "assistant"), None)
        if assistant is None:
            continue
        turn_idx = int(turn["turn_idx"])
        tool_calls = _msg_tool_calls(assistant)
        if tool_calls:
            target_answer = _tool_calls_to_blocks(tool_calls)
            if not target_answer:
                continue
            relevant_tools = []
            for call in tool_calls:
                block = _ensure_dict(call)
                fn = _ensure_dict(block.get("function"))
                name = str(fn.get("name") or block.get("name") or "").strip()
                if name:
                    relevant_tools.append(name)
            candidates.append(
                {
                    "turn_idx": turn_idx,
                    "gt_action": "B",
                    "gt_action_name": ACTION_MAP["B"],
                    "uq_target_type": "medium",
                    "reason_type": "observed_ready_to_execute",
                    "rationale": "The observed assistant next action is a concrete tool call with available arguments.",
                    "target_answer": target_answer,
                    "chosen_response": target_answer,
                    "relevant_tools": relevant_tools,
                    "missing_slots": [],
                    "timing_tag": "execute_now",
                    "recovery_hint": "call_tool_now",
                    "use_uq_reward": True,
                }
            )
            continue
        target_answer = _strip_think(_msg_content(assistant))
        if not target_answer:
            continue
        label, action_name, reason_type, recovery_hint = _classify_non_tool_response(target_answer)
        candidates.append(
            {
                "turn_idx": turn_idx,
                "gt_action": label,
                "gt_action_name": action_name,
                "uq_target_type": "medium" if label in {"C", "D"} else "low",
                "reason_type": reason_type,
                "rationale": "The observed assistant next action is a non-tool response at this turn.",
                "target_answer": target_answer,
                "chosen_response": target_answer,
                "relevant_tools": sorted(tool_names) if label == "D" else [],
                "missing_slots": [],
                "timing_tag": "respond_now",
                "recovery_hint": recovery_hint,
                "use_uq_reward": True,
            }
        )
    return candidates


def _format_balance_hint(label_counts: dict[str, int] | None) -> str:
    counts = {label: int((label_counts or {}).get(label, 0)) for label in ACTION_MAP}
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    underrepresented = [label for label, count in counts.items() if count == min_count]
    overrepresented = [label for label, count in counts.items() if count == max_count]
    return (
        "Dataset balance guidance:\n"
        f"- Current accepted annotation counts: A={counts['A']}, B={counts['B']}, C={counts['C']}, D={counts['D']}\n"
        f"- Underrepresented labels right now: {underrepresented}\n"
        f"- Overrepresented labels right now: {overrepresented}\n"
        "- If multiple different turn_idx values in the same trajectory are all reasonable to label, prefer the turn whose correct gt_action is currently underrepresented.\n"
        "- Prefer non-B candidates when they are valid and B is already overrepresented.\n"
        "- Prefer a small set of high-value turns per trajectory rather than many redundant turns.\n"
        "- Do not force a label that is unsupported by the trajectory.\n"
    )


def _build_user_prompt_with_balance(row: pd.Series, label_counts: dict[str, int] | None) -> str:
    messages = _extract_full_messages(row)
    tools = _extract_tools(row)
    checklists = _extract_checklists(row)
    payload = {
        "tools": tools,
        "turn_summaries": _summarize_turns(messages),
        "candidate_decision_points": _observed_action_candidates(row),
        "checklists": checklists,
    }
    return (
        _format_balance_hint(label_counts)
        + "\nTrajectory payload:\n"
        + json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)
    )


def _normalize_annotation(annotation: Any) -> dict[str, Any] | None:
    ann = _ensure_dict(annotation)
    gt_action = str(ann.get("gt_action") or "").strip().upper()
    if gt_action not in ACTION_MAP:
        return None
    try:
        turn_idx = int(ann.get("turn_idx"))
    except Exception:
        return None
    return {
        "turn_idx": turn_idx,
        "gt_action": gt_action,
        "gt_action_name": ACTION_MAP[gt_action],
        "uq_target_type": str(ann.get("uq_target_type") or "ambiguous").strip().lower(),
        "reason_type": str(ann.get("reason_type") or "other").strip(),
        "rationale": str(ann.get("rationale") or "").strip(),
        "target_answer": str(ann.get("target_answer") or "").strip(),
        "chosen_response": str(ann.get("chosen_response") or ann.get("target_answer") or "").strip(),
        "relevant_tools": [str(x) for x in _ensure_list(ann.get("relevant_tools")) if str(x).strip()],
        "missing_slots": [str(x) for x in _ensure_list(ann.get("missing_slots")) if str(x).strip()],
        "timing_tag": str(ann.get("timing_tag") or "").strip(),
        "recovery_hint": str(ann.get("recovery_hint") or "").strip(),
        "use_uq_reward": True,
    }


def _count_annotation_labels(annotations: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for annotation in annotations:
        label = str((annotation or {}).get("gt_action") or "").strip().upper()
        if label in ACTION_MAP:
            counts[label] += 1
    return counts


def _annotation_priority(annotation: dict[str, Any], label_counts: dict[str, int] | Counter[str]) -> tuple[int, int, int, str]:
    label = str(annotation.get("gt_action") or "").strip().upper()
    count = int(label_counts.get(label, 0))
    uq_rank = {"high": 0, "medium": 1, "low": 2, "ambiguous": 3}.get(
        str(annotation.get("uq_target_type") or "").strip().lower(),
        4,
    )
    try:
        turn_idx = int(annotation.get("turn_idx"))
    except Exception:
        turn_idx = 10**9
    return (count, uq_rank, turn_idx, label)


def _select_balanced_annotations(
    annotations: list[dict[str, Any]],
    label_counts: dict[str, int] | Counter[str],
    *,
    max_annotations_per_row: int = 1,
) -> list[dict[str, Any]]:
    if not annotations:
        return []
    unique_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for annotation in annotations:
        try:
            turn_idx = int(annotation.get("turn_idx"))
        except Exception:
            turn_idx = -1
        key = (
            turn_idx,
            str(annotation.get("gt_action") or "").strip().upper(),
            str(annotation.get("target_answer") or annotation.get("chosen_response") or "").strip(),
        )
        if key not in unique_by_key:
            unique_by_key[key] = annotation
    annotations = list(unique_by_key.values())
    if max_annotations_per_row <= 0 or len(annotations) <= max_annotations_per_row:
        return annotations
    available = list(annotations)
    selected: list[dict[str, Any]] = []
    running_counts = Counter(label_counts)
    while available and len(selected) < max_annotations_per_row:
        best = min(available, key=lambda ann: _annotation_priority(ann, running_counts))
        selected.append(best)
        running_counts.update(_count_annotation_labels([best]))
        available.remove(best)
    return selected


def _merge_model_and_observed_annotations(
    model_annotations: list[dict[str, Any]],
    observed_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    # Put observed annotations first so real trajectory actions survive if the model duplicates or drifts.
    for annotation in [*observed_annotations, *model_annotations]:
        item = _normalize_annotation(annotation)
        if item is None:
            continue
        key = (
            int(item["turn_idx"]),
            str(item["gt_action"]),
            str(item.get("target_answer") or item.get("chosen_response") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _collect_existing_label_counts(df: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    if "extra_info" not in df.columns:
        return counts
    for value in df["extra_info"].tolist():
        extra = _ensure_dict(value)
        annotations = [
            item
            for item in (_normalize_annotation(raw) for raw in _ensure_list(extra.get("when2call_annotations")))
            if item is not None
        ]
        counts.update(_count_annotation_labels(annotations))
    return counts


def parse_label_response(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            payload = json.loads(raw[start : end + 1])
        except Exception:
            return []
    annotations = _ensure_list(_ensure_dict(payload).get("annotations"))
    normalized = []
    for annotation in annotations:
        item = _normalize_annotation(annotation)
        if item is not None:
            normalized.append(item)
    normalized.sort(key=lambda x: (x["turn_idx"], x["gt_action"]))
    return normalized


def _call_label_model(
    client: OpenAI,
    *,
    model: str,
    row_prompt: str,
    max_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row_prompt},
        ],
    )
    # print(response.choices[0].message.content) 
    text = response.choices[0].message.content or ""
    return parse_label_response(text)


def augment_dataframe(
    df: pd.DataFrame,
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    workers: int,
    max_tokens: int,
    temperature: float,
    output_path: Path | None = None,
    limit: int = -1,
    save_every: int = 100,
    save_every_seconds: int = 120,
    save_parquet_every: int = 0,
) -> pd.DataFrame:
    api_key = os.getenv(api_key_env, "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)
    out = df.copy()
    label_counts = _collect_existing_label_counts(out)
    max_annotations_per_row = int(os.getenv("LABEL_MAX_ANNOTATIONS_PER_ROW", "2"))
    initial_pending = [idx for idx in out.index if not _has_existing_annotations(out.loc[idx])]
    target_indices = list(initial_pending)
    if limit > 0:
        target_indices = target_indices[:limit]
    skipped = len(out.index) - len(initial_pending)
    if limit > 0:
        skipped += max(0, len(initial_pending) - len(target_indices))

    print(
        f"[resume] total={len(out)} already_labeled={len(out) - len(initial_pending)} "
        f"targeting={len(target_indices)} skipped={skipped}"
    )
    print(
        "[resume] existing annotation label counts: "
        + ", ".join(f"{label}={label_counts.get(label, 0)}" for label in ACTION_MAP)
    )

    completed = 0
    last_save_time = time.time()
    resume_log = _resume_log_path(output_path) if output_path is not None else None

    if output_path is not None and not target_indices:
        _write_output(out, output_path)
        return out

    def _submit(executor: ThreadPoolExecutor, idx: int):
        label_snapshot = {label: int(label_counts.get(label, 0)) for label in ACTION_MAP}
        return executor.submit(
            _call_label_model,
            client,
            model=model,
            row_prompt=_build_user_prompt_with_balance(out.loc[idx], label_snapshot),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_idx: dict[Any, int] = {}
        submit_ptr = 0
        max_in_flight = max(1, workers) * 2
        while submit_ptr < len(target_indices) and len(future_to_idx) < max_in_flight:
            idx = target_indices[submit_ptr]
            future_to_idx[_submit(executor, idx)] = idx
            submit_ptr += 1
        with tqdm(total=len(target_indices), desc="Label CM2 turns", dynamic_ncols=True) as pbar:
            while future_to_idx:
                done, _ = wait(list(future_to_idx.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    idx = future_to_idx.pop(future)
                    extra_info = _ensure_dict(out.at[idx, "extra_info"])
                    try:
                        model_annotations = future.result()
                    except Exception:
                        model_annotations = []
                    observed_annotations = _observed_action_candidates(out.loc[idx])
                    annotations = _merge_model_and_observed_annotations(model_annotations, observed_annotations)
                    annotations = _select_balanced_annotations(
                        annotations,
                        label_counts,
                        max_annotations_per_row=max_annotations_per_row,
                    )
                    if annotations:
                        extra_info["when2call_annotations"] = annotations
                        extra_info["use_uq_reward"] = True
                        label_counts.update(_count_annotation_labels(annotations))
                        _append_resume_record(resume_log, idx, annotations, True)
                    out.at[idx, "extra_info"] = extra_info
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix_str(
                        " ".join(f"{label}={label_counts.get(label, 0)}" for label in ACTION_MAP)
                    )
                    now = time.time()
                    should_save = completed % max(1, save_every) == 0 or (now - last_save_time) >= max(1, save_every_seconds)
                    if output_path is not None and save_parquet_every > 0 and should_save and completed % save_parquet_every == 0:
                        _write_output(out, output_path)
                        last_save_time = now
                        pbar.set_postfix_str(f"saved={completed}")
                    while submit_ptr < len(target_indices) and len(future_to_idx) < max_in_flight:
                        next_idx = target_indices[submit_ptr]
                        future_to_idx[_submit(executor, next_idx)] = next_idx
                        submit_ptr += 1

    if output_path is not None:
        _write_output(out, output_path)
    return out


def main() -> None:
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Label CM2 trajectories with trajectory-level when2call annotations using an LLM.")
    parser.add_argument("--input", required=True, help="Prepared CM2 parquet path.")
    parser.add_argument(
        "--output",
        default=str(paths.cm2_augmented_dir / "cm2_when2call_augmented.parquet"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--save-every-seconds", type=int, default=120)
    parser.add_argument("--save-parquet-every", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    input_df = _load_dataframe(input_path)
    resume_df = _load_dataframe(output_path) if output_path.exists() else None
    df = _merge_resume_log(_merge_resume_dataframe(input_df, resume_df), _resume_log_path(output_path))
    augmented = augment_dataframe(
        df,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        workers=args.workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_path=output_path,
        limit=args.limit,
        save_every=args.save_every,
        save_every_seconds=args.save_every_seconds,
        save_parquet_every=args.save_parquet_every,
    )
    labeled = int(
        sum(
            1
            for value in augmented["extra_info"].tolist()
            if _ensure_dict(value).get("when2call_annotations")
        )
    )
    print(f"Wrote {len(augmented)} rows to {output_path}")
    print(f"Labeled rows with when2call annotations: {labeled}")
    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
