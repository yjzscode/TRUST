from __future__ import annotations

import json
import re
from typing import Any, Optional

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ACTION_XML_RE = re.compile(
    r"<action>\s*(direct_answer|tool_call|request_for_info|cannot_answer)\s*</action>",
    re.DOTALL | re.IGNORECASE,
)
_LABEL_XML_RE = re.compile(r"<label>\s*([ABCD])\s*</label>", re.DOTALL | re.IGNORECASE)
_ANSWER_XML_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_LEGACY_INLINE_RE = re.compile(
    r"(direct_answer|tool_call|request_for_info|cannot_answer)\s*<([ABCD])>",
    re.IGNORECASE,
)
_SUGGESTED_ACTION_RE = re.compile(
    r"suggested_action\s*=\s*(direct_answer|tool_call|request_for_info|cannot_answer)\s*<([ABCD])>",
    re.IGNORECASE,
)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)

ACTION_TO_LABEL = {
    "direct_answer": "A",
    "tool_call": "B",
    "request_for_info": "C",
    "cannot_answer": "D",
}
LABEL_TO_ACTION = {v: k for k, v in ACTION_TO_LABEL.items()}


def _strip_wrappers(text: str) -> str:
    body = (text or "").strip()
    body = _THINK_RE.sub("", body).strip()
    body = _ACTION_XML_RE.sub("", body).strip()
    body = _LABEL_XML_RE.sub("", body).strip()
    body = _LEGACY_INLINE_RE.sub("", body).strip()
    return body


def _parse_json_maybe(payload: str) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return None


def _normalize_tool_call_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if "function" in item and isinstance(item["function"], dict):
        fn = item["function"]
        args = fn.get("arguments", {})
        if isinstance(args, str):
            parsed = _parse_json_maybe(args)
            if isinstance(parsed, dict):
                args = parsed
        if not isinstance(args, dict):
            args = {}
        name = str(fn.get("name", "") or "").strip()
        if not name:
            return None
        return {"name": name, "arguments": args}
    name = str(item.get("name", "") or "").strip()
    if not name:
        return None
    args = item.get("arguments", {})
    if isinstance(args, str):
        parsed = _parse_json_maybe(args)
        if isinstance(parsed, dict):
            args = parsed
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def parse_tool_call_blocks(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text or ""):
        parsed = _parse_json_maybe(match.group(1).strip())
        if isinstance(parsed, list):
            for item in parsed:
                normalized = _normalize_tool_call_item(item)
                if normalized:
                    out.append(normalized)
        else:
            normalized = _normalize_tool_call_item(parsed)
            if normalized:
                out.append(normalized)
    return out


def render_tool_call_blocks(tool_calls: Any) -> str:
    items: list[dict[str, Any]] = []
    if isinstance(tool_calls, list):
        candidates = tool_calls
    else:
        candidates = [tool_calls]
    for item in candidates:
        normalized = _normalize_tool_call_item(item)
        if normalized:
            items.append(normalized)
    return "\n".join(
        f"<tool_call>\n{json.dumps(item, ensure_ascii=False, sort_keys=True)}\n</tool_call>"
        for item in items
    ).strip()


def extract_think(text: str) -> Optional[str]:
    m = _THINK_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_action(text: str) -> Optional[str]:
    raw = text or ""
    m = _ACTION_XML_RE.search(raw)
    if m:
        return m.group(1).lower()
    m = _LEGACY_INLINE_RE.search(raw)
    if m:
        return m.group(1).lower()
    think = extract_think(raw) or ""
    m = _SUGGESTED_ACTION_RE.search(think)
    if m:
        return m.group(1).lower()
    if parse_tool_call_blocks(raw):
        return "tool_call"
    return None


def extract_label(text: str) -> Optional[str]:
    raw = text or ""
    m = _LABEL_XML_RE.search(raw)
    if m:
        return m.group(1).upper()
    m = _LEGACY_INLINE_RE.search(raw)
    if m:
        return m.group(2).upper()
    think = extract_think(raw) or ""
    m = _SUGGESTED_ACTION_RE.search(think)
    if m:
        return m.group(2).upper()
    if parse_tool_call_blocks(raw):
        return "B"
    action = extract_action(raw)
    return ACTION_TO_LABEL.get(action or "")


def extract_answer(text: str) -> Optional[str]:
    raw = text or ""
    tool_blocks = parse_tool_call_blocks(raw)
    if tool_blocks:
        return render_tool_call_blocks(tool_blocks)
    m = _ANSWER_XML_RE.search(raw)
    if m:
        return m.group(1).strip()
    stripped = _strip_wrappers(raw)
    return stripped or None


def normalize_unified_output(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw
    action = extract_action(raw) or ""
    label = extract_label(raw) or ACTION_TO_LABEL.get(action, "")
    think = extract_think(raw) or ""
    answer = extract_answer(raw)
    if answer is None:
        answer = _strip_wrappers(raw)
    parts = []
    if think:
        parts.append(f"<think>{think}</think>")
    if action:
        parts.append(f"<action>{action}</action>")
    if label:
        parts.append(f"<label>{label}</label>")
    if answer:
        parts.append(answer.strip())
    return "\n".join(parts).strip()
