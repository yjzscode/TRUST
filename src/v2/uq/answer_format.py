"""Parse unified When2Call output:

<think>...</think> <direct_answer|tool_call|request_for_info|cannot_answer><A|B|C|D> <answer>...</answer>

Backward compatibility:
- Legacy [uq]...[/uq]
- Legacy [answer]...[/answer]
"""
from __future__ import annotations

import re
from typing import Optional

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_XML_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_UQ_RE = re.compile(r"\[uq\](.*?)\[/uq\]", re.DOTALL | re.IGNORECASE)  # legacy
_ANSWER_BRACKET_RE = re.compile(r"\[answer\](.*?)\[/answer\]", re.DOTALL | re.IGNORECASE)  # legacy
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)

# Unified action tag between think and answer
_ACTION_TAG_RE = re.compile(
    r"(direct_answer|tool_call|request_for_info|cannot_answer)\s*<([ABCD])>",
    re.IGNORECASE,
)

# Legacy suggested_action=...
_SUGGESTED_ACTION_FULL_RE = re.compile(
    r"suggested_action\s*=\s*(direct_answer|tool_call|request_for_info|cannot_answer)\s*<([ABCD])>",
    re.IGNORECASE,
)
_SUGGESTED_LETTER_ONLY_RE = re.compile(
    r"suggested_action\s*=\s*<?([ABCD])>?", re.IGNORECASE,
)

# Legacy tail in [answer]
_ANSWER_TAIL_ACTION_RE = re.compile(
    r"(direct_answer|tool_call|request_for_info|cannot_answer)\s*<([ABCD])>\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TAG_LETTER_RE = re.compile(r"<([ABCD])>\s*$", re.IGNORECASE | re.MULTILINE)

_INTERNAL_UQ_RE = re.compile(r"internal_uq\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_EXTERNAL_UQ_RE = re.compile(r"external_uq\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def extract_uq_inner(text: str) -> Optional[str]:
    """For compatibility: treat <think> as uq block; fallback to [uq]."""
    mt = _THINK_RE.search(text or "")
    if mt:
        return mt.group(1).strip()
    m = _UQ_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_answer_inner(text: str) -> Optional[str]:
    """Content inside first <answer>...</answer> (or legacy [answer])."""
    tool_blocks = list(_TOOL_CALL_BLOCK_RE.finditer(text or ""))
    if tool_blocks:
        return "\n".join(match.group(0).strip() for match in tool_blocks)
    m = _ANSWER_XML_RE.search(text or "")
    if m:
        return m.group(1).strip()
    m = _ANSWER_BRACKET_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_internal_uq(text: str) -> Optional[float]:
    """Parse internal_uq from [uq] block."""
    inner = extract_uq_inner(text)
    if not inner:
        return None
    m = _INTERNAL_UQ_RE.search(inner)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def extract_external_uq(text: str) -> Optional[float]:
    """Parse external_uq from [uq] block."""
    inner = extract_uq_inner(text)
    if not inner:
        return None
    m = _EXTERNAL_UQ_RE.search(inner)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def extract_source(text: str) -> Optional[str]:
    """Extract reasoning source, compatible with think/uq legacy."""
    inner = extract_uq_inner(text)
    if not inner:
        return None
    m = re.search(
        r"source\s*=\s*(.+?)\s+suggested_action\s*=",
        inner,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"source\s*=\s*([^\n]+)", inner, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_suggested_action(text: str) -> Optional[str]:
    """Letter A–D from unified action tag, or legacy suggested_action."""
    action_m = _ACTION_TAG_RE.search(text or "")
    if action_m:
        return action_m.group(2).upper()

    inner = extract_uq_inner(text) or ""
    m = _SUGGESTED_ACTION_FULL_RE.search(inner)
    if m:
        return m.group(2).upper()
    m = _SUGGESTED_LETTER_ONLY_RE.search(inner)
    return m.group(1).upper() if m else None


def extract_suggested_action_category(text: str) -> Optional[str]:
    """Category name from unified action tag, or legacy suggested_action."""
    action_m = _ACTION_TAG_RE.search(text or "")
    if action_m:
        return action_m.group(1).lower()

    inner = extract_uq_inner(text) or ""
    m = _SUGGESTED_ACTION_FULL_RE.search(inner)
    if m:
        return m.group(1).lower()
    return None


def extract_mcq_letter_from_text(text: str) -> Optional[str]:
    """Prefer unified action tag; then parse from answer/legacy variants."""
    action_m = _ACTION_TAG_RE.search(text or "")
    if action_m:
        return action_m.group(2).upper()

    inner = extract_answer_inner(text)
    if inner is not None:
        stripped = inner.strip()
        m = _ANSWER_TAIL_ACTION_RE.search(stripped)
        if m:
            return m.group(2).upper()
        m = _TAG_LETTER_RE.search(stripped)
        if m:
            return m.group(1).upper()
        # Prefer last non-empty line for \\b[A-D]\\b to avoid picking a letter from JSON in the middle
        for line in reversed(inner.splitlines()):
            line = line.strip()
            if not line:
                continue
            matches = list(re.finditer(r"\b([ABCD])\b", line, re.IGNORECASE))
            if matches:
                return matches[-1].group(1).upper()
    matches = list(re.finditer(r"\b([ABCD])\b", text or "", re.IGNORECASE))
    return matches[-1].group(1).upper() if matches else None


def extract_answer_tail_category(text: str) -> Optional[str]:
    """Category from action tag, fallback to legacy [answer] tail."""
    action_m = _ACTION_TAG_RE.search(text or "")
    if action_m:
        return action_m.group(1).lower()

    inner = extract_answer_inner(text)
    if not inner:
        return None
    m = _ANSWER_TAIL_ACTION_RE.search(inner.strip())
    return m.group(1).lower() if m else None


def response_text_for_llm_judge(full_text: str) -> str:
    """Return text for judge: prefer <answer>; strip think/uq wrappers and trailing tags."""
    t = full_text or ""
    rest = _THINK_RE.sub("", t, count=1).strip()
    rest = _UQ_RE.sub("", rest, count=1).strip()
    rest = _ACTION_TAG_RE.sub("", rest, count=1).strip()
    inner = extract_answer_inner(t)
    if inner is not None:
        body = inner.strip()
        m = _ANSWER_TAIL_ACTION_RE.search(body)
        if m:
            body = body[: m.start()].rstrip()
        return body.strip() if body else inner.strip()
    return rest

def strip_uq_and_answer_blocks(full_text: str) -> str:
    """Remove think/answer wrappers for callers needing plain tail."""
    t = full_text or ""
    t = _THINK_RE.sub("", t)
    t = _UQ_RE.sub("", t)
    t = _ANSWER_XML_RE.sub("", t)
    t = _ANSWER_BRACKET_RE.sub("", t)
    t = _ACTION_TAG_RE.sub("", t)
    return t.strip()
