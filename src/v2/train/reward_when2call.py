"""When2Call GRPO reward aligned with the TRUST paper.

Unified output:
  <think>...</think> <direct_answer|tool_call|request_for_info|cannot_answer><A|B|C|D> <answer>...</answer>

Final formula:
  raw = r_fmt + r_ans + cert * r_cls
  r_ans = r_align + lambda * r_ppl_action
  cert = sigmoid((ppl_neg - ppl_gt) / tau)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

_V2_ROOT = Path(__file__).resolve().parents[1]
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

from uq.answer_format import (
    extract_answer_inner,
    extract_mcq_letter_from_text,
    extract_suggested_action,
    extract_source,
    extract_suggested_action_category,
    extract_answer_tail_category,
    extract_uq_inner,
)

_CONFIDENT_LETTERS = frozenset({"A", "B"})
_LETTER_TO_ACTION = {
    "A": "direct_answer",
    "B": "tool_call",
    "C": "request_for_info",
    "D": "cannot_answer",
}
DEFAULT_TAU = 0.10
DEFAULT_LAMBDA = 1.0
_DROP_ENV = "REWARD_DROP_COMPONENTS"


# ---------------------------------------------------------------------------
# r_fmt: 格式合规 + 自一致性  ∈ [0, 2.15]
# ---------------------------------------------------------------------------
def _format_reward(text: str, final_letter: str | None) -> tuple[float, dict]:
    details: dict = {}
    score = 0.0

    has_uq = extract_uq_inner(text) is not None
    has_answer = extract_answer_inner(text) is not None
    if has_uq:
        score += 0.5
    if has_answer:
        score += 0.5
    details["has_uq_block"] = has_uq
    details["has_answer_block"] = has_answer

    has_letter_tag = final_letter is not None
    if has_letter_tag:
        score += 0.3
    details["has_letter_tag"] = has_letter_tag

    source = extract_source(text)
    # source 为自由文本（对 UQ 与任务的简述）；非空且不太短即给分
    valid_source = bool(source and len(source.strip()) >= 4)
    if valid_source:
        score += 0.3
    details["source"] = (source or "")[:200]
    details["valid_source"] = valid_source

    sug_cat = extract_suggested_action_category(text)
    ans_cat = extract_answer_tail_category(text)
    category_match = (
        sug_cat is not None and ans_cat is not None and sug_cat == ans_cat
    )
    if category_match:
        score += 0.15
    details["suggested_category"] = sug_cat or ""
    details["answer_category"] = ans_cat or ""
    details["category_match"] = category_match

    suggested = extract_suggested_action(text)
    self_consistent = (
        suggested is not None
        and final_letter is not None
        and suggested == final_letter
    )
    if self_consistent:
        score += 0.4
    details["suggested_action"] = suggested or ""
    details["self_consistent"] = self_consistent

    return score, details


# ---------------------------------------------------------------------------
# r_cls: 正确性  ∈ [0, 3.0]
# ---------------------------------------------------------------------------
def _correctness_reward(final_letter: str | None, gt: str) -> tuple[float, str]:
    if not final_letter or not gt:
        return 0.0, "missing"

    if final_letter == gt:
        return 2.0, "exact"

    final_confident = final_letter in _CONFIDENT_LETTERS
    gt_confident = gt in _CONFIDENT_LETTERS
    if final_confident == gt_confident:
        return 1.0, "direction"

    return 0.0, "wrong"


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _safe_float(v: object) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _get_hyper(extra_info: dict, key: str, default: float) -> float:
    v = _safe_float(extra_info.get(key))
    if v is None or not math.isfinite(v):
        return default
    return float(v)


def _drop_components() -> set[str]:
    raw = os.getenv(_DROP_ENV, "").strip().lower()
    if not raw:
        return set()
    out = set()
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.add(p)
    return out


def _calc_margin_terms(extra_info: dict) -> tuple[float, float, float]:
    """Return (ppl_gt, ppl_neg, margin). Missing values fallback to neutral margin=0."""
    ppl_gt = _safe_float(extra_info.get("ppl_gt"))
    ppl_neg = _safe_float(extra_info.get("ppl_neg"))
    if ppl_gt is None or ppl_neg is None or not math.isfinite(ppl_gt) or not math.isfinite(ppl_neg):
        return float("nan"), float("nan"), 0.0
    return float(ppl_gt), float(ppl_neg), float(ppl_neg - ppl_gt)


def _tokenize_for_f1(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    return toks


def _token_f1(a: str, b: str) -> float:
    ta, tb = _tokenize_for_f1(a), _tokenize_for_f1(b)
    if not ta or not tb:
        return 0.0
    ca = {}
    cb = {}
    for t in ta:
        ca[t] = ca.get(t, 0) + 1
    for t in tb:
        cb[t] = cb.get(t, 0) + 1
    inter = 0
    for k, va in ca.items():
        inter += min(va, cb.get(k, 0))
    if inter <= 0:
        return 0.0
    p = inter / len(ta)
    r = inter / len(tb)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _extract_tool_name_and_args(text: str) -> tuple[str, dict]:
    s = (text or "").strip()
    if "<toolcall>" in s.lower() or "<tool_call>" in s.lower():
        m = re.search(r"<(?:toolcall|tool_call)>\s*(.*?)\s*</(?:toolcall|tool_call)>", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except Exception:
        obj = None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if isinstance(obj, dict):
        name = str(obj.get("name", "")).strip()
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        return name, args
    return "", {}


def _is_question(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    if "?" in s:
        return True
    cues = ("could you", "can you", "would you", "please provide", "please specify", "what ", "which ", "how ")
    return any(c in s for c in cues)


def _is_refusal(text: str) -> bool:
    s = (text or "").strip().lower()
    cues = ("unable", "cannot", "can't", "sorry", "apologies", "i'm unable", "i do not have")
    return any(c in s for c in cues)


def _action_from_letter(letter: str | None) -> str:
    if not letter:
        return ""
    return _LETTER_TO_ACTION.get(letter.upper(), "")


def _r_align(pred_action: str, pred_answer: str, gold_answer: str, tools: list[str] | None) -> float:
    """Single-score content alignment in [0,1]."""
    pa = (pred_answer or "").strip()
    ga = (gold_answer or "").strip()
    if not pa or not ga:
        return 0.0

    if pred_action == "tool_call":
        p_name, p_args = _extract_tool_name_and_args(pa)
        g_name, g_args = _extract_tool_name_and_args(ga)
        if not p_name:
            return 0.0
        if tools:
            tool_names = set()
            for t in tools:
                try:
                    if isinstance(t, str):
                        d = json.loads(t)
                    else:
                        d = t
                    if isinstance(d, dict):
                        tool_names.add(str(d.get("name", "")).strip())
                except Exception:
                    continue
            if tool_names and p_name not in tool_names:
                return 0.0
        name_score = 1.0 if p_name == g_name and p_name else 0.0
        if not p_args and not g_args:
            args_score = 1.0
        else:
            args_score = _token_f1(json.dumps(p_args, ensure_ascii=False), json.dumps(g_args, ensure_ascii=False))
        return max(0.0, min(1.0, 0.5 * name_score + 0.5 * args_score))

    if pred_action == "request_for_info":
        if not _is_question(pa):
            return 0.0
        return max(0.0, min(1.0, _token_f1(pa, ga)))

    if pred_action == "cannot_answer":
        if _is_question(pa) or not _is_refusal(pa):
            return 0.0
        return max(0.0, min(1.0, _token_f1(pa, ga)))

    if pred_action == "direct_answer":
        return max(0.0, min(1.0, _token_f1(pa, ga)))

    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict:
    """Minimal reward with answer alignment + margin-driven cert."""
    extra = extra_info if isinstance(extra_info, dict) else {}
    raw_num_turns = extra.get("num_turns")
    try:
        num_turns = int(raw_num_turns) if raw_num_turns is not None else 1
    except Exception:
        num_turns = 1
    if num_turns <= 0:
        num_turns = 1

    if not solution_str or not isinstance(solution_str, str):
        return {
            "score": 0.0,
            "num_turns": num_turns,
            "max_num_turns": num_turns,
            "tool_call_success": True,
        }

    gt = (ground_truth or "").strip().upper()
    final_letter = extract_mcq_letter_from_text(solution_str)
    pred_action = _action_from_letter(final_letter)

    r_fmt, fmt_details = _format_reward(solution_str, final_letter)
    r_cls, cls_type = _correctness_reward(final_letter, gt)
    # r_align: chosen_response alignment + minimal anti-hacking gates
    pred_answer = extract_answer_inner(solution_str) or ""
    chosen_answer = str(extra.get("chosen_response", "") or "")
    tools = extra.get("tools")
    if not isinstance(tools, list):
        tools = None
    r_align = _r_align(pred_action, pred_answer, chosen_answer, tools)

    # r_ppl_action and cert from margin
    tau = _get_hyper(extra, "tau", DEFAULT_TAU)
    lam = _get_hyper(extra, "lambda", DEFAULT_LAMBDA)
    ppl_gt, ppl_neg, margin = _calc_margin_terms(extra)
    cert = _sigmoid(margin / max(1e-6, tau))
    r_ppl_action = cert

    # Minimal anti-hacking gate: invalid structure disables answer-level rewards.
    if not pred_answer.strip() or not final_letter:
        r_align = 0.0
        r_ppl_action = 0.0

    r_ans = max(0.0, min(2.0, r_align + lam * r_ppl_action))

    drops = _drop_components()
    if "r_fmt" in drops:
        r_fmt = 0.0
    if "r_ans" in drops:
        r_ans = 0.0
    if "r_cls" in drops:
        r_cls = 0.0

    raw = r_fmt + r_ans + cert * r_cls
    max_score = 2.15 + (1.0 + lam) + 2.0
    score = raw / max(1e-6, max_score)
    ppl_gt_ok = math.isfinite(ppl_gt)
    ppl_neg_ok = math.isfinite(ppl_neg)

    return {
        "score": round(float(score), 4),
        "num_turns": num_turns,
        "max_num_turns": num_turns,
        "tool_call_success": True,
        "raw_score": round(float(raw), 4),
        "max_score": round(float(max_score), 4),
        "r_fmt": round(float(r_fmt), 4),
        "r_ans": round(float(r_ans), 4),
        "r_align": round(float(r_align), 4),
        "r_ppl_action": round(float(r_ppl_action), 4),
        "r_cls": round(float(r_cls), 4),
        "cert": round(float(cert), 4),
        "tau": round(float(tau), 6),
        "lambda": round(float(lam), 6),
        "ppl_gt": round(float(ppl_gt), 6) if ppl_gt_ok else 0.0,
        "ppl_gt_ok": ppl_gt_ok,
        "ppl_neg": round(float(ppl_neg), 6) if ppl_neg_ok else 0.0,
        "ppl_neg_ok": ppl_neg_ok,
        "margin": round(float(margin), 6),
        "neg_type": str(extra.get("neg_type") or ""),
        "drop_components": ",".join(sorted(drops)),
        "cls_type": cls_type,
        "letter": final_letter or "",
        "pred_action": pred_action,
        "suggested_action": fmt_details.get("suggested_action", ""),
        "self_consistent": fmt_details.get("self_consistent", False),
        "category_match": fmt_details.get("category_match", False),
        "suggested_category": fmt_details.get("suggested_category", ""),
        "answer_category": fmt_details.get("answer_category", ""),
        "source": fmt_details.get("source", ""),
        "uq_value": round(float(_safe_float(extra.get("runtime_uq_value")) or 0.0), 4),
        "uq_source": "runtime_ppl" if extra.get("runtime_uq_value") is not None else "default",
    }
