"""When2Call GRPO 数据准备脚本（当前 reward 设计配套）。

- 生成 GRPO parquet（prompt + ground_truth + extra_info）
- 保留 chosen/rejected 原文与 tools，供 r_align 与调试使用

注意：
- 为避免 parquet 序列化错误，extra_info.tools 强制写成同构 list[str]。
"""
from __future__ import annotations

import json
import re
import sys
import os
from pathlib import Path

_V2_ROOT = Path(__file__).resolve().parents[1]
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

from uq.prompts import GRPO_UQ_ANSWER_INSTRUCTION, render_grpo_ppl_learn_system

ANSWER_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]
OPTION_LABELS = ["A", "B", "C", "D"]
_CATEGORY_TO_LETTER = {cat: lbl for cat, lbl in zip(ANSWER_ORDER, OPTION_LABELS)}

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. 
You have access to the following tools described in which you can use to answer the user's questions.
Only use a tool if it directly answers the user's question.
"""

TOOL_USE_INSTRUCTIONS = """To use a tool, return JSON in the following format:
{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2", ...}}
"""


def classify_response(content: str) -> str:
    """Classify assistant reply → direct|tool_call|request_for_info|cannot_answer."""
    text = (content or "").strip()
    if not text:
        return "direct"

    if "<toolcall>" in text.lower() or "<tool_call>" in text.lower() or '{"name":' in text or '"name":' in text:
        return "tool_call"

    refusal_phrases = [
        r"\bunable\b", r"\bcannot\b", r"\bcan't\b", r"\bsorry\b",
        r"i'm unable", r"i am unable", r"i don't have", r"i do not have",
        r"apologies", r"i'm sorry", r"i am sorry", r"i apologize",
        r"don't have the capability", r"lack the required", r"out of scope",
    ]
    for pat in refusal_phrases:
        if re.search(pat, text, re.IGNORECASE):
            return "cannot_answer"

    question_phrases = [
        r"\?$",
        r"could you (please )?provide", r"could you (please )?specify",
        r"could you (please )?tell", r"could you (please )?confirm",
        r"would you (please )?", r"can you (please )?provide",
        r"to proceed,? (i )?need", r"to assist you,? (i )?need",
        r"please (provide|specify|tell|confirm)", r"what (is|are) ",
        r"which (one|type|language)", r"how (would|do) you",
    ]
    for pat in question_phrases:
        if re.search(pat, text, re.IGNORECASE):
            return "request_for_info"

    return "direct"


def default_format_tools(tools: list) -> str:
    tool_strings = [f" {t} " if isinstance(t, str) else f" {json.dumps(t)} " for t in tools]
    return "\n\n".join(tool_strings)


def normalize_tools(tools_raw) -> list:
    """Normalize tools field to list for prompt rendering."""
    if tools_raw is None:
        return []
    if isinstance(tools_raw, list):
        return tools_raw
    if isinstance(tools_raw, (dict, str)):
        return [tools_raw]
    return []


def normalize_tools_for_extra_info(tools_raw) -> list[str]:
    """Normalize tools to a homogeneous list[str] for parquet stability."""
    tools = normalize_tools(tools_raw)
    out: list[str] = []
    for t in tools:
        if isinstance(t, str):
            out.append(t)
        else:
            try:
                out.append(json.dumps(t, ensure_ascii=False))
            except Exception:
                out.append(str(t))
    return out


def build_pref_prompt(item: dict) -> tuple[str, str, str] | None:
    """Build prompt. Returns (system_msg, user_content, gt_letter) or None."""
    tools = normalize_tools(item.get("tools"))
    messages = item.get("messages") or []
    chosen = item.get("chosen_response") or {}

    chosen_content = chosen.get("content", "") if isinstance(chosen, dict) else str(chosen)

    user_content_raw = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            user_content_raw = m.get("content", "")
            break
    if not user_content_raw:
        return None

    category = classify_response(chosen_content)
    gt_letter = _CATEGORY_TO_LETTER.get(category, "A")

    system_msg = render_grpo_ppl_learn_system()

    tool_string = default_format_tools(tools)
    user_content = (
        f"{DEFAULT_SYSTEM_PROMPT}\n{TOOL_USE_INSTRUCTIONS}\n\n{tool_string}\n\n"
        f"User question: {user_content_raw}\n\n"
        "Which response should the assistant give? Choose exactly one option (A, B, C, or D):\n"
        "- A (direct_answer): Answer from knowledge without tools.\n"
        "- B (tool_call): Call a tool with available info.\n"
        "- C (request_for_info): Ask the user for more info.\n"
        "- D (cannot_answer): Refuse (out of scope, insufficient tools).\n\n"
        f"{GRPO_UQ_ANSWER_INSTRUCTION}"
    )
    return system_msg, user_content, gt_letter


def main() -> None:
    root = Path(os.environ.get("FORMAL_ROOT", str(Path(__file__).resolve().parents[3])))
    input_root = Path(os.environ.get("WHEN2CALL_OFFICIAL_DATA_DIR", str(root / "data" / "when2call_official")))
    data_dir = input_root / "train"
    out_dir = Path(os.environ.get("WHEN2CALL_GRPO_OUTPUT_DIR", str(input_root / "grpo")))
    out_dir.mkdir(parents=True, exist_ok=True)

    src = data_dir / "when2call_train_pref.jsonl"
    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)

    rows = []
    skipped = 0
    parsed_items: list[tuple[int, dict]] = []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            parsed_items.append((i, item))

    for i, item in parsed_items:
        chosen = item.get("chosen_response") or {}
        rejected = item.get("rejected_response") or {}
        chosen_content = chosen.get("content", "") if isinstance(chosen, dict) else str(chosen)
        rejected_content = rejected.get("content", "") if isinstance(rejected, dict) else str(rejected)
        tools = normalize_tools(item.get("tools"))
        tools_for_extra = normalize_tools_for_extra_info(item.get("tools"))

        result = build_pref_prompt(item)
        if result is None:
            skipped += 1
            continue
        system_msg, user_content, gt_letter = result

        rows.append({
            "data_source": "when2call_pref",
            "prompt": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            "ability": "when2call",
            "reward_model": {"style": "mixed", "ground_truth": gt_letter},
            "extra_info": {
                "idx": i,
                "chosen_response": chosen_content,
                "rejected_response": rejected_content,
                "tools": tools_for_extra,
                "gt_letter": gt_letter,
            },
        })

    out_path = out_dir / "when2call_pref_grpo.parquet"
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path} (skipped {skipped})")
    cats = {}
    for r in rows:
        gl = r["extra_info"]["gt_letter"]
        cats[gl] = cats.get(gl, 0) + 1
    print(f"  GT distribution: {dict(sorted(cats.items()))}")


if __name__ == "__main__":
    main()
