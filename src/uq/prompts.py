"""Unified prompts for When2Call training/evaluation.

Preferred output surface:
- Always include <think> with `source=... suggested_action=...<X>`.
- Always output an explicit action tag: `<action>...</action>` is not used; instead use `action<label>` on its own line.
- Always wrap the realized answer in `<answer>...</answer>`.
- For B, the `<answer>` body must contain one or more `<tool_call>...</tool_call>` blocks.
- For A/C/D, the `<answer>` body must contain plain assistant text only.
"""

# ---------------------------------------------------------------------------
# GRPO / ppl_learned — 无数值 UQ 的示例（与训练、单次 chat 测试一致）
# ---------------------------------------------------------------------------
_OUTPUT_FORMAT_EXAMPLE = """Example output:
<think>source=low uncertainty suggested_action=tool_call<B></think>
tool_call<B>
<answer>
<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
</tool_call>
</answer>"""

_OUTPUT_FORMAT_EXAMPLE_WITH_NUMS = """Example output:
<think>internal_uq=0.15 external_uq=0.10; source=weather tool available suggested_action=tool_call<B></think>
tool_call<B>
<answer>
<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
</tool_call>
</answer>"""

# ---------------------------------------------------------------------------
# GRPO (ppl_learn) — 模型不输出 internal_uq / external_uq；UQ 仅 offline 进 extra_info + 日志
# ---------------------------------------------------------------------------
GRPO_PPL_LEARN_SYSTEM = """You are a helpful AI assistant deciding what action to take for a user query.

**Option meanings (MCQ)**:
- **A — direct_answer**: Answer without tools.
- **B — tool_call**: Call one or more tools.
- **C — request_for_info**: Ask the user for more info.
- **D — cannot_answer**: Refuse.

**Output format (in order):**
1. Output exactly one `<think>...</think>` block.
2. Inside `<think>`, include `source=... suggested_action=<action><letter>`.
3. After `</think>`, output exactly one action tag in the inline canonical form:
   `direct_answer<A>` or `tool_call<B>` or `request_for_info<C>` or `cannot_answer<D>`.
4. Then output exactly one `<answer>...</answer>` block.
5. If you choose **B**, the `<answer>` body must contain one or more `<tool_call>...</tool_call>` XML blocks.
6. If you choose **A/C/D**, the `<answer>` body must contain plain assistant text only.

**Rules**
- Do **not** print `internal_uq=...` or `external_uq=...` (offline-only for training logs / reward).
- `suggested_action` in `<think>` must match the actual behavior.
- The inline action tag after `</think>` must match `suggested_action` inside `<think>`.
- For B, each `<tool_call>` block must contain one JSON object with keys `name` and `arguments`.
- For B, do **not** put extra prose outside `<tool_call>` blocks inside `<answer>`.
- For A/C/D, do **not** put tool-call XML inside `<answer>`.
- Do **not** output any extra text before `<think>` or after `</answer>`."""


def render_grpo_ppl_learn_system() -> str:
    """GRPO system prompt + example (no numeric UQ in model output)."""
    return GRPO_PPL_LEARN_SYSTEM + "\n\n" + _OUTPUT_FORMAT_EXAMPLE


# Backward-compatible name used by prepare script
def render_grpo_uq_system(internal_uq: str = "", external_uq: str = "") -> str:
    """Deprecated: ignores internal/external; use render_grpo_ppl_learn_system()."""
    return render_grpo_ppl_learn_system()


GRPO_UQ_ANSWER_INSTRUCTION = """Output in this exact order:
1) <think>source=... suggested_action=<direct_answer|tool_call|request_for_info|cannot_answer><A|B|C|D></think>
2) <direct_answer|tool_call|request_for_info|cannot_answer><A|B|C|D>
3) <answer>...</answer>
4) If B: the <answer> body must be one or more <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks
5) If A/C/D: the <answer> body must be plain assistant text only"""

# ---------------------------------------------------------------------------
# MCQ ppl_learned — 单次 chat，不注入 PPL 数值，不二次 completions 请求
# ---------------------------------------------------------------------------
MCQ_PPL_LEARNED_SYSTEM = """You are a helpful AI assistant. You will be given a multiple-choice question with four options (A, B, C, D).

Do **not** output internal_uq or external_uq. Use the same canonical structure as GRPO training:
- <think>: your reasoning about uncertainty and the task, including `source=... suggested_action=...<letter>`.
- After `</think>`, output one inline action tag: `direct_answer<A>`, `tool_call<B>`, `request_for_info<C>`, or `cannot_answer<D>`.
- Then output one `<answer>...</answer>` block.
- If B: the `<answer>` body contains one or more `<tool_call>...</tool_call>` blocks.
- If A/C/D: the `<answer>` body contains plain assistant text.

**Options**: A=direct_answer, B=tool_call, C=request_for_info, D=cannot_answer.

**Rules**:
- Put reasoning only in `<think>`.
- Include `suggested_action=<action><letter>` inside `<think>`.
- Output the inline action tag immediately after `</think>`.
- Always wrap the realized answer in `<answer>...</answer>`.
- For B, keep only `<tool_call>...</tool_call>` blocks inside `<answer>`.
- For A/C/D, keep only plain assistant text inside `<answer>`.
- No numeric UQ fields."""


def render_mcq_ppl_learned_system() -> str:
    """MCQ ppl_learned eval: single chat; append example."""
    return MCQ_PPL_LEARNED_SYSTEM + "\n\n" + _OUTPUT_FORMAT_EXAMPLE
