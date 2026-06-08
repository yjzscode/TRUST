import asyncio
import json
from collections import defaultdict
from typing import Any
import os
import re
import httpx
import torch
from verl import DataProto

import logging
import threading
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SOFT_FAIL_ON_REWARD_CONNECT = os.getenv("CHECKLIST_SOFT_FAIL_ON_REWARD_CONNECT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
_REWARD_STATS_INTERVAL = max(1, int(os.getenv("CHECKLIST_REWARD_STATS_INTERVAL", "50")))
_REWARD_STATS = {
    "total": 0,
    "success": 0,
    "fail": 0,
    "network_fail": 0,
}
_REWARD_STATS_LOCK = threading.Lock()
_REWARD_PARSE_DEBUG_LIMIT = max(0, int(os.getenv("CHECKLIST_REWARD_PARSE_DEBUG_LIMIT", "5")))
_REWARD_PARSE_DEBUG_MAX_CHARS = max(200, int(os.getenv("CHECKLIST_REWARD_PARSE_DEBUG_MAX_CHARS", "2000")))
_REWARD_PARSE_DEBUG_COUNT = 0
_REWARD_PARSE_DEBUG_LOCK = threading.Lock()


def _record_reward_stats(success: bool, *, network_fail: bool = False) -> None:
    with _REWARD_STATS_LOCK:
        _REWARD_STATS["total"] += 1
        if success:
            _REWARD_STATS["success"] += 1
        else:
            _REWARD_STATS["fail"] += 1
            if network_fail:
                _REWARD_STATS["network_fail"] += 1

        total = _REWARD_STATS["total"]
        if (not success) or (total % _REWARD_STATS_INTERVAL == 0):
            fail_rate = (_REWARD_STATS["fail"] / total) if total > 0 else 0.0
            logger.warning(
                "[reward stats] total=%d success=%d fail=%d network_fail=%d fail_rate=%.4f",
                total,
                _REWARD_STATS["success"],
                _REWARD_STATS["fail"],
                _REWARD_STATS["network_fail"],
                fail_rate,
            )


def _build_openai_compatible_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    api_key = (
        os.getenv("JUDGE_API_KEY")
        or os.getenv("V3_JUDGE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def extract_last_json(text, parse=True):
    FENCE_JSON_RE = re.compile(
        r"```(?:\s*json)?\s*(.*?)\s*```",
        re.IGNORECASE | re.DOTALL
    )
    matches = FENCE_JSON_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    if not parse:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None  # 解析失败就返回原始字符串


def _as_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict):
                return item
    return None


def _iter_json_object_candidates(text: str):
    raw = (text or "").strip()
    if not raw:
        return

    try:
        parsed = json.loads(raw)
        obj = _as_json_object(parsed)
        if obj is not None:
            yield obj
    except Exception:
        pass

    FENCE_JSON_RE = re.compile(
        r"```(?:\s*json)?\s*(.*?)\s*```",
        re.IGNORECASE | re.DOTALL,
    )
    for fenced in reversed(FENCE_JSON_RE.findall(raw)):
        try:
            parsed = json.loads(fenced.strip())
            obj = _as_json_object(parsed)
            if obj is not None:
                yield obj
        except Exception:
            pass

    # Walk from the end so that the final verdict JSON wins over examples or
    # JSON-like snippets inside thinking text.
    starts = [idx for idx, ch in enumerate(raw) if ch == "{"]
    for start in reversed(starts):
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(raw)):
            ch = raw[end]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start : end + 1])
                        obj = _as_json_object(parsed)
                        if obj is not None:
                            yield obj
                    except Exception:
                        pass
                    break


def _log_reward_parse_failure(raw_content: Any, error: Exception) -> None:
    global _REWARD_PARSE_DEBUG_COUNT
    if _REWARD_PARSE_DEBUG_LIMIT <= 0:
        return
    with _REWARD_PARSE_DEBUG_LOCK:
        if _REWARD_PARSE_DEBUG_COUNT >= _REWARD_PARSE_DEBUG_LIMIT:
            return
        _REWARD_PARSE_DEBUG_COUNT += 1
        idx = _REWARD_PARSE_DEBUG_COUNT
    text = repr(raw_content)
    if len(text) > _REWARD_PARSE_DEBUG_MAX_CHARS:
        text = text[:_REWARD_PARSE_DEBUG_MAX_CHARS] + "...<truncated>"
    logger.warning(
        "[reward parse debug %d/%d] error=%r raw_content=%s",
        idx,
        _REWARD_PARSE_DEBUG_LIMIT,
        error,
        text,
    )


def _extract_choice_content(data: dict[str, Any]) -> Any:
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}

    message = first.get("message", {})
    if isinstance(message, dict) and "content" in message:
        content = message.get("content")
        if isinstance(content, list):
            # Handle OpenAI-style content parts: [{"type":"text","text":"..."}]
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(parts).strip() if parts else ""
        return content

    return first.get("text", "")


def _extract_bool_answer(raw_content: Any) -> bool:
    parsed: Any = None
    if isinstance(raw_content, dict):
        parsed = raw_content
    elif isinstance(raw_content, str):
        text = raw_content.strip()
        if not text:
            raise ValueError("empty reward response content")
        for candidate in _iter_json_object_candidates(text):
            parsed = candidate
            break
    if not isinstance(parsed, dict):
        raise ValueError("reward response is not a JSON object")

    ans = parsed.get("answer", parsed.get("result", None))
    if isinstance(ans, bool):
        return ans
    if isinstance(ans, str):
        lowered = ans.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    raise ValueError("answer/result is not boolean")


async def eval_one_check(client: httpx.AsyncClient, user_prompt: str, args: dict) -> bool:

    sglang_model = args.get("sglang_model")
    sglang_url = args.get("sglang_url")
    temperature = args.get("temperature")
    top_p = args.get("top_p")
    max_new_tokens = args.get("max_new_tokens")
    max_tokens = args.get("max_tokens")
    retry_times = args.get("retry_times")

    payload = {
        "model": sglang_model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "max_tokens": max_tokens,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        # "json_schema": {
        #     "type": "object",
        #     "properties": {
        #         "result": {"type": "boolean"}
        #     },
        #     "required": ["result"]
        # },
        # "response_format": {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "foo",
        #         "schema":{
        #             "type": "object",
        #             "properties": {
        #                 "result": {"type": "boolean"}
        #             },
        #             "required": ["result"]
        #         }
        #     }
        # }
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "title": "Checklist Evaluation Verdict",
                    "description": "Structured evaluation result for a single assistant turn against checklist criteria.",
                    "properties": {
                        "high_level_understanding_of_the_question": {
                            "type": "string",
                        },
                        "analysis_of_if_focus_on": {
                            "type": "string",
                        },
                        "analysis_of_pass_condition": {
                            "type": "string",
                        },
                        "analysis_of_failure_examples": {
                            "type": "string",
                        },
                        "answer": {
                            "type": "boolean",
                        }
                    },
                    "required": [
                        "high_level_understanding_of_the_question",
                        "analysis_of_if_focus_on",
                        "analysis_of_pass_condition",
                        "analysis_of_failure_examples",
                        "answer"
                    ],
                    "additionalProperties": False
                }
            }
        }
    }

    try:
        resp = await _post_with_retries(client, sglang_url, payload, retry_times)
        try:
            data = resp.json()
            text = _extract_choice_content(data)
            ans = _extract_bool_answer(text)
            _record_reward_stats(True)
            return ans, True
        except Exception as e:
            logger.warning(f"text can not be parsed in reward (call passed): {repr(e)}")
            try:
                _log_reward_parse_failure(text, e)  # type: ignore[name-defined]
            except Exception:
                pass
            _record_reward_stats(False, network_fail=False)
            return False, False
    except Exception as e:
        if _SOFT_FAIL_ON_REWARD_CONNECT and isinstance(
            e,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
            ),
        ):
            logger.warning(f"text can not be parsed in reward (soft-fail network): {repr(e)}")
            # Keep training/rollout alive when judge endpoint is flaky: assign False but mark call as passed.
            _record_reward_stats(False, network_fail=True)
            return False, True
        logger.warning(f"text can not be parsed in reward (call not passed): {repr(e)}")
        _record_reward_stats(False, network_fail=True)
        return False, False

def get_input_prompt(messages_str_before_this_step: str, this_step_message_str: str, following_tool_response_str: str, this_turn_checklist: list[dict[str, Any]]) -> str:
    reference_snippet = [evidence['snippet'] for evidence in this_turn_checklist['evidence']]
    input_prompt = (
        "# Instructions\n"
        "You are a strict checklist evaluator.\n"
        "You will be give a checklist for the assistant's new response.\n"
        "Checklist contains question, focus_on, pass_condition, failure_examples and reference snippet.\n"
        "Focus on is the part of the assistant's new response that the question is about.\n"
        "If the assistant's response follows the checklist's question, return true. Otherwise, return false.\n"
        "If the focus on is not in the assistant's new response or following tool response, return false.\n"
        "\n"
        "# Checklist:\n"
        f"Question: {this_turn_checklist['question']}\n"
        f"Focus on: {this_turn_checklist['focus_on']}\n"
        f"Pass condition: {this_turn_checklist['pass_condition']}\n"
        f"Failure examples: {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=False, indent=0)}\n"
        f"Reference snippet: {json.dumps(reference_snippet, ensure_ascii=False, indent=0)}\n"
        "\n"
        "# Previous messages:\n" + messages_str_before_this_step + "\n"
        "\n"
        "# Assistant's new response:\n" + this_step_message_str + "\n"
        "\n"
        "# Following tool response:\n" + following_tool_response_str + "\n"
        "# Response format:\n"
        "Return in JSON format: {'result': true/false}"
    )
    return input_prompt

def get_input_prompt_v2(messages_str_before_this_turn, messages_str_in_this_turn: str, this_turn_checklist: list[dict[str, Any]]) -> str:
    reference_snippet = [evidence['snippet'] for evidence in this_turn_checklist['evidence']]
    # input_prompt = (
    #     "# Instructions\n"
    #     "You are a strict checklist evaluator.\n"
    #     "You will be give messages between user, assistant and tools. And you will also be given a checklist for the assistant's response.\n"
    #     "Checklist contains question, focus_on, pass_condition, failure_examples and reference snippet.\n"
    #     "Focus on is the part of the assistant's response that the question is about.\n"
    #     "If the assistant's response follows the checklist's question and pass condition, return true. Otherwise, return false.\n"
    #     "If the focus on is not in the messages, return false.\n"
    #     "\n"
    #     "# Checklist:\n"
    #     f"Question: {this_turn_checklist['question']}\n"
    #     f"Focus on: {this_turn_checklist['focus_on']}\n"
    #     f"Pass condition: {this_turn_checklist['pass_condition']}\n"
    #     f"Failure examples: {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=False, indent=0)}\n"
    #     f"Reference snippet: {json.dumps(reference_snippet, ensure_ascii=False, indent=0)}\n"
    #     "\n"
    #     "# Messages:\n" + messages_str_in_this_turn + "\n"
    #     "# Response format:\n"
    #     "Return only in JSON format: {'result': true/false}"
    # )
    input_prompt = (
        "# Role\n"
        "You are a precise checklist evaluator. Your sole task is to judge whether the messages between user, assistant and tool satisfie the provided criteria.\n"
        "\n"
        "# Objective\n"
        "Produce a strict JSON verdict (no extra text) based on the instructions below.\n"
        "\n"
        "# Criteria\n"
        f"**Question:** {this_turn_checklist['question']}\n"
        f"**Focus on:** {this_turn_checklist['focus_on']}\n"
        f"**Pass condition:** {this_turn_checklist['pass_condition']}\n"
        f"**Failure examples:** {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=True, indent=2)}\n"
        f"**Reference snippet:** {json.dumps(reference_snippet, ensure_ascii=True, indent=2)}\n"
        "\n"
        "# Previous Messages\n"
        + messages_str_before_this_turn +
        "# Current Messages to Evaluate\n"
        + messages_str_in_this_turn +
        "\n"
        "# Special rule of tool call\n"
        "If there is no tool call in tool_call part but there are some tool calls in content.thinking part, it means these tools' format are not correct and all tool calls are not valid."
        "If there is error in tool response. The previous tool calls in latest assistant (only the latest one) are not valid."
        "# Evaluation Process (Align each step to a JSON output field)\n"
        "1. high_level_understanding_of_the_question:\n"
        "   - Briefly restate what is being evaluated (the intent of the question + what compliance means here).\n"
        "2. analysis_of_if_focus_on:\n"
        "   - Check whether Focus on part presents in the Current Messages.\n"
        "3. analysis_of_pass_condition:\n"
        "   - Determine if the 'Pass condition' is fully satisfied.\n"
        "4. analysis_of_failure_examples:\n"
        "   - For EACH failure example pattern: state clearly 'triggered' or 'not triggered' with a brief justification.\n"
        "5. answer:\n"
        "   - Return true ONLY IF:\n"
        "     * Focus on part is present.\n"
        "     * The 'Pass condition' is fully met.\n"
        "     * No failure example pattern is triggered.\n"
        "   - Otherwise return false.\n"
        "\n"
        "# Output Format\n"
        "Return ONLY a single JSON object with exactly these keys:\n"
        "{\n"
        "  \"high_level_understanding_of_the_question\": str,\n"
        "  \"analysis_of_if_focus_on\": str,\n"
        "  \"analysis_of_pass_condition\": str,\n"
        "  \"analysis_of_failure_examples\": str,\n"
        "  \"answer\": bool\n"
        "}"
    )

    # input_prompt = (
    #     "# Task\n"
    #     "You are a precise checklist evaluator that determines whether the messages between user, assistant and tool meet specific criteria.\n"
    #     "\n"
    #     "# Evaluation\n"
    #     "1. Locate the 'Focus on' content in the messages\n"
    #     "2. Check if the messages satisfies the 'Pass condition' of the question\n"
    #     "3. Compare against 'Failure examples' to avoid common mistakes\n"
    #     "4. Use 'Reference snippet' as a benchmark for expected quality if needed\n"
    #     "\n"
    #     "# Evaluation Criteria\n"
    #     f"**Question:** {this_turn_checklist['question']}\n"
    #     f"**Focus on:** {this_turn_checklist['focus_on']}\n"
    #     f"**Pass condition:** {this_turn_checklist['pass_condition']}\n"
    #     f"**Failure examples:** {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=True, indent=2)}\n"
    #     f"**Reference snippet:** {json.dumps(reference_snippet, ensure_ascii=True, indent=2)}\n"
    #     "\n"
    #     "# Decision Rules\n"
    #     "- Return `true` ONLY if:\n"
    #     "  * The 'Focus on' content is present in the assistant's response\n"
    #     "  * The response meets the 'Pass condition'\n"
    #     "  * The response avoids patterns shown in 'Failure examples'\n"
    #     "- Return `false` if:\n"
    #     "  * The 'Focus on' content is missing\n"
    #     "  * The 'Pass condition' is not satisfied\n"
    #     "  * The response matches any 'Failure examples'\n"
    #     "# Special criteria: Tool call format\n"
    #     "Each tool call made by the assistant must strictly adhere to the following format:\n"
    #     "<tool_call>\n{\"name\": \"tool_name\", \"arguments\": { \"key\": \"value\" }}\n</tool_call>\n"
    #     "If a tool call deviates from this format, this tool call is not a valid tool call and should be considered incorrect.\n"
    #     "\n"
    #     "# Previous Messages\n"
    #     + messages_str_before_this_turn + 
    #     "# Messages to Evaluate\n"
    #     + messages_str_in_this_turn + 
    #     "\n\n# Output\n"
        
    #     "{\"high_level_understanding_of_the_question\": str,
    # "analysis_of_if_focus_on": str,
    # "analysis_of_pass_condition": str,
    # "analysis_of_failure_examples": str,
    # "answer": bool}"    
    
    return input_prompt

def get_messages_str_v1(messages: list[dict[str, Any]], step_num: int=None, max_length: int=40000) -> str:

    # TODO: limit the length of the messages_str
    if step_num is not None and _message_role(messages[0]) == "assistant":
        assert len(messages) == 1, "Only one message is allowed when step_num is not None"
    turn = -1
    step = 0
    thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    messages_str = ""

    for i, message in enumerate(messages):
        role = _message_role(message)
        content = _message_content(message)
        if role == "assistant":
            if_thinking = thinking_regex.search(content)
            if if_thinking:
                thinking = if_thinking.group(1)
                user_visible_reply = content.split("</think>")[1].strip()
                if user_visible_reply == "":
                    user_visible_reply = "None"
            else:
                thinking = content
                user_visible_reply = "None"
            tool_calls = json.dumps(_message_tool_calls(message))
        
        if role == "system":
            messages_str += f"Role: system\ncontent: {content}\n"
            step = 0
        elif role == "user":
            turn += 1
            step = 0
            messages_str += f"# Turn: {turn}\nRole: user\ncontent: {content}\n"
        elif role == "assistant":
            if step_num is not None:
                _step = step_num
            else:
                _step = step
            this_step_message = f"## Step: {_step}\nRole: assistant\ncontent.thinking: {thinking}\ncontent.user_visible_reply: {user_visible_reply}\ntool_call: {tool_calls}\n"
            # for single_checklist in checklist[turn]:
            #     user_prompt = get_user_prompt_per_step(messages_before_this_step, this_step_message, single_checklist)
            #     all_step_results.append(eval_one_check(client, user_prompt, args))
            messages_str += this_step_message
            step += 1
        elif role == "observation" or role == "tool":
            messages_str += f"Role: tool\ncontent: {content}\n"
    return messages_str


def get_messages_str_v2(messages: list[dict[str, Any]], step_num: int=None, max_length: int=40000) -> str:

    # TODO: limit the length of the messages_str
    if step_num is not None and _message_role(messages[0]) == "assistant":
        assert len(messages) == 1, "Only one message is allowed when step_num is not None"
    turn = -1
    step = 0
    thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    messages_str = ""

    for i, message in enumerate(messages):
        role = _message_role(message)
        content = _message_content(message)
        if role == "assistant":
            if_thinking = thinking_regex.search(content)
            if if_thinking:
                thinking = if_thinking.group(1)
                user_visible_reply = content.split(thinking+"</think>")[1].strip()
                if user_visible_reply == "":
                    user_visible_reply = "None"
            else:
                thinking = content
                user_visible_reply = "None"
            tool_calls_data = _message_tool_calls(message)
            if tool_calls_data is not None:
                tool_calls = json.dumps(tool_calls_data)
            elif "<tool_call>" in user_visible_reply or "</tool_call>" in user_visible_reply:
                tool_calls = user_visible_reply.replace("<tool_call>", "<|tool_call_start|>").replace("</tool_call>", "<|tool_call_end|>")
                user_visible_reply = "None"
            else:
                tool_calls = "None"
        
        if role == "system":
            messages_str += f"Role: system\ncontent: {content}\n"
            step = 0
        elif role == "user":
            turn += 1
            step = 0
            messages_str += f"# Turn: {turn}\nRole: user\ncontent: {content}\n"
        elif role == "assistant":
            if step_num is not None:
                _step = step_num
            else:
                _step = step
            this_step_message = f"## Step: {_step}\nRole: assistant\ncontent.thinking: {thinking}\ncontent.user_visible_reply: {user_visible_reply}\ntool_call: {tool_calls}\n"
            # for single_checklist in checklist[turn]:
            #     user_prompt = get_user_prompt_per_step(messages_before_this_step, this_step_message, single_checklist)
            #     all_step_results.append(eval_one_check(client, user_prompt, args))
            messages_str += this_step_message
            step += 1
        elif role == "observation" or role == "tool":
            messages_str += f"Role: tool\ncontent: {content}\n"
    return messages_str

# async def get_checklist_scores_per_step(messages_before_this_step: list[dict[str, Any]], this_step_message: str, this_turn_checklist: list[dict[str, Any]], args: dict) -> tuple[list[float], list[int]]:
#     turn = -1
#     step = 0
#     thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
#     tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

#     messages_before_this_step = ""
#     all_step_results = []
#     # semaphore = asyncio.Semaphore(self._semaphore_size)
#     async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=1200.0, read=1200.0, write=1200.0, connect=1200.0)) as client:
#         for i, message in enumerate(messages_before_this_step):
#             role = message.role
#             content = message.content
#             if role == "assistant":
#                 if_thinking = thinking_regex.search(content)
#                 if if_thinking:
#                     thinking = if_thinking.group(1)
#                     user_visible_reply = content.split("</think>")[1]
#                 else:
#                     thinking = content
#                     user_visible_reply = ""
#                 tool_calls = json.dumps([fm.model_dump() for fm in message.tool_calls] if message.tool_calls else [])
            
#             if role == "system":
#                 messages_before_this_step += f"Role: system\nContent: {content}\n"
#                 step = 0
#             elif role == "user":
#                 turn += 1
#                 step = 0
#                 messages_before_this_step += f"# Turn: {turn}\nRole: user\nContent: {content}\n"
#             elif role == "assistant":
#                 this_step_message = f"## Step: {step}\nRole: assistant\nThinking: {thinking}\nUser visible reply: {user_visible_reply}\nTool calls: {tool_calls}\n"
#                 for single_checklist in checklist[turn]:
#                     user_prompt = get_user_prompt_per_step(messages_before_this_step, this_step_message, single_checklist)
#                     all_step_results.append(eval_one_check(client, user_prompt, args))
#                 messages_before_this_step += this_step_message
#                 step += 1
#             elif role == "observation" or role == "tool":
#                 messages_before_this_step += f"Role: tool response\nContent: {content}\n"

    
#         all_step_results = await asyncio.gather(*all_step_results)

#     turn = -1
#     step = 0
#     start = 0
#     all_step_scores = []
#     turns = []
#     for i, message in enumerate(messages):
#         role = message.role
        
#         if role == "user":
#             step = 0
#             turn += 1
#             this_turn_checklist_mask = [1] * len(checklist[turn])
#         elif role == "assistant":
#             this_turn_checklist = checklist[turn]
#             end = start + len(this_turn_checklist)
#             this_step_results = all_step_results[start:end]
#             weights = [float(single_checklist["weight"]) for single_checklist in this_turn_checklist]
#             this_step_score = sum([weight * result * mask for weight, result, mask in zip(weights, this_step_results, this_turn_checklist_mask)])
#             this_turn_checklist_mask = [a*(1-b) for a,b in zip(this_turn_checklist_mask, this_step_results)]
#             this_step_score = round(this_step_score, 4)
#             all_step_scores.append(this_step_score)
#             turns.append(turn)
#             start = end
#             step += 1

#     return all_step_scores, turns



# def get_user_prompt_per_step(messages_before_this_step, this_step_message, this_turn_checklist):
#     user_prompt = (
#         "If the assistant's response follows the checklist, return True. Otherwise, return False.\n"
#         + "Checklist: " + json.dumps(this_turn_checklist, ensure_ascii=False, indent=0) + "\n"
#         + "Previous messages: " + json.dumps(messages_before_this_step, ensure_ascii=False, indent=0) + "\n"
#         + "Assistant's new response: " + json.dumps(this_step_message, ensure_ascii=False, indent=0) + "\n"
#         + "Return in JSON format: {'result': True/False}"
#     )
#     return user_prompt

async def get_checklist_scores_multiturn_multistep(
    messages,
    checklist,
    args: dict,
    calculated_rewards: list[float] | None = None,
    calculated_turns: list[int] | None = None,
    calculated_call_success: list[int] | None = None,
    client: httpx.AsyncClient | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[float], list[int]]:
    # Count the number of assistant messages
    assistant_count = sum(1 for message in messages if _message_role(message) == "assistant")
    if calculated_rewards is not None:
        if assistant_count == len(calculated_rewards):
            return calculated_rewards, calculated_turns, calculated_call_success

    turns = []
    all_step_scores = []
    call_success = []

    # if calculated_turns is not None:
    #     if len(calculated_turns) == 0:
    #         start_turn = 0
    #     else:
    #         start_turn = calculated_turns[-1]

    #     for i, calculated_turn in enumerate(calculated_turns):
    #         if calculated_turn < start_turn:
    #             all_step_scores.append(calculated_rewards[i])
    #             turns.append(calculated_turn)
    #         else:
    #             break

    if calculated_rewards is not None:
        if len(calculated_rewards) == 0:
            start_step = 0
        else:
            start_step = len(calculated_rewards)

        for i, calculated_reward in enumerate(calculated_rewards):
            all_step_scores.append(calculated_reward)
            turns.append(calculated_turns[i])
            call_success.append(calculated_call_success[i])

    turn = -1
    step = 0
    
    # Setup shared client and semaphore if not provided by caller
    # local_client: httpx.AsyncClient | None = None
    # if client is None:
    #     semaphore_size = int(args.get("semaphore_size", 64))
    #     timeout_seconds = float(args.get("timeout_seconds", 120.0))
    #     limits = httpx.Limits(
    #         max_connections=max(16, semaphore_size),
    #         max_keepalive_connections=max(8, semaphore_size // 2),
    #     )
    #     timeout = httpx.Timeout(timeout=timeout_seconds, read=timeout_seconds, write=timeout_seconds, connect=timeout_seconds)
    #     local_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    #     client = local_client
    # if semaphore is None:
    #     semaphore = asyncio.Semaphore(int(args.get("semaphore_size", 64)))

    all_step_results = []
    step_eval_batch_size = max(1, int(args.get("step_eval_batch_size", 4)))
    global_assistant_count = 0
    try:
        for i, message in enumerate(messages):
            role = _message_role(message)
            
            if role == "system":
                step = 0
            elif role == "user":
                turn += 1
                step = 0
            elif role == "assistant":
                if calculated_rewards is not None and start_step > global_assistant_count:
                        pass # already calculated
                else:
                    this_step_message = [messages[i]]
                    messages_before_this_step = messages[:i]
                    this_step_message_str = get_messages_str_v2(this_step_message, step)
                    messages_str_before_this_step = get_messages_str_v2(messages_before_this_step)
                    # Get following tool response if next message is a tool
                    following_tool_response_str = "No following tool response"
                    last_user_message_idx = -1
                    for msg_idx in range(len(messages)-1, -1, -1):
                        if _message_role(messages[msg_idx]) == "user":
                            last_user_message_idx = msg_idx
                            break
                    assert last_user_message_idx != -1
                    messages_str_before_this_turn = get_messages_str_v2(messages[:last_user_message_idx])
                    this_turn_messages_util_now = messages[last_user_message_idx:i+1]
                    tool_call_failed = False
                    if i + 1 < len(messages) and _message_role(messages[i + 1]) in ["observation", "tool"]:
                        tool_messages = []
                        j = i + 1
                        while j < len(messages) and _message_role(messages[j]) in ["observation", "tool"]:
                            if _message_has_error_tool_call(messages[j]):
                                tool_call_failed = True
                            tool_messages.append(messages[j])
                            this_turn_messages_util_now.append(messages[j])
                            j += 1
                        following_tool_response_str = get_messages_str_v2(tool_messages)
                    for single_step_checklist in checklist[turn]:
                        # input_prompt = get_input_prompt(messages_str_before_this_step, this_step_message_str, following_tool_response_str, single_step_checklist)
                        messages_str_in_this_turn_until_now = get_messages_str_v2(this_turn_messages_util_now)
                        input_prompt = get_input_prompt_v2(messages_str_before_this_turn, messages_str_in_this_turn_until_now, single_step_checklist)
                        async def _guarded_eval(prompt: str) -> bool:
                            async with semaphore:  # type: ignore[arg-type]
                                return await eval_one_check(client, prompt, args)  # type: ignore[arg-type]
                        async def _guarded_eval_tool_error() -> bool:
                            return False, True  # type: ignore[arg-type]
                        if (single_step_checklist["focus_on"]=="assistant.tool_calls" or single_step_checklist["focus_on"]=="tool.content") and tool_call_failed:
                            all_step_results.append(_guarded_eval_tool_error())
                        else:
                            all_step_results.append(_guarded_eval(input_prompt))
                step += 1
                global_assistant_count += 1
            elif role == "observation" or role == "tool":
                pass

        if all_step_results:
            batched_results = []
            for start_idx in range(0, len(all_step_results), step_eval_batch_size):
                batched_results.extend(await asyncio.gather(*all_step_results[start_idx : start_idx + step_eval_batch_size]))
            all_step_results = batched_results
    finally:
        # if local_client is not None:
        #     await local_client.aclose()
        pass


    turn = -1
    step = 0
    start = 0
    global_assistant_count = 0

    for i, message in enumerate(messages):
        role = _message_role(message)
        if role == "system":
            step = 0
        elif role == "user":
            step = 0
            turn += 1
            this_turn_checklist_mask = [1] * len(checklist[turn])
        elif role == "assistant":
            if calculated_rewards is not None and start_step > global_assistant_count:
                pass
            else:
                this_turn_checklist = checklist[turn]
                end = start + len(this_turn_checklist)
                this_step_results = all_step_results[start:end]
                all_step_scores.append([ x[0] for x in this_step_results])
                call_success.append([ x[1] for x in this_step_results])
                # weights = [float(single_checklist["weight"]) for single_checklist in this_turn_checklist]
                # this_step_score = sum([weight * result * mask for weight, result, mask in zip(weights, this_step_results, this_turn_checklist_mask)])
                # this_turn_checklist_mask = [a*(1-b) for a,b in zip(this_turn_checklist_mask, this_step_results)]
                # this_step_score = round(this_step_score, 8)
                # all_step_scores.append(this_step_score)
                turns.append(turn)
                start = end
            step += 1
            global_assistant_count += 1

    return all_step_scores, turns, call_success


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_has_error_tool_call(message: Any) -> bool:
    content = _message_content(message)
    if not isinstance(content, str) or not content:
        return False
    try:
        parsed_content = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed_content, dict) and "error_tool_call" in parsed_content


def _message_tool_calls(message: Any) -> list[dict[str, Any]] | None:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    else:
        tool_calls = getattr(message, "tool_calls", None)

    if tool_calls is None:
        return None

    serialized = []
    for tool_call in tool_calls:
        if hasattr(tool_call, "model_dump"):
            serialized.append(tool_call.model_dump(exclude_unset=True, exclude_none=True))
        elif isinstance(tool_call, dict):
            serialized.append(tool_call)
    return serialized

async def _post_with_retries(client: httpx.AsyncClient, url: str, json_payload: dict, retry_times: int = 3) -> httpx.Response:
    """Post with retries and basic backoff. Caller should handle failures."""

    def _endpoint_candidates(u: str) -> list[str]:
        candidates = [u]
        if isinstance(u, str) and u.endswith("/v1"):
            candidates.append(u + "/chat/completions")
            candidates.append(u + "/completions")
        elif isinstance(u, str) and u.endswith("/v1/chat/completions"):
            candidates.append(u[: -len("/v1/chat/completions")] + "/chat/completions")
            candidates.append(u[: -len("/chat/completions")] + "/completions")
        elif isinstance(u, str) and u.endswith("/chat/completions"):
            candidates.append(u[: -len("/chat/completions")] + "/v1/chat/completions")
            candidates.append(u[: -len("/chat/completions")] + "/completions")
        elif isinstance(u, str) and u.endswith("/completions"):
            candidates.append(u[: -len("/completions")] + "/chat/completions")
            candidates.append(u[: -len("/completions")] + "/v1/chat/completions")
        # de-dup and keep order
        dedup: list[str] = []
        seen = set()
        for item in candidates:
            if item not in seen:
                seen.add(item)
                dedup.append(item)
        return dedup

    last_exc: Exception | None = None
    attempts = max(1, int(retry_times))
    urls = _endpoint_candidates(url)
    headers = _build_openai_compatible_headers()
    for attempt in range(attempts):
        for candidate_url in urls:
            try:
                resp = await client.post(candidate_url, json=json_payload, headers=headers)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as e:
                last_exc = e
                # Continue trying alternate endpoint only on 404 route mismatch.
                if e.response is None or e.response.status_code != 404:
                    break
            except Exception as e:  # type: ignore[attr-defined]
                last_exc = e
                break
        try:
            await asyncio.sleep(min((2 ** attempt) / 10, 1))
        except Exception:
            pass
    # If all retries failed, re-raise the last exception
    assert last_exc is not None
    raise last_exc
