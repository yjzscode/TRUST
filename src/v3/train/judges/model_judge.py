from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)
_JUDGE_PARSE_DEBUG_LIMIT = max(0, int(os.getenv("V3_JUDGE_PARSE_DEBUG_LIMIT", "5")))
_JUDGE_PARSE_DEBUG_MAX_CHARS = max(200, int(os.getenv("V3_JUDGE_PARSE_DEBUG_MAX_CHARS", "2000")))
_JUDGE_PARSE_DEBUG_COUNT = 0


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

    fence_re = re.compile(r"```(?:\s*json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    for fenced in reversed(fence_re.findall(raw)):
        try:
            parsed = json.loads(fenced.strip())
            obj = _as_json_object(parsed)
            if obj is not None:
                yield obj
        except Exception:
            pass

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


def _log_parse_failure(text: str) -> None:
    global _JUDGE_PARSE_DEBUG_COUNT
    if _JUDGE_PARSE_DEBUG_LIMIT <= 0:
        return
    if _JUDGE_PARSE_DEBUG_COUNT >= _JUDGE_PARSE_DEBUG_LIMIT:
        return
    _JUDGE_PARSE_DEBUG_COUNT += 1
    preview = repr(text)
    if len(preview) > _JUDGE_PARSE_DEBUG_MAX_CHARS:
        preview = preview[:_JUDGE_PARSE_DEBUG_MAX_CHARS] + "...<truncated>"
    logger.warning(
        "[v3 judge parse debug %d/%d] raw_content=%s",
        _JUDGE_PARSE_DEBUG_COUNT,
        _JUDGE_PARSE_DEBUG_LIMIT,
        preview,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    for candidate in _iter_json_object_candidates(text):
        return candidate
    _log_parse_failure(text)
    return {}


def _default_base_url_from_chat_url(chat_url: str) -> str:
    raw = (chat_url or "").strip()
    if raw.endswith("/chat/completions"):
        return raw[: -len("/chat/completions")]
    return raw


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _chat_messages(system_prompt: str, user_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": json.dumps(_json_safe(user_payload), ensure_ascii=False, indent=2)},
    ]


class ModelJudge:
    def __init__(self) -> None:
        self.backend = os.getenv("V3_JUDGE_BACKEND", "openai").strip().lower()
        self.base_url = os.getenv("V3_JUDGE_BASE_URL") or _default_base_url_from_chat_url(os.getenv("SGLANG_URL", ""))
        self.model_name = os.getenv("V3_JUDGE_MODEL", "")
        self.api_key = os.getenv("V3_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
        self.max_tokens = int(os.getenv("V3_JUDGE_MAX_TOKENS", "512"))
        self.temperature = float(os.getenv("V3_JUDGE_TEMPERATURE", "0.0"))
        self.soft_fail = os.getenv("V3_JUDGE_SOFT_FAIL", "1").strip().lower() not in {"0", "false", "no"}
        self._failure_count = 0
        self._failure_log_limit = max(0, int(os.getenv("V3_JUDGE_FAILURE_LOG_LIMIT", "20")))
        self.model_path = os.getenv("V3_JUDGE_MODEL_PATH", "")
        self.device = os.getenv("V3_JUDGE_DEVICE", "cuda")
        self._client = None
        self._tokenizer = None
        self._model = None
        if self.backend == "openai" and self.base_url and self.model_name:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=httpx.Client(trust_env=self.base_url.lower().startswith("https:")),
            )
        elif self.backend == "hf" and self.model_path:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
        print(
            "[v3 judge] "
            f"backend={self.backend} enabled={self.enabled} base_url={self.base_url!r} "
            f"model={self.model_name!r} model_path={self.model_path!r} soft_fail={self.soft_fail}"
        )

    @property
    def enabled(self) -> bool:
        return self._client is not None or (self._tokenizer is not None and self._model is not None)

    def generate_json(self, *, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=_chat_messages(system_prompt, user_payload),
                )
            except Exception as exc:
                if not self.soft_fail:
                    raise
                self._failure_count += 1
                if self._failure_count <= self._failure_log_limit:
                    logger.warning(
                        "[v3 judge] soft-failed remote judge call %d/%d: %s: %s",
                        self._failure_count,
                        self._failure_log_limit,
                        type(exc).__name__,
                        str(exc).split("\n")[0],
                    )
                return {}
            content = str(response.choices[0].message.content or "")
            return _extract_json_object(content)
        assert self._tokenizer is not None and self._model is not None
        messages = _chat_messages(system_prompt, user_payload)
        prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)
        output = self._model.generate(
            **encoded,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        new_tokens = output[0][encoded["input_ids"].shape[1] :]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _extract_json_object(content)


@lru_cache(maxsize=1)
def get_model_judge() -> ModelJudge:
    return ModelJudge()
