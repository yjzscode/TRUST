from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import time
import uuid
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v3.train.judges import get_model_judge
from v3.uq.answer_format import extract_answer, parse_tool_call_blocks, render_tool_call_blocks

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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


def _msg_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", "user") or "user").strip()
    return str(getattr(msg, "role", "user") or "user").strip()


def _msg_content(msg: Any) -> str:
    if isinstance(msg, dict):
        value = msg.get("content", "")
    else:
        value = getattr(msg, "content", "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(_json_safe(value)).strip()


def _normalize_prompt_messages(prompt_messages: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in ensure_list(prompt_messages):
        normalized.append(
            {
                "role": _msg_role(msg),
                "content": _msg_content(msg),
                "tool_calls": _json_safe(msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)),
            }
        )
    return normalized


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    raw = str(text or "")
    if not raw.strip():
        return []
    spans: list[tuple[str, int, int]] = []
    start = 0
    n = len(raw)
    for idx, ch in enumerate(raw):
        if ch in ".!?;\n":
            end = idx + 1
            sentence = raw[start:end].strip()
            if sentence:
                spans.append((sentence, start, end))
            start = end
    if start < n:
        sentence = raw[start:n].strip()
        if sentence:
            spans.append((sentence, start, n))
    return spans


class V2When2CallRuntime:
    def __init__(self) -> None:
        self._hard_negative_map = {
            "A": "B",
            "B": "A",
            "C": "A",
            "D": "C",
        }
        self._rand = random.Random(0)
        self._tau = float(os.getenv("REWARD_TAU", "0.10"))
        self._lambda = float(os.getenv("REWARD_LAMBDA", "1.00"))
        self._neg_route_mode = os.getenv("REWARD_NEG_ROUTE_MODE", "default").strip().lower()
        if self._neg_route_mode not in ("default", "none"):
            self._neg_route_mode = "default"
        self._group_margin_enabled = os.getenv("REWARD_GROUP_MARGIN_ENABLE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self._group_margin_timeout_s = float(os.getenv("REWARD_GROUP_MARGIN_TIMEOUT_S", "2.0"))
        self._group_margin_expected = self._infer_group_margin_expected()
        self._group_margin_state: dict[str, dict[str, Any]] = {}
        self._strict_backend = os.getenv("STRICT_PPL_BACKEND", "remote").strip().lower()
        if self._strict_backend not in {"remote", "hf", "none"}:
            self._strict_backend = "remote"
        self._strict_model_path = os.getenv("STRICT_PPL_MODEL_PATH", os.getenv("MODEL", ""))
        self._strict_tokenizer_path = os.getenv("STRICT_PPL_TOKENIZER_PATH", self._strict_model_path)
        self._strict_device = self._resolve_strict_device(os.getenv("STRICT_PPL_DEVICE", "auto"))
        self._strict_base_url = self._resolve_strict_base_url()
        self._strict_model_name = (
            os.getenv("STRICT_PPL_MODEL")
            or os.getenv("V3_JUDGE_MODEL")
            or os.getenv("MODEL_NAME")
            or ""
        ).strip()
        self._strict_api_key = os.getenv("STRICT_PPL_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
        self._strict_enabled = self._strict_backend != "none"
        self._strict_tokenizer = None
        self._strict_model = None
        self._strict_client = None
        if self._strict_tokenizer_path:
            try:
                self._strict_tokenizer = AutoTokenizer.from_pretrained(self._strict_tokenizer_path, trust_remote_code=True)
                logger.warning(
                    "Initialized strict PPL tokenizer path=%s",
                    self._strict_tokenizer_path,
                )
            except Exception:
                logger.exception("Failed to initialize strict PPL tokenizer")
        if self._strict_enabled and self._strict_backend == "remote":
            if self._strict_base_url and self._strict_model_name:
                try:
                    self._strict_client = OpenAI(
                        base_url=self._strict_base_url,
                        api_key=self._strict_api_key,
                        http_client=httpx.Client(trust_env=self._strict_base_url.lower().startswith("https:")),
                    )
                    logger.warning(
                        "Initialized strict PPL remote scorer base_url=%s model=%s",
                        self._strict_base_url,
                        self._strict_model_name,
                    )
                except Exception:
                    logger.exception("Failed to initialize strict PPL remote client; strict scoring disabled")
                    self._strict_enabled = False
            else:
                logger.warning("Strict PPL remote scorer not configured; strict scoring disabled")
                self._strict_enabled = False
        elif self._strict_enabled and self._strict_backend == "hf":
            if self._strict_model_path:
                try:
                    self._strict_model = AutoModelForCausalLM.from_pretrained(
                        self._strict_model_path,
                        torch_dtype="auto",
                        trust_remote_code=True,
                    ).to(self._strict_device)
                    self._strict_model.eval()
                    logger.warning(
                        "Initialized strict PPL HF scorer on device=%s model=%s",
                        self._strict_device,
                        self._strict_model_path,
                    )
                except Exception:
                    logger.exception("Failed to initialize strict PPL HF scorer; strict scoring disabled")
                    self._strict_enabled = False
            else:
                logger.warning("STRICT_PPL_BACKEND=hf but STRICT_PPL_MODEL_PATH is empty; strict scoring disabled")
                self._strict_enabled = False
        self._judge = get_model_judge()

    @staticmethod
    def _infer_group_margin_expected() -> int:
        raw = os.getenv("REWARD_GROUP_MARGIN_EXPECTED_SIZE", "").strip()
        if raw:
            try:
                val = int(raw)
                if val > 0:
                    return val
            except ValueError:
                pass
        for key in ("N_SAMPLES", "actor_rollout_ref.rollout.n"):
            raw_val = os.getenv(key, "").strip()
            if not raw_val:
                continue
            try:
                val = int(raw_val)
                if val > 0:
                    return val
            except ValueError:
                continue
        return 1

    @staticmethod
    def _resolve_strict_device(requested_device: str) -> str:
        requested = str(requested_device or "auto").strip().lower()
        manual_visible_devices = os.getenv("STRICT_PPL_CUDA_VISIBLE_DEVICES", "").strip()
        if manual_visible_devices and requested in {"", "auto"} | ({requested} if requested.startswith("cuda") else set()):
            os.environ["CUDA_VISIBLE_DEVICES"] = manual_visible_devices
            logger.warning(
                "Using STRICT_PPL_CUDA_VISIBLE_DEVICES=%s for strict PPL scorer",
                manual_visible_devices,
            )
        cuda_available = bool(torch.cuda.is_available())
        if requested in {"", "auto"}:
            return "cuda" if cuda_available else "cpu"
        if requested.startswith("cuda") and not cuda_available:
            logger.warning(
                "STRICT_PPL_DEVICE=%s requested but CUDA is not available in this process; falling back to cpu",
                requested_device,
            )
            return "cpu"
        return requested

    @staticmethod
    def _resolve_strict_base_url() -> str:
        candidates = [
            os.getenv("STRICT_PPL_BASE_URL", ""),
            os.getenv("V3_JUDGE_BASE_URL", ""),
            os.getenv("SGLANG_URL", ""),
        ]
        for raw in candidates:
            url = str(raw or "").strip()
            if not url:
                continue
            if url.endswith("/v1/chat/completions"):
                return url[: -len("/v1/chat/completions")] + "/v1"
            if url.endswith("/chat/completions"):
                return url[: -len("/chat/completions")]
            return url
        return ""

    @property
    def judge_enabled(self) -> bool:
        return bool(self._judge.enabled)

    @staticmethod
    def action_from_letter(letter: str) -> str:
        return {
            "A": "direct_answer",
            "B": "tool_call",
            "C": "request_for_info",
            "D": "cannot_answer",
        }.get((letter or "").upper(), "")

    @staticmethod
    def render_prompt_text(prompt_messages: list[Any]) -> str:
        lines: list[str] = []
        for msg in _normalize_prompt_messages(prompt_messages):
            role = str(msg.get("role", "user") or "user").strip()
            content = str(msg.get("content", "") or "").strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    def _route_type(self, key: str) -> str:
        if self._neg_route_mode == "none":
            return "none"
        digest = hashlib.md5(key.encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / float(2**64)
        if u < 0.25:
            return "hard"
        if u < 0.50:
            return "random"
        return "none"

    def _pick_random_negative(self, gt_letter: str, key: str) -> str:
        choices = [letter for letter in ("A", "B", "C", "D") if letter != gt_letter]
        seed = int(hashlib.md5((key + gt_letter).encode("utf-8")).hexdigest()[:8], 16)
        self._rand.seed(seed)
        return self._rand.choice(choices)

    @staticmethod
    def _group_key(extra_info: dict[str, Any], rollout_key: str) -> str:
        uid = str(extra_info.get("uid") or "").strip()
        if uid:
            return f"uid:{uid}"
        idx = extra_info.get("idx")
        if idx is not None:
            return f"idx:{idx}"
        return f"rollout:{rollout_key}"

    def _cleanup_group_margin_state(self) -> None:
        if not self._group_margin_state:
            return
        now = time.time()
        expire_before = now - max(5.0, self._group_margin_timeout_s * 10.0)
        stale = [k for k, v in self._group_margin_state.items() if v.get("created_at", now) < expire_before]
        for key in stale:
            self._group_margin_state.pop(key, None)

    def _await_group_margin_terms(
        self,
        *,
        group_key: str,
        sample_id: str,
        gt_letter: str,
        pred_letter: str,
        seq_ppl: float | None,
    ) -> tuple[float | None, float | None, str]:
        if not self._group_margin_enabled or self._group_margin_expected <= 1:
            return None, None, "disabled"

        self._cleanup_group_margin_state()
        state = self._group_margin_state.get(group_key)
        if state is None:
            state = {
                "created_at": time.time(),
                "items": {},
            }
            self._group_margin_state[group_key] = state

        state["items"][sample_id] = {
            "gt_letter": str(gt_letter or "").strip().upper(),
            "pred_letter": str(pred_letter or "").strip().upper(),
            "seq_ppl": float(seq_ppl) if seq_ppl is not None else None,
        }
        deadline = time.time() + self._group_margin_timeout_s
        waited = "timeout"
        while time.time() < deadline:
            if len(state["items"]) >= self._group_margin_expected:
                waited = "complete"
                break
            time.sleep(0.01)

        items = list(state["items"].values())
        correct_ppls = [
            float(item["seq_ppl"])
            for item in items
            if item.get("pred_letter") == item.get("gt_letter") and item.get("seq_ppl") is not None
        ]
        wrong_ppls = [
            float(item["seq_ppl"])
            for item in items
            if item.get("pred_letter")
            and item.get("pred_letter") != item.get("gt_letter")
            and item.get("seq_ppl") is not None
        ]
        cur = state["items"].get(sample_id, {})
        cur_seq_ppl = cur.get("seq_ppl")
        cur_pred_letter = cur.get("pred_letter", "")

        ppl_gt = None
        ppl_neg = None
        if correct_ppls and wrong_ppls:
            mean_correct = float(sum(correct_ppls) / len(correct_ppls))
            mean_wrong = float(sum(wrong_ppls) / len(wrong_ppls))
            if cur_pred_letter == gt_letter and cur_seq_ppl is not None:
                ppl_gt = float(cur_seq_ppl)
                ppl_neg = mean_wrong
            elif cur_pred_letter and cur_pred_letter != gt_letter and cur_seq_ppl is not None:
                ppl_gt = mean_correct
                ppl_neg = float(cur_seq_ppl)
            else:
                ppl_gt = mean_correct
                ppl_neg = mean_wrong
        elif correct_ppls and not wrong_ppls:
            mean_correct = float(sum(correct_ppls) / len(correct_ppls))
            if cur_pred_letter == gt_letter and cur_seq_ppl is not None:
                ppl_gt = float(cur_seq_ppl)
            else:
                ppl_gt = mean_correct
            ppl_neg = 1.0
        elif wrong_ppls and not correct_ppls:
            mean_wrong = float(sum(wrong_ppls) / len(wrong_ppls))
            ppl_gt = 1.0
            if cur_pred_letter and cur_pred_letter != gt_letter and cur_seq_ppl is not None:
                ppl_neg = float(cur_seq_ppl)
            else:
                ppl_neg = mean_wrong
        else:
            # No usable correct/wrong evidence in the group.
            ppl_gt = 1.0
            ppl_neg = 1.0

        if len(state["items"]) >= self._group_margin_expected or waited == "timeout":
            state["items"].pop(sample_id, None)
            if not state["items"]:
                self._group_margin_state.pop(group_key, None)

        return ppl_gt, ppl_neg, f"group_rollout:{waited}"

    @staticmethod
    def _force_action_in_response(response_text: str, forced_action: str, forced_letter: str) -> str:
        replacement = f"{forced_action}<{forced_letter}>"
        pattern = re.compile(r"\b(direct_answer|tool_call|request_for_info|cannot_answer)\s*<\s*([ABCD])\s*>", re.IGNORECASE)
        if pattern.search(response_text or ""):
            return pattern.sub(replacement, response_text, count=1)
        return str(response_text or "")

    def _strict_ppl(self, prompt_text: str, response_text: str) -> float | None:
        if not self._strict_enabled:
            return None
        try:
            prefix = prompt_text + "\nassistant: "
            response = str(response_text or "")
            if self._strict_client is not None and self._strict_tokenizer is not None:
                return self._strict_ppl_remote(prefix, response)
            if self._strict_model is not None and self._strict_tokenizer is not None:
                return self._strict_ppl_hf(prefix, response)
        except Exception:
            logger.debug("strict ppl scoring failed", exc_info=True)
            return None
        return None

    def _strict_ppl_hf(self, prefix: str, response_text: str) -> float | None:
        full = prefix + response_text
        tok = self._strict_tokenizer
        full_ids = tok(full, return_tensors="pt", add_special_tokens=True)
        prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=True)
        input_ids = full_ids["input_ids"].to(self._strict_device)
        attention_mask = full_ids.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._strict_device)
        labels = input_ids.clone()
        pref_len = int(prefix_ids["input_ids"].shape[1])
        labels[:, :pref_len] = -100
        with torch.no_grad():
            out = self._strict_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return float(torch.exp(torch.tensor(float(out.loss.item()))).item())

    def _strict_token_nlls_hf(self, prefix: str, response_text: str) -> list[float]:
        full = prefix + response_text
        tok = self._strict_tokenizer
        full_ids = tok(full, return_tensors="pt", add_special_tokens=True)
        prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=True)
        input_ids = full_ids["input_ids"].to(self._strict_device)
        attention_mask = full_ids.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._strict_device)
        with torch.no_grad():
            logits = self._strict_model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        labels = input_ids[0]
        prefix_len = int(prefix_ids["input_ids"].shape[1])
        token_nlls: list[float] = []
        for pos in range(prefix_len, labels.shape[0]):
            prev_pos = pos - 1
            token_id = int(labels[pos].item())
            log_probs = torch.log_softmax(logits[prev_pos], dim=-1)
            token_nlls.append(float(-log_probs[token_id].item()))
        return token_nlls

    def _strict_ppl_remote(self, prefix: str, response_text: str) -> float | None:
        if self._strict_client is None or self._strict_tokenizer is None or not self._strict_model_name:
            return None
        full = prefix + response_text
        prefix_ids = self._strict_tokenizer(prefix, add_special_tokens=True).get("input_ids", [])
        response_ids = self._strict_tokenizer(response_text, add_special_tokens=False).get("input_ids", [])
        response_len = len(response_ids)
        if response_len <= 0:
            return None
        completion = self._strict_client.completions.create(
            model=self._strict_model_name,
            prompt=full,
            max_tokens=0,
            temperature=0.0,
            echo=True,
            logprobs=1,
        )
        choice = completion.choices[0]
        logprob_payload = getattr(choice, "logprobs", None)
        token_logprobs = list(getattr(logprob_payload, "token_logprobs", None) or [])
        if not token_logprobs:
            return None
        # `echo=True` may include a leading BOS/null logprob; align from the tail to preserve the response span.
        expected_total = len(prefix_ids) + response_len
        if len(token_logprobs) >= expected_total:
            aligned = token_logprobs[-expected_total:]
        else:
            aligned = token_logprobs
        response_logprobs = aligned[-response_len:]
        usable = [float(lp) for lp in response_logprobs if lp is not None]
        if len(usable) != response_len:
            return None
        mean_nll = -sum(usable) / max(1, len(usable))
        return float(torch.exp(torch.tensor(mean_nll)).item())

    def _strict_token_nlls_remote(self, prefix: str, response_text: str) -> list[float]:
        if self._strict_client is None or self._strict_tokenizer is None or not self._strict_model_name:
            return []
        full = prefix + response_text
        prefix_ids = self._strict_tokenizer(prefix, add_special_tokens=True).get("input_ids", [])
        response_ids = self._strict_tokenizer(response_text, add_special_tokens=False).get("input_ids", [])
        response_len = len(response_ids)
        if response_len <= 0:
            return []
        completion = self._strict_client.completions.create(
            model=self._strict_model_name,
            prompt=full,
            max_tokens=0,
            temperature=0.0,
            echo=True,
            logprobs=1,
        )
        choice = completion.choices[0]
        logprob_payload = getattr(choice, "logprobs", None)
        token_logprobs = list(getattr(logprob_payload, "token_logprobs", None) or [])
        if not token_logprobs:
            return []
        expected_total = len(prefix_ids) + response_len
        if len(token_logprobs) >= expected_total:
            aligned = token_logprobs[-expected_total:]
        else:
            aligned = token_logprobs
        response_logprobs = aligned[-response_len:]
        token_nlls: list[float] = []
        for lp in response_logprobs:
            if lp is None:
                continue
            token_nlls.append(float(-float(lp)))
        return token_nlls

    def _strict_action_ppl(self, prompt_text: str, forced_letter: str, response_text: str) -> float | None:
        forced_action = self.action_from_letter(forced_letter)
        if not forced_action:
            return None
        if forced_letter == "B":
            force_inst = (
                "\n\n[Forced Action]\n"
                "You MUST perform a tool call.\n"
                "You MUST output only <tool_call>...</tool_call> blocks.\n"
                "You MUST NOT output action tags, <answer> tags, or extra prose.\n"
            )
        else:
            force_inst = (
                "\n\n[Forced Action]\n"
                f"You MUST output plain assistant text consistent with action {forced_action}<{forced_letter}>.\n"
                "You MUST NOT output action tags, <answer> tags, or any tool-call XML unless the action is tool_call<B>.\n"
            )
        forced_response = self._normalize_response_for_forced_surface(str(response_text or ""), forced_letter)
        return self._strict_ppl(prompt_text + force_inst, forced_response)

    @staticmethod
    def _normalize_response_for_forced_surface(response_text: str, forced_letter: str) -> str:
        raw = str(response_text or "").strip()
        if forced_letter == "B":
            tool_blocks = parse_tool_call_blocks(raw)
            if tool_blocks:
                return render_tool_call_blocks(tool_blocks)
            answer = extract_answer(raw)
            if answer:
                answer_blocks = parse_tool_call_blocks(answer)
                if answer_blocks:
                    return render_tool_call_blocks(answer_blocks)
            return raw
        answer = extract_answer(raw)
        if answer:
            return answer.strip()
        normalized = V2When2CallRuntime._force_action_in_response(raw, V2When2CallRuntime.action_from_letter(forced_letter), forced_letter)
        normalized = re.sub(r"</?answer>", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(
            r"\b(direct_answer|tool_call|request_for_info|cannot_answer)\s*<\s*([ABCD])\s*>",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        return normalized

    def _sentence_ppl_pairs_from_token_nlls(self, response_text: str, token_nlls: list[float]) -> list[dict[str, Any]]:
        if not response_text or not token_nlls or self._strict_tokenizer is None:
            return []
        tok = self._strict_tokenizer
        token_ids = list(tok(response_text, add_special_tokens=False).get("input_ids", []))
        if not token_ids:
            return []
        usable = min(len(token_ids), len(token_nlls))
        token_ids = token_ids[:usable]
        token_nlls = token_nlls[:usable]
        full_text = tok.decode(token_ids, skip_special_tokens=True)
        if not full_text:
            return []
        spans = _split_sentences_with_spans(full_text)
        if not spans:
            return []
        token_texts = [tok.decode([tid], skip_special_tokens=False) for tid in token_ids]
        pos = 0
        token_ranges: list[tuple[int, int]] = []
        for piece in token_texts:
            if not piece:
                token_ranges.append((pos, pos))
                continue
            idx = full_text.find(piece, pos)
            if idx < 0:
                idx = pos
            end = idx + len(piece)
            token_ranges.append((idx, end))
            pos = end
        out: list[dict[str, Any]] = []
        for sentence, s_start, s_end in spans:
            idxs: list[int] = []
            for i, (t_start, t_end) in enumerate(token_ranges):
                if t_end <= s_start:
                    continue
                if t_start >= s_end:
                    break
                idxs.append(i)
            if not idxs:
                continue
            mean_nll = float(sum(token_nlls[i] for i in idxs) / max(1, len(idxs)))
            out.append(
                {
                    "sentence": sentence,
                    "ppl": float(math.exp(mean_nll)),
                    "nll": mean_nll,
                    "token_count": int(len(idxs)),
                }
            )
        return out

    def _token_uq_summary(self, prompt_messages: list[Any], response_text: str) -> dict[str, Any]:
        if not self._strict_enabled:
            return {
                "mean_ppl": 0.0,
                "max_ppl": 0.0,
                "sequence_ppl": 0.0,
                "sequence_nll": 0.0,
                "token_count": 0,
                "sentence_ppl_pairs": [],
            }
        prompt_text = self.render_prompt_text(prompt_messages)
        seq_ppl = self._strict_ppl(prompt_text, response_text)
        if seq_ppl is None or seq_ppl <= 0.0:
            return {
                "mean_ppl": 0.0,
                "max_ppl": 0.0,
                "sequence_ppl": 0.0,
                "sequence_nll": 0.0,
                "token_count": 0,
                "sentence_ppl_pairs": [],
            }
        sentence_ppl_pairs: list[dict[str, Any]] = []
        try:
            prefix = prompt_text + "\nassistant: "
            response = str(response_text or "")
            token_nlls: list[float] = []
            if self._strict_client is not None and self._strict_tokenizer is not None:
                token_nlls = self._strict_token_nlls_remote(prefix, response)
            elif self._strict_model is not None and self._strict_tokenizer is not None:
                token_nlls = self._strict_token_nlls_hf(prefix, response)
            sentence_ppl_pairs = self._sentence_ppl_pairs_from_token_nlls(response, token_nlls)
        except Exception:
            logger.debug("sentence-level strict ppl summary failed", exc_info=True)
        return {
            "mean_ppl": float(seq_ppl),
            "max_ppl": float(seq_ppl),
            "sequence_ppl": float(seq_ppl),
            "sequence_nll": float(math.log(max(seq_ppl, 1e-12))),
            "token_count": 0,
            "sentence_ppl_pairs": sentence_ppl_pairs,
            "uq_level": "sequence",
            "uq_source": "strict_ppl",
        }

    def _token_uq_summary_from_rollout(
        self,
        *,
        response_ids: list[int] | None,
        response_logprobs: list[float] | None,
        response_mask: list[int] | None,
    ) -> dict[str, Any]:
        if not response_ids or not response_logprobs:
            return {
                "mean_ppl": 0.0,
                "max_ppl": 0.0,
                "sequence_ppl": 0.0,
                "sequence_nll": 0.0,
                "token_count": 0,
                "sentence_ppl_pairs": [],
            }
        try:
            usable = min(len(response_ids), len(response_logprobs))
            if usable <= 0:
                return {
                    "mean_ppl": 0.0,
                    "max_ppl": 0.0,
                    "sequence_ppl": 0.0,
                    "sequence_nll": 0.0,
                    "token_count": 0,
                    "sentence_ppl_pairs": [],
                }
            token_nlls: list[float] = []
            filtered_response_ids: list[int] = []
            for idx in range(usable):
                if response_mask is not None and idx < len(response_mask) and int(response_mask[idx]) == 0:
                    continue
                token_nlls.append(float(-response_logprobs[idx]))
                filtered_response_ids.append(int(response_ids[idx]))
            if not token_nlls:
                return {
                    "mean_ppl": 0.0,
                    "max_ppl": 0.0,
                    "sequence_ppl": 0.0,
                    "sequence_nll": 0.0,
                    "token_count": 0,
                    "sentence_ppl_pairs": [],
                }
            mean_nll = float(sum(token_nlls) / max(1, len(token_nlls)))
            seq_ppl = float(math.exp(mean_nll))
            sentence_ppl_pairs: list[dict[str, Any]] = []
            try:
                if self._strict_tokenizer is not None and filtered_response_ids:
                    response_text = self._strict_tokenizer.decode(filtered_response_ids, skip_special_tokens=True)
                    sentence_ppl_pairs = self._sentence_ppl_pairs_from_token_nlls(response_text, token_nlls)
            except Exception:
                logger.debug("sentence-level rollout ppl summary failed", exc_info=True)
            return {
                "mean_ppl": seq_ppl,
                "max_ppl": seq_ppl,
                "sequence_ppl": seq_ppl,
                "sequence_nll": mean_nll,
                "token_count": int(len(token_nlls)),
                "sentence_ppl_pairs": sentence_ppl_pairs,
                "uq_level": "sequence",
                "uq_source": "rollout_logprobs",
            }
        except Exception:
            logger.debug("rollout-logprob sequence uq failed", exc_info=True)
            return {
                "mean_ppl": 0.0,
                "max_ppl": 0.0,
                "sequence_ppl": 0.0,
                "sequence_nll": 0.0,
                "token_count": 0,
                "sentence_ppl_pairs": [],
            }

    def inject_runtime_fields(
        self,
        *,
        prompt_messages: list[Any],
        response_text: str,
        ground_truth: str,
        extra_info: dict[str, Any] | None,
        rollout_key: str,
        response_ids: list[int] | None = None,
        response_logprobs: list[float] | None = None,
        response_mask: list[int] | None = None,
    ) -> dict[str, Any]:
        extra = ensure_dict(extra_info)
        extra["tau"] = self._tau
        extra["lambda"] = self._lambda
        extra["rollout_key"] = rollout_key
        gt_letter = str(ground_truth or extra.get("gt_letter") or "").strip().upper()
        pred_letter = str(extra.get("pred_letter") or "").strip().upper()
        if not pred_letter:
            pred_letter = self._extract_pred_letter(response_text)
        route_type = self._route_type(rollout_key)
        if route_type == "hard":
            neg_letter = self._hard_negative_map.get(gt_letter, "A")
        elif route_type == "random":
            neg_letter = self._pick_random_negative(gt_letter, rollout_key)
        else:
            neg_letter = ""
        prompt_text = self.render_prompt_text(prompt_messages)
        ppl_gt = None
        ppl_neg = None
        margin_source = route_type if route_type != "none" else "neutral"
        seq_ppl = None
        if response_logprobs:
            try:
                nlls = [-float(lp) for lp in response_logprobs]
                seq_ppl = float(torch.exp(torch.tensor(sum(nlls) / max(len(nlls), 1), dtype=torch.float32)).item())
            except Exception:
                seq_ppl = None
        if seq_ppl is not None and pred_letter:
            if pred_letter == gt_letter:
                ppl_gt = float(seq_ppl)
            else:
                ppl_neg = float(seq_ppl)
        if route_type != "none":
            strict_gt = self._strict_action_ppl(prompt_text, gt_letter, response_text)
            strict_neg = self._strict_action_ppl(prompt_text, neg_letter, response_text) if neg_letter else None
            if ppl_gt is None and strict_gt is not None:
                ppl_gt = float(strict_gt)
            if ppl_neg is None and strict_neg is not None:
                ppl_neg = float(strict_neg)
        if ppl_gt is None or ppl_neg is None:
            group_ppl_gt, group_ppl_neg, group_source = self._await_group_margin_terms(
                group_key=self._group_key(extra, rollout_key),
                sample_id=str(uuid.uuid4()),
                gt_letter=gt_letter,
                pred_letter=pred_letter,
                seq_ppl=seq_ppl,
            )
            if ppl_gt is None and group_ppl_gt is not None:
                ppl_gt = float(group_ppl_gt)
            if ppl_neg is None and group_ppl_neg is not None:
                ppl_neg = float(group_ppl_neg)
            if group_ppl_gt is not None and group_ppl_neg is not None:
                margin_source = group_source
        margin = float(ppl_neg - ppl_gt) if (ppl_gt is not None and ppl_neg is not None) else 0.0
        extra["neg_type"] = route_type
        extra["margin_source"] = margin_source
        extra["gt_letter"] = gt_letter
        if neg_letter:
            extra["neg_letter"] = neg_letter
        if ppl_gt is not None:
            extra["ppl_gt"] = round(float(ppl_gt), 6)
        if ppl_neg is not None:
            extra["ppl_neg"] = round(float(ppl_neg), 6)
        extra["margin"] = round(float(margin), 6)
        return extra

    @staticmethod
    def _extract_pred_letter(response_text: str) -> str:
        text = str(response_text or "")
        match = re.search(
            r"\b(?:direct_answer|tool_call|request_for_info|cannot_answer)\s*<\s*([ABCD])\s*>",
            text,
            re.IGNORECASE,
        )
        if match:
            return str(match.group(1) or "").strip().upper()
        return ""


@lru_cache(maxsize=1)
def get_v2_when2call_runtime() -> V2When2CallRuntime:
    return V2When2CallRuntime()
