"""PPL-aware reward manager for When2Call GRPO.

This reward manager enriches each rollout sample with:
1. runtime sequence-level PPL / normalized UQ from rollout logprobs
2. group-level margin terms ``ppl_gt`` and ``ppl_neg`` used by
   ``src/v2/train/reward_when2call.py``

The open-source release uses the current VERL ``workers.reward_manager`` API,
so registration must happen through ``verl.workers.reward_manager.register``.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import numbers
import os
import random
import re
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except Exception:
            return value
    return value


def _coerce_dict(value: Any) -> dict[str, Any]:
    value = _maybe_json_loads(value)
    return value if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    value = _maybe_json_loads(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
        except Exception:
            converted = None
        if isinstance(converted, list):
            return converted
    return []


@register("ppl_uq")
class PPLRewardManager(AbstractRewardManager):
    """Inject runtime PPL/UQ and margin terms for When2Call GRPO."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_fn_key = reward_fn_key
        self.reward_router_address = kwargs.get("reward_router_address")
        self.reward_model_tokenizer = kwargs.get("reward_model_tokenizer")

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
            logger.warning("Unknown REWARD_NEG_ROUTE_MODE=%s, fallback to default", self._neg_route_mode)
            self._neg_route_mode = "default"

        self._group_margin_expected = self._infer_group_margin_expected()
        self._strict_model_path = os.getenv("STRICT_PPL_MODEL_PATH", os.getenv("MODEL", ""))
        self._strict_device = os.getenv("STRICT_PPL_DEVICE", "cpu")
        self._strict_enabled = bool(self._strict_model_path) and self._neg_route_mode != "none"
        self._strict_tokenizer = None
        self._strict_model = None
        if self._strict_enabled:
            try:
                logger.warning(
                    "Loading strict PPL scorer model from %s on %s",
                    self._strict_model_path,
                    self._strict_device,
                )
                self._strict_tokenizer = AutoTokenizer.from_pretrained(
                    self._strict_model_path,
                    trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self._strict_model_path,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
                self._strict_model = model.to(self._strict_device)
                self._strict_model.eval()
            except Exception:
                logger.exception("Failed to init strict PPL scorer; strict scoring disabled")
                self._strict_enabled = False

    @staticmethod
    def _infer_group_margin_expected() -> int:
        for key in ("REWARD_GROUP_MARGIN_EXPECTED_SIZE", "N_SAMPLES"):
            raw = os.getenv(key, "").strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value
        return 1

    def _compute_seq_ppl_and_uq(self, data_item) -> tuple[float | None, float | None]:
        try:
            batch = data_item.batch
            if "rollout_log_probs" not in batch:
                return None, None

            prompt_ids = batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = int(batch["attention_mask"][prompt_length:].sum())
            rollout_lp = batch["rollout_log_probs"]
            if rollout_lp is None or valid_response_length <= 0:
                return None, None

            from uq.ppl import ppl_from_token_logprobs, ppl_normalized

            token_logprobs = rollout_lp[:valid_response_length]
            if hasattr(token_logprobs, "tolist"):
                token_logprobs = token_logprobs.tolist()
            values = [float(x) for x in token_logprobs]
            if not values:
                return None, None
            seq_ppl = ppl_from_token_logprobs(values)
            uq = ppl_normalized(seq_ppl)
            return float(seq_ppl), float(uq)
        except Exception:
            logger.debug("runtime UQ extraction failed", exc_info=True)
            return None, None

    @staticmethod
    def _rollout_key(extra_info: dict[str, Any], data_item) -> str:
        idx = extra_info.get("idx")
        if idx is not None:
            return f"idx:{idx}"
        question = data_item.non_tensor_batch.get("question", "")
        return f"q:{hashlib.md5(str(question).encode('utf-8')).hexdigest()}"

    @staticmethod
    def _group_key(extra_info: dict[str, Any], data_item) -> str:
        uid = data_item.non_tensor_batch.get("uid")
        if uid:
            return f"uid:{uid}"
        idx = extra_info.get("idx")
        if idx is not None:
            return f"idx:{idx}"
        question = data_item.non_tensor_batch.get("question", "")
        return f"q:{hashlib.md5(str(question).encode('utf-8')).hexdigest()}"

    @staticmethod
    def _letter_from_gt(ground_truth: str) -> str:
        gt = (ground_truth or "").strip().upper()
        if gt in ("A", "B", "C", "D"):
            return gt
        mapping = {
            "direct_answer": "A",
            "direct": "A",
            "tool_call": "B",
            "request_for_info": "C",
            "cannot_answer": "D",
        }
        return mapping.get(gt.lower(), "A")

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
        choices = [x for x in ["A", "B", "C", "D"] if x != gt_letter]
        seed = int(hashlib.md5((key + gt_letter).encode("utf-8")).hexdigest()[:8], 16)
        self._rand.seed(seed)
        return self._rand.choice(choices)

    @staticmethod
    def _render_prompt_text(non_tensor_batch: dict[str, Any]) -> str:
        prompt = _coerce_list(non_tensor_batch.get("prompt"))
        if prompt:
            lines = []
            for message in prompt:
                if isinstance(message, dict):
                    role = str(message.get("role", "user")).strip()
                    content = str(message.get("content", "")).strip()
                    lines.append(f"{role}: {content}")
            if lines:
                return "\n".join(lines).strip()
        question = non_tensor_batch.get("question")
        return str(question or "").strip()

    @staticmethod
    def _action_from_letter(letter: str) -> str:
        return {
            "A": "direct_answer",
            "B": "tool_call",
            "C": "request_for_info",
            "D": "cannot_answer",
        }.get((letter or "").upper(), "")

    @staticmethod
    def _force_action_in_response(response_text: str, forced_action: str, forced_letter: str) -> str:
        text = str(response_text or "")
        replacement = f"{forced_action}<{forced_letter}>"
        pattern = re.compile(
            r"\b(direct_answer|tool_call|request_for_info|cannot_answer)\s*<\s*([ABCD])\s*>",
            re.IGNORECASE,
        )
        if pattern.search(text):
            return pattern.sub(replacement, text, count=1)
        return text

    def _strict_ppl(self, prompt_text: str, forced_letter: str, response_text: str) -> float | None:
        if not self._strict_enabled or self._strict_model is None or self._strict_tokenizer is None:
            return None
        try:
            forced_action = self._action_from_letter(forced_letter)
            if not forced_action:
                return None
            force_instruction = (
                "\n\n[Forced Action]\n"
                f"You MUST output action tag {forced_action}<{forced_letter}> and keep content consistent with this action.\n"
            )
            forced_response = self._force_action_in_response(response_text, forced_action, forced_letter)
            prefix = prompt_text + force_instruction
            full_text = prefix + "\nassistant: " + forced_response

            full_ids = self._strict_tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
            prefix_ids = self._strict_tokenizer(prefix + "\nassistant: ", return_tensors="pt", add_special_tokens=True)

            input_ids = full_ids["input_ids"].to(self._strict_device)
            attention_mask = full_ids.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._strict_device)
            labels = input_ids.clone()
            prefix_len = int(prefix_ids["input_ids"].shape[1])
            labels[:, :prefix_len] = -100

            with torch.no_grad():
                outputs = self._strict_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = float(outputs.loss.item())
            return float(torch.exp(torch.tensor(loss)).item())
        except Exception:
            logger.debug("strict ppl scoring failed", exc_info=True)
            return None

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return isinstance(value, numbers.Number) and torch.isfinite(torch.tensor(float(value))).item()

    def _apply_group_margin_fallback(self, item_infos: list[dict[str, Any]]) -> None:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in item_infos:
            groups[item["group_key"]].append(item)

        for group_items in groups.values():
            correct_ppls = [
                float(item["seq_ppl"])
                for item in group_items
                if item.get("pred_letter") == item.get("gt_letter") and item.get("seq_ppl") is not None
            ]
            wrong_ppls = [
                float(item["seq_ppl"])
                for item in group_items
                if item.get("pred_letter")
                and item.get("pred_letter") != item.get("gt_letter")
                and item.get("seq_ppl") is not None
            ]

            mean_correct = float(sum(correct_ppls) / len(correct_ppls)) if correct_ppls else None
            mean_wrong = float(sum(wrong_ppls) / len(wrong_ppls)) if wrong_ppls else None
            margin_source = (
                "group_rollout:complete"
                if len(group_items) >= self._group_margin_expected
                else "group_rollout:partial"
            )

            for item in group_items:
                cur_seq_ppl = item.get("seq_ppl")
                cur_pred_letter = item.get("pred_letter") or ""
                gt_letter = item["gt_letter"]

                if item["ppl_gt"] is None:
                    if mean_correct is not None:
                        if cur_pred_letter == gt_letter and cur_seq_ppl is not None:
                            item["ppl_gt"] = float(cur_seq_ppl)
                        else:
                            item["ppl_gt"] = float(mean_correct)
                    else:
                        item["ppl_gt"] = 1.0

                if item["ppl_neg"] is None:
                    if mean_wrong is not None:
                        if cur_pred_letter and cur_pred_letter != gt_letter and cur_seq_ppl is not None:
                            item["ppl_neg"] = float(cur_seq_ppl)
                        else:
                            item["ppl_neg"] = float(mean_wrong)
                    else:
                        item["ppl_neg"] = 1.0

                if item["margin_source"] == "neutral" or item["margin_source"] == item["route_type"]:
                    item["margin_source"] = margin_source

    def _compute_sample_reward(
        self,
        data_source: str,
        response_str: str,
        ground_truth: str,
        extra_info: dict[str, Any],
    ) -> Any:
        extra_reward_kwargs: dict[str, Any] = {}
        if self.reward_router_address is not None:
            extra_reward_kwargs["reward_router_address"] = self.reward_router_address
        if self.reward_model_tokenizer is not None:
            extra_reward_kwargs["reward_model_tokenizer"] = self.reward_model_tokenizer

        if self.is_async_reward_score:
            import asyncio

            return asyncio.run(
                self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                )
            )

        return self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **extra_reward_kwargs,
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list[Any]] = defaultdict(list)
        item_infos: list[dict[str, Any]] = []

        from uq.answer_format import extract_mcq_letter_from_text

        for index in range(len(data)):
            data_item = data[index]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            reward_model = _coerce_dict(data_item.non_tensor_batch.get("reward_model"))
            ground_truth = str(reward_model.get("ground_truth", ""))
            extra_info = dict(_coerce_dict(data_item.non_tensor_batch.get("extra_info")))
            tool_extra_fields = _coerce_dict(data_item.non_tensor_batch.get("tool_extra_fields"))
            if tool_extra_fields:
                extra_info.update(tool_extra_fields)

            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["tau"] = self._tau
            extra_info["lambda"] = self._lambda

            seq_ppl, runtime_uq = self._compute_seq_ppl_and_uq(data_item)
            if seq_ppl is not None:
                extra_info["runtime_seq_ppl"] = round(seq_ppl, 6)
            if runtime_uq is not None:
                extra_info["runtime_uq_value"] = round(runtime_uq, 6)

            pred_letter = extract_mcq_letter_from_text(response_str)
            gt_letter = self._letter_from_gt(ground_truth)
            rollout_key = self._rollout_key(extra_info, data_item)
            group_key = self._group_key(extra_info, data_item)
            route_type = self._route_type(rollout_key)
            if route_type == "hard":
                neg_letter = self._hard_negative_map.get(gt_letter, "A")
            elif route_type == "random":
                neg_letter = self._pick_random_negative(gt_letter, rollout_key)
            else:
                neg_letter = ""

            ppl_gt = None
            ppl_neg = None
            margin_source = route_type if route_type != "none" else "neutral"
            if seq_ppl is not None and pred_letter:
                if pred_letter == gt_letter:
                    ppl_gt = float(seq_ppl)
                else:
                    ppl_neg = float(seq_ppl)

            if route_type != "none":
                prompt_text = self._render_prompt_text(data_item.non_tensor_batch)
                strict_gt = self._strict_ppl(prompt_text, gt_letter, response_str)
                strict_neg = self._strict_ppl(prompt_text, neg_letter, response_str) if neg_letter else None
                if ppl_gt is None and strict_gt is not None:
                    ppl_gt = float(strict_gt)
                if ppl_neg is None and strict_neg is not None:
                    ppl_neg = float(strict_neg)

            item_infos.append(
                {
                    "index": index,
                    "data_source": data_item.non_tensor_batch.get(self.reward_fn_key, ""),
                    "ground_truth": ground_truth,
                    "extra_info": extra_info,
                    "response_str": response_str,
                    "prompt_str": prompt_str,
                    "valid_response_length": valid_response_length,
                    "route_type": route_type,
                    "neg_letter": neg_letter,
                    "pred_letter": pred_letter,
                    "gt_letter": gt_letter,
                    "rollout_key": rollout_key,
                    "group_key": group_key,
                    "seq_ppl": seq_ppl,
                    "runtime_uq": runtime_uq,
                    "ppl_gt": ppl_gt,
                    "ppl_neg": ppl_neg,
                    "margin_source": margin_source,
                }
            )

        self._apply_group_margin_fallback(item_infos)

        already_print_data_sources: dict[str, int] = {}
        max_num_turns: list[float] = []
        tool_call_success: list[bool] = []

        for item in item_infos:
            extra_info = dict(item["extra_info"])
            route_type = item["route_type"]
            extra_info["neg_type"] = route_type
            extra_info["margin_source"] = item["margin_source"]
            extra_info["gt_letter"] = item["gt_letter"]
            if item["neg_letter"]:
                extra_info["neg_letter"] = item["neg_letter"]

            ppl_gt = item["ppl_gt"]
            ppl_neg = item["ppl_neg"]
            margin = float(ppl_neg - ppl_gt) if (ppl_gt is not None and ppl_neg is not None) else 0.0
            if ppl_gt is not None:
                extra_info["ppl_gt"] = round(float(ppl_gt), 6)
            if ppl_neg is not None:
                extra_info["ppl_neg"] = round(float(ppl_neg), 6)
            extra_info["margin"] = round(float(margin), 6)

            result = self._compute_sample_reward(
                data_source=str(item["data_source"]),
                response_str=item["response_str"],
                ground_truth=item["ground_truth"],
                extra_info=extra_info,
            )

            if isinstance(result, dict):
                reward = result["score"]
                for key, value in result.items():
                    if value is None:
                        reward_extra_info[key].append(None)
                    elif isinstance(value, (str, bool, numbers.Number)):
                        reward_extra_info[key].append(value)
            else:
                reward = result
                reward_extra_info["acc"].append(reward)

            valid_response_length = int(item["valid_response_length"])
            if valid_response_length > 0:
                reward_tensor[item["index"], valid_response_length - 1] = float(reward)

            num_turns_value = extra_info.get("num_turns")
            try:
                num_turns_value = int(num_turns_value) if num_turns_value is not None else 1
            except Exception:
                num_turns_value = 1
            if num_turns_value <= 0:
                num_turns_value = 1

            max_num_turns.append(float(num_turns_value))
            tool_call_success.append(True)

            data_source = str(item["data_source"])
            already_print_data_sources.setdefault(data_source, 0)
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", item["prompt_str"])
                print("[response]", item["response_str"])
                print("[ground_truth]", item["ground_truth"])
                print("[route_type]", route_type)
                print("[margin_source]", item["margin_source"])
                if self._is_finite_number(ppl_gt):
                    print("[ppl_gt]", round(float(ppl_gt), 6))
                if self._is_finite_number(ppl_neg):
                    print("[ppl_neg]", round(float(ppl_neg), 6))
                print("[score]", reward)

        reward_extra_info["max_num_turns"] = torch.as_tensor(
            max_num_turns,
            device=reward_tensor.device,
            dtype=reward_tensor.dtype,
        )
        reward_extra_info["tool_call_success"] = torch.as_tensor(
            tool_call_success,
            device=reward_tensor.device,
            dtype=torch.bool,
        )

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
