"""PPL-aware RewardManager for GRPO.

Usage in verl config:
  reward.reward_manager.source=importlib
  reward.reward_manager.name=PPLRewardManager
  reward.reward_manager.module.path=$ROOT/train/ppl_reward_manager.py

This manager does two things:
1) runtime_uq_value from rollout sequence logprobs (same as before)
2) strict per-sample GT-vs-NEG sequence-level PPL scoring on routed samples:
   - 25%: GT + hard-negative
   - 25%: GT + random-negative
   - 50%: no strict margin term
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
import time
from typing import Any

import torch
import asyncio
from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("ppl_uq")
class PPLRewardManager(RewardLoopManagerBase):
    """NaiveRewardManager + runtime sequence-PPL UQ injection from rollout logprobs."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self._hard_negative_map = {
            "A": "B",  # direct -> tool_call
            "B": "A",  # tool_call -> direct
            "C": "A",  # ask -> direct
            "D": "C",  # cannot -> ask
        }
        self._rand = random.Random(0)
        self._tau = float(os.getenv("REWARD_TAU", "0.10"))
        self._lambda = float(os.getenv("REWARD_LAMBDA", "1.00"))
        self._neg_route_mode = os.getenv("REWARD_NEG_ROUTE_MODE", "default").strip().lower()
        if self._neg_route_mode not in ("default", "none"):
            logger.warning("Unknown REWARD_NEG_ROUTE_MODE=%s, fallback to default", self._neg_route_mode)
            self._neg_route_mode = "default"
        self._group_margin_enabled = os.getenv("REWARD_GROUP_MARGIN_ENABLE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        self._group_margin_timeout_s = float(os.getenv("REWARD_GROUP_MARGIN_TIMEOUT_S", "2.0"))
        self._group_margin_expected = self._infer_group_margin_expected(config)
        self._group_margin_state: dict[str, dict[str, Any]] = {}
        self._strict_model_path = os.getenv("STRICT_PPL_MODEL_PATH", os.getenv("MODEL", ""))
        self._strict_device = os.getenv("STRICT_PPL_DEVICE", "cpu")
        self._strict_enabled = bool(self._strict_model_path)
        self._strict_tokenizer = None
        self._strict_model = None
        if self._strict_enabled:
            try:
                logger.warning("Loading strict PPL scorer model from %s on %s", self._strict_model_path, self._strict_device)
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
    def _infer_group_margin_expected(config) -> int:
        raw = os.getenv("REWARD_GROUP_MARGIN_EXPECTED_SIZE", "").strip()
        if raw:
            try:
                val = int(raw)
                if val > 0:
                    return val
            except ValueError:
                pass
        try:
            val = int(config.actor_rollout_ref.rollout.n)
            if val > 0:
                return val
        except Exception:
            pass
        return 1

    def _compute_seq_ppl_and_uq(self, data_item) -> tuple[float | None, float | None]:
        """Compute sequence-level PPL and normalized UQ from rollout token logprobs.

        Keep consistent with eval `ppl` mode:
        1) sequence PPL from per-token logprobs
        2) normalized to [0, 1] with `ppl_normalized`
        """
        try:
            batch = data_item.batch
            if "rollout_log_probs" not in batch:
                return None, None

            response_ids = batch["responses"]
            response_length = response_ids.shape[-1]
            valid_length = int(batch["attention_mask"][-response_length:].sum())
            rollout_lp = batch["rollout_log_probs"]

            if rollout_lp is None or valid_length <= 0:
                return None, None

            from uq.ppl import ppl_from_token_logprobs, ppl_normalized

            token_logprobs = rollout_lp[:valid_length]
            if hasattr(token_logprobs, "tolist"):
                token_logprobs = token_logprobs.tolist()
            vals = [float(x) for x in token_logprobs]
            if not vals:
                return None, None
            seq_ppl = ppl_from_token_logprobs(vals)
            uq = ppl_normalized(seq_ppl)
            return float(seq_ppl), float(uq)
        except Exception:
            logger.debug("runtime UQ extraction failed", exc_info=True)
            return None, None

    @staticmethod
    def _rollout_key(extra_info: dict, data_item) -> str:
        idx = extra_info.get("idx")
        if idx is not None:
            return f"idx:{idx}"
        q = data_item.non_tensor_batch.get("question", "")
        return f"q:{hashlib.md5(str(q).encode('utf-8')).hexdigest()}"

    @staticmethod
    def _group_key(extra_info: dict, data_item) -> str:
        uid = data_item.non_tensor_batch.get("uid")
        if uid:
            return f"uid:{uid}"
        idx = extra_info.get("idx")
        if idx is not None:
            return f"idx:{idx}"
        q = data_item.non_tensor_batch.get("question", "")
        return f"q:{hashlib.md5(str(q).encode('utf-8')).hexdigest()}"

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
        h = hashlib.md5(key.encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / float(2**64)
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
    def _render_prompt_text(non_tensor_batch: dict) -> str:
        prompt = non_tensor_batch.get("prompt")
        if isinstance(prompt, list):
            lines = []
            for m in prompt:
                if isinstance(m, dict):
                    role = str(m.get("role", "user")).strip()
                    content = str(m.get("content", "")).strip()
                    lines.append(f"{role}: {content}")
            return "\n".join(lines).strip()
        q = non_tensor_batch.get("question")
        return str(q or "").strip()

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
        repl = f"{forced_action}<{forced_letter}>"
        # New unified format: direct_answer<A> / tool_call<B> / request_for_info<C> / cannot_answer<D>
        pat = re.compile(r"\b(direct_answer|tool_call|request_for_info|cannot_answer)\s*<\s*([ABCD])\s*>", re.IGNORECASE)
        if pat.search(text):
            return pat.sub(repl, text, count=1)
        return text

    def _strict_ppl(self, prompt_text: str, forced_letter: str, response_text: str) -> float | None:
        if not self._strict_enabled or self._strict_model is None or self._strict_tokenizer is None:
            return None
        try:
            forced_action = self._action_from_letter(forced_letter)
            if not forced_action:
                return None
            force_inst = (
                f"\n\n[Forced Action]\n"
                f"You MUST output action tag {forced_action}<{forced_letter}> and keep content consistent with this action.\n"
            )
            forced_response = self._force_action_in_response(response_text, forced_action, forced_letter)
            prefix = prompt_text + force_inst
            full = prefix + "\nassistant: " + forced_response

            tok = self._strict_tokenizer
            model = self._strict_model
            full_ids = tok(full, return_tensors="pt", add_special_tokens=True)
            pref_ids = tok(prefix + "\nassistant: ", return_tensors="pt", add_special_tokens=True)

            input_ids = full_ids["input_ids"].to(self._strict_device)
            attention_mask = full_ids.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._strict_device)
            labels = input_ids.clone()
            pref_len = int(pref_ids["input_ids"].shape[1])
            labels[:, :pref_len] = -100

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = float(out.loss.item())
            ppl = float(torch.exp(torch.tensor(loss)).item())
            return ppl
        except Exception:
            logger.debug("strict ppl scoring failed", exc_info=True)
            return None

    @staticmethod
    def _is_finite_number(val: Any) -> bool:
        return isinstance(val, numbers.Number) and torch.isfinite(torch.tensor(float(val))).item()

    def _cleanup_group_margin_state(self) -> None:
        if not self._group_margin_state:
            return
        now = time.time()
        expire_before = now - max(5.0, self._group_margin_timeout_s * 10.0)
        stale_keys = [k for k, v in self._group_margin_state.items() if v.get("created_at", now) < expire_before]
        for key in stale_keys:
            self._group_margin_state.pop(key, None)

    async def _await_group_margin_terms(
        self,
        group_key: str,
        sample_id: str,
        gt_letter: str,
        pred_letter: str | None,
        seq_ppl: float | None,
    ) -> tuple[float | None, float | None, str]:
        if not self._group_margin_enabled or self._group_margin_expected <= 1:
            return None, None, "disabled"

        self._cleanup_group_margin_state()
        state = self._group_margin_state.get(group_key)
        if state is None:
            state = {
                "created_at": time.time(),
                "event": asyncio.Event(),
                "items": {},
            }
            self._group_margin_state[group_key] = state

        state["items"][sample_id] = {
            "gt_letter": gt_letter,
            "pred_letter": (pred_letter or "").strip().upper(),
            "seq_ppl": float(seq_ppl) if seq_ppl is not None else None,
        }

        if len(state["items"]) >= self._group_margin_expected:
            state["event"].set()

        try:
            await asyncio.wait_for(state["event"].wait(), timeout=self._group_margin_timeout_s)
            waited = "complete"
        except asyncio.TimeoutError:
            waited = "timeout"

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

        if state["event"].is_set() or waited == "timeout":
            state["items"].pop(sample_id, None)
            if not state["items"]:
                self._group_margin_state.pop(group_key, None)

        return ppl_gt, ppl_neg, f"group_rollout:{waited}"

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

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

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # 25% hard + 25% random route tagging and margin fields
        from uq.answer_format import extract_mcq_letter_from_text

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
            # Strict per-sample scoring: same data prompt, force GT/NEG action and score sequence PPL.
            prompt_text = self._render_prompt_text(data_item.non_tensor_batch)
            strict_gt = self._strict_ppl(prompt_text, gt_letter, response_str)
            strict_neg = self._strict_ppl(prompt_text, neg_letter, response_str) if neg_letter else None
            if ppl_gt is None and strict_gt is not None:
                ppl_gt = float(strict_gt)
            if ppl_neg is None and strict_neg is not None:
                ppl_neg = float(strict_neg)

        if ppl_gt is None or ppl_neg is None:
            sample_id = str(uuid.uuid4())
            group_ppl_gt, group_ppl_neg, group_source = await self._await_group_margin_terms(
                group_key=group_key,
                sample_id=sample_id,
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

        margin = (float(ppl_neg - ppl_gt) if (ppl_gt is not None and ppl_neg is not None) else 0.0)
        extra_info["neg_type"] = route_type
        extra_info["margin_source"] = margin_source
        extra_info["gt_letter"] = gt_letter
        if neg_letter:
            extra_info["neg_letter"] = neg_letter
        if ppl_gt is not None:
            extra_info["ppl_gt"] = round(float(ppl_gt), 6)
        if ppl_neg is not None:
            extra_info["ppl_neg"] = round(float(ppl_neg), 6)
        extra_info["margin"] = round(float(margin), 6)

        extra_reward_kwargs: dict[str, Any] = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info: dict[str, Any] = {}
        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                # VERL validation computes numpy mean on collected values.
                # Skip None and non-scalar containers to avoid TypeError.
                if value is None:
                    continue
                if isinstance(value, (str, bool, numbers.Number)):
                    reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        if runtime_uq is not None:
            reward_extra_info["runtime_uq_value"] = round(runtime_uq, 6)
        if seq_ppl is not None:
            reward_extra_info["runtime_seq_ppl"] = round(seq_ppl, 6)
        reward_extra_info["neg_type"] = route_type
        reward_extra_info["margin_source"] = margin_source
        reward_extra_info["margin"] = round(float(margin), 6)
        reward_extra_info["tau"] = self._tau
        reward_extra_info["lambda"] = self._lambda
        if ppl_gt is not None:
            reward_extra_info["ppl_gt"] = round(float(ppl_gt), 6)
        if ppl_neg is not None:
            reward_extra_info["ppl_neg"] = round(float(ppl_neg), 6)

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
