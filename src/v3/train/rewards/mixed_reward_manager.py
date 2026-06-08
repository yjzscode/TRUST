from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import json

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.checklist import ChecklistRewardManager

from v3.train.rewards.trajectory_when2call_reward import compute_trajectory_when2call_reward
from v3.train.rewards.v2_when2call_adapter import score_with_v2_when2call_reward
from v3.train.rewards.v2_when2call_runtime import get_v2_when2call_runtime


_DEBUG_REWARD = str(__import__("os").getenv("V3_DEBUG_REWARD", "1")).strip().lower() not in {"0", "false", "no"}


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
        except Exception:
            converted = None
        if converted is not None:
            return _json_safe(converted)
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


def _to_batch_safe_value(value: Any) -> Any:
    normalized = _json_safe(value)
    if isinstance(normalized, (list, dict)):
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return normalized


def _is_cm2_like_source(source: Any) -> bool:
    source_str = str(source)
    return (source_str.startswith("cm2") and source_str != "cm2_turn_action") or source_str == "nvidia_nemotron_checklist"


@register("mixed")
class MixedRewardManager(AbstractRewardManager):
    """Route CM2 checklist samples and v3 next-action samples to different reward logic."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs: Any) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self._cm2_when2call_reward_weight = float(kwargs.pop("cm2_when2call_reward_weight", 1.0))
        self._when2call_runtime = get_v2_when2call_runtime()
        self._debug_calls = 0
        self._checklist_manager = ChecklistRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            **kwargs,
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        batch_size = len(data_sources)
        reward_extra_info: dict[str, list[Any]] = {}

        def ensure_batch_key(key: str, fill_value: Any = None) -> list[Any]:
            if key not in reward_extra_info:
                reward_extra_info[key] = [fill_value] * batch_size
            return reward_extra_info[key]

        max_num_turns = torch.ones(batch_size, dtype=reward_tensor.dtype, device=reward_tensor.device)
        tool_call_success = torch.ones(batch_size, dtype=torch.bool, device=reward_tensor.device)

        cm2_mask = np.array([_is_cm2_like_source(src) for src in data_sources], dtype=bool)
        action_mask = np.logical_not(cm2_mask)

        cm2_annotation_count = 0
        if cm2_mask.any() and "extra_info" in data.non_tensor_batch:
            for global_idx in np.where(cm2_mask)[0].tolist():
                extra_info = _coerce_dict(data.non_tensor_batch["extra_info"][global_idx])
                cm2_annotation_count += len(_coerce_list(extra_info.get("when2call_annotations")))
        self._debug_calls += 1
        if _DEBUG_REWARD and self._debug_calls <= 20:
            print(
                "[mixed reward] "
                f"batch={batch_size} cm2={int(cm2_mask.sum())} action={int(action_mask.sum())} "
                f"cm2_annotations={cm2_annotation_count} judge_enabled={self._when2call_runtime.judge_enabled}"
            )

        if cm2_mask.any():
            cm2_data = data.select_idxs(cm2_mask)
            cm2_result = self._checklist_manager(cm2_data, return_dict=True)
            cm2_reward_tensor = cm2_result["reward_tensor"]
            cm2_extra_info = cm2_result.get("reward_extra_info", {})
            cm2_global_indices = np.where(cm2_mask)[0].tolist()

            turn_end_tensors = cm2_extra_info.get("turn_end_tensor")
            cm2_max_num_turns = cm2_extra_info.get("max_num_turns")
            if cm2_max_num_turns is not None:
                cm2_max_num_turns = torch.as_tensor(
                    cm2_max_num_turns, device=reward_tensor.device, dtype=reward_tensor.dtype
                )
                max_num_turns[cm2_global_indices] = cm2_max_num_turns

            cm2_tool_call_success = cm2_extra_info.get("tool_call_success")
            if cm2_tool_call_success is not None:
                cm2_tool_call_success = torch.as_tensor(
                    cm2_tool_call_success, device=reward_tensor.device, dtype=torch.bool
                )
                tool_call_success[cm2_global_indices] = cm2_tool_call_success

            for local_idx, global_idx in enumerate(cm2_global_indices):
                reward_slice = cm2_reward_tensor[local_idx]
                if turn_end_tensors is not None:
                    extra_info = _coerce_dict(cm2_data.non_tensor_batch.get("extra_info", [])[local_idx])
                    when2call_annotations = extra_info.get("when2call_annotations", [])
                    if when2call_annotations:
                        aug_reward, aug_details = compute_trajectory_when2call_reward(
                            cm2_data.non_tensor_batch.get("messages", [])[local_idx],
                            when2call_annotations,
                            turn_end_tensors[local_idx],
                            runtime=self._when2call_runtime,
                            default_use_uq_reward=bool(extra_info.get("use_uq_reward", False)),
                            reward_weight=self._cm2_when2call_reward_weight,
                        )
                        reward_slice = reward_slice + aug_reward.to(reward_slice.device, dtype=reward_slice.dtype)
                        for key, value in aug_details.items():
                            ensure_batch_key(f"cm2_when2call_{key}")[global_idx] = _to_batch_safe_value(value)
                reward_tensor[global_idx] = reward_slice

            for key, values in cm2_result.get("reward_extra_info", {}).items():
                batch_values = ensure_batch_key(f"cm2_{key}")
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().tolist()
                elif isinstance(values, np.ndarray):
                    values = values.tolist()
                elif isinstance(values, tuple):
                    values = list(values)
                if isinstance(values, list) and len(values) == len(cm2_global_indices):
                    for local_idx, global_idx in enumerate(cm2_global_indices):
                        batch_values[global_idx] = _to_batch_safe_value(values[local_idx])

        for idx in np.where(action_mask)[0].tolist():
            data_item = data[idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch.get("response_mask")
            rollout_log_probs = data_item.batch.get("rollout_log_probs")
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_ids_list = [int(token_id) for token_id in valid_response_ids.detach().cpu().tolist()]
            response_mask_list = None
            if response_mask is not None:
                response_mask_list = [int(mask) for mask in response_mask[:valid_response_length].detach().cpu().tolist()]
            rollout_log_probs_list = None
            if rollout_log_probs is not None:
                rollout_log_probs_list = [
                    float(log_prob) for log_prob in rollout_log_probs[:valid_response_length].detach().cpu().tolist()
                ]

            reward_model = _coerce_dict(data_item.non_tensor_batch.get("reward_model"))
            ground_truth = reward_model.get("ground_truth", "")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "")
            extra_info = _coerce_dict(data_item.non_tensor_batch.get("extra_info"))
            prompt_messages = _coerce_list(data_item.non_tensor_batch.get("messages"))
            if not prompt_messages:
                prompt_messages = _coerce_list(data_item.non_tensor_batch.get("prompt"))
            rollout_key = str(extra_info.get("rollout_key") or extra_info.get("idx") or idx)
            if extra_info.get("is_negative_augmented"):
                enriched_extra = dict(extra_info)
            else:
                enriched_extra = self._when2call_runtime.inject_runtime_fields(
                    prompt_messages=prompt_messages,
                    response_text=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    rollout_key=rollout_key,
                    response_ids=response_ids_list,
                    response_logprobs=rollout_log_probs_list,
                    response_mask=response_mask_list,
                )
            result = score_with_v2_when2call_reward(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=enriched_extra,
            )
            reward_tensor[idx, max(valid_response_length - 1, 0)] = float(result.get("score", 0.0))
            for key, value in result.items():
                if isinstance(value, (str, bool, int, float)):
                    ensure_batch_key(key)[idx] = value

        reward_extra_info["max_num_turns"] = max_num_turns
        reward_extra_info["tool_call_success"] = tool_call_success

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
