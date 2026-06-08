"""Custom RLHF dataset for When2Call GRPO: parses extra_info and prompt from JSON strings.

HuggingFace datasets loads parquet with dict/list columns as JSON strings.
This subclass wraps the dataframe to parse these columns on access.
"""
import json
import os
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


def _normalize_token_ids(tokenized_prompt: Any) -> list[int]:
    """Normalize tokenizer/chat-template outputs to a flat list of token ids."""
    value = tokenized_prompt
    if isinstance(value, dict):
        value = value.get("input_ids", [])
    elif hasattr(value, "input_ids"):
        value = getattr(value, "input_ids")

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:
            pass

    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]

    if isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list):
        return []

    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            return []
    return out


def _ensure_dict(val: Any) -> dict:
    """Parse JSON string to dict if needed; return empty dict for None."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}
    return {}


def _ensure_list(val: Any) -> list:
    """Parse JSON string to list if needed; return empty list for None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _parse_row(row) -> dict:
    """Parse JSON string columns in a row to dict/list. Support mixed reward fields."""
    if not isinstance(row, dict):
        return row
    out = dict(row)
    if "extra_info" in out:
        ei = _ensure_dict(out["extra_info"])
        if "idx" in ei and "index" not in ei:
            ei["index"] = ei["idx"]
        out["extra_info"] = ei
    if "prompt" in out:
        out["prompt"] = _ensure_list(out["prompt"])
    if "reward_model" in out and isinstance(out["reward_model"], str):
        out["reward_model"] = _ensure_dict(out["reward_model"])
    return out


class _ParsedDataset:
    """Wrapper that parses extra_info, prompt, reward_model when they are JSON strings."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, item):
        return _parse_row(self._dataset[item])

    def __len__(self):
        return len(self._dataset)

    def select(self, indices):
        return _ParsedDataset(self._dataset.select(indices))

    def filter(self, *args, **kwargs):
        return _ParsedDataset(self._dataset.filter(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._dataset, name)


class When2CallGRPODataset(RLHFDataset):
    """RLHFDataset that parses extra_info and prompt from JSON when loaded as strings."""

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        filter_limit = self.max_prompt_length
        response_budget = 0
        try:
            response_budget = int(self.config.get("max_response_length", 0) or 0)
        except Exception:
            response_budget = 0
        env_limits = []
        for key in ("ROLLOUT_MAX_MODEL_LEN", "ROLLOUT_MAX_BATCHED_TOKENS"):
            raw = os.environ.get(key)
            if not raw:
                continue
            try:
                env_limits.append(int(raw))
            except ValueError:
                continue
        if env_limits:
            rollout_limit = min(env_limits)
            if response_budget > 0:
                rollout_limit = max(1, rollout_limit - response_budget)
            filter_limit = min(filter_limit, rollout_limit)

        if filter_limit < self.max_prompt_length:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key

            def parsed_doc2len(doc) -> int:
                try:
                    parsed = _parse_row(doc)
                    apply_kwargs = dict(**self.apply_chat_template_kwargs)
                    if self.tool_schemas is not None:
                        apply_kwargs["tools"] = self.tool_schemas
                    apply_kwargs.pop("tokenize", None)
                    apply_kwargs.pop("return_dict", None)
                    apply_kwargs.pop("return_tensors", None)
                    tokenized_prompt = tokenizer.apply_chat_template(
                        parsed[prompt_key], add_generation_prompt=True, tokenize=True, **apply_kwargs
                    )
                    return len(_normalize_token_ids(tokenized_prompt))
                except Exception:
                    return filter_limit + 1

            self.dataframe = self.dataframe.filter(
                lambda doc: parsed_doc2len(doc) <= filter_limit,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than rollout limit {filter_limit} tokens",
            )

        # Wrap after full init so __getitem__ and split() receive parsed rows.
        # Must wrap before any DataLoader access; filter already ran on raw data
        # (our prepare script has short prompts, so filter keeps all).
        self.dataframe = _ParsedDataset(self.dataframe)
