"""Runtime PPL→UQ utilities for GRPO reward.

Core function: from a single token's logprob (the chosen A/B/C/D letter),
approximate the 4-option PPL distribution and convert to UQ ∈ [0, 1].

Approximation: given only p_chosen = exp(logprob), assume the remaining
probability mass (1 − p_chosen) distributes uniformly among the other 3 options.
Then compute entropy → PPL → normalized UQ.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch


def uq_from_chosen_logprob(logprob: float, num_options: int = 4) -> float:
    """Approximate UQ from a single chosen-token logprob.

    Returns 0 (fully confident) to 1 (fully uncertain / uniform).
    """
    p_chosen = min(1.0, max(1e-10, math.exp(float(logprob))))
    p_other = max(0.0, (1.0 - p_chosen) / max(1, num_options - 1))

    entropy = 0.0
    if p_chosen > 1e-30:
        entropy -= p_chosen * math.log(p_chosen)
    if p_other > 1e-30:
        entropy -= (num_options - 1) * p_other * math.log(p_other)

    ppl = math.exp(entropy)
    return max(0.0, min(1.0, (ppl - 1.0) / max(1, num_options - 1)))


@lru_cache(maxsize=4)
def _letter_token_ids(tokenizer_name_or_path: str) -> dict[int, str]:
    """Build {token_id: letter} mapping — cached per tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return _letter_token_ids_from_tokenizer(tok)


def _letter_token_ids_from_tokenizer(tokenizer) -> dict[int, str]:
    """Build {token_id: letter} from a live tokenizer object."""
    mapping: dict[int, str] = {}
    for letter in "ABCD":
        for variant in (letter, f" {letter}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            tid = ids[-1]
            decoded = tokenizer.decode([tid]).strip()
            if decoded == letter:
                mapping[tid] = letter
    return mapping


def find_last_letter_position(
    response_ids: torch.Tensor,
    tokenizer,
    *,
    valid_length: Optional[int] = None,
) -> Optional[tuple[int, str]]:
    """Find the last A/B/C/D token in *response_ids*.

    Returns ``(position, letter)`` or ``None``.
    ``position`` is the index within *response_ids* (0-based).
    """
    mapping = _letter_token_ids_from_tokenizer(tokenizer)
    if not mapping:
        return None

    ids = response_ids.tolist() if isinstance(response_ids, torch.Tensor) else list(response_ids)
    n = valid_length if valid_length is not None else len(ids)

    for pos in range(n - 1, -1, -1):
        letter = mapping.get(ids[pos])
        if letter is not None:
            return (pos, letter)
    return None


def extract_runtime_uq(
    response_ids: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    tokenizer,
    valid_response_length: int,
) -> Optional[float]:
    """End-to-end: from rollout tensors → runtime UQ value or None."""
    result = find_last_letter_position(
        response_ids, tokenizer, valid_length=valid_response_length
    )
    if result is None:
        return None

    pos, _letter = result
    logprob = float(rollout_log_probs[pos])
    return uq_from_chosen_logprob(logprob)
