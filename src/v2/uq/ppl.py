from __future__ import annotations

import math
from typing import Iterable, Optional


def ppl_from_token_logprobs(token_logprobs: Iterable[float]) -> float:
    """Perplexity from per-token log-probabilities (natural log).

    PPL = exp(mean(-log p_t))
    """
    vals = [float(x) for x in token_logprobs]
    if not vals:
        return float("nan")
    nll = sum(-x for x in vals) / len(vals)
    return float(math.exp(nll))


def ppl_from_option_logprobs(logprobs_abcd: Iterable[float]) -> float:
    """PPL of 4-option MCQ distribution: softmax(logprobs) then exp(entropy).

    Returns 1 when confident (one option dominates), up to 4 when uniform.
    """
    lp = [float(x) for x in logprobs_abcd]
    if len(lp) < 4:
        return float("nan")
    m = max(lp)
    exps = [math.exp(x - m) for x in lp]
    s = sum(exps)
    if s <= 0:
        return float("nan")
    probs = [e / s for e in exps]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return float(math.exp(entropy))


def ppl_normalized(ppl: float, num_options: int = 4) -> float:
    """将 PPL 归一化到 [0,1]：(ppl - 1) / (num_options - 1)。1→0（确定），4→1（均匀）。"""
    if math.isnan(ppl) or ppl <= 1:
        return 0.0
    return min(1.0, (ppl - 1) / (num_options - 1))


def uq_from_ppl(ppl: float, num_options: int = 4) -> float:
    """Normalize PPL to [0,1] UQ: (ppl - 1) / (num_options - 1)."""
    return ppl_normalized(ppl, num_options)


if __name__ == "__main__":
    # If p=0.5 for each token, log p = ln(0.5) ~ -0.693 -> PPL = exp(0.693)=2
    lp = [math.log(0.5)] * 10
    ppl = ppl_from_token_logprobs(lp)
    print("ppl:", ppl)
    assert 1.9 < ppl < 2.1

