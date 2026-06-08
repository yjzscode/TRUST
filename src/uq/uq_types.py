"""UQ value types (avoid naming 'types' to not shadow stdlib)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UQValues:
    """Generic container for UQ outputs used by v2 prompt tests.

    internal_uq: current-step uncertainty (0..1, higher = more uncertain)
    external_uq: propagated / history uncertainty (0..1)
    details: method-specific raw values
    """

    internal_uq: float
    external_uq: float
    source: str
    next_action_suggestion: str
    details: dict[str, Any]


def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.5
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


if __name__ == "__main__":
    u = UQValues(0.2, 0.7, "demo", "clarify", {"ppl": 12.3})
    assert 0 <= u.internal_uq <= 1
    assert clamp01(-1) == 0.0 and clamp01(2) == 1.0
    print("uq_types ok")
