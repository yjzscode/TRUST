from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from common.paths import get_formal_paths

_FORMAL = get_formal_paths()
ROOT = _FORMAL.src / "v3"
UQ_ROOT = _FORMAL.root
CM2_ROOT = Path(
    os.getenv(
        "V3_CM2_ROOT",
        str(_FORMAL.src / "cm2_core"),
    )
)
V2_ROOT = Path(
    os.getenv(
        "V3_V2_ROOT",
        str(_FORMAL.src / "v2"),
    )
)


@dataclass(frozen=True)
class V3Paths:
    root: Path = ROOT
    cm2_root: Path = CM2_ROOT
    v2_root: Path = V2_ROOT
    data_dir: Path = UQ_ROOT / "data"
    mixed_dir: Path = UQ_ROOT / "data" / "mixed"
    when2call_dir: Path = UQ_ROOT / "data" / "when2call"
    cm2_augmented_dir: Path = UQ_ROOT / "data" / "cm2_augmented"
    cm2_turn_action_dir: Path = UQ_ROOT / "data" / "cm2_turn_action"
    checkpoints_dir: Path = UQ_ROOT / "outputs" / "checkpoints"


def get_paths() -> V3Paths:
    return V3Paths()
