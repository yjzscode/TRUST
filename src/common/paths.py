from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FormalPaths:
    root: Path
    src: Path
    configs: Path
    scripts: Path
    outputs: Path
    logs: Path
    third_party: Path
    data: Path


def get_formal_paths() -> FormalPaths:
    root = Path(__file__).resolve().parents[2]
    return FormalPaths(
        root=root,
        src=root / "src",
        configs=root / "configs",
        scripts=root / "scripts",
        outputs=root / "outputs",
        logs=root / "logs",
        third_party=root / "third_party",
        data=root / "data",
    )
