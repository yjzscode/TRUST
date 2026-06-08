from __future__ import annotations

import os
from pathlib import Path

from common.env import run_bash
from common.paths import get_formal_paths


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    paths = get_formal_paths()
    script = paths.root / "scripts" / "train" / "run_v3_cm2_aug_grpo.sh"
    train_output_root = Path(
        os.environ.get(
            "TRAIN_OUTPUT_ROOT",
            "/mnt/shared-storage-user/ai4good1-share/zhouyijin/uq2as_formal",
        )
    )
    # Keep full inherited env from common.runner exports (including SGLANG_URL/JUDGE_MODEL_NAME/etc),
    # then override a few workflow-local defaults.
    env = dict(os.environ)
    env.update(
        {
        "FORMAL_ROOT": paths.root,
        "TRAIN_OUTPUT_ROOT": str(train_output_root),
        "V3_CM2_ROOT": str(paths.src / "cm2_core"),
        "V3_V2_ROOT": str(paths.src / "v2"),
        "CHECKPOINT_ROOT": os.environ.get("CHECKPOINT_ROOT", str(train_output_root / "checkpoints" / "v3_grpo")),
        }
    )
    if _bool_env("BUILD_DATA_ONLY"):
        env["BUILD_DATA_ONLY"] = "1"
    return run_bash(f"bash {script}", workdir=paths.root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
