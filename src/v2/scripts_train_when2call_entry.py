from __future__ import annotations

import os
import sys
from pathlib import Path

from common.env import run_bash
from common.paths import get_formal_paths


def main() -> int:
    paths = get_formal_paths()
    train_root = paths.src / "v2" / "train"
    if str(paths.src) not in sys.path:
        sys.path.insert(0, str(paths.src))
    if str(train_root.parent) not in sys.path:
        sys.path.insert(0, str(train_root.parent))

    # Keep the original training implementation untouched; only normalize data/output roots.
    data_root = Path(os.environ.get("WHEN2CALL_DATA_ROOT", str(paths.root / "data" / "when2call_official")))
    train_output_root = Path(os.environ.get("TRAIN_OUTPUT_ROOT", str(paths.outputs / "when2call_open")))
    grpo_output_dir = Path(os.environ.get("WHEN2CALL_GRPO_OUTPUT_DIR", str(train_output_root / "grpo")))
    os.environ.setdefault("FORMAL_ROOT", str(paths.root))
    os.environ.setdefault("V2_RESULTS_DIR", str(paths.outputs / "results"))
    os.environ.setdefault("WHEN2CALL_OFFICIAL_DATA_DIR", str(data_root))
    os.environ.setdefault("TRAIN_OUTPUT_ROOT", str(train_output_root))
    os.environ.setdefault("WHEN2CALL_GRPO_OUTPUT_DIR", str(grpo_output_dir))

    if os.environ.get("PREPARE_ONLY", "0").strip() in {"1", "true", "yes"}:
        os.environ["PREPARE_ONLY_AFTER_DATA"] = "1"

    script = paths.root / "scripts" / "train" / "run_when2call_grpo.sh"
    env = dict(os.environ)
    env.update(
        {
            "FORMAL_ROOT": str(paths.root),
            "TRAIN_OUTPUT_ROOT": str(train_output_root),
            "WHEN2CALL_OFFICIAL_DATA_DIR": str(data_root),
            "WHEN2CALL_GRPO_OUTPUT_DIR": str(grpo_output_dir),
        }
    )
    return run_bash(f"bash {script}", workdir=paths.root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
