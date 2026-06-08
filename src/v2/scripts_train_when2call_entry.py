from __future__ import annotations

import os
import sys
from pathlib import Path

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
    os.environ.setdefault("FORMAL_ROOT", str(paths.root))
    os.environ.setdefault("V2_RESULTS_DIR", str(paths.outputs / "results"))
    os.environ.setdefault("WHEN2CALL_OFFICIAL_DATA_DIR", str(data_root))

    if os.environ.get("PREPARE_ONLY", "0").strip() in {"1", "true", "yes"}:
        os.environ["PREPARE_ONLY_AFTER_DATA"] = "1"

    script = paths.root / "scripts" / "train" / "run_when2call_grpo.sh"
    return os.system(f"bash {script}") >> 8


if __name__ == "__main__":
    raise SystemExit(main())
