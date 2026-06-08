from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import load_config
from training.dispatch import run_training_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified runner for TRUST open-source code")
    parser.add_argument("mode", choices=["train"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == "train":
        return run_training_job(cfg)
    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
