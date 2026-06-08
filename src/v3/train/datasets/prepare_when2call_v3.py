from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v3.config import get_paths
from v3.train.datasets.build_mixed_dataset import _ensure_jsonable_columns
from v3.train.datasets.v2_alignment import build_when2call_rows


def main() -> None:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description="Convert When2Call preference data into a v3 parquet that is schema-aligned with v2 reward and v3 mixed-data training."
    )
    parser.add_argument(
        "--input",
        default=str(paths.data_dir / "when2call_official" / "train" / "when2call_train_pref.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(paths.when2call_dir / "when2call_v3.parquet"),
    )
    args = parser.parse_args()

    rows = build_when2call_rows(Path(args.input).resolve())
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_jsonable_columns(pd.DataFrame(rows)).to_parquet(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
