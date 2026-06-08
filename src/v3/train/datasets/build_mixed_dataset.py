from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from v3.config import get_paths
from v3.train.datasets.v2_alignment import enrich_cm2_annotations_with_v2_fields

ACTION_LABELS = ("A", "B", "C", "D")


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _ensure_jsonable_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _to_jsonable(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                converted = value.tolist()
            except Exception:
                converted = None
            if converted is not None:
                return _to_jsonable(converted)
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return value.item()
            except Exception:
                pass
        return value

    for column in ("messages", "prompt", "tools", "reward_model", "extra_info"):
        if column in out.columns:
            def _serialize(value: Any) -> Any:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return None
                if isinstance(value, str):
                    return value
                normalized = _to_jsonable(value)
                if isinstance(normalized, (list, dict, tuple)):
                    return json.dumps(normalized, ensure_ascii=False)
                return str(normalized)

            out[column] = out[column].apply(_serialize)
    return out


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _is_cm2_source(source: Any) -> bool:
    src = str(source)
    return (src.startswith("cm2") and src != "cm2_turn_action") or src == "nvidia_nemotron_checklist"


def _cm2_annotation_actions(extra_info: Any) -> list[str]:
    extra = _ensure_dict(extra_info)
    actions: list[str] = []
    for raw_annotation in _ensure_list(extra.get("when2call_annotations")):
        annotation = _ensure_dict(raw_annotation)
        action = str(annotation.get("gt_action") or annotation.get("label") or "").strip().upper()
        if action in ACTION_LABELS:
            actions.append(action)
    return actions


def _has_cm2_annotations(extra_info: Any) -> bool:
    return bool(_cm2_annotation_actions(extra_info))


def _primary_cm2_gt_action(actions: list[str], action_counts: dict[str, int]) -> str:
    unique_actions = sorted(set(actions))
    if not unique_actions:
        return ""
    return min(unique_actions, key=lambda label: (action_counts.get(label, 0), ACTION_LABELS.index(label)))


def _balance_cm2_by_gt_action(df: pd.DataFrame, *, seed: int, shuffle: bool) -> pd.DataFrame:
    """Downsample CM2 rows so A/B buckets match the minimum bucket size among A/B/C/D.

    Rows can contain multiple key-turn annotations. To avoid changing reward logic, each row is assigned to
    the rarest action it contains. Non-CM2 and unannotated rows are preserved unchanged.
    """
    if df.empty or "data_source" not in df.columns or "extra_info" not in df.columns:
        return df

    source_series = df["data_source"].astype(str)
    cm2_mask = source_series.apply(_is_cm2_source)
    if not bool(cm2_mask.any()):
        return df

    action_lists = df.loc[cm2_mask, "extra_info"].apply(_cm2_annotation_actions)
    action_counts = {label: 0 for label in ACTION_LABELS}
    for actions in action_lists:
        for action in actions:
            action_counts[action] += 1

    bucket_by_index: dict[Any, str] = {}
    for idx, actions in action_lists.items():
        bucket = _primary_cm2_gt_action(actions, action_counts)
        if bucket:
            bucket_by_index[idx] = bucket
    if not bucket_by_index:
        return df

    bucket_counts: dict[str, int] = {}
    for label in ACTION_LABELS:
        indices = [idx for idx, bucket in bucket_by_index.items() if bucket == label]
        if not indices:
            continue
        bucket_counts[label] = len(indices)
    if len(bucket_counts) <= 1:
        return df

    target_count = min(bucket_counts.values())
    cm2_annotated_parts: list[pd.DataFrame] = []
    rng_seed = seed
    for label in ACTION_LABELS:
        indices = [idx for idx, bucket in bucket_by_index.items() if bucket == label]
        if not indices:
            continue
        group = df.loc[indices]
        if label in ("A", "B") and len(group) > target_count:
            group = group.sample(
                n=target_count,
                replace=False,
                random_state=rng_seed + ACTION_LABELS.index(label),
            )
        cm2_annotated_parts.append(group)

    annotated_indices = set(bucket_by_index)
    untouched = df.drop(index=list(annotated_indices))
    out = pd.concat([untouched, *cm2_annotated_parts], ignore_index=True)
    if shuffle:
        out = _shuffle_df(out, seed=seed)
    else:
        out = out.reset_index(drop=True)

    balanced_counts = out.loc[
        out["data_source"].apply(_is_cm2_source) & out["extra_info"].apply(lambda value: bool(_cm2_annotation_actions(value))),
        "extra_info",
    ].apply(_cm2_annotation_actions)
    final_annotation_counts = {label: 0 for label in ACTION_LABELS}
    for actions in balanced_counts:
        for action in actions:
            final_annotation_counts[action] += 1
    print(
        "CM2 gt_action balance: "
        f"row_buckets_before={bucket_counts} target_min_abcd_bucket={target_count} "
        f"annotation_counts_before={action_counts} annotation_counts_after={final_annotation_counts}"
    )
    return out


def _keep_cm2_with_annotations_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "data_source" not in df.columns or "extra_info" not in df.columns:
        return df
    source_series = df["data_source"].astype(str)
    keep_mask = source_series.apply(_is_cm2_source) & df["extra_info"].apply(_has_cm2_annotations)
    kept = df.loc[keep_mask].reset_index(drop=True)
    print(f"Kept CM2 rows with when2call annotations: {len(kept)}/{len(df)}")
    return kept


def _strip_when2call_fields_from_extra_info(value: Any) -> Any:
    extra = _ensure_dict(value)
    if not extra:
        return value
    stripped = dict(extra)
    for key in (
        "when2call_annotations",
        "use_uq_reward",
        "uq_target_type",
        "uq_target_turn_idx",
        "uq_target_answer",
        "uq_candidate_answers",
        "uq_reference_answer",
        "uq_reference_action",
    ):
        stripped.pop(key, None)
    return stripped


def _strip_when2call_reward_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "extra_info" in out.columns:
        out["extra_info"] = out["extra_info"].apply(_strip_when2call_fields_from_extra_info)
    return out


def _shrink_extra_info(value: Any) -> Any:
    extra = _ensure_dict(value)
    if not extra:
        return value

    tools = _ensure_list(extra.get("tools"))
    annotations = _ensure_list(extra.get("when2call_annotations"))
    if annotations:
        slim_annotations: list[Any] = []
        for annotation in annotations:
            ann = _ensure_dict(annotation)
            if not ann:
                slim_annotations.append(annotation)
                continue
            ann = dict(ann)
            ann.pop("tools", None)
            slim_annotations.append(ann)
        extra["when2call_annotations"] = slim_annotations
    if tools:
        extra["tools"] = tools
    return extra


def _restore_top_level_tools(row: pd.Series) -> pd.Series:
    out = row.copy()
    top_tools = _ensure_list(out.get("tools"))
    extra = _ensure_dict(out.get("extra_info"))
    extra_tools = _ensure_list(extra.get("tools"))
    if not top_tools and extra_tools:
        out["tools"] = extra_tools
    if extra:
        out["extra_info"] = extra
    return out


def _parse_source_ratios(spec: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    raw = (spec or "").strip()
    if not raw:
        return ratios
    for part in raw.split(","):
        item = part.strip()
        if not item or "=" not in item:
            raise ValueError(f"Invalid source ratio item: {item!r}")
        key, value = item.split("=", 1)
        source = key.strip()
        try:
            ratio = float(value.strip())
        except Exception as exc:
            raise ValueError(f"Invalid ratio for source {source!r}: {value!r}") from exc
        if not source or ratio <= 0:
            raise ValueError(f"Source ratio must use non-empty source and positive value: {item!r}")
        ratios[source] = ratio
    return ratios


def _shuffle_df(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _apply_source_ratios(df: pd.DataFrame, *, ratios: dict[str, float], seed: int, shuffle: bool) -> pd.DataFrame:
    if not ratios or df.empty or "data_source" not in df.columns:
        return df
    source_series = df["data_source"].astype(str)
    available_counts = source_series.value_counts().to_dict()
    alias_groups: dict[str, list[str]] = {}
    for source in ratios:
        if source in available_counts:
            alias_groups[source] = [source]
        elif source == "cm2":
            matched = [name for name in available_counts if name.startswith("cm2") or name == "nvidia_nemotron_checklist"]
            if matched:
                alias_groups[source] = matched
        elif source == "when2call":
            matched = [name for name in available_counts if name.startswith("when2call")]
            if matched:
                alias_groups[source] = matched

    missing = [source for source in ratios if source not in alias_groups]
    if missing:
        raise ValueError(
            f"Requested source ratios for missing data_source values: {missing}. "
            f"Available sources: {sorted(available_counts)}"
        )

    alias_counts = {
        alias: sum(available_counts[name] for name in names)
        for alias, names in alias_groups.items()
    }
    limiting_scale = min(alias_counts[alias] / ratio for alias, ratio in ratios.items())
    target_counts: dict[str, int] = {}
    for alias, ratio in ratios.items():
        alias_target = min(alias_counts[alias], max(1, int(math.floor(limiting_scale * ratio))))
        names = alias_groups[alias]
        remaining = alias_target
        for idx, name in enumerate(names):
            count = available_counts[name]
            if idx == len(names) - 1:
                target_counts[name] = min(count, remaining)
            else:
                share = int(math.floor(alias_target * count / alias_counts[alias]))
                share = min(count, share)
                target_counts[name] = share
                remaining -= share

    sampled_parts: list[pd.DataFrame] = []
    for source, group in df.groupby(source_series, sort=False):
        if source not in target_counts:
            sampled = group.copy()
        else:
            n = target_counts[source]
            if n >= len(group):
                sampled = group.copy()
            else:
                sampled = group.sample(n=n, random_state=seed)
        sampled_parts.append(sampled)

    out = pd.concat(sampled_parts, ignore_index=True)
    return _shuffle_df(out, seed=seed) if shuffle else out.reset_index(drop=True)


def _compute_stratified_val_counts(counts: dict[str, int], val_size: int) -> dict[str, int]:
    total = sum(counts.values())
    if total <= 0 or val_size <= 0:
        return {source: 0 for source in counts}
    val_size = min(val_size, total)
    raw_targets = {source: (count * val_size / total) for source, count in counts.items()}
    allocated = {
        source: min(count, int(math.floor(raw_targets[source])))
        for source, count in counts.items()
    }
    used = sum(allocated.values())
    remainders = sorted(
        ((raw_targets[source] - allocated[source], source) for source in counts),
        reverse=True,
    )
    idx = 0
    while used < val_size and idx < len(remainders):
        _, source = remainders[idx]
        if allocated[source] < counts[source]:
            allocated[source] += 1
            used += 1
        idx += 1
        if idx == len(remainders) and used < val_size:
            idx = 0
    return allocated


def _split_train_val(
    df: pd.DataFrame,
    *,
    val_size: int,
    stratify_by_source: bool,
    seed: int,
    shuffle: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    if not stratify_by_source or "data_source" not in df.columns:
        ordered = _shuffle_df(df, seed=seed) if shuffle else df.reset_index(drop=True)
        clipped_val_size = max(0, min(val_size, len(ordered)))
        return ordered.iloc[clipped_val_size:].copy(), ordered.iloc[:clipped_val_size].copy()

    ordered = _shuffle_df(df, seed=seed) if shuffle else df.reset_index(drop=True)
    counts = ordered["data_source"].astype(str).value_counts().to_dict()
    val_counts = _compute_stratified_val_counts(counts, max(0, min(val_size, len(ordered))))

    val_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []
    for source, group in ordered.groupby(ordered["data_source"].astype(str), sort=False):
        n_val = val_counts.get(source, 0)
        val_parts.append(group.iloc[:n_val].copy())
        train_parts.append(group.iloc[n_val:].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ordered.iloc[0:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else ordered.iloc[0:0].copy()
    if shuffle:
        train_df = _shuffle_df(train_df, seed=seed + 1)
        val_df = _shuffle_df(val_df, seed=seed + 2)
    return train_df, val_df


def main() -> None:
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Combine When2Call data with CM2 trajectory parquet, optionally when2call-augmented.")
    parser.add_argument("--cm2", "--cm2-file", dest="cm2", required=True, help="Prepared CM2 trajectory parquet path")
    parser.add_argument("--when2call", "--when2call-file", dest="when2call", default=str(paths.when2call_dir / "when2call_v3.parquet"))
    parser.add_argument("--include-when2call", action="store_true")
    parser.add_argument("--include-when2call-value", dest="include_when2call_value", default=None)
    parser.add_argument("--enable-uq", action="store_true")
    parser.add_argument("--enable-uq-value", dest="enable_uq_value", default=None)
    parser.add_argument("--output", "--output-file", dest="output", default=str(paths.mixed_dir / "mixed_v3.parquet"))
    parser.add_argument("--val-output", "--val-output-file", dest="val_output", default=str(paths.mixed_dir / "mixed_v3_val.parquet"))
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--mixed-shuffle", dest="mixed_shuffle_value", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-seed", dest="mixed_seed_value", default=None)
    parser.add_argument(
        "--stratify-by-source",
        action="store_true",
        help="Split validation set by data_source proportion instead of taking the first rows.",
    )
    parser.add_argument("--mixed-stratify-by-source", dest="mixed_stratify_by_source_value", default=None)
    parser.add_argument(
        "--source-ratios",
        default="",
        help="Optional relative downsampling ratios by data_source, e.g. 'cm2=3,when2call_pref=1'.",
    )
    parser.add_argument("--mixed-source-ratios", dest="mixed_source_ratios_value", default=None)
    parser.add_argument(
        "--balance-cm2-gt-action",
        action="store_true",
        help="Oversample CM2 rows by rarest when2call key-turn gt_action bucket before source-ratio sampling.",
    )
    parser.add_argument("--mixed-balance-cm2-gt-action", dest="mixed_balance_cm2_gt_action_value", default=None)
    parser.add_argument(
        "--cm2-annotated-only",
        action="store_true",
        help="Keep only CM2 rows that have when2call_annotations.",
    )
    parser.add_argument("--cm2-annotated-only-value", dest="cm2_annotated_only_value", default=None)
    parser.add_argument(
        "--strip-when2call-reward-fields",
        action="store_true",
        help="Remove when2call annotation/UQ reward fields from extra_info after optional filtering/balancing.",
    )
    parser.add_argument("--strip-when2call-reward-fields-value", dest="strip_when2call_reward_fields_value", default=None)
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Write the full mixed dataframe to --output without splitting train/val.",
    )
    args = parser.parse_args()

    include_when2call = args.include_when2call or _bool_from_any(args.include_when2call_value)
    enable_uq = args.enable_uq or _bool_from_any(args.enable_uq_value)
    shuffle = args.shuffle or _bool_from_any(args.mixed_shuffle_value)
    stratify_by_source = args.stratify_by_source or _bool_from_any(args.mixed_stratify_by_source_value)
    balance_cm2_gt_action = args.balance_cm2_gt_action or _bool_from_any(args.mixed_balance_cm2_gt_action_value)
    cm2_annotated_only = args.cm2_annotated_only or _bool_from_any(args.cm2_annotated_only_value)
    strip_when2call_reward_fields = args.strip_when2call_reward_fields or _bool_from_any(
        args.strip_when2call_reward_fields_value
    )
    seed = int(args.mixed_seed_value) if args.mixed_seed_value is not None else args.seed
    source_ratios_spec = (
        args.mixed_source_ratios_value if args.mixed_source_ratios_value is not None else args.source_ratios
    )

    cm2_df = _load_df(Path(args.cm2).resolve())
    cm2_df = enrich_cm2_annotations_with_v2_fields(cm2_df)
    frames = [cm2_df]
    if include_when2call:
        frames.append(_load_df(Path(args.when2call).resolve()))

    df = pd.concat(frames, ignore_index=True)
    if "extra_info" in df.columns:
        df["extra_info"] = df["extra_info"].apply(_shrink_extra_info)
    df = df.apply(_restore_top_level_tools, axis=1)
    if cm2_annotated_only:
        df = _keep_cm2_with_annotations_only(df)
    if balance_cm2_gt_action:
        df = _balance_cm2_by_gt_action(df, seed=seed, shuffle=shuffle)
    source_ratios = _parse_source_ratios(source_ratios_spec)
    if source_ratios:
        df = _apply_source_ratios(df, ratios=source_ratios, seed=seed, shuffle=shuffle)
    if enable_uq and "extra_info" in df.columns:
        def _enable_uq(value: Any) -> Any:
            cloned = _ensure_dict(value)
            if not cloned:
                return value
            if cloned.get("uq_target_type"):
                cloned["use_uq_reward"] = True
            annotations = cloned.get("when2call_annotations")
            if isinstance(annotations, list):
                updated_annotations = []
                for annotation in annotations:
                    if isinstance(annotation, dict):
                        ann = dict(annotation)
                        ann["use_uq_reward"] = True
                        updated_annotations.append(ann)
                    else:
                        updated_annotations.append(annotation)
                cloned["when2call_annotations"] = updated_annotations
                if updated_annotations:
                    cloned["use_uq_reward"] = True
            return cloned
        df["extra_info"] = df["extra_info"].apply(_enable_uq)
    if strip_when2call_reward_fields:
        df = _strip_when2call_reward_fields(df)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_split:
        _ensure_jsonable_columns(df).to_parquet(output_path, index=False)
        print(f"Wrote full={len(df)} to {output_path}")
        if "data_source" in df.columns:
            print(f"Full source counts: {df['data_source'].astype(str).value_counts().to_dict()}")
        return

    train_df, val_df = _split_train_val(
        df,
        val_size=args.val_size,
        stratify_by_source=stratify_by_source,
        seed=seed,
        shuffle=shuffle,
    )

    val_output_path = Path(args.val_output).resolve()
    val_output_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_jsonable_columns(train_df).to_parquet(output_path, index=False)
    _ensure_jsonable_columns(val_df).to_parquet(val_output_path, index=False)
    print(f"Wrote train={len(train_df)} to {output_path}")
    print(f"Wrote val={len(val_df)} to {val_output_path}")
    if "data_source" in df.columns:
        print(f"Final source counts: {df['data_source'].astype(str).value_counts().to_dict()}")
        print(f"Train source counts: {train_df['data_source'].astype(str).value_counts().to_dict()}")
        print(f"Val source counts: {val_df['data_source'].astype(str).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
