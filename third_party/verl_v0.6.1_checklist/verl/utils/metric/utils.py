# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics utils.
"""

from typing import Any

import numpy as np


def _reduce_metric_values(values: list[Any], reduce_op: str) -> Any:
    """Robustly reduce scalar-like metric values collected from multiple workers.

    Some metrics are logged as Python scalars while others may arrive as numpy arrays,
    torch scalars converted to arrays, or short lists. We flatten each item to numeric
    scalars first, then apply the requested reduction across the merged values.
    """
    flattened: list[float] = []
    for value in values:
        if value is None:
            continue

        arr = np.asarray(value)
        if arr.dtype == object:
            arr = np.asarray(arr.tolist(), dtype=np.float64)
        else:
            arr = arr.astype(np.float64, copy=False)

        if arr.ndim == 0:
            flattened.append(float(arr.item()))
        else:
            flattened.extend(arr.reshape(-1).tolist())

    if len(flattened) == 0:
        return np.nan

    if reduce_op == "max":
        return float(np.nanmax(flattened))
    if reduce_op == "min":
        return float(np.nanmin(flattened))
    return float(np.nanmean(flattened))


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = _reduce_metric_values(val, reduce_op="max")
        elif "min" in key:
            metrics[key] = _reduce_metric_values(val, reduce_op="min")
        else:
            metrics[key] = _reduce_metric_values(val, reduce_op="mean")
    return metrics
