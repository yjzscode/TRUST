#!/usr/bin/env bash
set -euo pipefail
FORMAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FORMAL_ROOT"
echo "[quickstart] build v3 annotation/mixed data with configs/train/v3_build_cm2_aug_balanced.yaml"
bash "$FORMAL_ROOT/run_job.sh" train "configs/train/v3_build_cm2_aug_balanced.yaml"
