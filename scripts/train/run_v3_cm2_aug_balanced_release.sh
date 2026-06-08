#!/usr/bin/env bash
set -euo pipefail
FORMAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FORMAL_ROOT"
echo "[quickstart] train v3 unified TRUST with configs/train/v3_train_cm2_aug_balanced_no_neg.yaml"
bash "$FORMAL_ROOT/run_job.sh" train "configs/train/v3_train_cm2_aug_balanced_no_neg.yaml"
