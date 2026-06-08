#!/usr/bin/env bash
set -euo pipefail
FORMAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FORMAL_ROOT"
echo "[quickstart] train v2 When2Call with configs/train/when2call_full_no_neg_rollout.yaml"
bash "$FORMAL_ROOT/run_job.sh" train "configs/train/when2call_full_no_neg_rollout.yaml"
