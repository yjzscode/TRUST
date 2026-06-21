#!/usr/bin/env bash
set -euo pipefail
RUN_JOB_START_DELAY="${RUN_JOB_START_DELAY:-0}"
if [[ "$RUN_JOB_START_DELAY" != "0" ]]; then
  echo "[run_job] sleeping ${RUN_JOB_START_DELAY}s before start"
  sleep "$RUN_JOB_START_DELAY"
fi
FORMAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$FORMAL_ROOT/src:$FORMAL_ROOT/third_party/verl_v0.6.1_checklist:$FORMAL_ROOT/third_party:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/trust_ray}"
MODE="${1:?usage: bash run_job.sh <train> <config.yaml>}"
CONFIG_INPUT="${2:?usage: bash run_job.sh <train> <config.yaml>}"
if [[ "$CONFIG_INPUT" = /* ]]; then
  CONFIG="$CONFIG_INPUT"
elif [[ -f "$CONFIG_INPUT" ]]; then
  CONFIG="$(cd "$(dirname "$CONFIG_INPUT")" && pwd)/$(basename "$CONFIG_INPUT")"
elif [[ -f "$FORMAL_ROOT/$CONFIG_INPUT" ]]; then
  CONFIG="$FORMAL_ROOT/$CONFIG_INPUT"
else
  CONFIG="$CONFIG_INPUT"
fi
cd "$FORMAL_ROOT"
echo "[run_job] mode=$MODE config=$CONFIG root=$FORMAL_ROOT python=$(command -v python3)"
python3 -m common.runner "$MODE" "$CONFIG"
