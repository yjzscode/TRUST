#!/usr/bin/env bash
set -euo pipefail
source /home/zhouyijin/miniconda3/etc/profile.d/conda.sh
conda activate uq2as_train
FORMAL_ROOT="${FORMAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-/mnt/shared-storage-user/ai4good1-share/zhouyijin/uq2as_formal}"
MCP_SERVER_PID=""
export PYTHONPATH="$FORMAL_ROOT/src:$FORMAL_ROOT/third_party/verl_v0.6.1_checklist:$FORMAL_ROOT/third_party:${PYTHONPATH:-}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TRAIN_OUTPUT_ROOT/tensorboard/${PROJECT_NAME:-v3_agent_uq}/${EXPERIMENT_NAME:-stage1_cm2_aug_only_balanced}}"
export CHECKLIST_SOFT_FAIL_ON_REWARD_CONNECT="${CHECKLIST_SOFT_FAIL_ON_REWARD_CONNECT:-1}"
cd "$FORMAL_ROOT"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[v3_cm2_aug_grpo] unset PYTORCH_CUDA_ALLOC_CONF for SGLang compatibility: ${PYTORCH_CUDA_ALLOC_CONF}"
  unset PYTORCH_CUDA_ALLOC_CONF
fi

# v3 has several consumers for the same judge endpoint/model:
# reward_kwargs uses SGLANG_URL/JUDGE_MODEL_NAME, while multi-turn interaction
# config uses JUDGE_API_URL/LLM_AS_A_JUDGE_NAME. Keep them in sync for direct
# `run_job.sh train ...` runs that do not go through submit-time URL rewriting.
export JUDGE_API_URL="${JUDGE_API_URL:-${SGLANG_URL:?set SGLANG_URL}}"
export LLM_AS_A_JUDGE_NAME="${LLM_AS_A_JUDGE_NAME:-${JUDGE_MODEL_NAME:?set JUDGE_MODEL_NAME}}"

judge_base_url="${JUDGE_BASE_URL:-${SGLANG_URL%/chat/completions}}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-$judge_base_url}"
export V3_JUDGE_BASE_URL="${V3_JUDGE_BASE_URL:-$judge_base_url}"
export V3_JUDGE_MODEL="${V3_JUDGE_MODEL:-$JUDGE_MODEL_NAME}"
export STRICT_PPL_BASE_URL="${STRICT_PPL_BASE_URL:-$judge_base_url}"
export STRICT_PPL_MODEL="${STRICT_PPL_MODEL:-$JUDGE_MODEL_NAME}"
export V3_JUDGE_API_KEY="${V3_JUDGE_API_KEY:-${JUDGE_API_KEY:-${LABEL_API_KEY:-${OPENAI_API_KEY:-EMPTY}}}}"
export STRICT_PPL_API_KEY="${STRICT_PPL_API_KEY:-$V3_JUDGE_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$V3_JUDGE_API_KEY}"
export REWARD_DROP_COMPONENTS="${REWARD_DROP_COMPONENTS:-r_fmt}"

enable_proxy_for_https_endpoints() {
  local needs_proxy=0
  for endpoint in "${SGLANG_URL:-}" "${LABEL_BASE_URL:-}"; do
    case "$endpoint" in
      https:*) needs_proxy=1 ;;
    esac
  done
  if (( needs_proxy )); then
    local proxy_setup_url="${V3_PROXY_SETUP_URL:-http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh}"
    echo "[v3_cm2_aug_grpo] https endpoint detected; sourcing proxy setup: $proxy_setup_url"
    source <(curl -sSL "$proxy_setup_url") >/dev/null
  fi
}

enable_proxy_for_https_endpoints

# Avoid routing in-cluster judge traffic through HTTP(S)_PROXY. Some clients do
# not understand CIDR entries in no_proxy, so add the concrete host for non-HTTPS
# in-cluster endpoints. HTTPS endpoints are intentionally left proxy-routable.
eval "$(python3 - "${SGLANG_URL:?set SGLANG_URL}" <<'PY'
import os
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
host = parsed.hostname
if not host:
    raise SystemExit(0)
is_https = parsed.scheme.lower() == "https"

defaults = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "10.0.0.0/8",
    "100.96.0.0/12",
]
if not is_https:
    defaults.append(".pjlab.org.cn")

def extend(value: str) -> str:
    items = [x.strip() for x in value.split(",") if x.strip()]
    candidates = [*defaults] if is_https else [host, *defaults]
    for item in candidates:
        if item not in items:
            items.append(item)
    return ",".join(items)

no_proxy = extend(os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or "")
print(f"export no_proxy={no_proxy!r}")
print(f"export NO_PROXY={no_proxy!r}")
PY
)"
echo "[v3_cm2_aug_grpo] judge_url=$SGLANG_URL judge_model=$JUDGE_MODEL_NAME judge_api_url=$JUDGE_API_URL no_proxy=$no_proxy"

# Fail fast if judge endpoint is unreachable to avoid noisy runtime warnings.
# HTTPS endpoints can require the proxy above, so avoid raw socket probing for
# them and let the HTTP preflight use proxy environment variables.
python3 - "${SGLANG_URL:?set SGLANG_URL}" "${JUDGE_URL_PREFLIGHT_TIMEOUT:-5}" <<'PY'
import socket
import sys
from urllib.parse import urlparse

url = sys.argv[1]
timeout = float(sys.argv[2])
parsed = urlparse(url)
host = parsed.hostname
port = parsed.port or (443 if parsed.scheme == "https" else 80)
if not host:
    raise SystemExit(f"[v3_cm2_aug_grpo] invalid SGLANG_URL: {url}")
if parsed.scheme.lower() == "https":
    print(f"[v3_cm2_aug_grpo] skip raw socket preflight for proxied HTTPS endpoint: {host}:{port}")
    raise SystemExit(0)
try:
    with socket.create_connection((host, port), timeout=timeout):
        pass
    print(f"[v3_cm2_aug_grpo] judge endpoint reachable: {host}:{port}")
except OSError as exc:
    raise SystemExit(f"[v3_cm2_aug_grpo] judge endpoint unreachable: {host}:{port} ({exc})")
PY

# HTTP-level preflight uses proxy env for external HTTPS endpoints and includes
# the same bearer key used by the v3 judge clients.
python3 - "${SGLANG_BASE_URL:?set SGLANG_BASE_URL}" "${JUDGE_MODEL_NAME:?set JUDGE_MODEL_NAME}" "${JUDGE_URL_PREFLIGHT_TIMEOUT:-5}" <<'PY'
import os
import sys

import httpx

base_url, model, timeout_s = sys.argv[1].rstrip("/"), sys.argv[2], float(sys.argv[3])
trust_env = base_url.lower().startswith("https:")
models_url = base_url + "/models" if base_url.endswith("/v1") else base_url + "/v1/models"
api_key = (
    os.getenv("V3_JUDGE_API_KEY")
    or os.getenv("JUDGE_API_KEY")
    or os.getenv("LABEL_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
headers = {}
if api_key and api_key != "EMPTY":
    headers["Authorization"] = f"Bearer {api_key}"
try:
    with httpx.Client(timeout=timeout_s, trust_env=trust_env) as client:
        resp = client.get(models_url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
except Exception as exc:
    raise SystemExit(f"[v3_cm2_aug_grpo] judge HTTP preflight failed: {models_url} ({exc!r})")

ids = []
if isinstance(payload, dict):
    data = payload.get("data", [])
    if isinstance(data, list):
        ids = [str(x.get("id", "")) for x in data if isinstance(x, dict)]
if ids and model not in ids:
    print(f"[v3_cm2_aug_grpo] warning: judge model {model!r} not listed by {models_url}; available={ids}")
print(f"[v3_cm2_aug_grpo] judge HTTP preflight ok: {models_url}")
PY

if [[ "${JUDGE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[v3_cm2_aug_grpo] JUDGE_PREFLIGHT_ONLY=1; exit before data build/training"
  exit 0
fi

mkdir -p \
  "$(dirname "${WHEN2CALL_OUTPUT_FILE:?set WHEN2CALL_OUTPUT_FILE}")" \
  "$(dirname "${CM2_AUG_OUTPUT_FILE:?set CM2_AUG_OUTPUT_FILE}")" \
  "$(dirname "${CM2_VAL_AUG_OUTPUT_FILE:-$CM2_AUG_OUTPUT_FILE}")" \
  "$(dirname "${CM2_AUG_BALANCED_OUTPUT_FILE:?set CM2_AUG_BALANCED_OUTPUT_FILE}")" \
  "$(dirname "${CM2_AUG_BALANCED_VAL_OUTPUT_FILE:?set CM2_AUG_BALANCED_VAL_OUTPUT_FILE}")"

python3 -m v3.train.datasets.prepare_when2call_v3 \
  --input "${WHEN2CALL_INPUT_FILE:?set WHEN2CALL_INPUT_FILE}" \
  --output "${WHEN2CALL_OUTPUT_FILE:?set WHEN2CALL_OUTPUT_FILE}"

run_cm2_labeling() {
  python3 -m v3.train.labeling.label_cm2_trajectory_when2call \
    --input "${CM2_INPUT_FILE:?set CM2_INPUT_FILE}" \
    --output "${CM2_AUG_OUTPUT_FILE:?set CM2_AUG_OUTPUT_FILE}" \
    --model "${LABEL_MODEL_NAME:?set LABEL_MODEL_NAME}" \
    --base-url "${LABEL_BASE_URL:?set LABEL_BASE_URL}" \
    --api-key-env "${LABEL_API_KEY_ENV:-LABEL_API_KEY}" \
    --workers "${LABEL_WORKERS:-16}" \
    --max-tokens "${LABEL_MAX_TOKENS:-1600}" \
    --temperature "${LABEL_TEMPERATURE:-0.0}" \
    --limit "${LABEL_LIMIT:--1}"

  if [[ -n "${CM2_VAL_INPUT_FILE:-}" ]]; then
    python3 -m v3.train.labeling.label_cm2_trajectory_when2call \
      --input "${CM2_VAL_INPUT_FILE}" \
      --output "${CM2_VAL_AUG_OUTPUT_FILE:?set CM2_VAL_AUG_OUTPUT_FILE}" \
      --model "${LABEL_MODEL_NAME:?set LABEL_MODEL_NAME}" \
      --base-url "${LABEL_BASE_URL:?set LABEL_BASE_URL}" \
      --api-key-env "${LABEL_API_KEY_ENV:-LABEL_API_KEY}" \
      --workers "${LABEL_WORKERS:-16}" \
      --max-tokens "${LABEL_MAX_TOKENS:-1600}" \
      --temperature "${LABEL_TEMPERATURE:-0.0}" \
      --limit "${LABEL_VAL_LIMIT:--1}"
  fi
}

if [[ "${SKIP_CM2_LABELING:-0}" == "1" ]]; then
  missing_cm2_aug=0
  if [[ ! -s "${CM2_AUG_OUTPUT_FILE:?set CM2_AUG_OUTPUT_FILE}" ]]; then
    missing_cm2_aug=1
  fi
  if [[ -n "${CM2_VAL_INPUT_FILE:-}" && ! -s "${CM2_VAL_AUG_OUTPUT_FILE:?set CM2_VAL_AUG_OUTPUT_FILE}" ]]; then
    missing_cm2_aug=1
  fi

  if (( missing_cm2_aug )); then
    cm2_aug_lock="${CM2_AUG_OUTPUT_FILE}.lock"
    echo "[v3_cm2_aug_grpo] SKIP_CM2_LABELING=1 but CM2 augmented parquet is missing; building it once under lock=$cm2_aug_lock"
    (
      flock 9
      still_missing=0
      if [[ ! -s "${CM2_AUG_OUTPUT_FILE:?set CM2_AUG_OUTPUT_FILE}" ]]; then
        still_missing=1
      fi
      if [[ -n "${CM2_VAL_INPUT_FILE:-}" && ! -s "${CM2_VAL_AUG_OUTPUT_FILE:?set CM2_VAL_AUG_OUTPUT_FILE}" ]]; then
        still_missing=1
      fi
      if (( still_missing )); then
        run_cm2_labeling
      else
        echo "[v3_cm2_aug_grpo] CM2 augmented parquet became available while waiting for lock; reusing existing files"
      fi
    ) 9>"$cm2_aug_lock"
  else
    echo "[v3_cm2_aug_grpo] SKIP_CM2_LABELING=1; reusing existing CM2 augmented parquet"
  fi
else
  run_cm2_labeling
fi

python3 -m v3.train.datasets.build_mixed_dataset \
  --cm2-file "${CM2_AUG_OUTPUT_FILE:?set CM2_AUG_OUTPUT_FILE}" \
  --when2call-file "${WHEN2CALL_OUTPUT_FILE:?set WHEN2CALL_OUTPUT_FILE}" \
  --output-file "${CM2_AUG_BALANCED_OUTPUT_FILE:?set CM2_AUG_BALANCED_OUTPUT_FILE}" \
  --enable-uq-value "${ENABLE_UQ:-1}" \
  --include-when2call-value 0 \
  --mixed-shuffle "${MIXED_SHUFFLE:-1}" \
  --mixed-seed "${MIXED_SEED:-42}" \
  --mixed-stratify-by-source "${MIXED_STRATIFY_BY_SOURCE:-1}" \
  --mixed-source-ratios "" \
  --mixed-balance-cm2-gt-action 1

if [[ -n "${CM2_VAL_INPUT_FILE:-}" ]]; then
  python3 -m v3.train.datasets.build_mixed_dataset \
    --cm2-file "${CM2_VAL_AUG_OUTPUT_FILE:?set CM2_VAL_AUG_OUTPUT_FILE}" \
    --when2call-file "${WHEN2CALL_OUTPUT_FILE:?set WHEN2CALL_OUTPUT_FILE}" \
    --output-file "${CM2_AUG_BALANCED_VAL_OUTPUT_FILE:?set CM2_AUG_BALANCED_VAL_OUTPUT_FILE}" \
    --enable-uq-value "${ENABLE_UQ:-1}" \
    --include-when2call-value 0 \
    --mixed-shuffle "${MIXED_SHUFFLE:-1}" \
    --mixed-seed "${MIXED_SEED:-42}" \
    --mixed-stratify-by-source 0 \
    --mixed-source-ratios "" \
    --mixed-balance-cm2-gt-action 1 \
    --no-split
else
  python3 -m v3.train.datasets.build_mixed_dataset \
    --cm2-file "${CM2_AUG_OUTPUT_FILE:?set CM2_AUG_OUTPUT_FILE}" \
    --when2call-file "${WHEN2CALL_OUTPUT_FILE:?set WHEN2CALL_OUTPUT_FILE}" \
    --output-file "${CM2_AUG_BALANCED_OUTPUT_FILE:?set CM2_AUG_BALANCED_OUTPUT_FILE}" \
    --val-output-file "${CM2_AUG_BALANCED_VAL_OUTPUT_FILE:?set CM2_AUG_BALANCED_VAL_OUTPUT_FILE}" \
    --val-size "${MIXED_VAL_SIZE:-64}" \
    --enable-uq-value "${ENABLE_UQ:-1}" \
    --include-when2call-value 0 \
    --mixed-shuffle "${MIXED_SHUFFLE:-1}" \
    --mixed-seed "${MIXED_SEED:-42}" \
    --mixed-stratify-by-source "${MIXED_STRATIFY_BY_SOURCE:-1}" \
    --mixed-source-ratios "" \
    --mixed-balance-cm2-gt-action 1
fi

if [[ "${BUILD_DATA_ONLY:-0}" == "1" ]]; then
  exit 0
fi

gcd() {
  local a="$1"
  local b="$2"
  while (( b != 0 )); do
    local t=$((a % b))
    a="$b"
    b="$t"
  done
  echo "$a"
}

cleanup_mcp_server() {
  if [[ -n "${MCP_SERVER_PID:-}" ]] && kill -0 "$MCP_SERVER_PID" 2>/dev/null; then
    echo "[v3_cm2_aug_grpo] stopping local MCP server pid=$MCP_SERVER_PID"
    kill "$MCP_SERVER_PID" 2>/dev/null || true
    wait "$MCP_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_mcp_server EXIT

CHECKLIST_MCP_BIND_HOST="${CHECKLIST_MCP_BIND_HOST:-127.0.0.1}"
CHECKLIST_MCP_CLIENT_HOST="${CHECKLIST_MCP_CLIENT_HOST:-127.0.0.1}"
CHECKLIST_MCP_PORT="${CHECKLIST_MCP_PORT:-$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}"
CHECKLIST_MCP_RUNTIME_DIR="${CHECKLIST_MCP_RUNTIME_DIR:-${RAY_TMPDIR:-/mnt/shared-storage-user/zhouyijin/tmp}/v3_cm2_aug_mcp}"
mkdir -p "$CHECKLIST_MCP_RUNTIME_DIR"

if [[ -z "${CHECKLIST_DATASET_PATH:-}" ]]; then
  CHECKLIST_DATASET_PATH="$CHECKLIST_MCP_RUNTIME_DIR/checklist_dataset_${CHECKLIST_MCP_PORT}.json"
  checklist_dataset_train_file="${TRAIN_FILE:-$CM2_AUG_BALANCED_OUTPUT_FILE}"
  checklist_dataset_val_file="${VAL_FILE:-$CM2_AUG_BALANCED_VAL_OUTPUT_FILE}"
  python3 - "$CHECKLIST_DATASET_PATH" "$checklist_dataset_train_file" "$checklist_dataset_val_file" <<'PY'
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

out_path = Path(sys.argv[1])
input_paths = [Path(p) for p in sys.argv[2:] if p]


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def _json_safe(value: Any) -> Any:
    if _is_null(value):
        return None
    value = _maybe_json_loads(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _ensure_dict(value: Any) -> dict[str, Any]:
    value = _json_safe(value)
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> list[Any]:
    value = _json_safe(value)
    if value is None:
        return []
    return value if isinstance(value, list) else []


records: list[dict[str, Any]] = []
seen: set[str] = set()
rows_with_tools = 0
missing_required: list[str] = []
for path in input_paths:
    if not path.is_file():
        continue
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        extra = _ensure_dict(row.get("extra_info"))
        original_index = str(extra.get("original_index") or row.get("uuid") or "").strip()
        tools = _ensure_list(extra.get("tools")) or _ensure_list(row.get("tools"))
        messages = _ensure_list(extra.get("messages")) or _ensure_list(row.get("messages")) or _ensure_list(row.get("prompt"))
        if not tools:
            continue
        rows_with_tools += 1
        if not original_index or not messages:
            missing_required.append(str(path))
            continue
        if original_index in seen:
            continue
        seen.add(original_index)
        merged_extra = dict(extra)
        merged_extra["original_index"] = original_index
        merged_extra["tools"] = json.dumps(_json_safe(tools), ensure_ascii=False)
        merged_extra["messages"] = _json_safe(messages)
        records.append({"extra_info": merged_extra})

if missing_required:
    sample = ", ".join(sorted(set(missing_required))[:3])
    raise SystemExit(
        "Cannot build CHECKLIST_DATASET_PATH: rows with tools are missing "
        f"original_index or messages. Example files: {sample}"
    )
if rows_with_tools > 0 and not records:
    raise SystemExit("Cannot build CHECKLIST_DATASET_PATH: no usable tool rows were exported.")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False)
    f.write("\n")
print(
    f"[v3_cm2_aug_grpo] wrote CHECKLIST_DATASET_PATH={out_path} "
    f"records={len(records)} rows_with_tools={rows_with_tools}"
)
PY
fi
export CHECKLIST_DATASET_PATH
echo "[v3_cm2_aug_grpo] checklist_dataset_path=$CHECKLIST_DATASET_PATH"

if [[ "${CHECKLIST_MCP_AUTO_START:-1}" != "0" ]]; then
  CHECKLIST_MCP_SERVER_CONFIG="${CHECKLIST_MCP_SERVER_CONFIG:-$CHECKLIST_MCP_RUNTIME_DIR/checklist_mcp_server_${CHECKLIST_MCP_PORT}.json}"
  export CHECKLIST_MCP_SERVER_CONFIG

  python3 - "$CHECKLIST_MCP_SERVER_CONFIG" "$CHECKLIST_MCP_CLIENT_HOST" "$CHECKLIST_MCP_PORT" <<'PY'
import json
import sys
path, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = {
    "mcpServers": {
        "SGLang Router MCP": {
            "url": f"http://{host}:{port}/sse",
            "auth_token": "checklist",
        }
    }
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

  if python3 - "$CHECKLIST_MCP_CLIENT_HOST" "$CHECKLIST_MCP_PORT" <<'PY'
import socket
import sys
host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1.0):
        pass
except OSError:
    raise SystemExit(1)
PY
  then
    echo "[v3_cm2_aug_grpo] reusing existing MCP server at ${CHECKLIST_MCP_CLIENT_HOST}:${CHECKLIST_MCP_PORT}"
  else
    MCP_SERVER_LOG="$CHECKLIST_MCP_RUNTIME_DIR/mcp_server_${CHECKLIST_MCP_PORT}.log"
    SGLANG_BASE_URL="${SGLANG_BASE_URL:-${SGLANG_URL%/chat/completions}}"
    echo "[v3_cm2_aug_grpo] starting local MCP server url=http://${CHECKLIST_MCP_CLIENT_HOST}:${CHECKLIST_MCP_PORT}/sse log=$MCP_SERVER_LOG"
    python3 "$FORMAL_ROOT/src/cm2_core/5_rl_training/sglang_router_mcp_server_sse.py" \
      --host "$CHECKLIST_MCP_BIND_HOST" \
      --port "$CHECKLIST_MCP_PORT" \
      --dataset-path "$CHECKLIST_DATASET_PATH" \
      --sglang-url "$SGLANG_BASE_URL" \
      --model "${JUDGE_MODEL_NAME:?set JUDGE_MODEL_NAME}" \
      --temperature "${MCP_TEMPERATURE:-0.6}" \
      --max-generated-tokens "${MCP_MAX_GENERATED_TOKENS:-2048}" \
      --retry-attempts "${MCP_RETRY_ATTEMPTS:-1}" \
      >"$MCP_SERVER_LOG" 2>&1 &
    MCP_SERVER_PID=$!

    python3 - "$CHECKLIST_MCP_CLIENT_HOST" "$CHECKLIST_MCP_PORT" "$MCP_SERVER_PID" "$MCP_SERVER_LOG" <<'PY'
import os
import socket
import sys
import time

host, port, pid, log_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
deadline = time.time() + 60.0
last_error = None
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit(f"MCP server process exited early; see log: {log_path}")
    try:
        with socket.create_connection((host, port), timeout=1.0):
            print(f"[v3_cm2_aug_grpo] MCP server ready at {host}:{port}")
            raise SystemExit(0)
    except OSError as exc:
        last_error = exc
        time.sleep(1.0)
raise SystemExit(f"MCP server did not become ready at {host}:{port}: {last_error}; see log: {log_path}")
PY
  fi
  echo "[v3_cm2_aug_grpo] checklist_mcp_server_config=$CHECKLIST_MCP_SERVER_CONFIG"
else
  echo "[v3_cm2_aug_grpo] CHECKLIST_MCP_AUTO_START=0; using CHECKLIST_MCP_SERVER_CONFIG=${CHECKLIST_MCP_SERVER_CONFIG:-unset}"
fi

TRAIN_BATCH_SIZE_EFF="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N_EFF="${ROLLOUT_N:-16}"
NUM_GPUS_PER_NODE_EFF="${NUM_GPUS_PER_NODE:-8}"

if (( TRAIN_BATCH_SIZE_EFF <= 0 || ROLLOUT_N_EFF <= 0 || NUM_GPUS_PER_NODE_EFF <= 0 )); then
  echo "[v3_cm2_aug_grpo] invalid batch/gpu settings: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE_EFF ROLLOUT_N=$ROLLOUT_N_EFF NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE_EFF" >&2
  exit 1
fi

REAL_TRAIN_BATCH_SIZE=$((TRAIN_BATCH_SIZE_EFF * ROLLOUT_N_EFF))
MINIMAL_BSZ="$NUM_GPUS_PER_NODE_EFF"
if (( REAL_TRAIN_BATCH_SIZE % MINIMAL_BSZ != 0 )); then
  d="$(gcd "$ROLLOUT_N_EFF" "$MINIMAL_BSZ")"
  step=$((MINIMAL_BSZ / d))
  adjusted=$((((TRAIN_BATCH_SIZE_EFF + step - 1) / step) * step))
  echo "[v3_cm2_aug_grpo] adjust TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE_EFF} -> ${adjusted} (require TRAIN_BATCH_SIZE*ROLLOUT_N divisible by NUM_GPUS_PER_NODE)" >&2
  TRAIN_BATCH_SIZE_EFF="$adjusted"
  REAL_TRAIN_BATCH_SIZE=$((TRAIN_BATCH_SIZE_EFF * ROLLOUT_N_EFF))
fi
echo "[v3_cm2_aug_grpo] preflight batch: train_batch=${TRAIN_BATCH_SIZE_EFF} rollout_n=${ROLLOUT_N_EFF} real_train_batch=${REAL_TRAIN_BATCH_SIZE} n_gpus=${NUM_GPUS_PER_NODE_EFF}"

python3 -m v3.train.launch_grpo \
  --config-path="$FORMAL_ROOT/src/cm2_core/5_rl_training/config" \
  --config-name='checklist' \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_token_veto_threshold=1e-4 \
  algorithm.use_kl_in_reward=False \
  data.train_batch_size="${TRAIN_BATCH_SIZE_EFF}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-1536}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
  data.filter_overlong_prompts=True \
  data.prompt_key=messages \
  data.truncation='error' \
  data.return_raw_chat=True \
  data.shuffle=True \
  data.custom_cls.path=pkg://verl.utils.dataset.checklist_dataset \
  data.custom_cls.name=ChecklistDataset \
  data.train_files="${TRAIN_FILE:-$CM2_AUG_BALANCED_OUTPUT_FILE}" \
  data.val_files="${VAL_FILE:-$CM2_AUG_BALANCED_VAL_OUTPUT_FILE}" \
  +data.cache_dir="${VERL_DATA_CACHE_DIR:-/tmp/verl_rlhf}" \
  reward_model.reward_manager="${REWARD_MANAGER_NAME:-mixed}" \
  reward_model.launch_reward_fn_async="${LAUNCH_REWARD_FN_ASYNC:-0}" \
  +reward_model.reward_fn_num_gpus="${ASYNC_REWARD_NUM_GPUS:-0}" \
  +reward_model.reward_fn_num_cpus="${ASYNC_REWARD_NUM_CPUS:-1}" \
  +reward_model.reward_fn_share_with_actor_rollout="${ASYNC_REWARD_SHARE_WITH_ACTOR:-0}" \
  +reward_model.reward_fn_share_worker_idx="${ASYNC_REWARD_SHARE_WORKER_IDX:-0}" \
  +reward_model.reward_kwargs.sglang_url="[\"${SGLANG_URL:?set SGLANG_URL}\"]" \
  +reward_model.reward_kwargs.sglang_model="${JUDGE_MODEL_NAME:?set JUDGE_MODEL_NAME}" \
  +reward_model.reward_kwargs.retry_times="${JUDGE_RETRY_TIMES:-8}" \
  +reward_model.reward_kwargs.semaphore_size="${JUDGE_SEMAPHORE_SIZE:-2}" \
  +reward_model.reward_kwargs.timeout="${JUDGE_TIMEOUT_SECONDS:-1200}" \
  +reward_model.reward_kwargs.timeout_seconds="${JUDGE_TIMEOUT_SECONDS:-1200}" \
  +reward_model.reward_kwargs.temperature=0.0 \
  +reward_model.reward_kwargs.top_p=1.0 \
  +reward_model.reward_kwargs.max_new_tokens="${JUDGE_MAX_NEW_TOKENS:-512}" \
  +reward_model.reward_kwargs.max_tokens="${JUDGE_MAX_TOKENS:-512}" \
  +reward_model.reward_kwargs.step_eval_batch_size="${CHECKLIST_STEP_BATCH_SIZE:-2}" \
  +reward_model.reward_kwargs.eta=1.0 \
  +reward_model.reward_kwargs.strict_tool_success="${STRICT_TOOL_SUCCESS:-0}" \
  actor_rollout_ref.model.path="${MODEL_PATH:?set MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR:-3e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE_EFF}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-8}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS:-8192}" \
  actor_rollout_ref.rollout.mode="${ROLLOUT_MODE:-async}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS:-1}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.skip_tokenizer_init=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-2}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.75}" \
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE:-False}" \
  actor_rollout_ref.rollout.multi_stage_wake_up=True \
  actor_rollout_ref.rollout.n="${ROLLOUT_N_EFF}" \
  actor_rollout_ref.rollout.over_sample_rate=0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$FORMAL_ROOT/src/cm2_core/5_rl_training/config/checklist_tool_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="$FORMAL_ROOT/src/cm2_core/5_rl_training/config/checklist_interaction_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.format=qwen25 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="${MAX_PARALLEL_CALLS:-1}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS:-8}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS:-8}" \
  actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
  +critic.model.override_config.attn_implementation=eager \
  trainer.critic_warmup=0 \
  trainer.project_name="${PROJECT_NAME:-v3_agent_uq}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-stage1_cm2_aug_only_balanced}" \
  trainer.default_local_dir="${CHECKPOINT_ROOT:-$TRAIN_OUTPUT_ROOT/checkpoints/v3_grpo}/${PROJECT_NAME:-v3_agent_uq}/${EXPERIMENT_NAME:-stage1_cm2_aug_only_balanced}" \
  trainer.n_gpus_per_node="${NUM_GPUS_PER_NODE_EFF}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-null}" \
  trainer.max_critic_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-null}" \
  trainer.logger='["console","tensorboard"]' \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  +checkpoint.save_contents="['model','optimizer','extra']" \
  trainer.test_freq="${TEST_FREQ:-100}" \
  trainer.val_before_train=False \
  trainer.rollout_data_dir="${ROLLOUT_OUTPUT_ROOT:-$TRAIN_OUTPUT_ROOT}/rollout_results/${PROJECT_NAME:-v3_agent_uq}/${EXPERIMENT_NAME:-stage1_cm2_aug_only_balanced}/train" \
  trainer.validation_data_dir="${ROLLOUT_OUTPUT_ROOT:-$TRAIN_OUTPUT_ROOT}/rollout_results/${PROJECT_NAME:-v3_agent_uq}/${EXPERIMENT_NAME:-stage1_cm2_aug_only_balanced}/valid" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  actor_rollout_ref.nccl_timeout=900000 \
  +ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-32}" \
  +ray_kwargs.ray_init.num_gpus="${RAY_NUM_GPUS:-8}" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR:-/mnt/shared-storage-user/zhouyijin/tmp}" \
  +ray_kwargs.ray_init.object_store_memory="${OBJECT_STORE_MEMORY:-4294967296}"
