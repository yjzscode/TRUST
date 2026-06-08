#!/usr/bin/env bash
set -euo pipefail
source /home/zhouyijin/miniconda3/etc/profile.d/conda.sh
conda activate base
FORMAL_ROOT="${FORMAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-/mnt/shared-storage-user/ai4good1-share/zhouyijin/uq2as_formal}"
export PYTHONPATH="$FORMAL_ROOT/src:$FORMAL_ROOT/third_party/verl_v0.6.1_checklist:$FORMAL_ROOT/third_party:${PYTHONPATH:-}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TRAIN_OUTPUT_ROOT/tensorboard/${TRAINER_PROJECT_NAME:-when2call_grpo}/${TRAINER_EXPERIMENT_NAME:-when2call_default}}"
export WHEN2CALL_GRPO_OUTPUT_DIR="${WHEN2CALL_GRPO_OUTPUT_DIR:-$TRAIN_OUTPUT_ROOT/data/when2call_official/grpo}"
export HF_HOME="${HF_HOME:-/mnt/shared-storage-user/zhouyijin/tmp/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
cd "$FORMAL_ROOT"

python3 - <<'PY'
import v2.train.ppl_reward_manager  # noqa: F401
print("[when2call_grpo] registered reward manager: ppl_uq")
PY

is_writable_dir() {
  local dir="$1"
  local tmp
  mkdir -p "$dir" 2>/dev/null || return 1
  tmp="$(mktemp "$dir/.write_test.XXXXXX" 2>/dev/null)" || return 1
  rm -f "$tmp"
}

latest_checkpoint_dir() {
  python3 - "$1" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(0)

tracker = root / "latest_checkpointed_iteration.txt"
if tracker.is_file():
    text = tracker.read_text(encoding="utf-8", errors="ignore").strip()
    if text.isdigit() and (root / f"global_step_{text}").is_dir():
        print(root / f"global_step_{text}")
        raise SystemExit(0)

best = None
for path in root.glob("global_step_*"):
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match and path.is_dir():
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, path)
if best is not None:
    print(best[1])
PY
}

fallback_root="${WHEN2CALL_FALLBACK_OUTPUT_ROOT:-$FORMAL_ROOT/outputs/when2call_runtime}"

if ! is_writable_dir "$WHEN2CALL_GRPO_OUTPUT_DIR"; then
  echo "Warning: WHEN2CALL_GRPO_OUTPUT_DIR is not writable: $WHEN2CALL_GRPO_OUTPUT_DIR" >&2
  WHEN2CALL_GRPO_OUTPUT_DIR="$fallback_root/data/when2call_official/grpo"
  export WHEN2CALL_GRPO_OUTPUT_DIR
  mkdir -p "$WHEN2CALL_GRPO_OUTPUT_DIR"
  echo "Falling back to writable GRPO data dir: $WHEN2CALL_GRPO_OUTPUT_DIR" >&2
fi

if ! is_writable_dir "${RAY_TMPDIR:-/mnt/shared-storage-user/zhouyijin/tmp}"; then
  echo "Warning: RAY_TMPDIR is not writable: ${RAY_TMPDIR:-/mnt/shared-storage-user/zhouyijin/tmp}" >&2
  RAY_TMPDIR="$fallback_root/ray_tmp"
  export RAY_TMPDIR
  mkdir -p "$RAY_TMPDIR"
  echo "Falling back to writable Ray temp dir: $RAY_TMPDIR" >&2
fi

if ! is_writable_dir "$HF_HOME"; then
  echo "Warning: HF_HOME is not writable: $HF_HOME" >&2
  HF_HOME="$fallback_root/cache/hf"
  HF_DATASETS_CACHE="$HF_HOME/datasets"
  HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  TRANSFORMERS_CACHE="$HF_HOME/transformers"
  export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
  echo "Falling back to writable HF cache: $HF_HOME" >&2
fi

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

BATCH_SIZE="${BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-16}"
NUM_GPUS="${NUM_GPUS:-8}"

if (( NUM_GPUS <= 0 )); then
  echo "NUM_GPUS must be > 0, got: $NUM_GPUS" >&2
  exit 1
fi

GLOBAL_PPO_MINI_BATCH_SIZE=$(( BATCH_SIZE * N_SAMPLES ))
if (( GLOBAL_PPO_MINI_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "Invalid batch setup: BATCH_SIZE * N_SAMPLES must be divisible by NUM_GPUS" >&2
  echo "  BATCH_SIZE=$BATCH_SIZE N_SAMPLES=$N_SAMPLES NUM_GPUS=$NUM_GPUS" >&2
  exit 1
fi

NORMALIZED_PPO_MINI_BATCH_SIZE=$(( GLOBAL_PPO_MINI_BATCH_SIZE / NUM_GPUS ))
if (( NORMALIZED_PPO_MINI_BATCH_SIZE <= 0 )); then
  echo "Normalized PPO mini batch size must be > 0, got: $NORMALIZED_PPO_MINI_BATCH_SIZE" >&2
  exit 1
fi

if [[ -z "${PPO_MICRO_BATCH_SIZE_PER_GPU:-}" ]]; then
  PPO_MICRO_BATCH_SIZE_PER_GPU="$NORMALIZED_PPO_MINI_BATCH_SIZE"
  for candidate in 16 8 4 2 1; do
    if (( candidate <= NORMALIZED_PPO_MINI_BATCH_SIZE )) && (( NORMALIZED_PPO_MINI_BATCH_SIZE % candidate == 0 )); then
      PPO_MICRO_BATCH_SIZE_PER_GPU="$candidate"
      break
    fi
  done
fi

if (( PPO_MICRO_BATCH_SIZE_PER_GPU <= 0 )); then
  echo "PPO_MICRO_BATCH_SIZE_PER_GPU must be > 0, got: $PPO_MICRO_BATCH_SIZE_PER_GPU" >&2
  exit 1
fi

if (( NORMALIZED_PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  echo "Invalid PPO micro batch setup after normalization:" >&2
  echo "  normalized_ppo_mini_batch_size=$NORMALIZED_PPO_MINI_BATCH_SIZE" >&2
  echo "  PPO_MICRO_BATCH_SIZE_PER_GPU=$PPO_MICRO_BATCH_SIZE_PER_GPU" >&2
  exit 1
fi

GRADIENT_ACCUMULATION_STEPS=$(( NORMALIZED_PPO_MINI_BATCH_SIZE / PPO_MICRO_BATCH_SIZE_PER_GPU ))

TB_PARENT="$(dirname "$TENSORBOARD_DIR")"
FALLBACK_TB_ROOT="$fallback_root/tensorboard"
if ! mkdir -p "$TB_PARENT" 2>/dev/null || ! mkdir -p "$TENSORBOARD_DIR" 2>/dev/null || ! test -w "$TENSORBOARD_DIR"; then
  echo "Warning: TENSORBOARD_DIR is not writable: $TENSORBOARD_DIR" >&2
  TENSORBOARD_DIR="${FALLBACK_TB_ROOT}/${TRAINER_PROJECT_NAME:-when2call_grpo}/${TRAINER_EXPERIMENT_NAME:-when2call_default}"
  export TENSORBOARD_DIR
  mkdir -p "$TENSORBOARD_DIR"
  echo "Falling back to writable tensorboard dir: $TENSORBOARD_DIR" >&2
fi

TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-$TRAIN_OUTPUT_ROOT/checkpoints/when2call}"
if ! is_writable_dir "$TRAINER_DEFAULT_LOCAL_DIR"; then
  echo "Error: TRAINER_DEFAULT_LOCAL_DIR is not writable: $TRAINER_DEFAULT_LOCAL_DIR" >&2
  echo "Refusing to save model/optimizer checkpoints under $fallback_root." >&2
  echo "Fix the shared checkpoint directory permissions/mount, for example:" >&2
  echo "  sudo chown -R zhouyijin:zhouyijin '$TRAINER_DEFAULT_LOCAL_DIR'" >&2
  echo "  sudo chmod -R u+rwX,g+rwX '$TRAINER_DEFAULT_LOCAL_DIR'" >&2
  echo "If this is a read-only mount, chmod/chown will not help; remount or request writable ai4good1-share." >&2
  exit 1
fi

prepared_file="$WHEN2CALL_GRPO_OUTPUT_DIR/when2call_pref_grpo.parquet"
if [[ "${SKIP_WHEN2CALL_PREPARE_IF_EXISTS:-1}" == "1" && -s "$prepared_file" ]]; then
  echo "Using existing When2Call GRPO data: $prepared_file" >&2
else
  python3 "$FORMAL_ROOT/src/v2/train/prepare_when2call_pref.py"
fi

if [[ "${PREPARE_ONLY_AFTER_DATA:-0}" == "1" ]]; then
  echo "PREPARE_ONLY_AFTER_DATA=1; exit after data preparation" >&2
  exit 0
fi

echo "When2Call GRPO batch config:" >&2
echo "  BATCH_SIZE=$BATCH_SIZE" >&2
echo "  N_SAMPLES=$N_SAMPLES" >&2
echo "  NUM_GPUS=$NUM_GPUS" >&2
echo "  GLOBAL_PPO_MINI_BATCH_SIZE=$GLOBAL_PPO_MINI_BATCH_SIZE" >&2
echo "  NORMALIZED_PPO_MINI_BATCH_SIZE=$NORMALIZED_PPO_MINI_BATCH_SIZE" >&2
echo "  PPO_MICRO_BATCH_SIZE_PER_GPU=$PPO_MICRO_BATCH_SIZE_PER_GPU" >&2
echo "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS" >&2

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.val_before_train=False \
  data.train_files="${TRAIN_FILE:-$WHEN2CALL_GRPO_OUTPUT_DIR/when2call_pref_grpo.parquet}" \
  data.val_files="${VAL_FILE:-$WHEN2CALL_GRPO_OUTPUT_DIR/when2call_pref_grpo.parquet}" \
  data.custom_cls.path="$FORMAL_ROOT/src/v2/train/when2call_grpo_dataset.py" \
  data.custom_cls.name=When2CallGRPODataset \
  data.train_batch_size="$BATCH_SIZE" \
  data.max_prompt_length="${PROMPT_FILTER_MAX_LEN:-2048}" \
  data.max_response_length="${MAX_RESPONSE:-1024}" \
  data.filter_overlong_prompts=True \
  data.truncation="${DATA_TRUNCATION:-left}" \
  data.shuffle=True \
  actor_rollout_ref.model.path="${MODEL:?set MODEL}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr="${LR:-3e-5}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$NUM_GPUS" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.75}" \
  actor_rollout_ref.rollout.n="$N_SAMPLES" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-3072}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS:-6144}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${TRAINER_PROJECT_NAME:-when2call_grpo}" \
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME:-when2call_default}" \
  trainer.default_local_dir="${TRAINER_DEFAULT_LOCAL_DIR:-$TRAIN_OUTPUT_ROOT/checkpoints/when2call}" \
  trainer.resume_mode="${TRAINER_RESUME_MODE:-auto}" \
  trainer.resume_from_path="${TRAINER_RESUME_FROM_PATH:-null}" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.save_freq="${SAVE_FREQ:-80}" \
  +checkpoint.save_contents="['model','optimizer','extra']" \
  trainer.test_freq="${TEST_FREQ:-80}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  reward_model.enable=False \
  reward_model.reward_manager=ppl_uq \
  custom_reward_function.path="$FORMAL_ROOT/src/v2/train/reward_when2call.py" \
  custom_reward_function.name=compute_score \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.num_cpus="${NUM_CPUS:-4}" \
  +ray_kwargs.ray_init.num_gpus="$NUM_GPUS" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR:-/mnt/shared-storage-user/zhouyijin/tmp}" \
  +ray_kwargs.ray_init.object_store_memory="${OBJECT_STORE_MEMORY:-1073741824}" \
  "$@"
