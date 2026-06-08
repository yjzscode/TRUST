# PATCHES.md

This vendored copy of `verl_v0.6.1_checklist` is not a byte-for-byte mirror of the comparison baseline at:

`/mnt/shared-storage-user/zhouyijin/workspace/MyProj/UQ/CM2-RLCR-Tool-Agent-main/verl_v0.6.1_checklist`

It is a runtime-focused fork used by the TRUST open-source training pipeline. This document records the meaningful differences so future maintenance, upgrades, and audits are tractable.

## Scope

- Non-runtime directories were removed from the baseline copy for release hygiene:
  - `.gemini`
  - `.github`
  - `docker`
  - `docs`
  - `examples`
  - `recipe`
  - `tests`
- In addition to that pruning, there are 10 source files with behavior changes.

## Patch 1: `verl/experimental/agent_loop/tool_parser.py`

### Change

`Qwen25ToolParser.extract_tool_calls` now accepts both:

- a single JSON object
- a non-empty JSON list whose first element is a tool-call object

instead of requiring the decoded payload to be a dict immediately.

### Why

Some model outputs wrap tool calls as a single-element list. The patch prevents valid tool calls from being dropped purely because of that formatting variation.

### Risk

Low. The behavior is strictly more permissive for a narrow case and still rejects malformed payloads.

## Patch 2: `verl/interactions/checklist_interaction.py`

### Change

All `httpx.AsyncClient` instances are created with `trust_env=False`.

### Why

This prevents inherited proxy and other environment-driven HTTP settings from interfering with local or cluster-internal checklist services.

### Risk

Low. This is usually desirable for local service-to-service calls. It can be surprising only if a deployment intentionally relies on `HTTP_PROXY`/`HTTPS_PROXY`.

## Patch 3: `verl/tools/mcp_checklist_tool.py`

### Change

Shared `httpx.AsyncClient` creation and recreation paths now also use `trust_env=False`.

### Why

Same motivation as Patch 2, but in the tool execution path. This makes the mocked tool-response generation path less sensitive to host proxy settings.

### Risk

Low. Same caveat as Patch 2.

## Patch 4: `verl/trainer/main_ppo.py`

### Change

Before `ray.init`, the code monkey-patches Ray memory probing functions so that `FileNotFoundError` falls back to environment-configurable defaults:

- `UQ_FORMAL_RAY_SYSTEM_MEMORY_BYTES`
- `UQ_FORMAL_RAY_USED_MEMORY_BYTES`

### Why

Some constrained/containerized environments do not expose the usual `/proc`-backed memory information Ray expects. Without this patch, trainer startup can fail before any actual training logic runs.

### Risk

Medium. This is operationally useful, but it is a fork-specific runtime patch against Ray internals. If the fallback values are badly chosen, Ray resource accounting can become inaccurate.

## Patch 5: `verl/trainer/ppo/metric_utils.py`

### Change

Validation metric aggregation now skips invalid scalar values such as:

- `None`
- strings
- container types
- `NaN`

and only computes bootstrap / majority-vote statistics on valid numeric or boolean values.

### Why

The checklist validation pipeline can surface partial or missing auxiliary fields. This patch keeps metric aggregation from crashing or silently producing nonsense because one bad value entered a grouped statistic.

### Risk

Low to medium. This improves robustness, but it also means bad upstream data can be masked rather than fail-fast.

## Patch 6: `verl/trainer/ppo/ray_trainer.py`

### Change

Validation reward post-processing now tolerates missing or shape-mismatched `reward_extra_info` entries for:

- `max_num_turns`
- `tool_call_success`

When absent or malformed, batch-aligned default tensors are substituted.

### Why

This prevents validation from crashing when an upstream reward path returns incomplete metadata.

### Risk

Medium. This is a defensive patch, not a semantically neutral one. If upstream reward metadata is broken, validation still completes but the resulting non-normalized reward metrics are only approximate.

## Patch 7: `verl/utils/dataset/checklist_dataset.py`

### Change

Parquet loading now uses a compatibility helper with three fallback layers:

1. `datasets.Dataset.from_parquet(...)`
2. `datasets.load_dataset("parquet", ...)`
3. direct `pyarrow.parquet` read with HuggingFace metadata stripping before constructing `datasets.Dataset`

### Why

This makes dataset loading tolerant to parquet files written under different `datasets` versions, especially when embedded HuggingFace metadata becomes incompatible across environments.

### Risk

Low. This is a practical compatibility patch for dataset portability.

## Patch 8: `verl/utils/distributed.py`

### Change

Ray distributed initialization now explicitly rejects CPU-only execution in the accelerator-backed training path. The backend string is created through `_require_accelerator_backend()`, which raises with a clearer error if the worker only sees CPU.

### Why

The TRUST training pipeline expects CUDA/NPU-backed distributed execution. Failing early with a direct message is preferable to obscure downstream initialization errors.

### Risk

Low. This is a stricter guardrail, not a behavior expansion.

## Patch 9: `verl/utils/reward_score/checklist_reward.py`

### Change

This file contains the largest fork-specific patch set:

- richer reward-response JSON extraction logic
- parse-failure debug logging
- rolling reward success/failure statistics
- support for OpenAI-style `message.content` arrays
- removal of early JSON validation from `_post_with_retries`
- optional soft-fail behavior on reward endpoint connectivity failures, controlled by:
  - `CHECKLIST_SOFT_FAIL_ON_REWARD_CONNECT`

The soft-fail path currently defaults to enabled and returns `(False, True)` on certain network exceptions, meaning:

- checklist answer is treated as `False`
- call success is treated as `True`

### Why

Most of this patch improves robustness to response-format variation and flaky judge endpoints. It is clearly aimed at keeping long-running training jobs alive.

### Risk

High. The parsing robustness changes are reasonable, but the soft-fail semantics are not neutral:

- a reward-service outage can be converted into a valid sample with zero/false reward
- downstream strict-success filtering will not necessarily exclude that sample

This is the patch most likely to change training behavior rather than just improving compatibility.

## Patch 10: `verl/utils/vllm/utils.py`

### Change

The vLLM LoRA hijack was generalized to support API variation across vLLM versions:

- fallback import path for `LoRAModel`
- signature-based filtering of kwargs before calling vLLM helpers
- compatibility handling for optional `tensorizer_config_dict`
- safer handling of optional `lora_extra_vocab_size`
- broader expected-module handling for expert modules

### Why

The TRUST pipeline vendors a custom LoRA-loading path, and upstream vLLM APIs can shift across versions. This patch reduces breakage from those interface changes.

### Risk

Medium. This is a useful compatibility layer, but it also increases divergence from an upstream `verl` + `vllm` pairing and should be revalidated during upgrades.

## Maintenance Notes

- Treat this directory as a forked third-party dependency, not a pristine vendor drop.
- If upgrading from upstream `verl`, review these 10 files first.
- If the reward path is being hardened for reproducibility rather than uptime, revisit Patch 9 before relying on new training results.
