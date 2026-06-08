# TRUST: Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning

Paper: [arXiv](https://arxiv.org/pdf/2606.06976)

This repository contains the core open-source pipeline of TRUST. The current release focuses on:

- `v3` key-turn annotation on CM2 trajectories
- `v2` turn-level `When2Call` GRPO training
- `v3` unified / trajectory-level GRPO training

If this repository is useful for your research, please consider starring it.

## Updates

- [2026-06-08] Our paper and code are released!

## Introduction

![framework](assets/2-method.pdf)

TRUST studies agentic tool-calling decisions through uncertainty-aligned reinforcement learning. Instead of only optimizing task success after a tool trajectory is completed, TRUST explicitly models the next-action decision of whether the agent should answer directly, call a tool, ask the user for more information, or refuse. This release keeps the reproducible core needed to run the main annotation and training pipeline described in the paper.

Released components in this repository:

- CM2 trajectory key-turn annotation for `when2call_annotations`
- `When2Call` turn-level preference-to-GRPO data preparation and training
- Unified CM2 + When2Call trajectory-level training

Intentionally excluded from this release:

- evaluation code
- ablation codepaths
- unrelated local experiments
- caches, logs, and temporary artifacts

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yjzscode/TRUST
cd formal_version_open
```

### 2. Create the environment

The internal training environment used for this release is `uq2as_train`.

```bash
conda create -n uq2as_train python=3.10
conda activate uq2as_train
pip install -r requirements.txt
```

### 3. Prepare required assets

Before running any script, prepare the following items:

- model checkpoints to download / place locally
- CM2 train / val parquet files
- label service endpoint, API key, and model name
- judge service endpoint, API key, and model name
- output directories and storage requirements

Then edit the bundled configs under `configs/train/` and replace placeholder values such as:

- `MODEL`, `MODEL_PATH`, `REFERENCE_MODEL_PATH`
- `LABEL_BASE_URL`, `LABEL_API_KEY`, `LABEL_MODEL_NAME`
- `SGLANG_URL`, `JUDGE_API_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL_NAME`, `LLM_AS_A_JUDGE_NAME`
- `CM2_INPUT_FILE`, `CM2_VAL_INPUT_FILE`, `TRAIN_FILE`, `VAL_FILE`
- `TRAIN_OUTPUT_ROOT`, `CHECKPOINT_ROOT`, `TENSORBOARD_DIR`

## Usage

All commands below assume you are in this directory:

```bash
cd /mnt/shared-storage-user/zhouyijin/workspace/MyProj/UQ/formal_version_open
```

### 1. Build key-turn annotation and mixed training data

This stage:

- converts the official `When2Call` jsonl file to parquet
- labels CM2 trajectories with key-turn `when2call_annotations`
- builds the balanced CM2 training parquet used by the unified pipeline

```bash
bash scripts/data/run_v3_build_cm2_aug_balanced.sh
```

Equivalent config-driven entry:

```bash
bash run_job.sh train configs/train/v3_build_cm2_aug_balanced.yaml
```

### 2. Train `v2` turn-level `When2Call`

This is the paper-aligned turn-level GRPO path on the official `When2Call` training set.

```bash
bash scripts/train/run_when2call_release.sh
```

Equivalent config-driven entry:

```bash
bash run_job.sh train configs/train/when2call_full_no_neg_rollout.yaml
```

### 3. Train the `v3` unified TRUST model

This uses CM2 trajectories augmented with key-turn supervision and the unified TRUST reward path.

```bash
bash scripts/train/run_v3_cm2_aug_balanced_release.sh
```

Equivalent config-driven entry:

```bash
bash run_job.sh train configs/train/v3_train_cm2_aug_balanced_no_neg.yaml
```

## Important Notes

- `v3_build_cm2_aug_balanced.yaml` is a build-only config. It does not start training.
- `v3_train_cm2_aug_balanced_no_neg.yaml` defaults to `SKIP_CM2_LABELING=1`, so it expects the augmented CM2 parquet produced in the previous step.
- Tool calling in this release uses `<tool_call>...</tool_call>` tags. If you switch to a different model checkpoint, check that its `chat_template.jinja` is compatible and still renders / expects `<tool_call>` rather than another tool-call format.
- The low-level scripts in `scripts/train/` such as `run_when2call_grpo.sh` and `run_v3_cm2_aug_grpo.sh` are still available, but they expect many environment variables to already be set.
- The vendored VERL fork used here is under `third_party/verl_v0.6.1_checklist`, and release-specific changes are documented in `third_party/verl_v0.6.1_checklist/PATCHES.md`.

## Repository Structure

```text
formal_version_open/
|-- configs/train/                      # Config-driven entry points
|-- data/                               # Local data stubs and runtime configs
|-- scripts/data/                       # Data-building wrappers
|-- scripts/train/                      # Training wrappers
|-- src/v2/                             # Turn-level When2Call pipeline
|-- src/v3/                             # Unified / trajectory-level pipeline
|-- src/cm2_core/                       # CM2 checklist tool-training stack
`-- third_party/verl_v0.6.1_checklist/  # Vendored VERL fork used by this release
```

## Acknowledge

Leveraged part of data and code framework from [CM2-RLCR-Tool-Agent](https://github.com/namezhenzhang/CM2-RLCR-Tool-Agent).

## Citation

If you use this repository, please cite:

```bibtex
@misc{zhou2026exploringagentictoolcallingdecisions,
      title={Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning},
      author={Yijin Zhou and Linqian Zeng and Xiaoya Lu and Wenyuan Xie and Dongrui Liu and Junchi Yan and Jing Shao},
      year={2026},
      eprint={2606.06976},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.06976},
}
```
