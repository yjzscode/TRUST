from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from common.env import run_bash
from common.paths import get_formal_paths


def _exports(env_map: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in env_map.items():
        if value is None:
            continue
        parts.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join(parts)


def _pythonpath(paths) -> str:
    return ":".join(
        [
            str(paths.src),
            str(paths.third_party / "verl_v0.6.1_checklist"),
            str(paths.third_party),
        ]
    )


def _run_python_module(module: str, *, workdir: Path, env_map: dict[str, Any]) -> int:
    env_lines = dict(env_map)
    env_lines["PYTHONPATH"] = _pythonpath(get_formal_paths())
    command = f"""
set -euo pipefail
{_exports(env_lines)}
python3 -m {module}
"""
    return run_bash(command, workdir=workdir)


def _run_python_script(script: Path, *, workdir: Path, env_map: dict[str, Any], args: list[str] | None = None) -> int:
    env_lines = dict(env_map)
    env_lines["PYTHONPATH"] = _pythonpath(get_formal_paths())
    extra = ""
    if args:
        extra = " " + " ".join(shlex.quote(str(item)) for item in args)
    command = f"""
set -euo pipefail
{_exports(env_lines)}
python3 {shlex.quote(str(script))}{extra}
"""
    return run_bash(command, workdir=workdir)


def run_training_job(cfg: dict[str, Any]) -> int:
    paths = get_formal_paths()
    job = cfg.get("job", {})
    env_cfg = cfg.get("env", {})
    target = str(job.get("target", "")).strip()

    common_env = {
        "FORMAL_ROOT": paths.root,
        "FORMAL_OUTPUT_ROOT": cfg.get("paths", {}).get("output_root", paths.outputs),
        "FORMAL_LOG_ROOT": cfg.get("paths", {}).get("log_root", paths.logs),
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": 0,
        "RAY_TMPDIR": "/tmp/trust_ray",
    }
    common_env.update(env_cfg.get("vars", {}))

    if target == "when2call_grpo":
        return _run_python_script(
            paths.src / "v2" / "scripts_train_when2call_entry.py",
            workdir=paths.root,
            env_map={**common_env, **cfg.get("train", {})},
        )
    if target == "v3_cm2_aug_grpo":
        return _run_python_script(
            paths.src / "v3" / "run_v3_workflow.py",
            workdir=paths.root,
            env_map={**common_env, **cfg.get("train", {}), **cfg.get("data", {})},
        )
    raise ValueError(f"Unsupported training target: {target}")
