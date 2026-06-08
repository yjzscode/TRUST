from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping


def build_env(extra: Mapping[str, object] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            env[str(key)] = str(value)
    return env


def run_bash(
    command: str,
    *,
    workdir: str | Path,
    env: Mapping[str, object] | None = None,
) -> int:
    proc = subprocess.run(
        # Avoid login-shell startup files from /etc/profile.d, which may fail under strict nounset.
        ["bash", "-c", command],
        cwd=str(Path(workdir).resolve()),
        env=build_env(env),
        check=False,
    )
    return int(proc.returncode)
