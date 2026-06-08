from __future__ import annotations

# Import reward manager for registry side effects before VERL trainer startup.
from v3.train.rewards import mixed_reward_manager as _mixed_reward_manager  # noqa: F401
from verl.trainer.main_ppo import main


if __name__ == "__main__":
    raise SystemExit(main())
