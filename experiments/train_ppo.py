"""External PPO training loop for palaestrAI.

Runs `palaestrai experiment-start` N times with episodes=1.
Between runs, PPOBrain/PPOMuscle auto-load the checkpoint saved by the
previous run (via checkpoint_path in the YAML params).

Usage:
    python experiments/train_ppo.py [n_episodes] [config_yaml]

Defaults:
    n_episodes = 50
    config_yaml = experiments/soja_arl_ppo.yaml
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG = os.path.join(_BASE, "experiments", "soja_arl_ppo.yaml")
_CHECKPOINT_DIR = os.path.join(_BASE, "experiments", "checkpoints")


def _kill_stale_processes() -> None:
    for pattern in ("palaestrai", "spawn_main", "resource_tracker"):
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
    time.sleep(2)


def _run_episode(config: str, ep: int, total: int) -> bool:
    print(f"\n{'='*60}")
    print(f"  Episode {ep}/{total}")
    print(f"{'='*60}")
    result = subprocess.run(
        ["palaestrai", "experiment-start", config],
        cwd=_BASE,
    )
    return result.returncode == 0


def main() -> None:
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    config = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_CONFIG

    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

    print(f"PPO Training Loop")
    print(f"  Episodes : {n_episodes}")
    print(f"  Config   : {config}")
    print(f"  Checkpts : {_CHECKPOINT_DIR}")

    failed = 0
    for ep in range(1, n_episodes + 1):
        _kill_stale_processes()
        ok = _run_episode(config, ep, n_episodes)
        if not ok:
            failed += 1
            print(f"[train_ppo] Episode {ep} returned non-zero exit code.")
            if failed >= 3:
                print("[train_ppo] 3 consecutive failures — aborting.")
                sys.exit(1)
        else:
            failed = 0

    print(f"\n{'='*60}")
    print(f"  Training complete: {n_episodes} episodes")
    print(f"  Checkpoints in: {_CHECKPOINT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
