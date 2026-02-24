"""External PPO training loop for palaestrAI.

Runs `palaestrai experiment-start` N times with episodes=1.
Between runs, PPOBrain/PPOMuscle auto-load the checkpoint saved by the
previous run (via checkpoint_path in the YAML params).

Fortschritt wird in checkpoints/progress.json gespeichert — Training kann
jederzeit mit Ctrl+C unterbrochen und mit dem gleichen Befehl fortgesetzt werden.

Usage:
    python experiments/train_ppo.py [n_episodes] [config_yaml]

Defaults:
    n_episodes = 50
    config_yaml = experiments/soja_arl_ppo.yaml
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG = os.path.join(_BASE, "experiments", "soja_arl_ppo.yaml")
_CHECKPOINT_DIR = os.path.join(_BASE, "experiments", "checkpoints")
_PROGRESS_FILE = os.path.join(_CHECKPOINT_DIR, "progress.json")


def _load_progress() -> dict:
    if os.path.isfile(_PROGRESS_FILE):
        with open(_PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_episodes": 0, "total_episodes": 0}


def _save_progress(completed: int, total: int) -> None:
    with open(_PROGRESS_FILE, "w") as f:
        json.dump({"completed_episodes": completed, "total_episodes": total}, f, indent=2)


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

    progress = _load_progress()
    done = progress["completed_episodes"]

    # Wenn bereits mehr Episoden abgeschlossen als angefordert: fertig
    if done >= n_episodes:
        print(f"Bereits {done} Episoden abgeschlossen — nichts zu tun.")
        print(f"Zum Neustart: progress.json löschen oder n_episodes erhöhen.")
        return

    start_ep = done + 1
    print(f"PPO Training Loop")
    print(f"  Ziel      : {n_episodes} Episoden")
    print(f"  Bereits   : {done} abgeschlossen")
    print(f"  Fortsetzen: ab Episode {start_ep}")
    print(f"  Config    : {config}")
    print(f"  Checkpts  : {_CHECKPOINT_DIR}")

    failed = 0
    for ep in range(start_ep, n_episodes + 1):
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
            _save_progress(ep, n_episodes)

    print(f"\n{'='*60}")
    print(f"  Training complete: {n_episodes} Episoden")
    print(f"  Checkpoints in: {_CHECKPOINT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
