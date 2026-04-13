"""Curriculum-Training: erst vs. Random, dann Self-Play.

Phase 1a: PPO-Defender vs. RandomMuscle  (Defender lernt Grundabwehr)
Phase 1b: PPO-Attacker vs. RandomMuscle  (Attacker lernt Grundangriff)
Phase 2:  PPO-Defender vs. PPO-Attacker  (Self-Play mit vortrainierten Weights)

Die Curriculum-Checkpoints aus Phase 1 werden automatisch in die Self-Play-
Pfade kopiert, sodass Phase 2 mit vortrainierten Weights startet.

Usage:
    python experiments/train_curriculum.py [phase1_episodes] [phase2_episodes]

Defaults:
    phase1_episodes = 50   (je Agent, also 50 Def + 50 Att)
    phase2_episodes = 100
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


_BASE = Path(__file__).parent.parent
_CKPT = _BASE / "experiments" / "checkpoints"
_PALAESTRAI = Path(sys.executable).parent / "palaestrai"

_CFG_DEF = _BASE / "experiments" / "configs" / "soja_curriculum_def.yaml"
_CFG_ATT = _BASE / "experiments" / "configs" / "soja_curriculum_att.yaml"
_CFG_SELF = _BASE / "experiments" / "configs" / "soja_arl_ppo_vm.yaml"

_PROGRESS = _CKPT / "curriculum_progress.json"


def _load_progress() -> dict:
    if _PROGRESS.exists():
        return json.loads(_PROGRESS.read_text())
    return {"phase": "1a", "completed": 0, "phase1_episodes": 0, "phase2_episodes": 0}


def _save_progress(p: dict) -> None:
    _CKPT.mkdir(exist_ok=True)
    _PROGRESS.write_text(json.dumps(p, indent=2))


def _kill_stale() -> None:
    for pat in ("palaestrai", "spawn_main", "resource_tracker"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(2)


def _run_episode(config: Path, ep: int, label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}  Episode {ep}")
    print(f"{'='*60}")
    result = subprocess.run(
        [str(_PALAESTRAI), "experiment-start", str(config)],
        cwd=str(_BASE),
    )
    return result.returncode == 0


def _run_phase(
    config: Path,
    label: str,
    start_ep: int,
    total_ep: int,
    progress: dict,
    progress_key: str,
) -> int:
    """Fuehrt start_ep..total_ep Episoden durch; gibt abgeschlossene Anzahl zurueck."""
    failed = 0
    for ep in range(start_ep, total_ep + 1):
        _kill_stale()
        ok = _run_episode(config, ep, label)
        if not ok:
            failed += 1
            print(f"[curriculum] Warnung: Episode {ep} fehlgeschlagen (exit != 0)")
            if failed >= 3:
                print("[curriculum] 3 aufeinanderfolgende Fehler — Abbruch.")
                sys.exit(1)
        else:
            failed = 0
        progress[progress_key] = ep
        _save_progress(progress)
    return total_ep


def main() -> None:
    phase1_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    phase2_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    _CKPT.mkdir(exist_ok=True)
    progress = _load_progress()

    # Ziele aus Kommandozeilenargumenten ueberschreiben
    progress["phase1_episodes"] = phase1_ep
    progress["phase2_episodes"] = phase2_ep

    print("\nCurriculum-Training")
    print(f"  Phase 1a: {phase1_ep} Episoden  PPO-Defender vs. Random-Attacker")
    print(f"  Phase 1b: {phase1_ep} Episoden  PPO-Attacker vs. Random-Defender")
    print(f"  Phase 2:  {phase2_ep} Episoden  Self-Play (PPO vs. PPO)")
    print(f"  Fortschritt: {_PROGRESS}")

    # ------------------------------------------------------------------
    # Phase 1a: Defender vs. Random
    # ------------------------------------------------------------------
    if progress["phase"] == "1a":
        done = progress.get("completed", 0)
        start = done + 1
        if start > phase1_ep:
            print("\nPhase 1a bereits abgeschlossen.")
        else:
            print(f"\n--- Phase 1a: Defender vs. Random (ab Episode {start}/{phase1_ep}) ---")
            _run_phase(_CFG_DEF, "Phase1a [Def vs. Rand]", start, phase1_ep, progress, "completed")
        progress["phase"] = "1b"
        progress["completed"] = 0
        _save_progress(progress)

    # ------------------------------------------------------------------
    # Phase 1b: Attacker vs. Random
    # ------------------------------------------------------------------
    if progress["phase"] == "1b":
        done = progress.get("completed", 0)
        start = done + 1
        if start > phase1_ep:
            print("\nPhase 1b bereits abgeschlossen.")
        else:
            print(f"\n--- Phase 1b: Attacker vs. Random (ab Episode {start}/{phase1_ep}) ---")
            _run_phase(_CFG_ATT, "Phase1b [Att vs. Rand]", start, phase1_ep, progress, "completed")

        # Curriculum-Checkpoints in Self-Play-Pfade kopieren
        print("\n[curriculum] Kopiere Curriculum-Checkpoints → Self-Play-Pfade")
        for src, dst in [
            (_CKPT / "curriculum_defender.pt", _CKPT / "selfplay_defender.pt"),
            (_CKPT / "curriculum_attacker.pt", _CKPT / "selfplay_attacker.pt"),
        ]:
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  {src.name} → {dst.name}")
            else:
                print(f"  WARNUNG: {src.name} nicht gefunden — Self-Play startet mit Random-Weights")

        progress["phase"] = "2"
        progress["completed"] = 0
        _save_progress(progress)

    # ------------------------------------------------------------------
    # Phase 2: Self-Play mit vortrainierten Checkpoints
    # ------------------------------------------------------------------
    if progress["phase"] == "2":
        # Self-Play-Config auf selfplay_*.pt Checkpoints umstellen
        _patch_selfplay_config()

        done = progress.get("completed", 0)
        start = done + 1
        if start > phase2_ep:
            print("\nPhase 2 bereits abgeschlossen.")
        else:
            print(f"\n--- Phase 2: Self-Play (ab Episode {start}/{phase2_ep}) ---")
            cfg_selfplay = _BASE / "experiments" / "configs" / "soja_curriculum_selfplay.yaml"
            _run_phase(cfg_selfplay, "Phase2 [Self-Play]", start, phase2_ep, progress, "completed")

        progress["phase"] = "done"
        _save_progress(progress)

    print("\n" + "=" * 60)
    print("  Curriculum-Training abgeschlossen.")
    print(f"  Finale Checkpoints: {_CKPT}/selfplay_*.pt")
    print("=" * 60)


def _patch_selfplay_config() -> None:
    """Erzeugt soja_curriculum_selfplay.yaml mit selfplay_*.pt Pfaden."""
    import yaml
    src = _CFG_SELF
    doc = yaml.safe_load(src.read_text())
    doc["uid"] = "provider-soy-curriculum-selfplay"
    agents = doc["schedule"][0]["phase_train"]["agents"]
    for agent in agents:
        role = agent["name"]  # "attacker" oder "defender"
        new_ckpt = f"experiments/checkpoints/selfplay_{role}.pt"
        if "brain" in agent and "params" in agent["brain"]:
            agent["brain"]["params"]["checkpoint_path"] = new_ckpt
        if "muscle" in agent and "params" in agent["muscle"]:
            agent["muscle"]["params"]["checkpoint_path"] = new_ckpt

    out = _BASE / "experiments" / "configs" / "soja_curriculum_selfplay.yaml"
    out.write_text(yaml.dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False))
    print(f"[curriculum] {out.name} erzeugt")


if __name__ == "__main__":
    main()
