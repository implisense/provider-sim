"""Live-Demo: PROVIDER als palaestrAI Custom Environment.

Interaktives Skript fuer den Vortrag — demonstriert das Zero-Sum-Prinzip
durch Echtzeit-Visualisierung von Attacker vs. Defender.

Ausfuehren:
    cd palestrai_simulation
    python tutorials/run_live_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

_PDL_PATH = _BASE / "scenarios" / "s1-soja.pdl.yaml"
_BAR_WIDTH = 30
_TICKS = 20

# ANSI-Farben (werden abgeschaltet wenn Terminal kein TTY ist)
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

_RED    = lambda t: _c("31", t)
_GREEN  = lambda t: _c("32", t)
_YELLOW = lambda t: _c("33", t)
_BOLD   = lambda t: _c("1",  t)
_DIM    = lambda t: _c("2",  t)


def _bar(value: float, width: int = _BAR_WIDTH, color_fn=None) -> str:
    filled = round(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return color_fn(bar) if color_fn else bar


def _pause(msg: str = "") -> None:
    label = _DIM(f"  → [Enter] {msg}") if msg else _DIM("  → [Enter] weiter")
    input(label)
    print()


def _section(title: str) -> None:
    print()
    print(_BOLD("=" * 60))
    print(_BOLD(f"  {title}"))
    print(_BOLD("=" * 60))
    print()


def _ask_strength(label: str, default: float) -> float:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if 0.0 <= val <= 1.0:
                return val
            print(_YELLOW("  Bitte einen Wert zwischen 0.0 und 1.0 eingeben."))
        except ValueError:
            print(_YELLOW("  Ungueltige Eingabe."))


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Block 1 — Ueberblick
    # ------------------------------------------------------------------
    _section("PROVIDER — palaestrAI Custom Environment  |  Live-Demo")

    print("  Szenario : Soja-Futtermittel-Lieferkette (S1)")
    print("  Modell   : 20 Entities · 18 Events · 99 Sensoren · 40 Aktuatoren")
    print("  Ziel     : Zero-Sum ARL — Attacker vs. Defender")
    print()
    print(_DIM("  Das Environment implementiert das palaestrAI-ABC mit zwei Methoden:"))
    print(_DIM("  start_environment()  →  EnvironmentBaseline"))
    print(_DIM("  update(actuators)    →  EnvironmentState"))

    _pause("PDL-Szenario laden")

    # ------------------------------------------------------------------
    # Block 2 — PDL laden & Environment aufbauen
    # ------------------------------------------------------------------
    _section("Schritt 1 — PDL laden & Environment aufbauen")

    from provider_sim.pdl.parser import load_pdl
    from provider_sim.env.environment import ProviderEnvironment

    print("  Lade: scenarios/s1-soja.pdl.yaml …", end=" ", flush=True)
    doc = load_pdl(_PDL_PATH)
    print(_GREEN("OK"))
    print(f"  Entities : {len(doc.entities)}")
    print(f"  Events   : {len(doc.events)}")
    print()

    print("  Baue ProviderEnvironment …", end=" ", flush=True)
    env = ProviderEnvironment(_PDL_PATH, seed=42, max_ticks=_TICKS)
    print(_GREEN("OK"))
    print(f"  Sensoren    : {len(env.sensor_names)}")
    print(f"  Aktuatoren  : {len(env.actuator_names)}")
    print()

    print(_DIM("  Sensor-Beispiele:"))
    examples = [
        ("entity.brazil_farms.supply", "Box(0, 2)  — normiertes Angebot"),
        ("event.brazil_drought.active", "Discrete(2) — Ereignis aktiv?"),
        ("sim.tick",                    "Box(0, 20) — aktueller Tick"),
    ]
    for sid, desc in examples:
        print(f"    {_YELLOW(sid):50s}  {_DIM(desc)}")

    _pause("interaktive What-If-Demo starten")

    # ------------------------------------------------------------------
    # Block 3 — Interaktive Simulation
    # ------------------------------------------------------------------
    while True:
        _section("Schritt 2 — What-If: Angriffsstärke wählen")

        print("  Wie stark greift der Attacker an?")
        print(_DIM("  0.0 = kein Angriff   0.5 = mittlerer Angriff   1.0 = maximaler Angriff"))
        print()
        attack  = _ask_strength("Attacker-Stärke (0.0 – 1.0)", default=0.5)
        defend  = _ask_strength("Defender-Stärke (0.0 – 1.0)", default=0.0)

        print()
        print(f"  Starte {_TICKS} Ticks mit  "
              f"{_RED(f'Attacker={attack:.1f}')}  /  "
              f"{_GREEN(f'Defender={defend:.1f}')}")
        print()

        # Header
        header = (
            f"  {'Tick':>4}  "
            f"{'Attacker-Reward':>15}  "
            f"{'Defender-Reward':>15}  "
            f"{'Lieferkette':>{_BAR_WIDTH + 2}}"
        )
        print(_BOLD(header))
        print(_DIM("  " + "-" * (len(header) - 2)))

        obs, _ = env.reset_dict()
        history_health: list[float] = []

        for tick in range(1, _TICKS + 1):
            actions: dict = {}
            for ent in doc.entities:
                actions[f"attacker.{ent.id}"] = attack
                actions[f"defender.{ent.id}"] = defend

            obs, rewards, done = env.step_dict(actions)

            att_r = rewards["reward.attacker"]
            def_r = rewards["reward.defender"]
            history_health.append(def_r)

            # Balkenfarbe: gruen wenn stabil, rot wenn kritisch
            if def_r >= 0.7:
                bar_fn = _GREEN
            elif def_r >= 0.4:
                bar_fn = _YELLOW
            else:
                bar_fn = _RED

            bar = _bar(def_r, color_fn=bar_fn)
            print(
                f"  {tick:>4}  "
                f"{_RED(f'{att_r:>15.4f}')}  "
                f"{_GREEN(f'{def_r:>15.4f}')}  "
                f"  {bar}"
            )

            time.sleep(0.05)  # kurze Pause fuer Live-Gefuehl
            if done:
                break

        # Zusammenfassung
        mean_h = float(np.mean(history_health))
        min_h  = float(np.min(history_health))
        print()
        print(_BOLD("  Zusammenfassung:"))
        print(f"    Ø Defender-Reward : {_GREEN(f'{mean_h:.4f}')}")
        print(f"    Min Health        : {_RED(f'{min_h:.4f}') if min_h < 0.5 else _YELLOW(f'{min_h:.4f}')}")

        if mean_h >= 0.75:
            verdict = _GREEN("Lieferkette stabil")
        elif mean_h >= 0.50:
            verdict = _YELLOW("Lieferkette unter Druck")
        else:
            verdict = _RED("KRITISCHER VERSORGUNGSENGPASS")
        print(f"    Bewertung         : {verdict}")
        print()

        again = input(_DIM("  Nochmal mit anderen Werten? [j/N]: ")).strip().lower()
        if again not in ("j", "ja", "y", "yes"):
            break

    # ------------------------------------------------------------------
    # Block 4 — Abschluss & Takeaways
    # ------------------------------------------------------------------
    _section("Takeaways")

    takeaways = [
        ("1", "PDL → Sensoren/Aktuatoren  automatisch generierbar"),
        ("2", "ABC: nur start_environment() + update()  +  Stub-Fallback"),
        ("3", "Zero-Sum ARL:  AttackerObjective / DefenderObjective sauber getrennt"),
    ]
    for num, text in takeaways:
        print(f"  {_BOLD(num)}.  {text}")

    print()
    print(_DIM("  Tutorial-Skript  :  tutorials/run_provider_tutorial.py"))
    print(_DIM("  Dokumentation    :  tutorials/provider_environment_tutorial.md"))
    print()
    print(_BOLD("  Danke!"))
    print()


if __name__ == "__main__":
    main()
