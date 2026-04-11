 Self-Play im aktuellen Setup.md                      

  Was Self-Play bedeutet                                                                                                           
   
  Im klassischen Self-Play (AlphaZero) spielt ein Netz gegen sich selbst — gleiche Gewichte, beide Rollen. Hier ist das nicht      
  direkt möglich, weil Attacker und Defender asymmetrisch sind (Budget 0.8 vs. 0.4, unterschiedliche Semantik der Aktionen).     
                                                                                                                                 
  Aber es gibt drei pragmatische Varianten, die im aktuellen Setup testbar wären:

  ---
  Variante 1: Frozen Opponent (einfachste)

  Idee: Trainiere einen Agenten, während der andere eingefroren ist.

  Phase A (25 Ep): Attacker lernt  →  Defender = DummyBrain (zufällig)
  Phase B (25 Ep): Defender lernt  →  Attacker = letzter Checkpoint eingefroren

  Umsetzung: Zwei YAML-Configs — soja_attacker_only.yaml (DummyBrain für Defender) und soja_defender_only.yaml (DummyBrain für
  Attacker).

  Was das zeigt: "Best Response" — wie gut kann ein Agent werden, wenn der Gegner sich nicht anpasst?

  ---
  Variante 2: Checkpoint Opponent (Fictitious Self-Play)

  Idee: Trainiere den Attacker in jeder Episode gegen einen zufällig gesampelten vergangenen Defender-Checkpoint — und umgekehrt.

  Episode 1:  Attacker vs. Defender_ep0  (random)
  Episode 5:  Attacker vs. Defender_ep3  (zufällig aus Pool)
  Episode 10: Attacker vs. Defender_ep7
  ...

  Vorteil: Verhindert Overfitting auf eine einzige Gegner-Strategie ("Cycling" — Attacker lernt nur Defender_ep5 zu schlagen,
  verliert gegen alle anderen).

  Umsetzung: train_ppo.py würde nach jeder Episode einen nummerierten Checkpoint speichern (attacker_ep01.pt, attacker_ep02.pt,
  ...) und beim nächsten Run zufällig einen davon als Opponent laden.

  ---
  Variante 3: Alternating Training (nächster sinnvoller Schritt)

  Idee: Beide lernen, aber abwechselnd — einer lernt, während der andere eingefroren ist.

  Ep 1–10:  Nur Attacker lernt   (Defender lädt letzten Checkpoint, kein PPO-Update)
  Ep 11–20: Nur Defender lernt   (Attacker eingefroren)
  Ep 21–30: Nur Attacker lernt
  ...

  Was das zeigt: Ob ein Agent eine stabile Strategie entwickeln kann, bevor der Gegner reagiert. Ähnlich wie iteriertes
  Best-Response-Lernen.

  ---
  Was das aktuelle Setup bereits ist

  Das jetzige Training (beide lernen gleichzeitig) ist Concurrent Self-Play — die einfachste Form von MARL. Das Problem dabei:

  Beide Agenten verändern sich gleichzeitig → aus Sicht jedes Agenten ist die Umgebung nicht-stationär. PPO setzt aber eine
  stationäre Umgebung voraus.

  Das erklärt, warum die Rewards in den 3 Test-Episoden kaum variiert haben (~0.23–0.25 für Attacker) — beide Agenten "drehen sich"
   ohne echten Fortschritt.

  ---
  Empfehlung für einen konkreten Test

  Variante 1 ist in ~30 Minuten implementierbar:

  1. Kopie von soja_arl_ppo.yaml → soja_attacker_only.yaml: Defender bekommt DummyBrain/DummyMuscle
  2. 25 Episoden nur Attacker trainieren
  3. Attacker-Reward sollte deutlich über 0.25 steigen, wenn der Defender sich nicht anpasst
  4. Dann Defender mit dem trainierten Attacker als Opponent trainieren

  Das würde direkt zeigen, ob der PPO-Algorithmus überhaupt lernt — was im aktuellen simultanen Setup schwer zu isolieren ist.