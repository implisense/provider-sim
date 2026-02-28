"""
Generiert arl_loss_report.json aus gap-explorer SQLite-Embeddings.

Dummy-Logik: Szenarien mit hoher k-NN-Distanz (semantisch isoliert)
erhalten hohen Loss (Simulation kann sie schlecht generalisieren).

CLI:
    python palestrai_simulation/generate_dummy_losses.py \
        --db gap-explorer/data/scenarios.db \
        --output gap-explorer/data/arl_loss_report.json

Muss aus 04_Apps/ aufgerufen werden (relativer DB-Pfad).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def load_embeddings_from_db(db_path: str) -> tuple:
    """Lädt alle IDs und Embeddings aus der gap-explorer SQLite-DB."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM scenarios ORDER BY created_at"
            ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("DB nicht lesbar: %s -- %s", db_path, exc)
        return [], np.array([])
    if not rows:
        return [], np.array([])
    ids = [r[0] for r in rows]
    embeddings = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, embeddings


def compute_dummy_losses(ids: list, embeddings: np.ndarray) -> dict:
    """
    Berechnet Dummy-Verlustrate aus k-NN-Distanz.

    Isolierte Szenarien (hohe Avg-Distanz) -> hoher Loss.
    Normiert auf [0.1, 0.9].
    """
    if len(embeddings) < 4:
        return {sid: 0.5 for sid in ids}

    k = min(5, len(embeddings) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_distances = distances[:, 1:].mean(axis=1)

    # Normierung auf [0.1, 0.9]
    d_min, d_max = avg_distances.min(), avg_distances.max()
    if d_max == d_min:
        normalized = np.full_like(avg_distances, 0.5)
    else:
        normalized = 0.1 + 0.8 * (avg_distances - d_min) / (d_max - d_min)

    return {sid: round(float(loss), 4) for sid, loss in zip(ids, normalized)}


def generate_dummy_losses(db_path: str, output_path: str) -> dict:
    """Hauptfunktion: liest DB, schreibt JSON."""
    ids, embeddings = load_embeddings_from_db(db_path)
    if not ids:
        logger.warning("Keine Szenarien in DB: %s", db_path)
        report = {
            "timestamp": datetime.now().isoformat(),
            "scenario_losses": {},
            "meta": {
                "source": "dummy_embedding_density",
                "n_scenarios": 0,
                "db_path": db_path,
            },
        }
    else:
        losses = compute_dummy_losses(ids, embeddings)
        report = {
            "timestamp": datetime.now().isoformat(),
            "scenario_losses": losses,
            "meta": {
                "source": "dummy_embedding_density",
                "n_scenarios": len(ids),
                "db_path": db_path,
            },
        }
        logger.info("Dummy-Losses generiert fuer %d Szenarien", len(ids))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))
    logger.info("arl_loss_report.json geschrieben: %s", output_path)
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generiert Dummy-ARL-Verlustdaten")
    parser.add_argument("--db", default="gap-explorer/data/scenarios.db",
                        help="Pfad zur gap-explorer SQLite-DB")
    parser.add_argument("--output", default="gap-explorer/data/arl_loss_report.json",
                        help="Pfad zur Ausgabedatei")
    args = parser.parse_args()
    report = generate_dummy_losses(args.db, args.output)
    print(f"Fertig: {len(report['scenario_losses'])} Verlusteintraege nach {args.output}")


if __name__ == "__main__":
    main()
