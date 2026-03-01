"""
Generiert arl_loss_report.json aus gap-explorer SQLite-Embeddings.

Dummy-Logik: Szenarien mit hoher k-NN-Distanz (semantisch isoliert)
erhalten hohen Loss (Simulation kann sie schlecht generalisieren).
Zusätzlich wird ein Dimensions-Scarcity-Bonus berücksichtigt:
Szenarien mit seltenen Dimensionen erhalten einen höheren Loss.

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


def load_dimensions_from_db(db_path: str) -> dict:
    """Lädt scenario_id → [dimensions] aus der gap-explorer SQLite-DB."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT id, dimensions FROM scenarios ORDER BY created_at"
            ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("DB nicht lesbar: %s -- %s", db_path, exc)
        return {}
    result = {}
    for row_id, dims_raw in rows:
        if dims_raw:
            try:
                import json as _json
                result[row_id] = _json.loads(dims_raw)
            except Exception:
                result[row_id] = []
        else:
            result[row_id] = []
    return result


def compute_dimension_scarcity(scenario_dims: dict) -> dict:
    """
    Berechnet Scarcity-Score pro Szenario basierend auf Dimensions-Häufigkeit.

    Szenario mit seltenen Dimensionen → hoher Score (nahe 1.0).
    Szenario mit häufigen Dimensionen → niedriger Score (nahe 0.0).
    Szenario ohne Dimensionen → neutraler Score 0.5.

    Args:
        scenario_dims: {scenario_id: [dimension, ...]}

    Returns:
        {scenario_id: scarcity_score}
    """
    if not scenario_dims:
        return {}

    global_counts: dict = {}
    total = len(scenario_dims)
    for dims in scenario_dims.values():
        for d in dims:
            global_counts[d] = global_counts.get(d, 0) + 1

    dim_freq = {d: c / total for d, c in global_counts.items()}

    result = {}
    for sid, dims in scenario_dims.items():
        if not dims:
            result[sid] = 0.5
        else:
            result[sid] = float(np.mean([1.0 - dim_freq.get(d, 0.0) for d in dims]))
    return result


def compute_dummy_losses(
    ids: list,
    embeddings: np.ndarray,
    scenario_dims: dict = None,
    scarcity_weight: float = 0.3,
) -> dict:
    """
    Berechnet Dummy-Verlustrate aus k-NN-Distanz + optionalem Dimensions-Scarcity-Bonus.

    loss = (1 - scarcity_weight) * knn_loss + scarcity_weight * scarcity_bonus
    Normiert auf [0.1, 0.9].
    """
    if len(embeddings) < 4:
        return {sid: 0.5 for sid in ids}

    k = min(5, len(embeddings) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_distances = distances[:, 1:].mean(axis=1)

    # knn-Loss normieren auf [0, 1]
    d_min, d_max = avg_distances.min(), avg_distances.max()
    if d_max == d_min:
        knn_normalized = np.full_like(avg_distances, 0.5)
    else:
        knn_normalized = (avg_distances - d_min) / (d_max - d_min)

    # Scarcity berechnen (optional)
    if scenario_dims and scarcity_weight > 0.0:
        scarcity = compute_dimension_scarcity(scenario_dims)
        scarcity_arr = np.array([scarcity.get(sid, 0.5) for sid in ids])
    else:
        scarcity_arr = np.zeros(len(ids))
        scarcity_weight = 0.0

    combined = (1.0 - scarcity_weight) * knn_normalized + scarcity_weight * scarcity_arr

    # Auf [0.1, 0.9] normieren
    c_min, c_max = combined.min(), combined.max()
    if c_max == c_min:
        final = np.full_like(combined, 0.5)
    else:
        final = 0.1 + 0.8 * (combined - c_min) / (c_max - c_min)

    return {sid: round(float(loss), 4) for sid, loss in zip(ids, final)}


def generate_dummy_losses(db_path: str, output_path: str, scarcity_weight: float = 0.3) -> dict:
    """Hauptfunktion: liest DB, schreibt JSON."""
    ids, embeddings = load_embeddings_from_db(db_path)
    scenario_dims = load_dimensions_from_db(db_path)

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
        losses = compute_dummy_losses(ids, embeddings, scenario_dims, scarcity_weight)
        report = {
            "timestamp": datetime.now().isoformat(),
            "scenario_losses": losses,
            "meta": {
                "source": "dummy_embedding_density+dimension_scarcity",
                "n_scenarios": len(ids),
                "scarcity_weight": scarcity_weight,
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
    parser.add_argument(
        "--scarcity-weight", type=float, default=0.3,
        help="Gewichtung des Dimension-Scarcity-Bonus (0.0 = reiner knn-Loss)"
    )
    args = parser.parse_args()
    report = generate_dummy_losses(args.db, args.output, args.scarcity_weight)
    print(f"Fertig: {len(report['scenario_losses'])} Verlusteintraege nach {args.output}")


if __name__ == "__main__":
    main()
