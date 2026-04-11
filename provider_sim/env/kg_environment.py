"""KG-parametrisiertes ProviderEnvironment für palaestrAI.

Lädt beim Start KG-Schocks aus einer JSON-Datei (erzeugt von coypu-kg-analyser)
und injiziert sie als PDL-Events in das Szenario.

Verwendung in palaestrAI-Experiment-YAML::

    environments:
      - environment:
          name: provider_sim.env.kg_environment:KgShocksProviderEnvironment
          uid: provider_env
          params:
            pdl_source: /path/to/s1-soja.pdl.yaml
            shocks_json: /path/to/shocks_2026-03-04.json
            id_mapping: S1_KG_TO_PDL
            max_ticks: 365
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from provider_sim.adapters.kg_shocks import S1_KG_TO_PDL, apply_kg_shocks
from provider_sim.env.environment import ProviderEnvironment
from provider_sim.pdl.model import PdlDocument
from provider_sim.pdl.parser import load_pdl

_MAPPING_REGISTRY: dict[str, dict[str, str]] = {
    "S1_KG_TO_PDL": S1_KG_TO_PDL,
}


class KgShocksProviderEnvironment(ProviderEnvironment):
    """ProviderEnvironment mit automatischer KG-Schock-Anwendung beim Start.

    Zusätzliche Parameter (ergänzend zu ProviderEnvironment):

    Args:
        pdl_source: Pfad zur PDL-YAML-Datei oder PdlDocument-Objekt.
        shocks_json: Pfad zur JSON-Datei mit KG-Schocks
            (Format: {"shocks": [{"target_id": ..., "shock_type": ..., "magnitude": ...}, ...]}).
        id_mapping: Name des Mapping-Dicts. Erlaubte Werte: "S1_KG_TO_PDL" (default).
        duration_days: Dauer der injizierten KG-Events in Tagen (default: 365).
        Alle weiteren kwargs werden an ProviderEnvironment weitergegeben.
    """

    def __init__(
        self,
        pdl_source: Union[str, PdlDocument],
        shocks_json: str,
        id_mapping: str = "S1_KG_TO_PDL",
        duration_days: int = 365,
        **kwargs: Any,
    ) -> None:
        shocks = json.loads(Path(shocks_json).read_text(encoding="utf-8"))["shocks"]
        mapping = _MAPPING_REGISTRY.get(id_mapping, S1_KG_TO_PDL)
        doc = load_pdl(pdl_source) if isinstance(pdl_source, str) else pdl_source
        enriched_doc = apply_kg_shocks(
            doc, shocks, id_mapping=mapping, duration_days=duration_days
        )
        super().__init__(pdl_source=enriched_doc, **kwargs)
