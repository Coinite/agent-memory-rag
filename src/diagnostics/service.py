from __future__ import annotations

from typing import Any, Dict

from src.memory.long_term_store import LongTermMemoryStore


class DiagnosticsService:
    """Persist per-turn evaluation metrics for trend analysis."""

    def __init__(self, store: LongTermMemoryStore) -> None:
        self.store = store

    def record_scores(
        self,
        tenant_id: str,
        brand_id: str,
        scores: Dict[str, Any],
        session_id: str,
        turn_id: int,
        route: str,
    ) -> None:
        dimensions = {"session_id": session_id, "turn_id": turn_id, "route": route}
        for name, value in scores.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            self.store.record_metric(
                tenant_id=tenant_id,
                brand_id=brand_id,
                metric_name=f"agent_quality.{name}",
                metric_value=numeric,
                dimensions=dimensions,
            )
