from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from src.memory.long_term_store import LongTermMemoryStore
from src.models.embedding_client import OpenAICompatibleEmbeddingClient


class MemoryRetrievalService:
    """Tenant/brand-scoped hybrid semantic and lexical retrieval."""

    def __init__(
        self,
        store: LongTermMemoryStore,
        embedding_client: OpenAICompatibleEmbeddingClient,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client

    def search(
        self,
        tenant_id: str,
        brand_id: str,
        query: str,
        limit: int = 8,
        memory_types: Iterable[str] | None = None,
        session_id: str = "",
        turn_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        embedding = self.embedding_client.embed_query(query)
        latency_ms = (time.perf_counter() - started) * 1000.0
        vector = embedding.vectors[0] if embedding.vectors else []
        self.store.record_model_call(
            tenant_id=tenant_id,
            brand_id=brand_id,
            provider=embedding.provider,
            model_name=embedding.model_name,
            model_version=embedding.model_version,
            capability="embedding",
            session_id=session_id,
            turn_id=turn_id,
            used_fallback=embedding.used_fallback,
            latency_ms=latency_ms,
            input_text=query,
        )
        return self.store.search_memories(
            tenant_id=tenant_id,
            brand_id=brand_id,
            query=query,
            query_embedding=vector,
            limit=limit,
            memory_types=memory_types,
        )
