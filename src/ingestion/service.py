from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from src.memory.long_term_store import LongTermMemoryStore
from src.models.embedding_client import OpenAICompatibleEmbeddingClient


@dataclass(frozen=True)
class IngestionResult:
    source_id: int
    evidence_id: int
    memory_id: int
    deduplicated: bool
    content_hash: str


class EvidenceIngestionService:
    """Register sources, retain raw evidence, deduplicate, and index memory."""

    def __init__(
        self,
        store: LongTermMemoryStore,
        embedding_client: OpenAICompatibleEmbeddingClient,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client

    def ingest(
        self,
        tenant_id: str,
        brand_id: str,
        source_uri: str,
        raw_content: str,
        title: str = "",
        source_type: str = "web",
        extracted_text: str = "",
        metadata: Dict[str, Any] | None = None,
        observed_at: str | None = None,
        session_id: str = "",
        turn_id: int | None = None,
    ) -> IngestionResult:
        source_id = self.store.register_source(
            tenant_id=tenant_id,
            brand_id=brand_id,
            source_uri=source_uri,
            source_type=source_type,
            title=title,
            metadata=metadata,
        )
        evidence = self.store.ingest_evidence(
            tenant_id=tenant_id,
            brand_id=brand_id,
            source_id=source_id,
            raw_content=raw_content,
            extracted_text=extracted_text,
            metadata=metadata,
            observed_at=observed_at,
        )

        index_text = extracted_text.strip() or raw_content
        started = time.perf_counter()
        embedding = self.embedding_client.embed_query(index_text)
        latency_ms = (time.perf_counter() - started) * 1000.0
        vector = embedding.vectors[0] if embedding.vectors else []
        memory = self.store.store_memory(
            tenant_id=tenant_id,
            brand_id=brand_id,
            memory_type="evidence",
            content=index_text,
            embedding=vector,
            embedding_model=f"{embedding.provider}:{embedding.model_name}:{embedding.model_version}",
            source_evidence_id=evidence.evidence_id,
            importance=0.6,
            observed_at=observed_at,
        )
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
            input_text=index_text,
        )
        return IngestionResult(
            source_id=source_id,
            evidence_id=evidence.evidence_id,
            memory_id=memory.memory_id,
            deduplicated=not evidence.created,
            content_hash=evidence.content_hash,
        )
