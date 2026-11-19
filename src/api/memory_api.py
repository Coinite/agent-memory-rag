from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.config import settings
from src.ingestion.service import EvidenceIngestionService
from src.memory.long_term_store import LongTermMemoryStore
from src.models.embedding_client import OpenAICompatibleEmbeddingClient
from src.retrieval.service import MemoryRetrievalService


class Scope(BaseModel):
    tenant_id: str = Field(min_length=1)
    brand_id: str = Field(min_length=1)


class EvidenceIngestRequest(Scope):
    source_uri: str = Field(min_length=1)
    raw_content: str = Field(min_length=1)
    title: str = ""
    source_type: str = "web"
    extracted_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    observed_at: str | None = None


class MemorySearchRequest(Scope):
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=100)
    memory_types: List[str] = Field(default_factory=list)


class BrandFactRequest(Scope):
    fact_key: str = Field(min_length=1)
    fact_value: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_evidence_id: int | None = None
    valid_from: str | None = None
    valid_to: str | None = None


class MetricSnapshotRequest(Scope):
    metric_name: str = Field(min_length=1)
    metric_value: float
    unit: str = ""
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    observed_at: str | None = None


class MemoryAPI:
    """Application API for tenant/brand-scoped memory workflows."""

    def __init__(
        self,
        store: LongTermMemoryStore,
        embedding_client: OpenAICompatibleEmbeddingClient,
    ) -> None:
        self.store = store
        self.ingestion = EvidenceIngestionService(store, embedding_client)
        self.retrieval = MemoryRetrievalService(store, embedding_client)

    def ingest_evidence(self, request: EvidenceIngestRequest) -> Dict[str, Any]:
        result = self.ingestion.ingest(**request.model_dump())
        return {
            "source_id": result.source_id,
            "evidence_id": result.evidence_id,
            "memory_id": result.memory_id,
            "deduplicated": result.deduplicated,
            "content_hash": result.content_hash,
        }

    def search_memory(self, request: MemorySearchRequest) -> List[Dict[str, Any]]:
        return self.retrieval.search(
            tenant_id=request.tenant_id,
            brand_id=request.brand_id,
            query=request.query,
            limit=request.limit,
            memory_types=request.memory_types,
        )

    def upsert_brand_fact(self, request: BrandFactRequest) -> Dict[str, int]:
        fact_id = self.store.upsert_brand_fact(**request.model_dump())
        return {"fact_id": fact_id}

    def list_brand_facts(self, tenant_id: str, brand_id: str) -> List[Dict[str, Any]]:
        return self.store.get_brand_facts(tenant_id=tenant_id, brand_id=brand_id)

    def record_metric(self, request: MetricSnapshotRequest) -> Dict[str, int]:
        metric_id = self.store.record_metric(**request.model_dump())
        return {"metric_id": metric_id}


def build_memory_api(db_path: str | None = None) -> MemoryAPI:
    store = LongTermMemoryStore(db_path or settings.memory_db_path)
    embedding_client = OpenAICompatibleEmbeddingClient()
    return MemoryAPI(store=store, embedding_client=embedding_client)


def create_fastapi_app(api: MemoryAPI | None = None):
    """Create the optional HTTP API; install requirements-api.txt first."""

    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("FastAPI support requires: pip install -r requirements-api.txt") from exc

    memory_api = api or build_memory_api()
    app = FastAPI(title="Long-Term Memory Search API", version="1.0.0")

    @app.post("/v1/evidence")
    def ingest_evidence(request: EvidenceIngestRequest):
        return memory_api.ingest_evidence(request)

    @app.post("/v1/memory/search")
    def search_memory(request: MemorySearchRequest):
        return {"items": memory_api.search_memory(request)}

    @app.put("/v1/brand-facts")
    def upsert_brand_fact(request: BrandFactRequest):
        return memory_api.upsert_brand_fact(request)

    @app.get("/v1/brand-facts/{tenant_id}/{brand_id}")
    def list_brand_facts(tenant_id: str, brand_id: str):
        return {"items": memory_api.list_brand_facts(tenant_id, brand_id)}

    @app.post("/v1/metrics")
    def record_metric(request: MetricSnapshotRequest):
        return memory_api.record_metric(request)

    return app


app = None
