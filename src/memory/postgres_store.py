from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.memory.long_term_store import canonical_json, content_hash


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.10g}" for value in values) + "]"


class PostgresMemoryStore:
    """Optional production adapter for PostgreSQL, TimescaleDB, and pgvector."""

    def __init__(self, dsn: str) -> None:
        if not str(dsn).strip():
            raise ValueError("A PostgreSQL DSN is required")
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL support requires: pip install -r requirements-postgres.txt"
            ) from exc
        self.psycopg = psycopg
        self.dsn = dsn

    def _connect(self):
        return self.psycopg.connect(self.dsn)

    def initialize_schema(self, schema_path: str | None = None) -> None:
        path = Path(schema_path) if schema_path else Path(__file__).resolve().parents[2] / "schema" / "postgres.sql"
        sql = path.read_text(encoding="utf-8")
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)

    def ensure_scope(
        self,
        tenant_id: str,
        brand_id: str,
        tenant_name: str | None = None,
        brand_name: str | None = None,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tenants(tenant_id, display_name) VALUES(%s, %s)
                    ON CONFLICT(tenant_id) DO NOTHING
                    """,
                    (tenant_id, tenant_name or tenant_id),
                )
                cursor.execute(
                    """
                    INSERT INTO brands(tenant_id, brand_id, display_name) VALUES(%s, %s, %s)
                    ON CONFLICT(tenant_id, brand_id) DO NOTHING
                    """,
                    (tenant_id, brand_id, brand_name or brand_id),
                )

    def ingest_evidence(
        self,
        tenant_id: str,
        brand_id: str,
        source_uri: str,
        raw_content: str,
        title: str = "",
        source_type: str = "web",
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        self.ensure_scope(tenant_id, brand_id)
        source_digest = content_hash(source_uri, title, metadata or {})
        evidence_digest = content_hash(raw_content)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO source_registry(
                        tenant_id, brand_id, source_uri, source_type, title,
                        content_hash, metadata
                    ) VALUES(%s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(tenant_id, brand_id, source_uri) DO UPDATE SET
                        title=excluded.title,
                        content_hash=excluded.content_hash,
                        metadata=excluded.metadata,
                        updated_at=now()
                    RETURNING source_id
                    """,
                    (
                        tenant_id,
                        brand_id,
                        source_uri,
                        source_type,
                        title,
                        source_digest,
                        canonical_json(metadata or {}),
                    ),
                )
                source_id = cursor.fetchone()[0]
                cursor.execute(
                    """
                    INSERT INTO raw_evidence(
                        tenant_id, brand_id, source_id, content_hash,
                        raw_content, metadata
                    ) VALUES(%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(tenant_id, brand_id, content_hash) DO UPDATE SET
                        source_id=excluded.source_id
                    RETURNING evidence_id
                    """,
                    (
                        tenant_id,
                        brand_id,
                        source_id,
                        evidence_digest,
                        raw_content,
                        canonical_json(metadata or {}),
                    ),
                )
                return str(cursor.fetchone()[0])

    def store_memory(
        self,
        tenant_id: str,
        brand_id: str,
        memory_type: str,
        content: str,
        embedding: Sequence[float],
        embedding_model: str,
        source_evidence_id: str | None = None,
        importance: float = 0.5,
    ) -> str:
        digest = content_hash(content)
        vector = _vector_literal(embedding)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO memory_records(
                        tenant_id, brand_id, memory_type, content, content_hash,
                        embedding, embedding_model, source_evidence_id, importance
                    ) VALUES(%s, %s, %s, %s, %s, %s::vector, %s, %s, %s)
                    ON CONFLICT(tenant_id, brand_id, memory_type, content_hash)
                    DO UPDATE SET
                        embedding=excluded.embedding,
                        embedding_model=excluded.embedding_model,
                        updated_at=now()
                    RETURNING memory_id
                    """,
                    (
                        tenant_id,
                        brand_id,
                        memory_type,
                        content,
                        digest,
                        vector,
                        embedding_model,
                        source_evidence_id,
                        importance,
                    ),
                )
                return str(cursor.fetchone()[0])

    def search_memories(
        self,
        tenant_id: str,
        brand_id: str,
        query_embedding: Sequence[float],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        vector = _vector_literal(query_embedding)
        with self._connect() as conn:
            with conn.cursor(row_factory=self.psycopg.rows.dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT
                        memory_id, memory_type, content, embedding_model,
                        importance, observed_at,
                        1 - (embedding <=> %s::vector) AS semantic_score
                    FROM memory_records
                    WHERE tenant_id=%s AND brand_id=%s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, tenant_id, brand_id, vector, max(1, min(100, int(limit)))),
                )
                return [dict(row) for row in cursor.fetchall()]
