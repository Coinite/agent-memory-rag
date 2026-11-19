from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from src.models.embedding_client import cosine_similarity


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(*parts: Any) -> str:
    payload = "\n".join(part if isinstance(part, str) else canonical_json(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scope_value(value: str, name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{name} must not be empty")
    return clean


@dataclass(frozen=True)
class EvidenceWriteResult:
    evidence_id: int
    content_hash: str
    created: bool


@dataclass(frozen=True)
class MemoryWriteResult:
    memory_id: int
    content_hash: str
    created: bool


class LongTermMemoryStore:
    """Structured long-term memory with tenant and brand isolation.

    SQLite is the zero-infrastructure development backend. The same logical
    entities are represented by the PostgreSQL/TimescaleDB/pgvector schema in
    ``schema/postgres.sql`` for production deployments.
    """

    def __init__(self, db_path: str = ".agent_memory/session.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL").close()
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS source_registry (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                brand_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'web',
                title TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(tenant_id, brand_id, source_uri)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS raw_evidence (
                evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                brand_id TEXT NOT NULL,
                source_id INTEGER,
                content_hash TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                extracted_text TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                observed_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                UNIQUE(tenant_id, brand_id, content_hash),
                FOREIGN KEY(source_id) REFERENCES source_registry(source_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                brand_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_json TEXT NOT NULL DEFAULT '[]',
                embedding_model TEXT NOT NULL DEFAULT '',
                source_evidence_id INTEGER,
                importance REAL NOT NULL DEFAULT 0.5,
                observed_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(tenant_id, brand_id, memory_type, content_hash),
                FOREIGN KEY(source_evidence_id) REFERENCES raw_evidence(evidence_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS brand_facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                brand_id TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_evidence_id INTEGER,
                valid_from TEXT,
                valid_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(tenant_id, brand_id, fact_key),
                FOREIGN KEY(source_evidence_id) REFERENCES raw_evidence(evidence_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS metric_snapshots (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                brand_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                unit TEXT NOT NULL DEFAULT '',
                dimensions_json TEXT NOT NULL DEFAULT '{}',
                observed_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                capability TEXT NOT NULL,
                endpoint_hash TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(provider, model_name, model_version, capability, endpoint_hash)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_calls (
                call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                brand_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                turn_id INTEGER,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                capability TEXT NOT NULL,
                used_fallback INTEGER NOT NULL DEFAULT 0,
                latency_ms REAL,
                input_hash TEXT NOT NULL DEFAULT '',
                output_hash TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_sources_scope ON source_registry(tenant_id, brand_id)",
            "CREATE INDEX IF NOT EXISTS idx_evidence_scope_time ON raw_evidence(tenant_id, brand_id, observed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_scope_time ON memory_records(tenant_id, brand_id, observed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_scope_type ON memory_records(tenant_id, brand_id, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_facts_scope ON brand_facts(tenant_id, brand_id, fact_key)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_scope_time ON metric_snapshots(tenant_id, brand_id, metric_name, observed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_model_calls_scope_time ON model_calls(tenant_id, brand_id, created_at DESC)",
        ]
        with self._connect() as conn:
            for statement in statements:
                conn.execute(statement)

    def register_source(
        self,
        tenant_id: str,
        brand_id: str,
        source_uri: str,
        source_type: str = "web",
        title: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        uri = _scope_value(source_uri, "source_uri")
        now = utc_now()
        metadata_json = canonical_json(metadata or {})
        digest = content_hash(uri, title, metadata or {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_registry(
                    tenant_id, brand_id, source_uri, source_type, title,
                    content_hash, metadata_json, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, brand_id, source_uri) DO UPDATE SET
                    source_type=excluded.source_type,
                    title=excluded.title,
                    content_hash=excluded.content_hash,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (tenant, brand, uri, source_type, title, digest, metadata_json, now, now),
            )
            row = conn.execute(
                "SELECT source_id FROM source_registry WHERE tenant_id=? AND brand_id=? AND source_uri=?",
                (tenant, brand, uri),
            ).fetchone()
        return int(row["source_id"])

    def ingest_evidence(
        self,
        tenant_id: str,
        brand_id: str,
        raw_content: str,
        source_id: int | None = None,
        extracted_text: str = "",
        metadata: Dict[str, Any] | None = None,
        observed_at: str | None = None,
    ) -> EvidenceWriteResult:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        raw = _scope_value(raw_content, "raw_content")
        digest = content_hash(raw)
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO raw_evidence(
                    tenant_id, brand_id, source_id, content_hash, raw_content,
                    extracted_text, metadata_json, observed_at, ingested_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant,
                    brand,
                    source_id,
                    digest,
                    raw,
                    extracted_text,
                    canonical_json(metadata or {}),
                    observed_at or now,
                    now,
                ),
            )
            created = cursor.rowcount > 0
            row = conn.execute(
                "SELECT evidence_id FROM raw_evidence WHERE tenant_id=? AND brand_id=? AND content_hash=?",
                (tenant, brand, digest),
            ).fetchone()
        return EvidenceWriteResult(int(row["evidence_id"]), digest, created)

    def store_memory(
        self,
        tenant_id: str,
        brand_id: str,
        memory_type: str,
        content: str,
        embedding: Sequence[float] | None = None,
        embedding_model: str = "",
        source_evidence_id: int | None = None,
        importance: float = 0.5,
        observed_at: str | None = None,
    ) -> MemoryWriteResult:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        kind = _scope_value(memory_type, "memory_type")
        text = _scope_value(content, "content")
        digest = content_hash(text)
        now = utc_now()
        vector_json = canonical_json([float(value) for value in (embedding or [])])
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memory_records(
                    tenant_id, brand_id, memory_type, content, content_hash,
                    embedding_json, embedding_model, source_evidence_id,
                    importance, observed_at, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant,
                    brand,
                    kind,
                    text,
                    digest,
                    vector_json,
                    embedding_model,
                    source_evidence_id,
                    max(0.0, min(1.0, float(importance))),
                    observed_at or now,
                    now,
                    now,
                ),
            )
            created = cursor.rowcount > 0
            row = conn.execute(
                """
                SELECT memory_id FROM memory_records
                WHERE tenant_id=? AND brand_id=? AND memory_type=? AND content_hash=?
                """,
                (tenant, brand, kind, digest),
            ).fetchone()
        return MemoryWriteResult(int(row["memory_id"]), digest, created)

    def search_memories(
        self,
        tenant_id: str,
        brand_id: str,
        query: str,
        query_embedding: Sequence[float] | None = None,
        limit: int = 8,
        memory_types: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        type_list = [value for value in (memory_types or []) if str(value).strip()]
        sql = "SELECT * FROM memory_records WHERE tenant_id=? AND brand_id=?"
        params: List[Any] = [tenant, brand]
        if type_list:
            placeholders = ",".join("?" for _ in type_list)
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(type_list)
        sql += " ORDER BY observed_at DESC LIMIT 500"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        query_tokens = set(str(query).lower().split())
        vector = list(query_embedding or [])
        scored: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            content_tokens = set(str(item["content"]).lower().split())
            lexical = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            candidate_vector = json.loads(item.pop("embedding_json") or "[]")
            semantic = cosine_similarity(vector, candidate_vector) if vector and candidate_vector else 0.0
            importance = float(item.get("importance", 0.5))
            item["score"] = round(0.7 * semantic + 0.25 * lexical + 0.05 * importance, 6)
            item["semantic_score"] = round(semantic, 6)
            item["lexical_score"] = round(lexical, 6)
            scored.append(item)
        scored.sort(key=lambda item: (item["score"], item["observed_at"]), reverse=True)
        return scored[: max(1, min(100, int(limit)))]

    def upsert_brand_fact(
        self,
        tenant_id: str,
        brand_id: str,
        fact_key: str,
        fact_value: str,
        confidence: float = 0.5,
        source_evidence_id: int | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> int:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        key = _scope_value(fact_key, "fact_key")
        value = _scope_value(fact_value, "fact_value")
        now = utc_now()
        digest = content_hash(key, value)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO brand_facts(
                    tenant_id, brand_id, fact_key, fact_value, content_hash,
                    confidence, source_evidence_id, valid_from, valid_to,
                    created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, brand_id, fact_key) DO UPDATE SET
                    fact_value=excluded.fact_value,
                    content_hash=excluded.content_hash,
                    confidence=excluded.confidence,
                    source_evidence_id=excluded.source_evidence_id,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    updated_at=excluded.updated_at
                """,
                (
                    tenant,
                    brand,
                    key,
                    value,
                    digest,
                    max(0.0, min(1.0, float(confidence))),
                    source_evidence_id,
                    valid_from,
                    valid_to,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT fact_id FROM brand_facts WHERE tenant_id=? AND brand_id=? AND fact_key=?",
                (tenant, brand, key),
            ).fetchone()
        return int(row["fact_id"])

    def get_brand_facts(self, tenant_id: str, brand_id: str) -> List[Dict[str, Any]]:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM brand_facts
                WHERE tenant_id=? AND brand_id=?
                ORDER BY fact_key ASC
                """,
                (tenant, brand),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_metric(
        self,
        tenant_id: str,
        brand_id: str,
        metric_name: str,
        metric_value: float,
        unit: str = "",
        dimensions: Dict[str, Any] | None = None,
        observed_at: str | None = None,
    ) -> int:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        name = _scope_value(metric_name, "metric_name")
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO metric_snapshots(
                    tenant_id, brand_id, metric_name, metric_value, unit,
                    dimensions_json, observed_at, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant,
                    brand,
                    name,
                    float(metric_value),
                    unit,
                    canonical_json(dimensions or {}),
                    observed_at or now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_metric_snapshots(
        self,
        tenant_id: str,
        brand_id: str,
        metric_name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM metric_snapshots
                WHERE tenant_id=? AND brand_id=? AND metric_name=?
                ORDER BY observed_at DESC LIMIT ?
                """,
                (tenant, brand, metric_name, max(1, min(1000, int(limit)))),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["dimensions"] = json.loads(item.pop("dimensions_json") or "{}")
            output.append(item)
        return output

    def register_model(
        self,
        provider: str,
        model_name: str,
        model_version: str,
        capability: str,
        base_url: str,
    ) -> int:
        now = utc_now()
        endpoint_hash = content_hash(base_url.rstrip("/"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_registry(
                    provider, model_name, model_version, capability,
                    endpoint_hash, first_seen_at, last_seen_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, model_name, model_version, capability, endpoint_hash)
                DO UPDATE SET last_seen_at=excluded.last_seen_at
                """,
                (provider, model_name, model_version, capability, endpoint_hash, now, now),
            )
            row = conn.execute(
                """
                SELECT model_id FROM model_registry
                WHERE provider=? AND model_name=? AND model_version=?
                  AND capability=? AND endpoint_hash=?
                """,
                (provider, model_name, model_version, capability, endpoint_hash),
            ).fetchone()
        return int(row["model_id"])

    def record_model_call(
        self,
        tenant_id: str,
        brand_id: str,
        provider: str,
        model_name: str,
        model_version: str,
        capability: str,
        session_id: str = "",
        turn_id: int | None = None,
        used_fallback: bool = False,
        latency_ms: float | None = None,
        input_text: str = "",
        output_text: str = "",
    ) -> int:
        tenant = _scope_value(tenant_id, "tenant_id")
        brand = _scope_value(brand_id, "brand_id")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_calls(
                    tenant_id, brand_id, session_id, turn_id, provider,
                    model_name, model_version, capability, used_fallback,
                    latency_ms, input_hash, output_hash, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant,
                    brand,
                    session_id,
                    turn_id,
                    provider,
                    model_name,
                    model_version,
                    capability,
                    int(bool(used_fallback)),
                    latency_ms,
                    content_hash(input_text) if input_text else "",
                    content_hash(output_text) if output_text else "",
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)
