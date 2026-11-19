CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id text PRIMARY KEY,
    display_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS brands (
    tenant_id text NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    brand_id text NOT NULL,
    display_name text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, brand_id)
);

CREATE TABLE IF NOT EXISTS source_registry (
    source_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    brand_id text NOT NULL,
    source_uri text NOT NULL,
    source_type text NOT NULL DEFAULT 'web',
    title text NOT NULL DEFAULT '',
    content_hash char(64) NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, brand_id) REFERENCES brands(tenant_id, brand_id) ON DELETE CASCADE,
    UNIQUE (tenant_id, brand_id, source_uri)
);

CREATE TABLE IF NOT EXISTS raw_evidence (
    evidence_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    brand_id text NOT NULL,
    source_id uuid REFERENCES source_registry(source_id) ON DELETE SET NULL,
    content_hash char(64) NOT NULL,
    raw_content text NOT NULL,
    extracted_text text NOT NULL DEFAULT '',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    observed_at timestamptz NOT NULL DEFAULT now(),
    ingested_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, brand_id) REFERENCES brands(tenant_id, brand_id) ON DELETE CASCADE,
    UNIQUE (tenant_id, brand_id, content_hash)
);

CREATE TABLE IF NOT EXISTS memory_records (
    memory_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    brand_id text NOT NULL,
    memory_type text NOT NULL,
    content text NOT NULL,
    content_hash char(64) NOT NULL,
    embedding vector(384),
    embedding_model text NOT NULL DEFAULT '',
    source_evidence_id uuid REFERENCES raw_evidence(evidence_id) ON DELETE SET NULL,
    importance real NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    observed_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, brand_id) REFERENCES brands(tenant_id, brand_id) ON DELETE CASCADE,
    UNIQUE (tenant_id, brand_id, memory_type, content_hash)
);

CREATE INDEX IF NOT EXISTS memory_scope_time_idx
    ON memory_records (tenant_id, brand_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS memory_embedding_hnsw_idx
    ON memory_records USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS brand_facts (
    fact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    brand_id text NOT NULL,
    fact_key text NOT NULL,
    fact_value jsonb NOT NULL,
    content_hash char(64) NOT NULL,
    confidence real NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    source_evidence_id uuid REFERENCES raw_evidence(evidence_id) ON DELETE SET NULL,
    valid_from timestamptz,
    valid_to timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, brand_id) REFERENCES brands(tenant_id, brand_id) ON DELETE CASCADE,
    UNIQUE (tenant_id, brand_id, fact_key)
);

CREATE TABLE IF NOT EXISTS metric_snapshots (
    metric_id uuid NOT NULL DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    brand_id text NOT NULL,
    metric_name text NOT NULL,
    metric_value double precision NOT NULL,
    unit text NOT NULL DEFAULT '',
    dimensions jsonb NOT NULL DEFAULT '{}'::jsonb,
    observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (metric_id, observed_at),
    FOREIGN KEY (tenant_id, brand_id) REFERENCES brands(tenant_id, brand_id) ON DELETE CASCADE
);

SELECT create_hypertable(
    'metric_snapshots',
    by_range('observed_at'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS metric_scope_time_idx
    ON metric_snapshots (tenant_id, brand_id, metric_name, observed_at DESC);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    provider text NOT NULL,
    model_name text NOT NULL,
    model_version text NOT NULL,
    capability text NOT NULL,
    endpoint_hash char(64) NOT NULL,
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (provider, model_name, model_version, capability, endpoint_hash)
);

CREATE TABLE IF NOT EXISTS model_calls (
    call_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    brand_id text NOT NULL,
    session_id text NOT NULL DEFAULT '',
    turn_id bigint,
    provider text NOT NULL,
    model_name text NOT NULL,
    model_version text NOT NULL,
    capability text NOT NULL,
    used_fallback boolean NOT NULL DEFAULT false,
    latency_ms double precision,
    input_hash char(64),
    output_hash char(64),
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, brand_id) REFERENCES brands(tenant_id, brand_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS model_calls_scope_time_idx
    ON model_calls (tenant_id, brand_id, created_at DESC);
