# Agent Memory RAG
# Haotian Chen

An MVP agentic system for long-horizon research and automated brand operations.
The project separates orchestration, web ingestion, structured extraction,
persistent memory, retrieval, evaluation, and diagnostics into independently
testable modules. SQLite is the default local backend; PostgreSQL,
TimescaleDB, and pgvector are available as the production memory design.

## Prerequisites
1. Python 3.10+
2. An OpenAI-compatible model endpoint
3. For local use: LM Studio with its server enabled (example: `http://127.0.0.1:1234/v1`)

## Setup (Recommended: virtual environment)
1. Create venv:
   - `python -m venv .venv`
2. Activate venv (PowerShell):
   - `.\.venv\Scripts\activate`
3. Upgrade packaging tools:
   - `python -m pip install --upgrade pip setuptools wheel`
4. Install dependencies:
   - `pip install -r requirements.txt`
5. Configure environment:
   - copy `.env.example` to `.env`
   - set `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, and `MODEL_NAME` as needed

## Run
1. Start LM Studio local server.
2. Run app:
   - `python -m src.main`

## Notes on LM Studio port
- Default in this project: `LLM_BASE_URL=http://127.0.0.1:1234/v1`
- Config location:
  - `.env` (recommended)
  - fallback default in `src/config.py`

## Architecture

```text
User / downstream agent
        |
        v
LangGraph supervisor -----> specialized agents (web/news/weather/stock/code/file)
        |                                      |
        |                                      v
        |                           evidence + structured extraction
        v                                      |
persistent session memory <--------------------+
        |
        +--> source registry + raw evidence + SHA-256 deduplication
        +--> brand facts + timestamped memory records
        +--> hybrid semantic/lexical retrieval
        +--> metric snapshots + model/version call tracking
        |
        v
SQLite (local) / PostgreSQL + TimescaleDB + pgvector (production)
```

Main modules:

- `src/ingestion/`: source registration, raw evidence retention, content-hash
  deduplication, and memory indexing.
- `src/extraction/`: structured JSON extraction with deterministic failure
  output.
- `src/memory/`: session persistence, structured long-term memory, model
  registry, SQLite backend, and optional PostgreSQL adapter.
- `src/retrieval/`: tenant/brand-scoped hybrid semantic and lexical search.
- `src/diagnostics/`: timestamped quality and operational metric snapshots.
- `src/api/`: Python service API plus optional FastAPI endpoints.
- `src/agents/`: LangGraph supervisor and specialized execution agents.

## Structured Long-Term Memory

The local SQLite schema stores:

- `source_registry`: scoped source identity and metadata.
- `raw_evidence`: immutable raw content with SHA-256 deduplication.
- `memory_records`: timestamped semantic memories and embedding metadata.
- `brand_facts`: current scoped facts with confidence and validity windows.
- `metric_snapshots`: time-stamped quality or business metrics.
- `model_registry` and `model_calls`: provider, model/version, capability,
  fallback use, latency, and privacy-preserving input/output hashes.
- `sessions`, `turns`, and `agent_events`: conversation history and full agent
  traces.

Every search and write requires both `tenant_id` and `brand_id`. The default
scope is configured with `DEFAULT_TENANT_ID` and `DEFAULT_BRAND_ID`.

The retrieval pipeline uses the configured OpenAI-compatible embedding
endpoint when `EMBEDDING_REMOTE_ENABLED=true`. When it is unavailable or
disabled, a deterministic feature-hash embedding keeps local development and
tests functional without external services.

## Memory Search API

The dependency-free Python API is available from `src.api.memory_api.MemoryAPI`.
Optional HTTP endpoints are:

- `POST /v1/evidence`
- `POST /v1/memory/search`
- `PUT /v1/brand-facts`
- `GET /v1/brand-facts/{tenant_id}/{brand_id}`
- `POST /v1/metrics`

Install and run the optional FastAPI server:

```powershell
pip install -r requirements-api.txt
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

## Production PostgreSQL RAG Backend

`schema/postgres.sql` defines a production-oriented schema with:

- PostgreSQL tenant/brand isolation and foreign-key integrity.
- pgvector `vector(384)` semantic memories and an HNSW cosine index.
- TimescaleDB hypertable storage for temporal metric snapshots.
- source/evidence hash uniqueness, brand facts, model registry, and model-call
  diagnostics.

Install the optional adapter and initialize the schema against a PostgreSQL
instance that has TimescaleDB and pgvector available:

```powershell
pip install -r requirements-postgres.txt
```

Then set `POSTGRES_DSN` and call
`PostgresMemoryStore(settings.postgres_dsn).initialize_schema()`.

## Verification

The test suite uses temporary databases and requires no LM Studio or PostgreSQL
server:

```powershell
python -m unittest discover -s tests -v
```

It covers deterministic/offline embeddings, evidence deduplication,
tenant/brand isolation, semantic retrieval, brand facts, metric snapshots,
model tracking, the application API, end-to-end memory integration, and the
PostgreSQL schema contract.

## Dependency conflict troubleshooting
If your global Python already has many AI packages installed, use a fresh venv first.
If needed, force reinstall in current env:
- `pip install --upgrade --force-reinstall -r requirements.txt`

## Application Scenarios
1. News and incident monitoring
   - Track ongoing events, summarize latest developments, and keep source-grounded updates.
2. Data-to-chart automation
   - Retrieve public web data, save it as local CSV, auto-generate plotting scripts, and verify output artifacts.
3. Local research assistant
   - Collect multi-source evidence for technical, academic, or policy questions with traceable links.
4. Financial quick analysis
   - Pull stock quotes/history and produce compact trend summaries or chart scripts.
5. Weather and city-based planning
   - Query weather by location and produce actionable short summaries.
6. File-centric task automation
   - Create/read/write local files and run generated scripts end-to-end in one workflow.

## Agents
1. `supervisor`
   - Plans route and task structure for each turn.
   - Extracts constraints (freshness, depth, coverage, format).
   - Stabilizes file/code execution plan (`target_file`, `run_entrypoint`, `expected_artifacts`).
   - Initializes per-turn state and emits subtasks/logs.
2. `web_agent`
   - Runs web retrieval and produces evidence + draft answer.
   - Handles direct PDF download tasks into workspace.
   - For file+data/chart tasks, extracts structured data and stages CSV content for `file_agent`.
3. `news_agent`
   - Performs domain-filtered news retrieval with intent-aware search depth.
   - Applies constraint-driven coverage (`must_cover`, `breakdown_by`, `strict_coverage`).
4. `weather_agent`
   - Resolves city and fetches weather via Open-Meteo.
   - Produces concise weather summary with source evidence.
5. `stock_agent`
   - Resolves symbols/sector/time window, fetches quote/history data.
   - Writes normalized market series into `context_payload` for downstream code generation.
6. `code_agent`
   - Generates executable code (JSON schema with optional `extra_files`).
   - Uses retrieved evidence/context and enforces artifact-aware output when execution is requested.
   - Falls back to a CSV plotting template when data-file-driven code generation fails.
7. `file_agent`
   - Handles local file operations (`list/read/mkdir/write`).
   - Writes staged structured data as CSV during the data stage.
   - Writes generated code and any extra generated files.
8. `runner_agent`
   - Executes generated entry script in workspace.
   - Captures return code/stdout/stderr and verifies expected artifacts.
9. `judge_agent`
   - Scores groundedness/coverage quality.
   - Formats final answer and decides whether to trigger one verification web round.

## Routing Logic
1. Entry point is always `supervisor`.
2. Supervisor dispatch:
   - `route=file` + retrieval intent + `requires_code=true` -> `web_agent` first (`file_with_research`).
   - `route=file` + `requires_code=true` without retrieval -> `code_agent`.
   - `route=file` + no code -> `file_agent`.
   - `route=news|weather|stock` -> corresponding specialized agent.
   - `route=research|general` -> `web_agent`.
3. After context agents (`web/news/weather/stock`):
   - If `direct_action_done=true` -> end.
   - If `route=file` and `file_stage=data` with staged files -> `file_agent` writes CSV first.
   - Else if code is required -> `code_agent`.
   - Else -> `judge_agent`.
4. After `file_agent`:
   - If current stage is data and write succeeded -> `code_agent`.
   - Else if `run_after_write=true` and write succeeded -> `runner_agent`.
   - Else -> end.
5. After `judge_agent`:
   - If verification needed and not yet verified -> one more `web_agent` pass.
   - Otherwise -> end.

## Improvement Opportunities
1. Performance and runtime cost
   - Add LLM call budget controls per turn (max calls, per-agent token caps, timeout tiers by task type).
   - Introduce response/result caching for repeated planner, constraint extraction, and web-search steps.
   - Add asynchronous retrieval in `web_agent/news_agent` to reduce wall-clock latency for multi-query searches.
   - Add model fallback policies by task complexity (small model for routing, stronger model only for hard synthesis/code).
2. Reliability and efficiency
   - Add deterministic validation for generated code before write/run (AST parse, required artifact path checks, dependency checks).
   - Expand runner sandbox and preflight checks (module availability, file existence, write permissions) before execution.
   - Add structured error taxonomy across agents to improve automatic retries and debugging.
   - Add golden regression suites for routing + state transitions using recorded session traces.
3. Generalization and quality
   - Improve multilingual intent detection consistency and unify query normalization for mixed-language inputs.
   - Add entity disambiguation and temporal normalization modules (explicit date anchoring for "latest/today" requests).
   - Add confidence signals in final answers based on source count, source diversity, and evidence freshness.
   - Strengthen table extraction for data tasks with schema inference and numeric consistency checks.
6. New tools to support agents
   - Browser rendering/fetch tool (headless HTML render) for JS-heavy pages and anti-bot tolerant extraction.
   - Table extraction tool (HTML/PDF to normalized table) with row/column type inference.
   - Dependency resolver tool to auto-generate minimal `requirements.txt` from code imports.
   - Artifact inspection tool for CSV/PNG/PDF validation (shape checks, image existence, basic metadata checks).
   - Metrics/telemetry tool for per-agent latency, token usage, retry count, and success rate dashboards.
