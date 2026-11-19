from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from src.config import settings
from src.diagnostics.service import DiagnosticsService
from src.ingestion.service import EvidenceIngestionService
from src.memory.long_term_store import LongTermMemoryStore, canonical_json
from src.memory.session_store import SessionStore
from src.models.embedding_client import OpenAICompatibleEmbeddingClient
from src.retrieval.service import MemoryRetrievalService


class MemoryManager:
    def __init__(
        self,
        store: SessionStore,
        long_term_store: LongTermMemoryStore | None = None,
        embedding_client: OpenAICompatibleEmbeddingClient | None = None,
    ) -> None:
        self.store = store
        self.long_term_store = long_term_store or LongTermMemoryStore(str(store.db_path))
        self.embedding_client = embedding_client or OpenAICompatibleEmbeddingClient()
        self.ingestion = EvidenceIngestionService(self.long_term_store, self.embedding_client)
        self.retrieval = MemoryRetrievalService(self.long_term_store, self.embedding_client)
        self.diagnostics = DiagnosticsService(self.long_term_store)
        self.last_session_file = Path(".agent_memory/last_session_id.txt")
        self.last_session_file.parent.mkdir(parents=True, exist_ok=True)
        self.long_term_store.register_model(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            model_version=settings.model_version,
            capability="chat",
            base_url=settings.resolved_llm_base_url,
        )
        self.long_term_store.register_model(
            provider=settings.llm_provider,
            model_name=settings.embedding_model_name,
            model_version=settings.embedding_model_version,
            capability="embedding",
            base_url=settings.resolved_llm_base_url,
        )

    def load_or_create_session_id(self) -> str:
        if self.last_session_file.exists():
            sid = self.last_session_file.read_text(encoding="utf-8").strip()
            if sid:
                self.store.ensure_session(sid)
                return sid
        sid = "session-1"
        self.store.ensure_session(sid)
        self.last_session_file.write_text(sid, encoding="utf-8")
        return sid

    def set_current_session_id(self, session_id: str) -> None:
        self.store.ensure_session(session_id)
        self.last_session_file.write_text(session_id, encoding="utf-8")

    def new_session(self, session_id: str = "") -> str:
        sid = session_id.strip() if session_id else f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.set_current_session_id(sid)
        return sid

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.store.list_sessions(limit=limit)

    def get_summary(self, session_id: str) -> Dict[str, Any]:
        return self.store.get_session(session_id)

    def get_history(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self.store.get_turns(session_id=session_id, limit=limit)

    def get_trace(self, session_id: str, turn_id: int) -> List[Dict[str, Any]]:
        return self.store.get_agent_events(session_id=session_id, turn_id=turn_id)

    def build_context_pack(
        self,
        session_id: str,
        recent_limit: int = 5,
        query: str = "",
        tenant_id: str = "default",
        brand_id: str = "default",
        turn_id: int | None = None,
    ) -> Dict[str, Any]:
        session = self.store.get_session(session_id)
        recent_turns = self.store.get_recent_turns(session_id, limit=recent_limit)
        retrieved_memories: List[Dict[str, Any]] = []
        if query.strip():
            retrieved_memories = self.retrieval.search(
                tenant_id=tenant_id,
                brand_id=brand_id,
                query=query,
                limit=5,
                session_id=session_id,
                turn_id=turn_id,
            )
        return {
            "session_id": session_id,
            "summary_text": session.get("summary_text", ""),
            "active_goal": session.get("active_goal", ""),
            "recent_turns": recent_turns,
            "retrieved_memories": retrieved_memories,
        }

    def update_after_turn(
        self,
        session_id: str,
        turn_id: int,
        user_query: str,
        final_state: Dict[str, Any],
        tenant_id: str = "default",
        brand_id: str = "default",
    ) -> None:
        route = str(final_state.get("route", ""))
        final_answer = str(final_state.get("final_answer", ""))
        scores = final_state.get("scores", {}) or {}
        logs = final_state.get("logs", []) or []
        self.store.finalize_turn(
            session_id=session_id,
            turn_id=turn_id,
            route=route,
            final_answer=final_answer,
            scores=scores,
            logs=logs,
        )

        for evt in final_state.get("agent_events", []) or []:
            self.store.add_agent_event(
                session_id=session_id,
                turn_id=turn_id,
                agent_name=str(evt.get("agent_name", "unknown")),
                input_payload=evt.get("input", {}) or {},
                output_payload=evt.get("output", {}) or {},
                tool_calls=evt.get("tool_calls", []) or [],
            )

        summary = self._merge_summary(
            existing_summary=self.store.get_session(session_id).get("summary_text", ""),
            user_query=user_query,
            final_answer=final_answer,
            route=route,
            recent_turns=self.store.get_recent_turns(session_id, limit=5),
        )
        self.store.update_session_summary(session_id, summary_text=summary, active_goal=self._infer_active_goal(recent_turns=self.store.get_recent_turns(session_id, limit=3)))

        for index, item in enumerate((final_state.get("evidence", []) or [])[:20]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", ""))
            snippet = str(item.get("snippet", ""))
            raw_content = "\n".join(part for part in (title, snippet) if part.strip())
            if not raw_content:
                continue
            source_uri = str(item.get("url", "")).strip() or f"agent://{session_id}/{turn_id}/evidence/{index}"
            self.ingestion.ingest(
                tenant_id=tenant_id,
                brand_id=brand_id,
                source_uri=source_uri,
                source_type=str(item.get("source", "web")) or "web",
                title=title,
                raw_content=raw_content,
                extracted_text=snippet,
                metadata={"route": route, "session_id": session_id, "turn_id": turn_id},
                session_id=session_id,
                turn_id=turn_id,
            )

        interaction_text = canonical_json(
            {
                "user_intent": user_query,
                "route": route,
                "outcome": final_answer,
            }
        )
        embedding = self.embedding_client.embed_query(interaction_text)
        vector = embedding.vectors[0] if embedding.vectors else []
        self.long_term_store.store_memory(
            tenant_id=tenant_id,
            brand_id=brand_id,
            memory_type="interaction",
            content=interaction_text,
            embedding=vector,
            embedding_model=f"{embedding.provider}:{embedding.model_name}:{embedding.model_version}",
            importance=0.7,
        )
        self.long_term_store.record_model_call(
            tenant_id=tenant_id,
            brand_id=brand_id,
            provider=settings.llm_provider,
            model_name=settings.model_name,
            model_version=settings.model_version,
            capability="agent-workflow",
            session_id=session_id,
            turn_id=turn_id,
            used_fallback=any("fallback" in str(line).lower() for line in logs),
            input_text=user_query,
            output_text=final_answer,
        )
        self.diagnostics.record_scores(
            tenant_id=tenant_id,
            brand_id=brand_id,
            scores=scores,
            session_id=session_id,
            turn_id=turn_id,
            route=route,
        )

    def _merge_summary(
        self,
        existing_summary: str,
        user_query: str,
        final_answer: str,
        route: str,
        recent_turns: List[Dict[str, Any]],
    ) -> str:
        # Deterministic lightweight summary to avoid extra model dependency.
        lines: List[str] = []
        if existing_summary.strip():
            lines.append(existing_summary.strip())
        lines.append(f"Latest route: {route}")
        lines.append(f"Latest user intent: {user_query[:220]}")
        lines.append(f"Latest outcome: {final_answer[:260]}")
        if recent_turns:
            topic = " | ".join([str(t.get("user_query", ""))[:80] for t in recent_turns[-3:]])
            lines.append(f"Recent thread: {topic}")
        # Keep summary bounded.
        merged = "\n".join(lines)
        return merged[-2500:]

    def _infer_active_goal(self, recent_turns: List[Dict[str, Any]]) -> str:
        if not recent_turns:
            return ""
        # Use latest user query as active goal placeholder.
        return str(recent_turns[-1].get("user_query", ""))[:180]
