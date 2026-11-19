import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.memory.long_term_store import LongTermMemoryStore
from src.memory.memory_manager import MemoryManager
from src.memory.session_store import SessionStore
from src.models.embedding_client import OpenAICompatibleEmbeddingClient


class MemoryManagerIntegrationTests(unittest.TestCase):
    def test_turn_persists_structured_memory_and_retrieves_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "integrated.db"
            session_store = SessionStore(str(db_path))
            long_term_store = LongTermMemoryStore(str(db_path))
            embedder = OpenAICompatibleEmbeddingClient(dimension=64, remote_enabled=False)
            manager = MemoryManager(session_store, long_term_store, embedder)

            turn_id = session_store.create_turn("session-test", "What happened to the campaign?")
            manager.update_after_turn(
                session_id="session-test",
                turn_id=turn_id,
                user_query="What happened to the campaign?",
                tenant_id="tenant-test",
                brand_id="brand-test",
                final_state={
                    "route": "research",
                    "final_answer": "The campaign engagement increased.",
                    "scores": {"accuracy": 4.0, "groundedness": 5.0},
                    "logs": ["test=true"],
                    "agent_events": [],
                    "evidence": [
                        {
                            "title": "Campaign report",
                            "url": "https://example.com/campaign",
                            "snippet": "Engagement increased by 15 percent.",
                            "source": "web",
                        }
                    ],
                },
            )

            context = manager.build_context_pack(
                session_id="session-test",
                query="campaign engagement",
                tenant_id="tenant-test",
                brand_id="brand-test",
                turn_id=turn_id,
            )
            self.assertTrue(context["retrieved_memories"])

            with closing(sqlite3.connect(db_path)) as conn:
                self.assertGreaterEqual(conn.execute("SELECT COUNT(*) FROM raw_evidence").fetchone()[0], 1)
                self.assertGreaterEqual(conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0], 2)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM metric_snapshots").fetchone()[0], 2)
                self.assertGreaterEqual(conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0], 3)


if __name__ == "__main__":
    unittest.main()
