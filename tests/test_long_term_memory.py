import tempfile
import unittest
from pathlib import Path

from src.memory.long_term_store import LongTermMemoryStore
from src.models.embedding_client import deterministic_embedding


class LongTermMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.db"
        self.store = LongTermMemoryStore(str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scoped_evidence_deduplication_and_search(self):
        source_id = self.store.register_source(
            tenant_id="tenant-a",
            brand_id="brand-a",
            source_uri="https://example.com/report",
            title="Quarterly report",
        )
        same_source_id = self.store.register_source(
            tenant_id="tenant-a",
            brand_id="brand-a",
            source_uri="https://example.com/report",
            title="Quarterly report updated",
        )
        self.assertEqual(source_id, same_source_id)

        evidence = self.store.ingest_evidence(
            tenant_id="tenant-a",
            brand_id="brand-a",
            source_id=source_id,
            raw_content="Revenue increased by 20 percent.",
        )
        duplicate = self.store.ingest_evidence(
            tenant_id="tenant-a",
            brand_id="brand-a",
            source_id=source_id,
            raw_content="Revenue increased by 20 percent.",
        )
        self.assertTrue(evidence.created)
        self.assertFalse(duplicate.created)
        self.assertEqual(evidence.evidence_id, duplicate.evidence_id)

        memory = self.store.store_memory(
            tenant_id="tenant-a",
            brand_id="brand-a",
            memory_type="brand_fact",
            content="Revenue increased by 20 percent.",
            embedding=deterministic_embedding("revenue growth", 32),
            embedding_model="feature-hash-v1",
            source_evidence_id=evidence.evidence_id,
        )
        duplicate_memory = self.store.store_memory(
            tenant_id="tenant-a",
            brand_id="brand-a",
            memory_type="brand_fact",
            content="Revenue increased by 20 percent.",
            embedding=deterministic_embedding("revenue growth", 32),
        )
        self.assertTrue(memory.created)
        self.assertFalse(duplicate_memory.created)

        self.store.store_memory(
            tenant_id="tenant-a",
            brand_id="brand-b",
            memory_type="brand_fact",
            content="This must not leak across brands.",
            embedding=deterministic_embedding("revenue growth", 32),
        )
        results = self.store.search_memories(
            tenant_id="tenant-a",
            brand_id="brand-a",
            query="revenue growth",
            query_embedding=deterministic_embedding("revenue growth", 32),
        )
        self.assertEqual(len(results), 1)
        self.assertIn("Revenue increased", results[0]["content"])

    def test_facts_metrics_and_model_registry(self):
        fact_id = self.store.upsert_brand_fact(
            "tenant-a", "brand-a", "market", "North America", confidence=0.8
        )
        same_fact_id = self.store.upsert_brand_fact(
            "tenant-a", "brand-a", "market", "Global", confidence=0.9
        )
        self.assertEqual(fact_id, same_fact_id)
        self.assertEqual(self.store.get_brand_facts("tenant-a", "brand-a")[0]["fact_value"], "Global")

        self.store.record_metric(
            "tenant-a",
            "brand-a",
            "engagement_rate",
            0.42,
            unit="ratio",
            dimensions={"channel": "social"},
        )
        metrics = self.store.get_metric_snapshots("tenant-a", "brand-a", "engagement_rate")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["dimensions"]["channel"], "social")

        first_model_id = self.store.register_model("lmstudio", "chat-model", "v1", "chat", "http://local/v1")
        second_model_id = self.store.register_model("lmstudio", "chat-model", "v1", "chat", "http://local/v1")
        self.assertEqual(first_model_id, second_model_id)


if __name__ == "__main__":
    unittest.main()
