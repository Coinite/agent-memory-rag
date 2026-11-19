import tempfile
import unittest
from pathlib import Path

from src.api.memory_api import (
    BrandFactRequest,
    EvidenceIngestRequest,
    MemoryAPI,
    MemorySearchRequest,
    MetricSnapshotRequest,
    create_fastapi_app,
)
from src.memory.long_term_store import LongTermMemoryStore
from src.models.embedding_client import OpenAICompatibleEmbeddingClient


class MemoryAPITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        store = LongTermMemoryStore(str(Path(self.temp_dir.name) / "api.db"))
        embedder = OpenAICompatibleEmbeddingClient(dimension=64, remote_enabled=False)
        self.api = MemoryAPI(store, embedder)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scoped_api_workflow(self):
        request = EvidenceIngestRequest(
            tenant_id="tenant-1",
            brand_id="brand-1",
            source_uri="https://example.com/news",
            raw_content="The brand launched a sustainable product line.",
            title="Launch",
        )
        first = self.api.ingest_evidence(request)
        second = self.api.ingest_evidence(request)
        self.assertFalse(first["deduplicated"])
        self.assertTrue(second["deduplicated"])

        results = self.api.search_memory(
            MemorySearchRequest(
                tenant_id="tenant-1",
                brand_id="brand-1",
                query="sustainable product launch",
            )
        )
        self.assertTrue(results)

        fact = self.api.upsert_brand_fact(
            BrandFactRequest(
                tenant_id="tenant-1",
                brand_id="brand-1",
                fact_key="positioning",
                fact_value="sustainability",
            )
        )
        self.assertGreater(fact["fact_id"], 0)

        metric = self.api.record_metric(
            MetricSnapshotRequest(
                tenant_id="tenant-1",
                brand_id="brand-1",
                metric_name="share_of_voice",
                metric_value=12.5,
                unit="percent",
            )
        )
        self.assertGreater(metric["metric_id"], 0)

    def test_optional_fastapi_routes(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI optional dependencies are not installed")

        client = TestClient(create_fastapi_app(self.api))
        response = client.post(
            "/v1/evidence",
            json={
                "tenant_id": "tenant-http",
                "brand_id": "brand-http",
                "source_uri": "https://example.com/http",
                "raw_content": "HTTP API evidence",
            },
        )
        self.assertEqual(response.status_code, 200)
        search = client.post(
            "/v1/memory/search",
            json={
                "tenant_id": "tenant-http",
                "brand_id": "brand-http",
                "query": "HTTP evidence",
            },
        )
        self.assertEqual(search.status_code, 200)
        self.assertTrue(search.json()["items"])


if __name__ == "__main__":
    unittest.main()
