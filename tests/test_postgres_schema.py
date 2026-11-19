import unittest
from pathlib import Path


class PostgresSchemaTests(unittest.TestCase):
    def test_schema_contains_required_extensions_and_entities(self):
        sql = Path("schema/postgres.sql").read_text(encoding="utf-8").lower()
        for expected in (
            "create extension if not exists vector",
            "create extension if not exists timescaledb",
            "create table if not exists source_registry",
            "create table if not exists raw_evidence",
            "create table if not exists memory_records",
            "create table if not exists brand_facts",
            "create table if not exists metric_snapshots",
            "create_hypertable",
            "vector(384)",
            "vector_cosine_ops",
        ):
            self.assertIn(expected, sql)


if __name__ == "__main__":
    unittest.main()
