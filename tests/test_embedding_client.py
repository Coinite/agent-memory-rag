import math
import unittest

from src.models.embedding_client import (
    OpenAICompatibleEmbeddingClient,
    cosine_similarity,
    deterministic_embedding,
)


class EmbeddingClientTests(unittest.TestCase):
    def test_deterministic_embedding_is_stable_and_normalized(self):
        first = deterministic_embedding("brand revenue increased", dimension=64)
        second = deterministic_embedding("brand revenue increased", dimension=64)
        other = deterministic_embedding("weather forecast tomorrow", dimension=64)

        self.assertEqual(first, second)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0, places=6)
        self.assertGreater(cosine_similarity(first, second), cosine_similarity(first, other))

    def test_client_uses_offline_fallback_when_remote_disabled(self):
        client = OpenAICompatibleEmbeddingClient(dimension=32, remote_enabled=False)
        result = client.embed_query("persistent memory")

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.provider, "deterministic-local")
        self.assertEqual(len(result.vectors), 1)
        self.assertEqual(len(result.vectors[0]), 32)


if __name__ == "__main__":
    unittest.main()
