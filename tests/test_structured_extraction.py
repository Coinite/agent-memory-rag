import unittest

from src.extraction.service import StructuredExtractor, _json_object


class _FailingLLM:
    def invoke(self, _prompt):
        raise RuntimeError("provider unavailable")


class StructuredExtractionTests(unittest.TestCase):
    def test_json_object_accepts_wrapped_json(self):
        self.assertEqual(
            _json_object('result: {"category": "launch", "score": 0.9}'),
            {"category": "launch", "score": 0.9},
        )

    def test_provider_failure_returns_schema_shaped_fallback(self):
        extractor = StructuredExtractor.__new__(StructuredExtractor)
        extractor.llm = _FailingLLM()

        result = extractor.extract(
            "A product launch was announced.",
            {"category": "string", "confidence": "number"},
        )

        self.assertEqual(result, {"category": None, "confidence": None})


if __name__ == "__main__":
    unittest.main()
