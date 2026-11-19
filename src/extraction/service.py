from __future__ import annotations

import json
from typing import Any, Dict

from src.models.lmstudio_client import build_llm


def _json_object(text: str) -> Dict[str, Any]:
    clean = str(text or "").strip()
    try:
        value = json.loads(clean)
        return value if isinstance(value, dict) else {}
    except Exception:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(clean[start : end + 1])
                return value if isinstance(value, dict) else {}
            except Exception:
                pass
    return {}


class StructuredExtractor:
    """Schema-guided extraction with deterministic failure behavior."""

    def __init__(self) -> None:
        self.llm = build_llm(temp=0.0, response_format={"type": "json_object"})

    def extract(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "Extract only facts supported by the input. Return one JSON object.\n"
            f"Schema: {json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
            f"Input: {text}"
        )
        try:
            message = self.llm.invoke(prompt)
            payload = _json_object(getattr(message, "content", message))
            if payload:
                return payload
        except Exception:
            pass
        return {key: None for key in schema}
