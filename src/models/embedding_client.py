from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List

from openai import OpenAI

from src.config import settings


TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


def deterministic_embedding(text: str, dimension: int = 384) -> List[float]:
    """Create a stable local embedding for offline operation and tests.

    This is intentionally simple: tokens are feature-hashed into a fixed-size
    signed vector and L2-normalized. It is not a replacement for a production
    embedding model, but preserves deterministic retrieval when the configured
    OpenAI-compatible embedding endpoint is unavailable.
    """

    size = max(8, int(dimension))
    vector = [0.0] * size
    tokens = TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = list(left)
    b = list(right)
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: List[List[float]]
    provider: str
    model_name: str
    model_version: str
    used_fallback: bool


class OpenAICompatibleEmbeddingClient:
    """Embedding interface for LM Studio or any OpenAI-compatible provider."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        dimension: int | None = None,
        provider: str | None = None,
        timeout_sec: float | None = None,
        remote_enabled: bool | None = None,
    ) -> None:
        self.base_url = (base_url or settings.resolved_llm_base_url).rstrip("/")
        self.api_key = api_key or settings.resolved_llm_api_key
        self.model_name = model_name or settings.embedding_model_name
        self.model_version = model_version or settings.embedding_model_version
        self.dimension = max(8, int(dimension or settings.embedding_dimension))
        self.provider = provider or settings.llm_provider
        self.timeout_sec = float(timeout_sec or settings.lm_timeout_sec)
        self.remote_enabled = settings.embedding_remote_enabled if remote_enabled is None else bool(remote_enabled)

    def embed_texts(self, texts: Iterable[str], allow_fallback: bool = True) -> EmbeddingResult:
        items = [str(text) for text in texts]
        if not items:
            return EmbeddingResult([], self.provider, self.model_name, self.model_version, False)

        try:
            if not self.remote_enabled:
                raise RuntimeError("Remote embedding endpoint is disabled")
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_sec,
                max_retries=0,
            )
            response = client.embeddings.create(model=self.model_name, input=items)
            vectors = [list(item.embedding) for item in response.data]
            if len(vectors) != len(items):
                raise RuntimeError("Embedding endpoint returned an unexpected result count")
            return EmbeddingResult(
                vectors=vectors,
                provider=self.provider,
                model_name=self.model_name,
                model_version=self.model_version,
                used_fallback=False,
            )
        except Exception:
            if not allow_fallback:
                raise
            vectors = [deterministic_embedding(text, self.dimension) for text in items]
            return EmbeddingResult(
                vectors=vectors,
                provider="deterministic-local",
                model_name="feature-hash-v1",
                model_version="1",
                used_fallback=True,
            )

    def embed_query(self, text: str, allow_fallback: bool = True) -> EmbeddingResult:
        return self.embed_texts([text], allow_fallback=allow_fallback)
