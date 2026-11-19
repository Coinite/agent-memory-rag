import urllib.error
import urllib.request
from functools import lru_cache
from urllib.parse import urlparse

import httpx
from langchain_openai import ChatOpenAI

from src.config import settings


def _normalize_base_url(url: str) -> str:
    clean = url.rstrip("/")
    if clean.endswith("/v1"):
        return clean
    return f"{clean}/v1"


def _is_reachable(base_url: str, timeout_sec: float = 2.5) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


@lru_cache(maxsize=1)
def _resolve_base_url() -> str:
    configured = _normalize_base_url(settings.resolved_llm_base_url)
    parsed = urlparse(configured)
    candidates = [configured]

    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        port = parsed.port or 1234
        candidates.append(f"http://127.0.0.1:{port}/v1")
        candidates.append(f"http://localhost:{port}/v1")

    for candidate in candidates:
        if _is_reachable(candidate):
            if candidate != configured:
                print(f"[LM Studio] fallback base_url: {candidate} (configured: {configured})")
            return candidate
    return configured


def build_llm(
    temp: float = 0.2,
    model_override: str = "",
    max_tokens: int | None = None,
    response_format: dict | None = None,
    timeout_sec: float | None = None,
    max_retries: int | None = None,
):
    base_url = _resolve_base_url()
    resolved_timeout = float(timeout_sec) if timeout_sec and timeout_sec > 0 else float(settings.lm_timeout_sec)
    resolved_retries = settings.lm_max_retries if max_retries is None else max(0, int(max_retries))
    http_client = httpx.Client(timeout=resolved_timeout, trust_env=False)
    model_name = model_override.strip() or settings.model_name
    extra_kwargs = {}
    if max_tokens is not None:
        extra_kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        extra_kwargs["model_kwargs"] = {"response_format": response_format}
    return ChatOpenAI(
        base_url=base_url,
        api_key=settings.resolved_llm_api_key,
        model=model_name,
        temperature=temp,
        timeout=resolved_timeout,
        max_retries=resolved_retries,
        http_client=http_client,
        **extra_kwargs,
    )
