import json
from typing import Dict, List

from src.models.lmstudio_client import build_llm

judge_llm = build_llm(temp=0.0)

RUBRIC = (
    "Score 0-5 for: accuracy, coverage, groundedness. "
    "Return JSON only: {\"accuracy\":x,\"coverage\":x,\"groundedness\":x,\"total\":x}"
)


def _extract_json_object(text: str) -> Dict[str, float]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


def _clip_0_5(x: float) -> float:
    return max(0.0, min(5.0, float(x)))


def _compact_evidence(evidence: List[dict], limit: int = 8) -> List[dict]:
    out: List[dict] = []
    for e in evidence[:limit]:
        out.append(
            {
                "title": str(e.get("title", ""))[:120],
                "url": str(e.get("url", ""))[:240],
                "source": str(e.get("source", ""))[:64],
                "snippet": str(e.get("snippet", ""))[:180],
            }
        )
    return out


def _fallback_score(answer: str, evidence: List[dict]) -> Dict[str, float]:
    has_answer = 2.0 if answer.strip() else 0.0
    has_evidence = 2.0 if evidence else 0.5
    grounded = 2.0 if evidence and any(str(x.get("url", "")).startswith("http") for x in evidence) else 1.0
    total = round(has_answer + has_evidence + grounded, 2)
    return {
        "accuracy": has_answer,
        "coverage": has_evidence,
        "groundedness": grounded,
        "total": total,
    }


def score_answer(query: str, answer: str, evidence: List[dict]):
    prompt = f"{RUBRIC}\nQuery:{query}\nAnswer:{answer}\nEvidence:{_compact_evidence(evidence, limit=8)}"
    try:
        msg = judge_llm.invoke(prompt).content
        data = _extract_json_object(msg)
        if not data:
            return _fallback_score(answer, evidence)

        accuracy = _clip_0_5(data.get("accuracy", 0.0))
        coverage = _clip_0_5(data.get("coverage", 0.0))
        groundedness = _clip_0_5(data.get("groundedness", 0.0))
        total = round(accuracy + coverage + groundedness, 2)
        return {
            "accuracy": accuracy,
            "coverage": coverage,
            "groundedness": groundedness,
            "total": total,
        }
    except Exception:
        return _fallback_score(answer, evidence)
