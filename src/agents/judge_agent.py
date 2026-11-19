import json
import re
from typing import Any, Dict, List, Tuple

from src.agents.utils import append_agent_event
from src.eval.judge import score_answer
from src.models.lmstudio_client import build_llm

llm = build_llm(temp=0.0)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _invoke_text(prompt: str) -> str:
    try:
        msg = llm.invoke(prompt)
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        return str(content)
    except Exception:
        return ""


def _extract_json_object(text: str) -> Dict[str, Any]:
    clean = _strip_think(text)
    try:
        data = json.loads(clean)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(clean[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _has_reliable_sources(evidence: List[dict]) -> bool:
    urls = [str(x.get("url", "")).strip() for x in evidence]
    return any(u.startswith("http") for u in urls)


def _source_count(evidence: List[dict]) -> int:
    seen = set()
    for e in evidence:
        u = str(e.get("url", "")).strip()
        if u.startswith("http"):
            seen.add(u)
    return len(seen)


def _numeric_token_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text or ""))


def _required_cover_items(constraints: Dict[str, Any]) -> List[str]:
    raw = (constraints or {}).get("must_cover", [])
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
    elif isinstance(raw, str):
        for item in re.split(r"[,\n;|]+", raw):
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
    return out[:8]


def _breakdown_axis(constraints: Dict[str, Any]) -> str:
    return str((constraints or {}).get("breakdown_by", "") or "").strip()


def _covers_required_items(answer: str, required_items: List[str], strict: bool) -> bool:
    if not required_items:
        return True
    text = (answer or "").lower()
    matched = 0
    for item in required_items:
        tokens = [
            t
            for t in re.findall(r"[a-zA-Z0-9\u3400-\u9fff]+", item.lower())
            if (len(t) >= 3 or bool(re.search(r"[\u3400-\u9fff]", t)))
        ]
        if not tokens:
            continue
        token_hits = sum(1 for t in tokens if t in text)
        if token_hits >= max(1, min(2, len(tokens))):
            matched += 1
    if strict:
        return matched >= len(required_items)
    return matched >= max(1, len(required_items) // 2)


def _has_structured_breakdown(answer: str, breakdown_by: str) -> bool:
    if not breakdown_by:
        return True
    text = (answer or "").strip()
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    structured = [
        ln
        for ln in lines
        if re.match(r"^[-*•]\s+", ln) or re.match(r"^\d+\.\s+", ln) or (":" in ln)
    ]
    joined = " ".join(structured) if structured else text
    has_unknown_marker = bool(
        re.search(r"\b(not reliably reported|unknown|n/?a|insufficient evidence)\b", joined, flags=re.IGNORECASE)
    )
    if len(structured) >= 2 and (_numeric_token_count(joined) >= 2 or has_unknown_marker):
        return True
    if text.count(":") >= 2 and (_numeric_token_count(text) >= 2 or has_unknown_marker):
        return True
    return False


def _is_market_query(query: str) -> bool:
    q = (query or "").lower()
    terms = [
        "market",
        "economy",
        "economic",
        "oil",
        "energy",
        "inflation",
        "stocks",
        "equity",
        "bond",
        "forex",
        "fx",
        "commodity",
        "shipping",
        "trade",
        "supply chain",
        "global growth",
        "gdp",
        "recession",
    ]
    return any(t in q for t in terms)


def _has_market_evidence(evidence: List[dict]) -> bool:
    merged = " ".join(
        [
            str(x.get("title", "")) + " " + str(x.get("snippet", "")) + " " + str(x.get("url", ""))
            for x in evidence[:20]
        ]
    ).lower()
    hints = [
        "market",
        "business",
        "economy",
        "economic",
        "oil",
        "energy",
        "brent",
        "wti",
        "stocks",
        "s&p",
        "nasdaq",
        "bond",
        "yield",
        "dollar",
        "gold",
        "shipping",
        "trade",
        "inflation",
        "reuters.com/business",
        "bloomberg.com",
        "wsj.com",
        "ft.com",
    ]
    return any(h in merged for h in hints)


def _evidence_suggests_ongoing_conflict(evidence: List[dict]) -> bool:
    merged = " ".join(
        [str(x.get("title", "")) + " " + str(x.get("snippet", "")) for x in evidence[:20]]
    ).lower()
    ongoing_terms = ["war", "conflict", "strikes", "airstrikes", "retaliation", "fighting", "ongoing"]
    return sum(1 for t in ongoing_terms if t in merged) >= 2


def _needs_verify(
    query: str,
    scores: Dict[str, float],
    answer: str,
    evidence: List[dict],
    route: str,
    did_web_verify: bool,
    constraints: Dict[str, Any],
) -> bool:
    if did_web_verify:
        return False
    total = float(scores.get("total", 0.0))
    avg = total / 3.0 if total > 5.0 else total
    if avg < 3.0:
        return True
    if "TODO" in answer:
        return True
    if "<think>" in answer.lower():
        return True
    if route in {"weather", "news", "stock"} and not _has_reliable_sources(evidence):
        return True
    required_sources = 6
    try:
        min_sources = int(constraints.get("min_sources", 0) or 0)
        if min_sources > 0:
            required_sources = max(required_sources, min(30, min_sources))
    except Exception:
        pass
    if route == "news" and _source_count(evidence) < required_sources:
        return True
    if bool(constraints.get("must_cite", False)) and _source_count(evidence) == 0:
        return True
    if bool(constraints.get("need_data", False)) and _numeric_token_count(answer) < 2:
        return True
    if route == "news" and _is_market_query(query) and not _has_market_evidence(evidence):
        return True
    required_items = _required_cover_items(constraints)
    breakdown_by = _breakdown_axis(constraints)
    strict_coverage = bool(constraints.get("strict_coverage", False))
    if route == "news" and (strict_coverage or breakdown_by):
        if _source_count(evidence) < max(8, required_sources):
            return True
    if route == "news" and strict_coverage and not _covers_required_items(answer, required_items, strict=True):
        return True
    if route == "news" and breakdown_by and not _has_structured_breakdown(answer, breakdown_by):
        return True
    if "war" in (query or "").lower() and _evidence_suggests_ongoing_conflict(evidence):
        if re.search(r"\b(potential|possible|might|could start)\b", answer or "", flags=re.IGNORECASE):
            return True
    return False


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _parse_length_pref(query: str, constraints: Dict[str, Any] | None = None) -> Tuple[str, int]:
    c = constraints or {}
    try:
        w = int(c.get("answer_words", 0) or 0)
    except Exception:
        w = 0
    try:
        ch = int(c.get("answer_chars", 0) or 0)
    except Exception:
        ch = 0
    if w > 0:
        return "words", max(10, min(1200, w))
    if ch > 0:
        return "chars", max(20, min(1200, ch))

    required_items = _required_cover_items(c)
    breakdown_by = _breakdown_axis(c)
    if bool(c.get("strict_coverage", False)) and required_items:
        target = max(90, min(1200, 40 + len(required_items) * 25))
        return "words", target
    if breakdown_by:
        return "words", 110

    q = query or ""
    nums = [int(x) for x in re.findall(r"\d{1,4}", q)]

    m_words_zh = re.search(r"(\d{1,4})\s*(词|单词)", q, flags=re.IGNORECASE)
    if m_words_zh:
        n = max(10, min(200, int(m_words_zh.group(1))))
        return "words", n
    if ("词" in q or "单词" in q) and nums:
        n = max(10, min(200, nums[0]))
        return "words", n

    m_zh = re.search(r"(\d{1,4})\s*(字|字符)", q, flags=re.IGNORECASE)
    if m_zh:
        n = max(20, min(400, int(m_zh.group(1))))
        return "chars", n
    if ("字" in q or "字符" in q) and nums:
        n = max(20, min(400, nums[0]))
        return "chars", n

    m_chars = re.search(r"(\d{1,4})\s*(char|chars|character|characters)", q, flags=re.IGNORECASE)
    if m_chars:
        n = max(20, min(400, int(m_chars.group(1))))
        return "chars", n

    m_words = re.search(r"(\d{1,4})\s*(word|words)", q, flags=re.IGNORECASE)
    if m_words:
        n = max(10, min(200, int(m_words.group(1))))
        return "words", n

    # Default policy: around 60 Chinese chars for CJK queries, otherwise around 60 words.
    if _contains_cjk(q):
        return "chars", 60
    return "words", 60


def _finish_sentence(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    if t[-1] in ".!?。！？":
        return t
    return t + "."


def _trim_to_target(text: str, unit: str, target: int) -> str:
    t = text.strip()
    if not t:
        return t

    if unit == "words":
        words = t.split()
        if len(words) <= target:
            return _finish_sentence(t)
        trimmed = " ".join(words[:target]).strip()
        return _finish_sentence(trimmed)

    # chars: try not to cut mid-word/sentence.
    if len(t) <= target:
        return _finish_sentence(t)

    hard = t[:target]
    # Find natural boundary near the end.
    cut = max(hard.rfind("。"), hard.rfind("！"), hard.rfind("？"), hard.rfind("."), hard.rfind("!"), hard.rfind("?"))
    if cut >= int(target * 0.6):
        return _finish_sentence(hard[: cut + 1].strip())

    ws = hard.rfind(" ")
    if ws >= int(target * 0.6):
        return _finish_sentence(hard[:ws].strip())

    return _finish_sentence(hard.strip())


def _word_count(text: str) -> int:
    return len((text or "").split())


def _length_sufficient(text: str, unit: str, target: int) -> bool:
    if target <= 0:
        return True
    if unit == "words":
        return _word_count(text) >= int(target * 0.8)
    return len(text or "") >= int(target * 0.8)


def _dedup_urls(urls: List[str], limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls:
        s = str(u).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _format_sources_block(urls: List[str]) -> str:
    if not urls:
        return ""
    lines = ["Sources:"]
    for i, u in enumerate(urls, start=1):
        lines.append(f"{i}. {u}")
    return "\n".join(lines)


def _format_final_answer(
    query: str,
    answer: str,
    evidence: List[dict],
    local_time_iso: str,
    route: str,
    constraints: Dict[str, Any],
) -> str:
    unit, target = _parse_length_pref(query, constraints)
    need_data = bool(constraints.get("need_data", False))
    required_items = _required_cover_items(constraints)
    breakdown_by = _breakdown_axis(constraints)
    answer_style = str(constraints.get("answer_style", "auto") or "auto").strip().lower()
    strict_coverage = bool(constraints.get("strict_coverage", False))
    source_limit = 10 if route == "news" else 3
    if route == "news" and (strict_coverage or breakdown_by):
        source_limit = max(source_limit, 8)
    try:
        min_sources = int(constraints.get("min_sources", 0) or 0)
        if route == "news" and min_sources > 0:
            source_limit = max(source_limit, min(12, min_sources))
    except Exception:
        pass
    if bool(constraints.get("must_cite", False)) and route != "news":
        source_limit = max(source_limit, 4)

    compact_evidence = []
    for e in evidence[:8]:
        compact_evidence.append(
            {
                "title": str(e.get("title", ""))[:100],
                "url": str(e.get("url", ""))[:220],
                "source": str(e.get("source", ""))[:64],
                "snippet": str(e.get("snippet", ""))[:140],
            }
        )

    prompt = (
        'Return JSON only: {"final":"...","sources":["url1","url2"]}.\n'
        "Rules:\n"
        "1) no chain-of-thought, no <think>.\n"
        "2) final should be factual and directly answer the query.\n"
        f"3) final length target: about {target} {unit}, counting FINAL only (exclude sources).\n"
        f"4) include up to {source_limit} source URLs in sources array only.\n"
        f"5) need_data={str(need_data).lower()}; if true include concrete numbers and time anchors.\n"
        "6) For market-impact questions, include sector-level impact and key indicators.\n"
        f"7) must_cover={required_items[:6]}\n"
        f"8) breakdown_by={breakdown_by}\n"
        f"9) answer_style={answer_style}\n"
        f"10) strict_coverage={str(strict_coverage).lower()}\n"
        "11) If strict_coverage=true, explicitly address each required coverage item. "
        "If evidence is insufficient, state 'not reliably reported'.\n"
        "12) If breakdown_by is non-empty, include a labeled breakdown by that axis.\n"
        f"local_time={local_time_iso}\n"
        f"route={route}\n"
        f"query={query}\n"
        f"answer={answer}\n"
        f"evidence={compact_evidence}\n"
    )

    urls_from_evidence = _dedup_urls([str(x.get("url", "")).strip() for x in evidence], source_limit)
    ongoing_conflict = _evidence_suggests_ongoing_conflict(evidence) and "war" in (query or "").lower()

    for _ in range(2):
        obj = _extract_json_object(_invoke_text(prompt))
        final_raw = str(obj.get("final", "")).strip()
        model_urls = obj.get("sources", []) if isinstance(obj.get("sources", []), list) else []
        urls = _dedup_urls([str(u).strip() for u in model_urls] + urls_from_evidence, source_limit)

        if final_raw:
            if ongoing_conflict:
                final_raw = re.sub(r"\bpotential\b", "ongoing", final_raw, flags=re.IGNORECASE)
            if strict_coverage and required_items and not _covers_required_items(
                final_raw, required_items, strict=True
            ):
                continue
            if breakdown_by and not _has_structured_breakdown(final_raw, breakdown_by):
                continue
            if target >= 120 and not _length_sufficient(final_raw, unit, target):
                expand_prompt = (
                    'Return JSON only: {"final":"..."}.\n'
                    f"Expand the text to about {target} {unit} while keeping facts consistent.\n"
                    f"need_data={str(need_data).lower()}\n"
                    f"query={query}\n"
                    f"base_answer={final_raw}\n"
                    f"evidence={compact_evidence}\n"
                )
                expanded = _extract_json_object(_invoke_text(expand_prompt))
                expanded_text = str(expanded.get("final", "")).strip()
                if expanded_text:
                    final_raw = expanded_text
            final = _trim_to_target(final_raw, unit, target)
            src = _format_sources_block(urls)
            return final + ("\n" + src if src else "")

    brief = _trim_to_target(_strip_think(answer).split("\n")[0].strip(), unit, target)
    src = _format_sources_block(urls_from_evidence)
    return brief + ("\n" + src if src else "")


def judge_agent_node(state):
    answer = _strip_think(state.get("draft_answer", ""))
    evidence = state.get("evidence", [])
    route = state.get("route", "general")
    did_web_verify = bool(state.get("did_web_verify", False))
    constraints = state.get("constraints", {}) or {}

    scores = score_answer(state["query"], answer, evidence)
    state["scores"] = scores

    needs_verify = _needs_verify(state["query"], scores, answer, evidence, route, did_web_verify, constraints)
    state["needs_web_verify"] = needs_verify

    state["final_answer"] = _format_final_answer(
        query=state["query"],
        answer=answer,
        evidence=evidence,
        local_time_iso=state.get("local_time", {}).get("iso", ""),
        route=route,
        constraints=constraints,
    )

    state["logs"].append(f"judge_agent.total={scores.get('total', 0):.2f}")
    state["logs"].append(f"judge_agent.needs_web_verify={needs_verify}")
    state["logs"].append(f"judge_agent.source_count={_source_count(evidence)}")

    append_agent_event(
        state,
        agent_name="judge_agent",
        input_payload={
            "query": state["query"],
            "route": route,
            "did_web_verify": did_web_verify,
            "evidence_count": len(evidence),
            "constraints": constraints,
        },
        output_payload={
            "scores": scores,
            "needs_web_verify": needs_verify,
            "final_preview": state["final_answer"][:300],
        },
    )
    return state
