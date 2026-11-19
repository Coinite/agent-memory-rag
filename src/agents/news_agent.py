import json
import re
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Set

from src.agents.utils import append_agent_event
from src.models.lmstudio_client import build_llm
from src.tools.context_tools import get_location, get_time
from src.tools.web_tools import web_search

llm = build_llm(temp=0.0)

# Curated reliable domains.
CORE_NEWS_DOMAINS = [
    "reuters.com",
    "apnews.com",
    "npr.org",
    "bbc.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "bloomberg.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "economist.com",
    "theguardian.com",
    "axios.com",
    "politico.com",
    "cnn.com",
    "nbcnews.com",
    "abcnews.go.com",
    "cbsnews.com",
]

POLL_DOMAINS = [
    "pewresearch.org",
    "gallup.com",
    "yougov.com",
    "fivethirtyeight.com",
    "news.gallup.com",
]

MARKET_DATA_DOMAINS = [
    "fred.stlouisfed.org",
    "eia.gov",
    "iea.org",
    "opec.org",
    "imf.org",
    "worldbank.org",
    "ecb.europa.eu",
    "bis.org",
    "tradingeconomics.com",
    "spglobal.com",
    "lseg.com",
    "stooq.com",
]

RELIABLE_NEWS_DOMAINS = list(dict.fromkeys(CORE_NEWS_DOMAINS + POLL_DOMAINS + MARKET_DATA_DOMAINS))

SEARCH_TIME_BUDGET_SEC = 18.0
DEEP_SEARCH_TIME_BUDGET_SEC = 36.0
CACHE_TTL_SEC = 600.0
_NEWS_CACHE: Dict[str, Dict[str, Any]] = {}
DEFAULT_NEWS_SOURCE_LIMIT = 10


def _extract_json_object(text: str) -> Dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
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


def _invoke_text(prompt: str) -> str:
    try:
        msg = llm.invoke(prompt)
        content = getattr(msg, "content", "")
        return content if isinstance(content, str) else str(content)
    except Exception:
        return ""


def _extract_topic(query: str) -> str:
    prompt = (
        'Return JSON only: {"topic":"..."}.\n'
        "Extract a concise news search topic from user query.\n"
        f"query={query}\n"
    )
    for _ in range(2):
        data = _extract_json_object(_invoke_text(prompt))
        topic = str(data.get("topic", "")).strip()
        if topic:
            return topic
    return query


def _extract_host(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_reliable_host(host: str, allowed_domains: Set[str] | None = None) -> bool:
    if not host:
        return False
    domains = list(allowed_domains) if allowed_domains else RELIABLE_NEWS_DOMAINS
    for dom in domains:
        if host == dom or host.endswith("." + dom):
            return True
    return False


def _detect_news_intent(query: str, need_data: bool) -> str:
    q = (query or "").lower()
    market_terms = [
        "global market",
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
        "gdp",
        "recession",
        "global growth",
        "宏观",
        "市场",
        "经济",
        "油价",
        "通胀",
        "股市",
        "汇率",
        "供应链",
    ]
    poll_terms = [
        "majority",
        "support",
        "approval",
        "disapprove",
        "poll",
        "public opinion",
        "americans think",
        "民调",
        "支持率",
        "舆情",
    ]
    market_hit = any(t in q for t in market_terms)
    poll_hit = any(t in q for t in poll_terms)
    if market_hit:
        return "market_impact"
    if poll_hit:
        return "public_opinion"
    return "general_news"


def _domains_for_intent(intent: str, deep: bool) -> Set[str]:
    if intent == "market_impact":
        base = CORE_NEWS_DOMAINS + MARKET_DATA_DOMAINS
        if deep:
            base += POLL_DOMAINS[:2]
        return set(base)
    if intent == "public_opinion":
        return set(CORE_NEWS_DOMAINS + POLL_DOMAINS)
    return set(CORE_NEWS_DOMAINS)


def _query_cache_key(query: str) -> str:
    q = re.sub(r"\s+", " ", str(query or "").strip().lower())
    return q[:220]


def _constraint_focus_key(constraints: Dict[str, Any]) -> str:
    items: List[str] = []
    raw_cover = constraints.get("must_cover", [])
    if isinstance(raw_cover, list):
        for x in raw_cover:
            s = str(x or "").strip().lower()
            if s:
                items.append(s)
    breakdown = str(constraints.get("breakdown_by", "") or "").strip().lower()
    style = str(constraints.get("answer_style", "auto") or "auto").strip().lower()
    strict = "strict" if bool(constraints.get("strict_coverage", False)) else "loose"
    key_parts = [strict, style]
    if breakdown:
        key_parts.append(f"bd:{breakdown[:80]}")
    if items:
        key_parts.append("cover:" + "|".join(items[:5]))
    return "||".join(key_parts)[:280]


def _build_deep_queries(
    topic: str,
    deep: bool,
    freshness: str,
    local_date: str,
    intent: str,
    focus_items: List[str] | None = None,
    breakdown_by: str = "",
) -> List[str]:
    if intent == "market_impact":
        queries = [
            f"{topic} impact on global markets",
            f"{topic} impact on oil prices brent wti",
            f"{topic} effect on stocks bonds dollar gold",
            f"{topic} strait of hormuz shipping risk tanker rates",
            f"{topic} inflation and energy price outlook",
            f"{topic} central banks market reaction",
            f"{topic} global trade and supply chain impact",
            f"{topic} sector impact energy airlines shipping industrials",
        ]
    elif intent == "public_opinion":
        queries = [
            f"{topic} public opinion poll United States",
            f"{topic} approval rating Americans",
            f"{topic} Reuters Ipsos poll",
            f"{topic} Pew Research American views",
            f"{topic} Gallup survey",
            f"{topic} AP NORC poll",
            f"{topic} YouGov poll US",
            f"{topic} policy support among Americans",
        ]
    else:
        queries = [
            f"{topic} latest updates",
            f"{topic} timeline and key developments",
            f"{topic} international response",
            f"{topic} business and economic impact",
            f"{topic} conflict status now",
        ]

    if freshness == "recent":
        y = ""
        if local_date and len(local_date) >= 4:
            y = local_date[:4]
        if y:
            queries.extend(
                [
                    f"{topic} latest update {y}",
                    f"{topic} latest US news {y}",
                    f"{topic} breaking developments {y}",
                ]
            )
        else:
            queries.extend([f"{topic} latest update", f"{topic} latest US news", f"{topic} breaking developments"])

    for item in (focus_items or [])[:4]:
        s = str(item or "").strip()
        if s:
            queries.append(f"{topic} {s}")

    bd = str(breakdown_by or "").strip()
    if bd:
        queries.extend(
            [
                f"{topic} {bd} breakdown",
                f"{topic} {bd} latest figures",
                f"{topic} {bd} by group",
            ]
        )

    if deep:
        if intent == "market_impact":
            queries.extend(
                [
                    f"{topic} data chart oil gold sp500 dxy",
                    f"{topic} quantitative market analysis",
                    f"{topic} market volatility index and safe haven flows",
                    f"{topic} analysts estimate downside scenarios",
                ]
            )
        elif intent == "public_opinion":
            queries.extend(
                [
                    f"{topic} crosstab poll methodology",
                    f"{topic} trend over time survey",
                ]
            )
        else:
            queries.extend(
                [
                    f"{topic} market impact data analysis",
                    f"{topic} economists outlook quantitative estimates",
                ]
            )
    return queries


def _search_reliable_news(
    topic: str,
    source_limit: int = DEFAULT_NEWS_SOURCE_LIMIT,
    deep: bool = False,
    freshness: str = "any",
    local_date: str = "",
    intent: str = "general_news",
    query_key: str = "",
    directive_key: str = "",
    focus_items: List[str] | None = None,
    breakdown_by: str = "",
) -> tuple[List[Dict[str, str]], List[str]]:
    allowed_domains = _domains_for_intent(intent, deep)
    key = (
        f"{topic.strip().lower()}|{source_limit}|{str(deep).lower()}|{freshness}|{local_date}|"
        f"{intent}|{query_key}|{directive_key}"
    )
    now = time.time()
    cached = _NEWS_CACHE.get(key, {})
    if cached and (now - float(cached.get("ts", 0))) < CACHE_TTL_SEC:
        c_evd = cached.get("evidence", []) or []
        c_q = cached.get("queries", []) or []
        if len(c_evd) >= min(source_limit, 6):
            return c_evd[:source_limit], c_q

    evidence: List[Dict[str, str]] = []
    search_queries = _build_deep_queries(
        topic,
        deep=deep,
        freshness=freshness,
        local_date=local_date,
        intent=intent,
        focus_items=focus_items or [],
        breakdown_by=breakdown_by,
    )
    seen_urls = set()
    t0 = time.monotonic()
    time_budget = DEEP_SEARCH_TIME_BUDGET_SEC if deep else SEARCH_TIME_BUDGET_SEC
    k_per_query = 10 if deep else 8

    for q in search_queries:
        if (time.monotonic() - t0) > time_budget:
            break
        try:
            rows = web_search(q, k=k_per_query)
        except Exception:
            rows = []

        for r in rows:
            url = str(r.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            host = _extract_host(url)
            if not _is_reliable_host(host, allowed_domains):
                continue

            seen_urls.add(url)
            evidence.append(
                {
                    "title": str(r.get("title", "")).strip(),
                    "url": url,
                    "snippet": str(r.get("snippet", "")).strip(),
                    "source": host,
                }
            )

            if len(evidence) >= source_limit:
                break
        if len(evidence) >= source_limit:
            break

    # If results are still thin, run site-targeted retrieval across trusted domains.
    if len(evidence) < source_limit:
        domain_pool = sorted(list(allowed_domains))
        domain_limit = len(domain_pool) if deep else min(12, len(domain_pool))
        site_k = 6 if deep else 4
        for dom in domain_pool[:domain_limit]:
            if (time.monotonic() - t0) > time_budget:
                break
            q = f"{topic} site:{dom}"
            try:
                rows = web_search(q, k=site_k)
            except Exception:
                rows = []
            for r in rows:
                url = str(r.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                host = _extract_host(url)
                if not _is_reliable_host(host, allowed_domains):
                    continue
                seen_urls.add(url)
                evidence.append(
                    {
                        "title": str(r.get("title", "")).strip(),
                        "url": url,
                        "snippet": str(r.get("snippet", "")).strip(),
                        "source": host,
                    }
                )
                if len(evidence) >= source_limit:
                    break
            if len(evidence) >= source_limit:
                break

    # Final catch-up pass.
    if len(evidence) < source_limit:
        catch_up: List[str] = []
        for item in (focus_items or [])[:3]:
            s = str(item or "").strip()
            if s:
                catch_up.append(f"{topic} {s} latest")
        bd = str(breakdown_by or "").strip()
        if bd:
            catch_up.extend(
                [
                    f"{topic} {bd} breakdown latest",
                    f"{topic} {bd} estimates by group",
                ]
            )

        if not catch_up and intent == "market_impact":
            catch_up = [
                f"{topic} latest global market reaction data",
                f"{topic} oil price reaction brent wti",
                f"{topic} stocks bonds dollar gold moves",
            ]
        elif not catch_up and intent == "public_opinion":
            catch_up = [f"{topic} latest US poll", f"{topic} latest public opinion"]
        elif not catch_up:
            catch_up = [f"{topic} latest US news", f"{topic} latest update"]
        if deep:
            catch_up.extend([f"{topic} market data report", f"{topic} sector impact report"])
        for q in catch_up:
            if (time.monotonic() - t0) > time_budget:
                break
            try:
                rows = web_search(q, k=6)
            except Exception:
                rows = []
            for r in rows:
                url = str(r.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                host = _extract_host(url)
                if not _is_reliable_host(host, allowed_domains):
                    continue
                seen_urls.add(url)
                evidence.append(
                    {
                        "title": str(r.get("title", "")).strip(),
                        "url": url,
                        "snippet": str(r.get("snippet", "")).strip(),
                        "source": host,
                    }
                )
                if len(evidence) >= source_limit:
                    break
            if len(evidence) >= source_limit:
                break

    final_evidence = evidence[:source_limit]
    _NEWS_CACHE[key] = {"ts": now, "evidence": final_evidence, "queries": search_queries}
    return final_evidence, search_queries


def _infer_event_status(query: str, evidence: List[Dict[str, str]]) -> str:
    q = (query or "").lower()
    if "the war between" in q or ("war" in q and "potential" not in q and "possible" not in q):
        return "ongoing"

    text = " ".join(
        [str(x.get("title", "")) + " " + str(x.get("snippet", "")) for x in evidence[:12]]
    ).lower()
    ongoing_terms = [
        "war",
        "conflict",
        "strikes",
        "airstrike",
        "missile",
        "retaliation",
        "casualties",
        "fighting",
        "battle",
        "ongoing",
        "continued",
    ]
    potential_terms = [
        "potential",
        "possible",
        "risk",
        "could",
        "may",
        "fear",
        "concern",
        "might",
    ]
    ongoing_hits = sum(1 for t in ongoing_terms if t in text)
    potential_hits = sum(1 for t in potential_terms if t in text)
    if ongoing_hits >= potential_hits + 2:
        return "ongoing"
    if potential_hits >= ongoing_hits + 2:
        return "potential"
    return "uncertain"


def _compose_news_draft(
    query: str,
    topic: str,
    evidence: List[Dict[str, str]],
    local_iso: str,
    need_data: bool,
    intent: str,
    event_status: str,
    must_cover: List[str],
    breakdown_by: str,
    answer_style: str,
    strict_coverage: bool,
) -> str:
    compact = [
        {
            "title": str(x.get("title", ""))[:100],
            "url": str(x.get("url", ""))[:220],
            "source": str(x.get("source", ""))[:64],
            "snippet": str(x.get("snippet", ""))[:140],
        }
        for x in evidence[:12]
    ]
    prompt = (
        'Return JSON only: {"summary":"..."}.\n'
        "Rules:\n"
        "1) no chain-of-thought, no <think>.\n"
        "2) 5-8 sentences.\n"
        f"3) intent={intent}. If intent=market_impact, prioritize oil, equities, bonds/rates, FX, shipping/trade, and sectors.\n"
        f"4) event_status={event_status}. If event_status=ongoing, do NOT describe it as potential/hypothetical.\n"
        f"5) need_data={str(need_data).lower()}.\n"
        "6) If need_data=true, include at least four concrete numeric/directional data points tied to metric+direction+time.\n"
        "7) Avoid using public-opinion polling as primary evidence for market-impact questions.\n"
        f"8) must_cover={must_cover[:6]}.\n"
        f"9) breakdown_by={breakdown_by}\n"
        f"10) answer_style={answer_style}\n"
        f"11) strict_coverage={str(strict_coverage).lower()}.\n"
        "12) If strict_coverage=true, explicitly address each required coverage item; use "
        "'not reliably reported' when evidence is insufficient.\n"
        "13) If breakdown_by is non-empty, provide a labeled breakdown by that axis with group-level values/notes.\n"
        f"local_time={local_iso}\n"
        f"query={query}\n"
        f"topic={topic}\n"
        f"evidence={compact}\n"
    )
    for _ in range(2):
        data = _extract_json_object(_invoke_text(prompt))
        text = str(data.get("summary", "")).strip()
        if text:
            return text

    parts: List[str] = []
    for item in evidence[:6]:
        snippet = str(item.get("snippet", "")).strip()
        title = str(item.get("title", "")).strip()
        text = snippet or title
        if not text:
            continue
        text = text.replace("\n", " ").strip().rstrip(".")
        if text:
            parts.append(text)
    if parts:
        return ". ".join(parts[:4]) + "."
    return ""


def _to_int(value: Any, default: int = 0, low: int = 0, high: int = 50) -> int:
    try:
        x = int(value)
        return max(low, min(high, x))
    except Exception:
        return default


def news_agent_node(state):
    query = state["query"]
    local_time = state.get("local_time") or get_time()
    _ = state.get("local_location") or get_location()
    constraints = state.get("constraints", {}) or {}

    raw_cover = constraints.get("must_cover", [])
    must_cover: List[str] = []
    if isinstance(raw_cover, list):
        for x in raw_cover:
            s = str(x or "").strip()
            if s and s not in must_cover:
                must_cover.append(s)
    elif isinstance(raw_cover, str):
        s = raw_cover.strip()
        if s:
            must_cover.append(s)

    breakdown_by = str(constraints.get("breakdown_by", "") or "").strip()
    answer_style = str(constraints.get("answer_style", "auto") or "auto").strip().lower()
    strict_coverage = bool(constraints.get("strict_coverage", False))

    query_key = _query_cache_key(query)
    directive_key = _constraint_focus_key(constraints)
    min_sources = _to_int(constraints.get("min_sources", 0), default=0, low=0, high=50)
    depth = str(constraints.get("search_depth", "normal") or "normal").lower()
    freshness = str(constraints.get("freshness", "any") or "any").lower()
    need_data = bool(constraints.get("need_data", False)) or bool(breakdown_by)
    deep_mode = depth == "deep" or strict_coverage or bool(breakdown_by)
    intent = _detect_news_intent(query, need_data=bool(constraints.get("need_data", False)))
    source_limit = DEFAULT_NEWS_SOURCE_LIMIT
    if min_sources > 0:
        source_limit = max(source_limit, min_sources)
    if strict_coverage or breakdown_by or len(must_cover) >= 2:
        source_limit = max(source_limit, 12)
    if deep_mode:
        source_limit = max(source_limit, 14)
    source_limit = max(6, min(30, source_limit))

    topic = _extract_topic(query)

    evidence, search_queries = _search_reliable_news(
        topic,
        source_limit=source_limit,
        deep=deep_mode,
        freshness="recent" if freshness == "recent" else "any",
        local_date=local_time.get("date", ""),
        intent=intent,
        query_key=query_key,
        directive_key=directive_key,
        focus_items=must_cover,
        breakdown_by=breakdown_by,
    )
    if not evidence:
        state["draft_answer"] = f"Could not retrieve reliable news for topic: {topic}"
        state["evidence"] = []
        state["search_queries"] = search_queries
        state["logs"].append("news_agent.fetch=failed")
        append_agent_event(
            state,
            agent_name="news_agent",
            input_payload={"query": query, "topic": topic},
            output_payload={"status": "failed", "search_queries": search_queries},
        )
        return state

    event_status = _infer_event_status(query, evidence)
    draft = _compose_news_draft(
        query,
        topic,
        evidence,
        local_time.get("iso", ""),
        need_data=need_data,
        intent=intent,
        event_status=event_status,
        must_cover=must_cover,
        breakdown_by=breakdown_by,
        answer_style=answer_style,
        strict_coverage=strict_coverage,
    )
    state["search_queries"] = search_queries
    state["evidence"] = evidence
    state["draft_answer"] = draft
    state["logs"].append(f"news_agent.fetch=ok:{len(evidence)}")
    state["logs"].append(f"news_agent.depth={'deep' if deep_mode else 'normal'}")
    state["logs"].append(f"news_agent.source_target={source_limit}")
    state["logs"].append(f"news_agent.intent={intent}")
    state["logs"].append(f"news_agent.event_status={event_status}")
    if must_cover:
        state["logs"].append(
            "news_agent.must_cover="
            + json.dumps(must_cover[:6], ensure_ascii=False, separators=(",", ":"))
        )
    if breakdown_by:
        state["logs"].append(f"news_agent.breakdown_by={breakdown_by}")

    append_agent_event(
        state,
        agent_name="news_agent",
        input_payload={
            "query": query,
            "topic": topic,
            "constraints": {
                "min_sources": min_sources,
                "search_depth": depth,
                "freshness": freshness,
                "need_data": need_data,
                "must_cover": must_cover,
                "breakdown_by": breakdown_by,
                "answer_style": answer_style,
                "strict_coverage": strict_coverage,
            },
            "intent": intent,
            "event_status": event_status,
            "directive_key": directive_key,
        },
        output_payload={
            "evidence_count": len(evidence),
            "source_count": len({e.get('source', '') for e in evidence if e.get('source', '')}),
            "search_queries": search_queries,
            "draft_preview": state["draft_answer"][:400],
        },
    )
    return state
