import json
import math
import re
from typing import Any, Dict, List

from src.agents.utils import append_agent_event
from src.models.lmstudio_client import build_llm
from src.tools.context_tools import get_location, get_time
from src.tools.realtime_tools import get_stock_stooq, get_stock_stooq_history

llm = build_llm(temp=0.0)

SECTOR_UNIVERSE: Dict[str, List[str]] = {
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "KMI", "OXY"],
    "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "AVGO", "INTC"],
    "finance": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW"],
    "healthcare": ["LLY", "JNJ", "PFE", "MRK", "ABBV", "TMO", "UNH", "BMY"],
    "industrial": ["GE", "CAT", "HON", "DE", "BA", "LMT", "RTX", "MMM"],
}
DEFAULT_SECTOR = "energy"
DEFAULT_PICK_COUNT = 5
DEFAULT_LOOKBACK_HOURS = 24


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


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


def _invoke_text(prompt: str) -> str:
    try:
        msg = llm.invoke(prompt)
        content = getattr(msg, "content", "")
        return content if isinstance(content, str) else str(content)
    except Exception:
        return ""


def _normalize_symbols(items: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for v in items:
        s = re.sub(r"[^A-Za-z.]", "", str(v or "").strip().upper())
        if not s:
            continue
        if "." in s:
            s = s.split(".")[0]
        if 1 <= len(s) <= 5 and s not in out:
            out.append(s)
    return out


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_task_with_llm(query: str) -> Dict[str, Any]:
    prompt = (
        "Return JSON only with schema:\n"
        '{"count":5,"sector":"energy","symbols":["XOM"],"lookback_hours":24,"lookback_days":0}\n'
        "Rules:\n"
        "0) Use this as an EXAMPLE schema only, do not copy symbols unless user requests.\n"
        "1) symbols should include explicitly requested tickers only.\n"
        "2) lookback_days for 'past N days'; lookback_hours for 'past N hours'.\n"
        "3) If time range unspecified set lookback_hours=24.\n"
        f"query={query}\n"
    )
    data = _extract_json_object(_invoke_text(prompt))
    if not data:
        return {}
    count = max(1, min(10, _to_int(data.get("count", DEFAULT_PICK_COUNT), DEFAULT_PICK_COUNT)))
    sector = str(data.get("sector", DEFAULT_SECTOR) or DEFAULT_SECTOR).strip().lower()
    symbols = _normalize_symbols(data.get("symbols", []))
    lookback_days = max(0, min(90, _to_int(data.get("lookback_days", 0), 0)))
    lookback_hours = max(0, min(24 * 30, _to_int(data.get("lookback_hours", 0), 0)))
    return {
        "count": count,
        "sector": sector if sector else DEFAULT_SECTOR,
        "symbols": symbols,
        "lookback_days": lookback_days,
        "lookback_hours": lookback_hours,
    }


def _extract_stock_task(query: str) -> Dict[str, Any]:
    for _ in range(3):
        llm_task = _extract_task_with_llm(query)
        if llm_task:
            if llm_task.get("lookback_days", 0) <= 0 and llm_task.get("lookback_hours", 0) <= 0:
                llm_task["lookback_hours"] = DEFAULT_LOOKBACK_HOURS
            return llm_task
    return {
        "count": DEFAULT_PICK_COUNT,
        "sector": DEFAULT_SECTOR,
        "symbols": [],
        "lookback_days": 0,
        "lookback_hours": DEFAULT_LOOKBACK_HOURS,
    }


def _pick_symbols(task: Dict[str, Any]) -> List[str]:
    count = max(1, min(10, int(task.get("count", DEFAULT_PICK_COUNT))))
    chosen: List[str] = []
    for s in _normalize_symbols(task.get("symbols", [])):
        if s not in chosen:
            chosen.append(s)
        if len(chosen) >= count:
            return chosen

    sector = str(task.get("sector", DEFAULT_SECTOR) or DEFAULT_SECTOR).lower()
    universe = SECTOR_UNIVERSE.get(sector, SECTOR_UNIVERSE[DEFAULT_SECTOR])
    for s in universe:
        if s not in chosen:
            chosen.append(s)
        if len(chosen) >= count:
            break
    return chosen[:count]


def _resolve_lookback_days(task: Dict[str, Any]) -> int:
    days = int(task.get("lookback_days", 0) or 0)
    if days > 0:
        return max(1, min(90, days))

    hours = int(task.get("lookback_hours", DEFAULT_LOOKBACK_HOURS) or DEFAULT_LOOKBACK_HOURS)
    hours = max(1, min(24 * 30, hours))
    # Stooq provides daily bars; map hourly windows to a small daily span.
    return max(2, min(90, int(math.ceil(hours / 24.0)) + 1))


def _series_change_pct(rows: List[Dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    first = rows[0].get("Close")
    last = rows[-1].get("Close")
    if first in (None, 0) or last is None:
        return None
    try:
        return (float(last) - float(first)) / float(first) * 100.0
    except Exception:
        return None


def stock_agent_node(state):
    query = state["query"]
    local_time = state.get("local_time") or get_time()
    _ = state.get("local_location") or get_location()

    task = _extract_stock_task(query)
    symbols = _pick_symbols(task)
    lookback_days = _resolve_lookback_days(task)
    lookback_hours = int(task.get("lookback_hours", 0) or 0)
    if int(task.get("lookback_days", 0) or 0) > 0:
        lookback_hours = int(task["lookback_days"]) * 24
    if lookback_hours <= 0:
        lookback_hours = DEFAULT_LOOKBACK_HOURS

    series: List[Dict[str, Any]] = []
    evidence: List[Dict[str, str]] = []

    for sym in symbols:
        h = get_stock_stooq_history(sym, lookback_days=lookback_days)
        if h.get("ok"):
            rows = h.get("rows", []) or []
            if not rows:
                continue
            sym_clean = str(h.get("symbol", sym)).upper().replace(".US", "")
            change_pct = _series_change_pct(rows)
            latest = rows[-1]
            d0 = rows[0].get("Date", "")
            d1 = rows[-1].get("Date", "")
            change_text = "N/A" if change_pct is None else f"{change_pct:+.2f}%"
            summary = (
                f"{sym_clean} {d0}->{d1}: close={latest.get('Close','N/A')}, "
                f"change={change_text}, points={len(rows)}."
            )
            series.append(
                {
                    "symbol": sym_clean,
                    "rows": rows,
                    "source": h.get("source", ""),
                    "resolution": h.get("resolution", "1d"),
                }
            )
            evidence.append(
                {
                    "title": f"{sym_clean} historical prices",
                    "url": str(h.get("source", "")),
                    "snippet": summary,
                    "source": "stooq",
                }
            )
            continue

        # Fallback to latest quote when history fails.
        q = get_stock_stooq(sym)
        if q.get("ok"):
            quote = q.get("quote", {}) or {}
            sym_clean = str(q.get("symbol", sym)).upper().replace(".US", "")
            summary = (
                f"{sym_clean} latest: date={quote.get('Date','N/A')}, close={quote.get('Close','N/A')}, "
                f"open={quote.get('Open','N/A')}."
            )
            series.append(
                {
                    "symbol": sym_clean,
                    "rows": [
                        {
                            "Date": quote.get("Date"),
                            "Open": quote.get("Open"),
                            "High": quote.get("High"),
                            "Low": quote.get("Low"),
                            "Close": quote.get("Close"),
                            "Volume": quote.get("Volume"),
                        }
                    ],
                    "source": q.get("source", ""),
                    "resolution": "latest",
                }
            )
            evidence.append(
                {
                    "title": f"{sym_clean} latest quote",
                    "url": str(q.get("source", "")),
                    "snippet": summary,
                    "source": "stooq",
                }
            )

    if not series:
        state["draft_answer"] = "Could not retrieve stock data for requested symbols."
        state["evidence"] = []
        state["logs"].append("stock_agent.fetch=failed")
        append_agent_event(
            state,
            agent_name="stock_agent",
            input_payload={"query": query, "task": task, "symbols": symbols},
            output_payload={"status": "failed"},
        )
        return state

    payload = {
        "sector": task.get("sector", DEFAULT_SECTOR),
        "symbols": [x.get("symbol", "") for x in series],
        "lookback_hours": lookback_hours,
        "lookback_days": lookback_days,
        "resolution": "1d",
        "series": series,
    }
    context_payload = state.get("context_payload", {}) or {}
    context_payload["stock"] = payload
    state["context_payload"] = context_payload

    state["evidence"] = evidence
    state["draft_answer"] = (
        f"Collected stock data for {len(series)} symbols in {task.get('sector', DEFAULT_SECTOR)} "
        f"over ~{lookback_hours}h window (daily bars)."
    )
    state["logs"].append(f"stock_agent.fetch=ok:{len(series)}")
    state["logs"].append(f"stock_agent.symbols={','.join(payload['symbols'])}")
    state["logs"].append(f"stock_agent.lookback_hours={lookback_hours}")

    append_agent_event(
        state,
        agent_name="stock_agent",
        input_payload={"query": query, "task": task},
        output_payload={
            "series_count": len(series),
            "symbols": payload["symbols"],
            "lookback_hours": lookback_hours,
            "evidence_count": len(evidence),
            "draft_preview": state["draft_answer"][:240],
        },
    )
    return state
