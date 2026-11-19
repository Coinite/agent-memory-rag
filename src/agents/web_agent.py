import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.agents.utils import append_agent_event
from src.config import settings
from src.models.lmstudio_client import build_llm
from src.tools.fs_tools import make_dir, write_file
from src.tools.web_tools import (
    download_pdf_to_workspace,
    fetch_web_text,
    infer_pdf_filename,
    to_pdf_url,
    web_search,
)

llm_json = build_llm(temp=0.0, max_tokens=180, response_format={"type": "json_object"})
llm_plain = build_llm(temp=0.0, max_tokens=220)
llm_data_json = build_llm(temp=0.0, max_tokens=900, response_format={"type": "json_object"})

ROOT = Path(settings.workspace_root).resolve()


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _content_to_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _invoke_text(prompt: str) -> str:

    try:
        text = _content_to_text(llm_json.invoke(prompt))
        if text.strip():
            return text
    except Exception:
        pass
    try:
        return _content_to_text(llm_plain.invoke(prompt))
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


def _invoke_json(prompt: str) -> Dict[str, Any]:
    raw = _invoke_text(prompt)
    obj = _extract_json_object(raw)
    if obj:
        return obj
    repair_prompt = (
        "Fix and return valid JSON object only.\n"
        f"broken_output={raw[:1500]}\n"
    )
    return _extract_json_object(_invoke_text(repair_prompt))


def _invoke_data_json(prompt: str) -> Dict[str, Any]:
    try:
        raw = _content_to_text(llm_data_json.invoke(prompt))
        obj = _extract_json_object(raw)
        if obj:
            return obj
    except Exception:
        pass
    return _invoke_json(prompt)


def _normalize_queries(raw: Dict[str, Any], user_query: str) -> List[str]:
    queries = raw.get("queries", []) if isinstance(raw, dict) else []
    out: List[str] = []
    if isinstance(queries, list):
        for q in queries:
            s = str(q).strip()
            if s:
                out.append(s)
    if not out:
        out = [user_query]
    return out[:5]


def _normalize_answer(raw: Dict[str, Any]) -> str:
    summary = str(raw.get("summary", "")).strip() if isinstance(raw, dict) else ""
    return summary


def _synthesize_from_evidence(evidence: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for item in evidence[:4]:
        snippet = str(item.get("snippet", "")).strip()
        title = str(item.get("title", "")).strip()
        text = snippet or title
        if not text:
            continue
        text = text.replace("\n", " ").strip().rstrip(".")
        if text:
            parts.append(text)
    if not parts:
        return ""
    return ". ".join(parts) + "."


def _build_search_queries(
    query: str,
    today_date: str,
    verification_mode: bool,
    preferred_sources: List[str],
    constraints: Dict[str, Any],
) -> List[str]:
    search_depth = str(constraints.get("search_depth", "normal") or "normal").lower()
    freshness = str(constraints.get("freshness", "any") or "any").lower()
    need_data = bool(constraints.get("need_data", False))
    try:
        min_sources = int(constraints.get("min_sources", 0) or 0)
    except Exception:
        min_sources = 0
    query_cap = 8 if search_depth == "deep" else 5

    prompt = (
        "Return JSON only: {\"queries\":[\"...\"]}.\n"
        "Generate focused search queries for up-to-date factual retrieval.\n"
        f"today_date={today_date}\n"
        f"verification_mode={str(verification_mode).lower()}\n"
        f"search_depth={search_depth}\n"
        f"freshness={freshness}\n"
        f"need_data={str(need_data).lower()}\n"
        f"min_sources={min_sources}\n"
        f"user_query={query}\n"
    )
    for _ in range(2):
        raw = _invoke_json(prompt)
        queries = _normalize_queries(raw, query)
        if preferred_sources:
            enriched = []
            for q in queries:
                enriched.append(q)
                for src in preferred_sources[:5]:
                    host = src.replace("https://", "").replace("http://", "").split("/")[0]
                    if host:
                        enriched.append(f"{q} site:{host}")
            queries = enriched
        if queries:
            uniq = []
            seen = set()
            for item in queries:
                key = item.lower().strip()
                if key and key not in seen:
                    uniq.append(item)
                    seen.add(key)
            return uniq[:query_cap]
    return [query]


def _compose_answer(query: str, evidence: List[Dict[str, str]], today_date: str) -> str:
    prompt = (
        "Return JSON only: {\"summary\":\"...\",\"sources\":[\"url1\",\"url2\"]}.\n"
        "Rules:\n"
        "1) max 4 concise sentences.\n"
        "2) no chain-of-thought, no <think>.\n"
        "3) if query has time words like today/current, anchor answer to today_date.\n"
        "4) include 1-3 source URLs in sources.\n"
        f"today_date={today_date}\n"
        f"query={query}\n"
        f"evidence={evidence[:10]}\n"
    )
    for _ in range(2):
        raw = _invoke_json(prompt)
        answer = _normalize_answer(raw)
        if answer and "<think>" not in answer.lower():
            return answer
    return _synthesize_from_evidence(evidence)


def _compact_web_context(search_queries: List[str], evidence: List[Dict[str, str]]) -> Dict[str, Any]:
    compact_evidence: List[Dict[str, str]] = []
    for item in evidence[:6]:
        compact_evidence.append(
            {
                "title": str(item.get("title", "")).strip()[:90],
                "url": str(item.get("url", "")).strip()[:180],
                "snippet": str(item.get("snippet", "")).strip().replace("\n", " ")[:140],
                "source": str(item.get("source", "")).strip()[:60],
            }
        )
    return {
        "search_queries": [str(q).strip()[:120] for q in search_queries[:6]],
        "evidence": compact_evidence,
    }


def _slugify_name(text: str, fallback: str = "retrieved_data") -> str:
    s = re.sub(r"[^\w\-]+", "_", str(text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:80] if s else fallback


def _derive_data_csv_path(target_dir: str, target_file: str, query: str) -> str:
    td = str(target_dir or "").strip().replace("\\", "/").rstrip("/")
    tf = str(target_file or "").strip().replace("\\", "/")
    if td:
        base_dir = td
    else:
        parent = Path(tf).parent.as_posix() if tf else ""
        base_dir = "" if parent == "." else parent
    if tf:
        stem = Path(tf).stem
    else:
        stem = _slugify_name(query, fallback="retrieved_data")
    name = f"{_slugify_name(stem, fallback='retrieved_data')}.csv"
    path = f"{base_dir}/{name}" if base_dir else name
    return re.sub(r"/{2,}", "/", path)


def _query_wants_chart(query: str) -> bool:
    q = str(query or "").lower()
    terms = [
        "plot",
        "chart",
        "graph",
        "line chart",
        "visualize",
        "visualization",
        "draw",
        "绘图",
        "图表",
        "可视化",
        "折线图",
        "画图",
    ]
    return any(t in q for t in terms)


def _normalize_table_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    columns: List[str] = []
    rows: List[List[str]] = []

    raw_cols = raw.get("columns", []) if isinstance(raw, dict) else []
    if isinstance(raw_cols, list):
        for c in raw_cols:
            s = str(c or "").strip()
            if s:
                columns.append(s)

    raw_rows = raw.get("rows", []) if isinstance(raw, dict) else []
    if isinstance(raw_rows, list):
        for item in raw_rows[:80]:
            if isinstance(item, dict):
                if not columns:
                    columns = [str(k).strip() for k in item.keys() if str(k).strip()][:8]
                if not columns:
                    continue
                row = [str(item.get(col, "")).strip() for col in columns]
            elif isinstance(item, list):
                row = [str(x).strip() for x in item]
                if not columns:
                    columns = [f"col{i+1}" for i in range(len(row))]
            else:
                continue
            if any(x for x in row):
                rows.append(row[: len(columns)])

    if columns and rows:
        width = len(columns)
        rows = [r + ([""] * (width - len(r))) if len(r) < width else r[:width] for r in rows]
    return {"columns": columns[:10], "rows": rows[:60]}


def _table_to_csv(columns: List[str], rows: List[List[str]]) -> str:
    lines = [",".join(_csv_escape(c) for c in columns)]
    for row in rows:
        lines.append(",".join(_csv_escape(v) for v in row))
    return "\n".join(lines) + "\n"


def _extract_year_value_rows_from_text(chunks: List[str]) -> List[List[str]]:
    year_map: Dict[str, str] = {}
    pattern = re.compile(r"(19\d{2}|20\d{2})[^\d]{0,24}(\d{7,13}(?:\.\d+)?)")
    for text in chunks:
        for year, value in pattern.findall(str(text or "")):
            if year not in year_map:
                year_map[year] = value
    rows = [[y, year_map[y]] for y in sorted(year_map.keys())]
    return rows[:30]


def _extract_structured_data_csv(
    query: str,
    evidence: List[Dict[str, str]],
    target_dir: str,
    target_file: str,
) -> Dict[str, Any]:
    if not evidence:
        return {}

    sources: List[Dict[str, str]] = []
    for item in evidence[:6]:
        sources.append(
            {
                "title": str(item.get("title", "")).strip()[:120],
                "url": str(item.get("url", "")).strip()[:220],
                "snippet": str(item.get("snippet", "")).strip()[:220],
            }
        )

    page_context: List[Dict[str, str]] = []
    seen = set()
    for item in sources:
        url = item.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        fetched = fetch_web_text(url, timeout=15, max_chars=4000)
        if fetched.get("ok"):
            page_context.append({"url": url, "text": str(fetched.get("text", ""))[:1200]})
        if len(page_context) >= 3:
            break

    prompt = (
        'Return JSON only: {"columns":["..."],"rows":[["..."]]}.\n'
        "Extract structured numeric data suitable for plotting from source snippets/page text.\n"
        "Rules:\n"
        "1) Use only values present in provided sources.\n"
        "2) Keep columns short; prefer a year/time column and one or more numeric columns.\n"
        "3) Do not invent missing rows.\n"
        "4) If reliable extraction is impossible, return empty rows.\n"
        f"query={query}\n"
        f"sources={sources}\n"
        f"page_context={page_context}\n"
    )
    raw = _invoke_data_json(prompt)
    normalized = _normalize_table_payload(raw)
    cols = normalized.get("columns", [])
    rows = normalized.get("rows", [])
    if not cols or len(rows) < 3:
        text_chunks = [x.get("snippet", "") for x in sources]
        text_chunks.extend([x.get("text", "") for x in page_context])
        regex_rows = _extract_year_value_rows_from_text(text_chunks)
        if len(regex_rows) >= 3:
            cols = ["year", "value"]
            rows = regex_rows

    if not cols or not rows:
        cols = ["title", "url", "snippet"]
        rows = [[x.get("title", ""), x.get("url", ""), x.get("snippet", "")] for x in sources if x.get("url", "")]
        if not rows:
            return {}

    csv_path = _derive_data_csv_path(target_dir=target_dir, target_file=target_file, query=query)
    csv_content = _table_to_csv(cols, rows)
    return {"path": csv_path, "content": csv_content, "columns": cols, "row_count": len(rows)}


def _is_pdf_download_task(query: str) -> bool:
    q = (query or "").lower()
    download_terms = ["download", "save", "store", "fetch", "retrieve", "下载", "保存"]
    pdf_terms = ["pdf", "paper", "papers", "论文", "arxiv"]
    return any(t in q for t in download_terms) and any(t in q for t in pdf_terms)


def _extract_download_count(query: str, default: int = 5) -> int:
    q = query or ""
    m = re.search(r"\b(\d{1,2})\b", q)
    if m:
        try:
            return max(1, min(20, int(m.group(1))))
        except Exception:
            pass
    return default


def _resolve_download_dir(query: str, target_dir: str) -> str:
    td = str(target_dir or "").strip().replace("\\", "/")
    if td:
        return td

    q = query or ""
    m = re.search(r'folder called ["“]?([A-Za-z0-9_\-./ ]+)["”]?', q, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().replace(" ", "_")
        if candidate:
            return candidate
    return "downloads"


def _collect_pdf_candidates(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for r in rows:
        url = str(r.get("url", "")).strip()
        title = str(r.get("title", "")).strip()
        snippet = str(r.get("snippet", "")).strip()
        pdf = to_pdf_url(url)
        if not pdf or pdf in seen:
            continue
        seen.add(pdf)
        out.append({"title": title or "paper", "url": url, "pdf_url": pdf, "snippet": snippet})
    return out


def _to_rel(path_str: str) -> str:
    try:
        p = Path(path_str).resolve()
        if p == ROOT or ROOT in p.parents:
            return p.relative_to(ROOT).as_posix()
    except Exception:
        pass
    return str(path_str).replace("\\", "/")


def _csv_escape(s: str) -> str:
    x = str(s or "")
    return '"' + x.replace('"', '""') + '"'


def _write_csv_rows(path: str, headers: List[str], rows: List[Dict[str, Any]]) -> str:
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(_csv_escape(str(r.get(h, ""))) for h in headers))
    content = "\n".join(lines) + "\n"
    return write_file(path, content)


def _download_pdfs_from_web(
    query: str,
    target_dir: str,
    count: int,
    search_queries: List[str],
    initial_evidence: List[Dict[str, str]],
) -> Dict[str, Any]:
    make_dir(target_dir)
    candidates = _collect_pdf_candidates(initial_evidence)

    if len(candidates) < count:
        extra_queries = []
        for q in search_queries[:6]:
            extra_queries.append(f"{q} filetype:pdf")
            extra_queries.append(f"{q} arxiv pdf")
        seen_q = set()
        for q in extra_queries:
            key = q.lower().strip()
            if not key or key in seen_q:
                continue
            seen_q.add(key)
            try:
                rows = web_search(q, k=8)
            except Exception:
                rows = []
            candidates.extend(_collect_pdf_candidates(rows))
            if len(candidates) >= count * 2:
                break

    # Dedup by pdf_url while keeping order.
    uniq: List[Dict[str, str]] = []
    seen_pdf = set()
    for c in candidates:
        u = c.get("pdf_url", "")
        if not u or u in seen_pdf:
            continue
        seen_pdf.add(u)
        uniq.append(c)
    selected = uniq[:count]

    results: List[Dict[str, Any]] = []
    for i, c in enumerate(selected, start=1):
        name = infer_pdf_filename(c.get("title", "paper"), i)
        rel_path = f"{target_dir}/{name}".replace("\\", "/")
        d = download_pdf_to_workspace(c.get("pdf_url", ""), rel_path)
        results.append(
            {
                "rank": i,
                "title": c.get("title", ""),
                "source_url": c.get("url", ""),
                "pdf_url": c.get("pdf_url", ""),
                "status": "downloaded" if d.get("ok") else "failed",
                "saved_path": _to_rel(d.get("saved_path", "")) if d.get("ok") else "",
                "error": "" if d.get("ok") else str(d.get("error", "")),
            }
        )

    papers_csv = f"{target_dir}/papers_list.csv".replace("\\", "/")
    report_csv = f"{target_dir}/download_report.csv".replace("\\", "/")
    _write_csv_rows(
        papers_csv,
        ["rank", "title", "source_url", "pdf_url"],
        [
            {
                "rank": r["rank"],
                "title": r["title"],
                "source_url": r["source_url"],
                "pdf_url": r["pdf_url"],
            }
            for r in results
        ],
    )
    _write_csv_rows(
        report_csv,
        ["rank", "title", "status", "saved_path", "pdf_url", "error"],
        results,
    )

    ok_count = sum(1 for r in results if r["status"] == "downloaded")
    return {
        "target_dir": target_dir,
        "results": results,
        "ok_count": ok_count,
        "papers_csv": papers_csv,
        "report_csv": report_csv,
        "selected": selected,
    }


def web_agent_node(state):
    query = state["query"]
    memory_context = state.get("memory_context", {}) or {}
    memory_summary = str(memory_context.get("summary_text", ""))
    verification_mode = bool(state.get("needs_web_verify", False) and not state.get("did_web_verify", False))
    today_date = state.get("local_time", {}).get("date", "")
    constraints = state.get("constraints", {}) or {}

    route = state.get("route", "general")
    source_registry = state.get("source_registry", {}) or {}
    preferred_sources = source_registry.get(route, []) if isinstance(source_registry, dict) else []
    search_queries = _build_search_queries(query, today_date, verification_mode, preferred_sources, constraints)
    evidence: List[Dict[str, str]] = []
    deep_mode = str(constraints.get("search_depth", "normal") or "normal").lower() == "deep"
    top_k = 10 if (verification_mode or deep_mode) else 5
    for q in search_queries:
        try:
            evidence.extend(web_search(q, k=top_k))
        except Exception:
            continue

    if not evidence:
        state["draft_answer"] = "I could not retrieve web evidence for this query."
        state["evidence"] = []
        state["search_queries"] = search_queries
        state["logs"].append("web_agent.search=empty")
        return state

    if _is_pdf_download_task(query) and not verification_mode:
        count = _extract_download_count(query, default=5)
        target_dir = _resolve_download_dir(query, state.get("target_dir", ""))
        result = _download_pdfs_from_web(
            query=query,
            target_dir=target_dir,
            count=count,
            search_queries=search_queries,
            initial_evidence=evidence,
        )

        results = result["results"]
        ok_count = int(result["ok_count"])
        selected = result["selected"]
        report_rel = result["report_csv"]
        papers_rel = result["papers_csv"]
        saved_preview = [r["saved_path"] for r in results if r.get("saved_path")]
        preview_text = ", ".join(saved_preview[:5]) if saved_preview else "(none)"

        final = (
            f"PDF download completed: {ok_count}/{len(results)} files downloaded into `{target_dir}`.\n"
            f"Index CSV: `{papers_rel}`\n"
            f"Report CSV: `{report_rel}`\n"
            f"Saved files: {preview_text}"
        )

        state["search_queries"] = search_queries
        state["evidence"] = [
            {
                "title": c.get("title", ""),
                "url": c.get("pdf_url", ""),
                "snippet": c.get("snippet", ""),
                "source": "pdf_download",
            }
            for c in selected
        ]
        state["draft_answer"] = final
        state["final_answer"] = final
        state["requires_code"] = False
        state["run_after_write"] = False
        state["direct_action_done"] = True
        context_payload = state.get("context_payload", {}) or {}
        context_payload["web"] = _compact_web_context(search_queries, state["evidence"])
        state["context_payload"] = context_payload
        state["logs"].append(f"web_agent.pdf_download=ok:{ok_count}/{len(results)}")
        append_agent_event(
            state,
            agent_name="web_agent",
            input_payload={
                "query": query,
                "mode": "pdf_download",
                "count": count,
                "target_dir": target_dir,
                "memory_summary": memory_summary[:300],
            },
            output_payload={
                "ok_count": ok_count,
                "requested_count": len(results),
                "papers_csv": papers_rel,
                "report_csv": report_rel,
            },
        )
        return state

    draft = _compose_answer(query, evidence, today_date)
    state["search_queries"] = search_queries
    state["evidence"] = evidence
    state["draft_answer"] = _strip_think(draft)
    web_context = _compact_web_context(search_queries, evidence)

    should_stage_data_csv = (
        str(state.get("route", "")).lower() == "file"
        and bool(state.get("requires_code", False))
        and not verification_mode
        and (bool(constraints.get("need_data", False)) or _query_wants_chart(query))
    )
    staged_data_info: Dict[str, Any] = {}
    if should_stage_data_csv:
        staged_data_info = _extract_structured_data_csv(
            query=query,
            evidence=evidence,
            target_dir=str(state.get("target_dir", "")).strip(),
            target_file=str(state.get("target_file", "")).strip(),
        )
        if staged_data_info:
            path = str(staged_data_info.get("path", "")).strip()
            content = str(staged_data_info.get("content", ""))
            if path and content:
                staged_files = state.get("staged_data_files", {}) or {}
                staged_files[path] = content
                state["staged_data_files"] = staged_files
                state["file_stage"] = "data"
                web_context["data_file"] = path
                web_context["data_columns"] = staged_data_info.get("columns", [])
                web_context["data_rows"] = int(staged_data_info.get("row_count", 0))
                state["logs"].append(f"web_agent.data_file={path}")
                state["logs"].append(f"web_agent.data_rows={int(staged_data_info.get('row_count', 0))}")

    context_payload = state.get("context_payload", {}) or {}
    context_payload["web"] = web_context
    state["context_payload"] = context_payload
    if verification_mode:
        state["did_web_verify"] = True
        state["verify_round"] = int(state.get("verify_round", 0)) + 1
        state["logs"].append("web_agent.search=verify")
    else:
        state["logs"].append("web_agent.search=ok")
    append_agent_event(
        state,
        agent_name="web_agent",
        input_payload={
            "query": query,
            "verification_mode": verification_mode,
            "today_date": today_date,
            "memory_summary": memory_summary[:300],
        },
        output_payload={
            "search_queries": search_queries,
            "evidence_count": len(evidence),
            "draft_preview": state["draft_answer"][:300],
            "data_file": str(staged_data_info.get("path", "")),
            "data_rows": int(staged_data_info.get("row_count", 0) or 0),
        },
    )
    return state
