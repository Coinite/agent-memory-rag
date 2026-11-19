import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from src.agents.utils import append_agent_event
from src.config import settings
from src.models.lmstudio_client import build_llm
from src.tools.context_tools import get_location, get_time

planner_llm_json = build_llm(
    temp=0.0,
    model_override=settings.supervisor_model_name,
    max_tokens=96,
    response_format={"type": "json_object"},
)
planner_llm_plain = build_llm(
    temp=0.0,
    model_override=settings.supervisor_model_name,
    max_tokens=120,
)

ALLOWED_ROUTES = {"file", "research", "general", "weather", "news", "stock"}
SOURCE_REGISTRY: Dict[str, List[str]] = {
    "weather": [
        "https://api.open-meteo.com/v1/forecast",
        "https://geocoding-api.open-meteo.com/v1/search",
        "https://www.weather.gov/documentation/services-web-api",
    ],
    "news": [
        "https://www.reuters.com/",
        "https://apnews.com/",
        "https://www.npr.org/",
        "https://www.bbc.com/news",
        "https://www.nytimes.com/",
        "https://www.washingtonpost.com/",
        "https://www.wsj.com/",
        "https://www.bloomberg.com/",
        "https://www.ft.com/",
        "https://www.cnbc.com/",
        "https://www.marketwatch.com/",
        "https://www.eia.gov/",
        "https://www.iea.org/",
        "https://www.imf.org/",
        "https://www.worldbank.org/",
        "https://fred.stlouisfed.org/",
        "https://www.pewresearch.org/",
        "https://news.gallup.com/",
    ],
    "stock": [
        "https://stooq.com/q/l/",
        "https://www.alphavantage.co/documentation/",
        "https://polygon.io/docs",
    ],
}

SCHEMA_HINT = {
    "route": "file|research|general|weather|news|stock",
    "target_file": "string, empty if N/A",
    "target_dir": "string, empty if N/A",
    "requires_code": "boolean",
    "wants_requirements": "boolean",
    "code_spec": "one concise coding requirement sentence, empty if N/A",
    "run_after_write": "boolean",
    "run_entrypoint": "workspace-relative path, empty if N/A",
    "expected_artifacts": "array of workspace-relative paths, empty list if N/A",
}

DEFAULT_CONSTRAINTS: Dict[str, Any] = {
    "answer_words": 0,
    "answer_chars": 0,
    "min_sources": 0,
    "need_data": False,
    "search_depth": "normal",  # normal|deep
    "freshness": "any",  # any|recent
    "must_cite": False,
    "must_cover": [],
    "breakdown_by": "",
    "answer_style": "auto",  # auto|paragraph|bullets|numbered
    "strict_coverage": False,
}


@lru_cache(maxsize=1)
def _load_agent_catalog() -> Dict[str, Any]:
    catalog_path = Path(__file__).resolve().parents[2] / "agents_catalog.json"
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and isinstance(data.get("agents"), list):
            return data
    except Exception:
        pass

    return {
        "version": "fallback",
        "agents": [
            {"name": "supervisor", "purpose": "Route and split tasks."},
            {"name": "code_agent", "purpose": "Generate code content."},
            {"name": "file_agent", "purpose": "Read/write/list local files."},
            {"name": "web_agent", "purpose": "Search web and draft answer."},
            {"name": "judge_agent", "purpose": "Score and finalize answers."},
        ],
    }


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _bool_like(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def _extract_constraints_with_rules(query: str) -> Dict[str, Any]:
    q = query or ""
    c = dict(DEFAULT_CONSTRAINTS)

    m_words = re.search(r"(\d{1,4})\s*(word|words)\b", q, flags=re.IGNORECASE)
    if m_words:
        c["answer_words"] = max(20, min(1200, int(m_words.group(1))))

    m_chars_en = re.search(r"(\d{1,4})\s*(char|chars|character|characters)\b", q, flags=re.IGNORECASE)
    m_chars_zh = re.search(r"(\d{1,4})\s*(字|字符)\b", q, flags=re.IGNORECASE)
    if m_chars_en:
        c["answer_chars"] = max(40, min(1200, int(m_chars_en.group(1))))
    elif m_chars_zh:
        c["answer_chars"] = max(40, min(1200, int(m_chars_zh.group(1))))

    lowered = q.lower()
    need_data_terms = [
        "show me the data",
        "with data",
        "data points",
        "quantitative",
        "numbers",
        "numeric",
        "metrics",
        "statistics",
        "数据",
        "给我数据",
    ]
    if any(t in lowered for t in need_data_terms) or any(t in q for t in ["数据", "给我数据"]):
        c["need_data"] = True

    deep_terms = [
        "search more sources",
        "more sources",
        "search more webpages",
        "more accurate",
        "deep search",
        "deeper search",
        "更多来源",
        "更准确",
        "深度搜索",
    ]
    if any(t in lowered for t in deep_terms) or any(t in q for t in ["更多来源", "更准确", "深度搜索"]):
        c["search_depth"] = "deep"
        c["min_sources"] = max(c["min_sources"], 18)

    recent_terms = ["latest", "recent", "today", "current", "最新", "近期", "今天"]
    if any(t in lowered for t in recent_terms) or any(t in q for t in ["最新", "近期", "今天"]):
        c["freshness"] = "recent"

    if c["need_data"] or c["search_depth"] == "deep" or c["min_sources"] > 0:
        c["must_cite"] = True

    return c


def _extract_constraints_with_llm(query: str) -> Dict[str, Any]:
    prompt = (
        "Return ONE JSON object only with schema:\n"
        '{"answer_words":0,"answer_chars":0,"min_sources":0,"need_data":false,'
        '"search_depth":"normal|deep","freshness":"any|recent","must_cite":false,'
        '"must_cover":[],"breakdown_by":"","answer_style":"auto|paragraph|bullets|numbered","strict_coverage":false}\n'
        "Extract response constraints from user query. Use 0 when unspecified.\n"
        f"query={query}\n"
    )
    raw = _extract_json_object(_invoke_text(prompt))
    if not raw:
        return {}

    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    must_cover_raw = raw.get("must_cover", [])
    must_cover: List[str] = []
    if isinstance(must_cover_raw, list):
        for item in must_cover_raw:
            s = str(item or "").strip()
            if s and s not in must_cover:
                must_cover.append(s)
    elif isinstance(must_cover_raw, str):
        s = must_cover_raw.strip()
        if s:
            must_cover.append(s)

    return {
        "answer_words": _safe_int(raw.get("answer_words", 0), 0),
        "answer_chars": _safe_int(raw.get("answer_chars", 0), 0),
        "min_sources": _safe_int(raw.get("min_sources", 0), 0),
        "need_data": _bool_like(raw.get("need_data", False), False),
        "search_depth": str(raw.get("search_depth", "normal") or "normal").lower(),
        "freshness": str(raw.get("freshness", "any") or "any").lower(),
        "must_cite": _bool_like(raw.get("must_cite", False), False),
        "must_cover": must_cover,
        "breakdown_by": str(raw.get("breakdown_by", "") or "").strip(),
        "answer_style": str(raw.get("answer_style", "auto") or "auto").lower().strip(),
        "strict_coverage": _bool_like(raw.get("strict_coverage", False), False),
    }


def _normalize_constraints(c: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULT_CONSTRAINTS)
    out.update(c or {})
    out["answer_words"] = max(0, min(1200, int(out.get("answer_words", 0) or 0)))
    out["answer_chars"] = max(0, min(1200, int(out.get("answer_chars", 0) or 0)))
    out["min_sources"] = max(0, min(50, int(out.get("min_sources", 0) or 0)))
    out["need_data"] = _bool_like(out.get("need_data", False), False)
    out["must_cite"] = _bool_like(out.get("must_cite", False), False)
    out["strict_coverage"] = _bool_like(out.get("strict_coverage", False), False)

    sd = str(out.get("search_depth", "normal") or "normal").lower()
    out["search_depth"] = "deep" if sd == "deep" else "normal"

    fr = str(out.get("freshness", "any") or "any").lower()
    out["freshness"] = "recent" if fr == "recent" else "any"

    cover_items: List[str] = []
    raw_cover = out.get("must_cover", [])
    if isinstance(raw_cover, list):
        for item in raw_cover:
            s = str(item or "").strip()
            if s and s not in cover_items:
                cover_items.append(s)
    elif isinstance(raw_cover, str):
        for item in re.split(r"[,\n;|]+", raw_cover):
            s = str(item or "").strip()
            if s and s not in cover_items:
                cover_items.append(s)
    out["must_cover"] = cover_items[:8]

    out["breakdown_by"] = str(out.get("breakdown_by", "") or "").strip()[:120]

    style = str(out.get("answer_style", "auto") or "auto").lower().strip()
    out["answer_style"] = style if style in {"auto", "paragraph", "bullets", "numbered"} else "auto"

    # Keep one primary length constraint if both present.
    if out["answer_words"] > 0 and out["answer_chars"] > 0:
        # Prefer explicit words for English queries.
        out["answer_chars"] = 0

    if out["breakdown_by"] and out["answer_style"] == "auto":
        out["answer_style"] = "bullets"

    if out["strict_coverage"] and out["min_sources"] == 0:
        out["min_sources"] = 8

    if out["strict_coverage"] and not out["must_cite"]:
        out["must_cite"] = True

    return out


def _extract_constraints(query: str) -> Dict[str, Any]:
    rule_based = _extract_constraints_with_rules(query)

    llm_based = _extract_constraints_with_llm(query)
    merged = dict(rule_based)
    # LLM can fill what rules miss; rules stay dominant for explicit numeric constraints.
    for k, v in llm_based.items():
        if k in {"answer_words", "answer_chars", "min_sources"}:
            if int(merged.get(k, 0) or 0) == 0 and int(v or 0) > 0:
                merged[k] = int(v)
        elif k in {"need_data", "must_cite", "strict_coverage"}:
            if not _bool_like(merged.get(k, False), False) and _bool_like(v, False):
                merged[k] = True
        elif k == "search_depth":
            if str(merged.get(k, "normal")) == "normal" and str(v or "normal") == "deep":
                merged[k] = "deep"
        elif k == "freshness":
            if str(merged.get(k, "any")) == "any" and str(v or "any") == "recent":
                merged[k] = "recent"
        elif k == "must_cover":
            if not merged.get("must_cover") and isinstance(v, list) and v:
                merged["must_cover"] = v
        elif k == "breakdown_by":
            if not str(merged.get("breakdown_by", "")).strip() and str(v or "").strip():
                merged["breakdown_by"] = str(v).strip()
        elif k == "answer_style":
            if str(merged.get("answer_style", "auto")) == "auto" and str(v or "auto") != "auto":
                merged["answer_style"] = str(v).strip()
        else:
            if merged.get(k) in {False, "normal", "any"} and v not in {False, "normal", "any"}:
                merged[k] = v
    return _normalize_constraints(merged)


def _active_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (constraints or {}).items():
        if DEFAULT_CONSTRAINTS.get(k) != v:
            out[k] = v
    return out


def _query_has_local_output_intent(query: str) -> bool:
    q = str(query or "")
    q_lower = q.lower()

    action_terms = [
        "create",
        "write",
        "save",
        "store",
        "generate",
        "build",
        "make",
        "plot",
        "download",
        "edit",
        "modify",
        "change",
        "update",
        "run",
        "创建",
        "新建",
        "写入",
        "保存",
        "生成",
        "绘图",
        "下载",
        "修改",
        "更新",
        "运行",
    ]
    container_terms = [
        "file",
        "files",
        "folder",
        "directory",
        "path",
        "root directory",
        "project root",
        "current project",
        "workspace",
        "文件",
        "文件夹",
        "目录",
        "路径",
        "根目录",
        "当前项目",
        "工程",
    ]
    code_terms = [
        "python",
        "script",
        "program",
        "code",
        "notebook",
        "jupyter",
        "脚本",
        "程序",
        "代码",
    ]
    output_terms = [
        "csv",
        "json",
        "xlsx",
        "excel",
        "plot",
        "chart",
        "figure",
        "graph",
        "image",
        "png",
        "jpg",
        "jpeg",
        "pdf",
        "report",
        "导出",
        "图表",
        "可视化",
        "结果文件",
    ]

    has_action = any(t in q_lower for t in action_terms) or any(t in q for t in action_terms if not t.isascii())
    has_container = any(t in q_lower for t in container_terms) or any(
        t in q for t in container_terms if not t.isascii()
    )
    has_code_or_output = any(t in q_lower for t in code_terms + output_terms) or any(
        t in q for t in (code_terms + output_terms) if not t.isascii()
    )
    has_file_ext = bool(
        re.search(r"\.(py|csv|json|txt|md|html|js|ts|tsx|png|jpg|jpeg|svg|pdf|ipynb)\b", q, flags=re.IGNORECASE)
    )
    return bool(has_action and (has_container or has_file_ext or has_code_or_output))


def _query_has_retrieval_intent(query: str) -> bool:
    q = str(query or "")
    q_lower = q.lower()
    retrieval_terms = [
        "research",
        "search",
        "find",
        "look up",
        "collect",
        "gather",
        "web",
        "online",
        "latest",
        "recent",
        "查找",
        "搜索",
        "检索",
        "调研",
        "收集",
        "资料",
        "信息",
        "最新",
    ]
    return any(t in q_lower for t in retrieval_terms) or any(
        t in q for t in retrieval_terms if not t.isascii()
    )


def _clear_codegen_fields(plan: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(plan)
    out["requires_code"] = False
    out["wants_requirements"] = False
    out["target_file"] = ""
    out["target_dir"] = ""
    out["code_spec"] = ""
    out["run_after_write"] = False
    out["run_entrypoint"] = ""
    out["expected_artifacts"] = []
    return out


def _enforce_codegen_intent_guard(plan: Dict[str, Any], query: str) -> tuple[Dict[str, Any], str]:
    out = dict(plan)
    route = str(out.get("route", "general") or "general").strip().lower()
    requires_code = bool(out.get("requires_code", False))
    local_output_intent = _query_has_local_output_intent(query)
    has_local_target = bool(str(out.get("target_file", "")).strip() or str(out.get("target_dir", "")).strip())

    if not requires_code:
        if route == "file" and not local_output_intent and not has_local_target:
            out["route"] = "research" if _query_has_retrieval_intent(query) else "general"
            return _normalize_plan(out), "normalized_file_route_without_output_intent"
        if bool(out.get("wants_requirements", False)):
            out["wants_requirements"] = False
            return _normalize_plan(out), "cleared_orphan_requirements"
        return _normalize_plan(out), ""

    if local_output_intent:
        if route != "file":
            out["route"] = "file"
            return _normalize_plan(out), "force_file_route_for_codegen"
        return _normalize_plan(out), ""

    out = _clear_codegen_fields(out)
    if route == "file":
        out["route"] = "research" if _query_has_retrieval_intent(query) else "general"
    return _normalize_plan(out), "disabled_codegen_without_local_output"


def _invoke_text(prompt: str) -> str:
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

    # Try strict JSON mode first; fallback to plain completion for LM Studio compatibility.
    try:
        msg = planner_llm_json.invoke(prompt)
        text = _content_to_text(msg)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        msg = planner_llm_plain.invoke(prompt)
        return _content_to_text(msg)
    except Exception:
        return ""


def _extract_json_object(text: str) -> Dict[str, Any]:
    clean = _strip_think(text)
    if not clean:
        return {}
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


def _repair_plan_json(raw_text: str, user_query: str) -> Dict[str, Any]:
    repair_prompt = (
        "Fix and return one valid JSON object only.\n"
        f"Required schema: {json.dumps(SCHEMA_HINT, ensure_ascii=False)}\n"
        f"user_query={user_query}\n"
        f"broken_output={raw_text[:1200]}\n"
    )
    return _extract_json_object(_invoke_text(repair_prompt))


def _normalize_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes"}:
                return True
            if v in {"false", "0", "no"}:
                return False
        return default

    def _normalize_rel_path(path: str) -> str:
        p = str(path or "").strip().replace("\\", "/")
        p = p.strip("\"'")
        # Normalize common placeholder roots produced by local models.
        for prefix in [
            "/current/project/root/",
            "current/project/root/",
            "/workspace/",
            "workspace/",
        ]:
            if p.lower().startswith(prefix):
                p = p[len(prefix) :]
                break
        if p == "/current/project/root":
            p = ""
        # On Windows, local models often emit pseudo-root paths like /stock/file.py.
        # Treat them as workspace-relative, not filesystem root.
        workspace = Path(settings.workspace_root).resolve()
        if workspace.drive and p.startswith("/") and not re.match(r"^[A-Za-z]:/", p):
            p = p.lstrip("/")
        # Normalize accidental self-prefixed relative paths like "project/foo" -> "foo".
        ws_name = workspace.name.lower()
        if p.lower().startswith(ws_name + "/"):
            p = p[len(workspace.name) + 1 :]

        # Normalize absolute Windows/Unix paths into workspace-relative paths.
        if re.match(r"^[A-Za-z]:/", p) or p.startswith("/"):
            try:
                abs_path = Path(p).resolve()
                try:
                    p = abs_path.relative_to(workspace).as_posix()
                except Exception:
                    # Common one-level-up mistake: map from workspace parent into workspace.
                    rel_parent = abs_path.relative_to(workspace.parent)
                    parts = list(rel_parent.parts)
                    if parts and parts[0].lower() == workspace.name.lower():
                        parts = parts[1:]
                    p = Path(*parts).as_posix() if parts else ""
            except Exception:
                pass

        # Prefer workspace-relative paths in this project.
        if p.startswith("./"):
            p = p[2:]
        p = re.sub(r"/{2,}", "/", p)
        if len(p) > 1 and p.endswith("/"):
            p = p.rstrip("/")
        return p

    route_raw = str(raw.get("route", "") or "").strip().lower()
    if route_raw in ALLOWED_ROUTES:
        route = route_raw
    else:
        route = "general"
        for token in ["file", "research", "weather", "news", "stock", "general"]:
            if token in route_raw:
                route = token
                break

    target_file = _normalize_rel_path(raw.get("target_file", ""))
    target_dir = _normalize_rel_path(raw.get("target_dir", ""))
    run_entrypoint = _normalize_rel_path(raw.get("run_entrypoint", ""))

    raw_artifacts = raw.get("expected_artifacts", [])
    expected_artifacts: List[str] = []
    if isinstance(raw_artifacts, list):
        for item in raw_artifacts:
            p = _normalize_rel_path(item)
            if target_dir and p and "/" not in p:
                p = f"{target_dir}/{p}"
            if p and p not in expected_artifacts:
                expected_artifacts.append(p)
    if target_dir and target_file and "/" not in target_file:
        target_file = f"{target_dir}/{target_file}"
    if target_dir and run_entrypoint and "/" not in run_entrypoint:
        run_entrypoint = f"{target_dir}/{run_entrypoint}"

    requires_code = _to_bool(raw.get("requires_code", False), default=False)
    wants_requirements = _to_bool(raw.get("wants_requirements", False), default=False)
    run_after_write = _to_bool(raw.get("run_after_write", False), default=False)
    code_spec = str(raw.get("code_spec", "") or "").strip()
    if code_spec and not requires_code:
        requires_code = True
    if not requires_code:
        code_spec = ""
        run_after_write = False
        run_entrypoint = ""
        expected_artifacts = []
    elif run_after_write and not run_entrypoint and target_file:
        run_entrypoint = target_file

    return {
        "route": route,
        "target_file": target_file,
        "target_dir": target_dir,
        "requires_code": requires_code,
        "wants_requirements": wants_requirements,
        "code_spec": code_spec,
        "run_after_write": run_after_write,
        "run_entrypoint": run_entrypoint,
        "expected_artifacts": expected_artifacts,
    }


def _extract_plan_from_partial_text(text: str) -> Dict[str, Any]:
    clean = _strip_think(text)
    if not clean:
        return {}

    out: Dict[str, Any] = {}
    route_m = re.search(r'"route"\s*:\s*"([^"]+)', clean, flags=re.IGNORECASE)
    if route_m:
        out["route"] = route_m.group(1).strip()

    tf_m = re.search(r'"target_file"\s*:\s*"([^"]*)', clean, flags=re.IGNORECASE)
    if tf_m:
        out["target_file"] = tf_m.group(1).strip()

    td_m = re.search(r'"target_dir"\s*:\s*"([^"]*)', clean, flags=re.IGNORECASE)
    if td_m:
        out["target_dir"] = td_m.group(1).strip()

    rc_m = re.search(r'"requires_code"\s*:\s*(true|false)', clean, flags=re.IGNORECASE)
    if rc_m:
        out["requires_code"] = rc_m.group(1).lower() == "true"
    elif "requires_cod" in clean.lower():
        out["requires_code"] = True

    wr_m = re.search(r'"wants_requirements"\s*:\s*(true|false)', clean, flags=re.IGNORECASE)
    if wr_m:
        out["wants_requirements"] = wr_m.group(1).lower() == "true"

    cs_m = re.search(r'"code_spec"\s*:\s*"([^"]*)', clean, flags=re.IGNORECASE)
    if cs_m:
        out["code_spec"] = cs_m.group(1).strip()

    rw_m = re.search(r'"run_after_write"\s*:\s*(true|false)', clean, flags=re.IGNORECASE)
    if rw_m:
        out["run_after_write"] = rw_m.group(1).lower() == "true"

    re_m = re.search(r'"run_entrypoint"\s*:\s*"([^"]*)', clean, flags=re.IGNORECASE)
    if re_m:
        out["run_entrypoint"] = re_m.group(1).strip()

    return out


def _has_schema_signals(raw: Dict[str, Any]) -> bool:
    if not isinstance(raw, dict) or not raw:
        return False
    keys = set(raw.keys())
    required_markers = {
        "route",
        "requires_code",
        "wants_requirements",
        "target_file",
        "target_dir",
        "run_after_write",
        "run_entrypoint",
        "expected_artifacts",
    }
    return len(keys.intersection(required_markers)) >= 2


def _plan_needs_refine(plan: Dict[str, Any]) -> bool:
    tf = str(plan.get("target_file", "")).lower().strip()
    td = str(plan.get("target_dir", "")).strip()
    re_path = str(plan.get("run_entrypoint", "")).strip()
    artifacts = plan.get("expected_artifacts", []) if isinstance(plan.get("expected_artifacts", []), list) else []
    route = str(plan.get("route", ""))
    requires_code = bool(plan.get("requires_code", False))
    wants_requirements = bool(plan.get("wants_requirements", False))
    run_after_write = bool(plan.get("run_after_write", False))

    if route == "file" and requires_code and not tf:
        return True
    if tf.startswith("/current/project/root") or td.startswith("/current/project/root"):
        return True
    if re_path.startswith("/current/project/root"):
        return True
    if route == "file" and requires_code and not wants_requirements and tf.endswith("requirements.txt"):
        return True
    if run_after_write and not (re_path or tf):
        return True
    if run_after_write:
        for a in artifacts:
            s = str(a).strip().lower()
            if s.startswith("/current/project/root"):
                return True
    return False


def _refine_plan_with_llm(query: str, plan: Dict[str, Any], memory_summary: str) -> Dict[str, Any]:
    prompt = (
        "You are PlanRefiner. Return ONE strict JSON object only with this exact schema:\n"
        f"{json.dumps(SCHEMA_HINT, ensure_ascii=False)}\n"
        "Refine the plan to align with user intent.\n"
        "Hard constraints:\n"
        "1) Use workspace-relative paths only; do NOT use placeholders like /current/project/root.\n"
        "2) For file+code tasks, target_file must be the primary source file requested by user.\n"
        "3) requirements.txt is auxiliary and should not be target_file unless explicitly requested as the main task.\n"
        "4) Keep booleans as true/false JSON booleans.\n"
        "5) If run_after_write=true, provide run_entrypoint and expected_artifacts.\n"
        "6) If user did NOT request creating/updating local files, set requires_code=false and keep file fields empty.\n"
        f"user_query={query}\n"
        f"memory_summary={memory_summary[:300]}\n"
        f"current_plan={json.dumps(plan, ensure_ascii=False)}\n"
    )
    obj = _extract_json_object(_invoke_text(prompt))
    return _normalize_plan(obj)


def _fallback_plan_with_llm(query: str, memory_summary: str, recent_brief: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = (
        "Return ONE JSON object only with exact schema:\n"
        f"{json.dumps(SCHEMA_HINT, ensure_ascii=False)}\n"
        "Task: plan routing for a local multi-agent system.\n"
        "Hard constraints:\n"
        "1) JSON only, no markdown.\n"
        "2) Use workspace-relative paths only.\n"
        "3) If route=file and requires_code=true, target_file must be non-empty.\n"
        "4) wants_requirements=true only if user explicitly asks dependency file.\n"
        "5) If user asks to plot/store/output generated results, set run_after_write=true with expected_artifacts.\n"
        "6) If query is pure QA/retrieval without local file operation request, set requires_code=false and empty file fields.\n"
        f"query={query}\n"
        f"memory_summary={memory_summary[:300]}\n"
        f"recent={json.dumps(recent_brief, ensure_ascii=False)}\n"
    )
    return _normalize_plan(_extract_json_object(_invoke_text(prompt)))


def _infer_target_file_with_llm(query: str, target_dir: str, code_spec: str) -> str:
    prompt = (
        "Return JSON only: {\"target_file\":\"...\"}.\n"
        "Infer a concise workspace-relative file path for code generation.\n"
        "Rules:\n"
        "1) Return a single file path, not a directory.\n"
        "2) If target_dir is provided, place file under it.\n"
        "3) Prefer common entry file names for the language.\n"
        f"query={query}\n"
        f"target_dir={target_dir}\n"
        f"code_spec={code_spec}\n"
    )
    obj = _extract_json_object(_invoke_text(prompt))
    path = str(obj.get("target_file", "")).strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def _align_layout_with_llm(query: str, target_dir: str, target_file: str, code_spec: str) -> Dict[str, str]:
    prompt = (
        "Return JSON only: {\"target_dir\":\"...\",\"target_file\":\"...\"}.\n"
        "Align local file layout to user intent.\n"
        "Rules:\n"
        "1) Paths must be workspace-relative.\n"
        "2) If user asks to create a folder and a program inside it, target_file must be inside target_dir.\n"
        "3) Do not return placeholders like /current/project/root.\n"
        f"query={query}\n"
        f"target_dir={target_dir}\n"
        f"target_file={target_file}\n"
        f"code_spec={code_spec}\n"
    )
    obj = _extract_json_object(_invoke_text(prompt))
    td = str(obj.get("target_dir", "")).strip().replace("\\", "/")
    tf = str(obj.get("target_file", "")).strip().replace("\\", "/")
    if td.startswith("./"):
        td = td[2:]
    if tf.startswith("./"):
        tf = tf[2:]
    return {"target_dir": td, "target_file": tf}


def _infer_execution_with_llm(
    query: str, route: str, target_dir: str, target_file: str, code_spec: str
) -> Dict[str, Any]:
    prompt = (
        "Return JSON only with schema:\n"
        '{"run_after_write":false,"run_entrypoint":"","expected_artifacts":[]}\n'
        "Infer whether code should be executed immediately and what artifacts should exist.\n"
        "Rules:\n"
        "1) Paths must be workspace-relative.\n"
        "2) If run_after_write=true and run_entrypoint empty, use target_file.\n"
        "3) expected_artifacts should include files user explicitly requests to store/output.\n"
        f"route={route}\n"
        f"query={query}\n"
        f"target_dir={target_dir}\n"
        f"target_file={target_file}\n"
        f"code_spec={code_spec}\n"
    )
    obj = _extract_json_object(_invoke_text(prompt))
    if not isinstance(obj, dict):
        return {"run_after_write": False, "run_entrypoint": "", "expected_artifacts": []}

    run_after = _bool_like(obj.get("run_after_write", False), False)
    run_entry = str(obj.get("run_entrypoint", "") or "").strip().replace("\\", "/")
    if run_entry.startswith("./"):
        run_entry = run_entry[2:]

    artifacts: List[str] = []
    arr = obj.get("expected_artifacts", [])
    if isinstance(arr, list):
        for item in arr:
            p = str(item or "").strip().replace("\\", "/")
            if p.startswith("./"):
                p = p[2:]
            if p and p not in artifacts:
                artifacts.append(p)

    return {"run_after_write": run_after, "run_entrypoint": run_entry, "expected_artifacts": artifacts[:10]}


def _stabilize_plan_with_consistency(plan: Dict[str, Any], query: str) -> Dict[str, Any]:
    out = dict(plan)
    route = out.get("route", "general")
    target_file = str(out.get("target_file", "")).strip()
    target_dir = str(out.get("target_dir", "")).strip()
    requires_code = bool(out.get("requires_code", False))
    code_spec = str(out.get("code_spec", "")).strip()
    run_after_write = bool(out.get("run_after_write", False))
    run_entrypoint = str(out.get("run_entrypoint", "")).strip()
    expected_artifacts = out.get("expected_artifacts", []) if isinstance(out.get("expected_artifacts", []), list) else []
    had_explicit_target = bool(target_file or target_dir)
    code_exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp", ".c", ".cs", ".rb", ".php", ".sh"}
    artifact_exts = {
        ".csv",
        ".txt",
        ".json",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".pdf",
        ".md",
        ".html",
        ".xml",
    }

    # If plan already describes local file/code intent, force file route.
    if route == "general" and (target_file or target_dir):
        out["route"] = "file"
        route = "file"

    if code_spec and not requires_code:
        out["requires_code"] = True
        requires_code = True

    # Infer output layout only when user/plan already implies artifact output.
    # Avoid converting pure research Q&A into code generation tasks.
    can_infer_layout = (
        route == "file"
        or had_explicit_target
        or run_after_write
        or bool(expected_artifacts)
    )
    if requires_code and not target_file and can_infer_layout:
        aligned = _align_layout_with_llm(
            query=query,
            target_dir=target_dir,
            target_file=target_file,
            code_spec=code_spec or query,
        )
        aligned_td = str(aligned.get("target_dir", "")).strip()
        aligned_tf = str(aligned.get("target_file", "")).strip()
        if aligned_td and not out.get("target_dir"):
            out["target_dir"] = aligned_td
        if aligned_tf and not out.get("target_file"):
            out["target_file"] = aligned_tf
        target_file = str(out.get("target_file", "")).strip()
        target_dir = str(out.get("target_dir", "")).strip()

    # If artifact task has directory but no file, infer one with LLM.
    if requires_code and not target_file and target_dir and can_infer_layout:
        inferred = _infer_target_file_with_llm(query=query, target_dir=target_dir, code_spec=code_spec)
        if inferred:
            out["target_file"] = inferred if "/" in inferred else f"{target_dir}/{inferred}"

    # Retry target inference once more; keep empty if still unresolved.
    if requires_code and can_infer_layout and not str(out.get("target_file", "")).strip():
        inferred = _infer_target_file_with_llm(query=query, target_dir=target_dir, code_spec=code_spec or query)
        if inferred:
            out["target_file"] = inferred

    # Keep target_file as executable source path for code_agent; data/figure files should be extra outputs.
    old_target_for_entry = str(out.get("target_file", "")).strip().replace("\\", "/")
    tf_curr = old_target_for_entry
    if requires_code and tf_curr:
        ext = Path(tf_curr).suffix.lower()
        if ext in artifact_exts or (ext and ext not in code_exts):
            parent = Path(tf_curr).parent.as_posix()
            stem = Path(tf_curr).stem or "task_output"
            py_name = f"{stem}.py"
            out["target_file"] = f"{parent}/{py_name}" if parent and parent != "." else py_name
        elif not ext:
            out["target_file"] = f"{tf_curr}.py"

    new_target_for_entry = str(out.get("target_file", "")).strip().replace("\\", "/")
    if requires_code and run_entrypoint:
        entry_ext = Path(run_entrypoint).suffix.lower()
        if run_entrypoint == old_target_for_entry or entry_ext in artifact_exts:
            run_entrypoint = new_target_for_entry

    # Infer runtime execution plan for code tasks.
    if requires_code:
        route_now = str(out.get("route", route))
        inferred_exec = _infer_execution_with_llm(
            query=query,
            route=route_now,
            target_dir=str(out.get("target_dir", target_dir)),
            target_file=str(out.get("target_file", "")),
            code_spec=code_spec or query,
        )
        if not run_after_write:
            run_after_write = bool(inferred_exec.get("run_after_write", False))
        if not run_entrypoint:
            run_entrypoint = str(inferred_exec.get("run_entrypoint", "") or "").strip()
        if not expected_artifacts:
            arr = inferred_exec.get("expected_artifacts", [])
            if isinstance(arr, list):
                expected_artifacts = [str(x).strip() for x in arr if str(x).strip()]

        if run_after_write and not run_entrypoint:
            run_entrypoint = str(out.get("target_file", "")).strip()
        if run_after_write and run_entrypoint:
            tf_now = str(out.get("target_file", "")).strip().replace("\\", "/")
            rp_now = str(run_entrypoint).strip().replace("\\", "/")
            rp_ext = Path(rp_now).suffix.lower()
            # Keep user-explicit alternative entrypoint; otherwise default to target_file for consistency.
            if tf_now and rp_now != tf_now:
                q_lower = str(query or "").lower()
                rp_name = Path(rp_now).name.lower()
                if rp_ext in artifact_exts or (rp_name and rp_name not in q_lower):
                    run_entrypoint = tf_now

        out["run_after_write"] = run_after_write
        out["run_entrypoint"] = run_entrypoint
        out["expected_artifacts"] = expected_artifacts[:10]
    else:
        out["run_after_write"] = False
        out["run_entrypoint"] = ""
        out["expected_artifacts"] = []

    # Non-file routes without explicit artifact intent should remain pure retrieval/answer tasks.
    if (
        str(out.get("route", route)) != "file"
        and bool(out.get("requires_code", False))
        and not str(out.get("target_file", "")).strip()
        and not str(out.get("target_dir", "")).strip()
        and not bool(out.get("run_after_write", False))
    ):
        out["requires_code"] = False
        out["code_spec"] = ""

    # If model wrongly points primary target to requirements.txt, recover to a source file.
    tf_lower = str(out.get("target_file", "")).lower()
    if out.get("route") == "file" and bool(out.get("requires_code", False)) and tf_lower.endswith("requirements.txt"):
        td = str(out.get("target_dir", "")).strip()
        inferred = _infer_target_file_with_llm(query=query, target_dir=td, code_spec=code_spec)
        if inferred and not inferred.lower().endswith("requirements.txt"):
            out["target_file"] = inferred if "/" in inferred else (f"{td}/{inferred}" if td else inferred)

    # Ensure target_file sits under target_dir when both provided and file name is bare.
    tf = str(out.get("target_file", "")).strip()
    td = str(out.get("target_dir", "")).strip()
    if td and tf and "/" not in tf:
        out["target_file"] = f"{td}/{tf}"

    # Let LLM align folder/file layout when they are inconsistent.
    td = str(out.get("target_dir", "")).strip()
    tf = str(out.get("target_file", "")).strip()
    if out.get("route") == "file" and bool(out.get("requires_code", False)):
        misaligned = False
        if td and tf and not tf.startswith(f"{td}/"):
            misaligned = True
        if not td and tf:
            misaligned = True
        if misaligned:
            aligned = _align_layout_with_llm(
                query=query,
                target_dir=td,
                target_file=tf,
                code_spec=code_spec,
            )
            aligned_td = str(aligned.get("target_dir", "")).strip()
            aligned_tf = str(aligned.get("target_file", "")).strip()
            if aligned_td:
                out["target_dir"] = aligned_td
            if aligned_tf:
                out["target_file"] = aligned_tf
            if aligned_td and aligned_tf and "/" not in aligned_tf:
                out["target_file"] = f"{aligned_td}/{aligned_tf}"

    return _normalize_plan(out)


def _parent_dir(path: str) -> str:
    parent = Path(path).parent.as_posix()
    return "" if parent == "." else parent


def _extract_recent_target_files(recent_turns: List[Dict[str, Any]]) -> List[str]:
    targets: List[str] = []
    for t in recent_turns:
        logs = t.get("logs", []) or []
        for line in logs:
            s = str(line)
            if s.startswith("supervisor.target_file="):
                p = s.split("=", 1)[1].strip().replace("\\", "/")
                if p:
                    targets.append(p)
    return targets


def _inherit_file_target_from_recent(plan: Dict[str, Any], recent_turns: List[Dict[str, Any]]) -> tuple[Dict[str, Any], str]:
    out = dict(plan)
    if out.get("route") != "file":
        return _normalize_plan(out), ""

    recent_targets = _extract_recent_target_files(recent_turns)
    if not recent_targets:
        return _normalize_plan(out), ""

    latest = recent_targets[-1]
    tf = str(out.get("target_file", "")).strip().replace("\\", "/")
    td = str(out.get("target_dir", "")).strip().replace("\\", "/")

    # No file specified: reuse latest concrete file target.
    if not tf:
        out["target_file"] = latest
        out["target_dir"] = _parent_dir(latest)
        return _normalize_plan(out), f"empty->{latest}"

    # Bare filename: match recent file with same basename.
    if "/" not in tf and not td:
        for cand in reversed(recent_targets):
            if Path(cand).name.lower() == tf.lower():
                out["target_file"] = cand
                out["target_dir"] = _parent_dir(cand)
                return _normalize_plan(out), f"basename:{tf}->{cand}"
        if len(recent_targets) == 1:
            out["target_file"] = latest
            out["target_dir"] = _parent_dir(latest)
            return _normalize_plan(out), f"single_recent->{latest}"

    return _normalize_plan(out), ""


def _is_valid_plan(plan: Dict[str, Any]) -> bool:
    if plan.get("route") not in ALLOWED_ROUTES:
        return False
    if not isinstance(plan.get("requires_code"), bool):
        return False
    if not isinstance(plan.get("wants_requirements"), bool):
        return False
    if not isinstance(plan.get("run_after_write"), bool):
        return False
    if not isinstance(plan.get("expected_artifacts", []), list):
        return False
    if any(not isinstance(x, str) for x in plan.get("expected_artifacts", [])):
        return False
    for key in ["target_file", "target_dir", "code_spec", "run_entrypoint"]:
        if not isinstance(plan.get(key, ""), str):
            return False
    return True


def _build_subtasks(plan: Dict[str, Any], query: str = "", constraints: Dict[str, Any] | None = None) -> List[str]:
    subtasks: List[str] = []
    route = plan.get("route", "general")
    c = constraints or {}
    if route in {"research", "general"} and not bool(plan.get("requires_code", False)):
        subtasks.append("web_agent: collect context/evidence")
        return subtasks
    if bool(plan.get("requires_code", False)) and str(plan.get("target_file", "")).strip():
        provider = ""
        if route in {"stock", "news", "weather", "research", "general"}:
            provider = "web_agent" if route in {"research", "general"} else f"{route}_agent"
        elif route == "file" and _query_has_retrieval_intent(query):
            provider = "web_agent"
        if provider:
            subtasks.append(f"{provider}: collect context/evidence")
        if route == "file" and provider == "web_agent" and bool(c.get("need_data", False)):
            td = str(plan.get("target_dir", "")).strip().replace("\\", "/").rstrip("/")
            hint_csv = f"{td}/retrieved_data.csv" if td else "retrieved_data.csv"
            subtasks.append(f"file_agent: save structured data to `{hint_csv}` (CSV)")
        subtasks.append(f"code_agent: generate code for `{plan['target_file']}`")
        if plan["wants_requirements"]:
            subtasks.append("code_agent: include requirements.txt only when third-party dependencies are used")
        subtasks.append(f"file_agent: write generated files to `{plan['target_file']}`")
        if bool(plan.get("run_after_write", False)):
            entry = str(plan.get("run_entrypoint", "")).strip() or str(plan.get("target_file", "")).strip()
            subtasks.append(f"runner_agent: execute `{entry}` and verify artifacts")
    return subtasks


def supervisor_node(state):
    query = state["query"]
    memory_context = state.get("memory_context", {}) or {}
    memory_summary = str(memory_context.get("summary_text", ""))
    recent_turns = memory_context.get("recent_turns", []) or []
    retrieved_memories = memory_context.get("retrieved_memories", []) or []

    recent_brief = [
        {
            "turn_id": t.get("turn_id"),
            "query": str(t.get("user_query", ""))[:72],
            "route": t.get("route", ""),
            "result": str(t.get("final_answer", ""))[:72],
        }
        for t in recent_turns[-5:]
    ]
    recent_compact = recent_brief[-2:]
    semantic_compact = [
        {
            "type": str(item.get("memory_type", "")),
            "content": str(item.get("content", ""))[:180],
            "score": item.get("score", 0.0),
        }
        for item in retrieved_memories[:5]
    ]

    local_time = get_time()
    local_location = get_location()
    agent_catalog = _load_agent_catalog()
    constraints = _extract_constraints(query)

    catalog_compact = {
        "version": agent_catalog.get("version", ""),
        "agents": [
            {"name": str(a.get("name", "")), "purpose": str(a.get("purpose", ""))[:80]}
            for a in agent_catalog.get("agents", [])
        ],
    }

    prompt = (
        "You are PlannerAgent. Output STRICT JSON object only.\n"
        f"Output schema: {json.dumps(SCHEMA_HINT, ensure_ascii=False)}\n"
        f"Available agents and capabilities: {json.dumps(catalog_compact, ensure_ascii=False)}\n"
        "Rules:\n"
        "1) Never output explanation, markdown, chain-of-thought, or <think>.\n"
        "2) If query edits previous work, infer target_file from memory.\n"
        "3) code_spec must be one concise sentence when requires_code=true.\n"
        "4) For non-file routes keep file fields empty.\n"
        "5) For executable code tasks with requested outputs, set run_after_write=true and expected_artifacts.\n"
        f"Conversation summary: {memory_summary[:220]}\n"
        f"Recent turns: {json.dumps(recent_compact, ensure_ascii=False)}\n"
        f"Retrieved long-term memories: {json.dumps(semantic_compact, ensure_ascii=False)}\n"
        f"User query: {query}\n"
    )

    plan = {
        "route": "general",
        "target_file": "",
        "target_dir": "",
        "requires_code": False,
        "wants_requirements": False,
        "code_spec": "",
        "run_after_write": False,
        "run_entrypoint": "",
        "expected_artifacts": [],
    }

    had_parse_repair = False
    used_llm_fallback = False
    for _ in range(2):
        raw_text = _invoke_text(prompt)
        raw = _extract_json_object(raw_text)
        if not raw:
            raw = _extract_plan_from_partial_text(raw_text)
        candidate = _normalize_plan(raw)

        # Accept only when the model actually produced schema-like content.
        if _is_valid_plan(candidate) and _has_schema_signals(raw):
            plan = candidate
            break

        had_parse_repair = True
        repaired = _repair_plan_json(raw_text, query)
        if not repaired:
            repaired = _extract_plan_from_partial_text(raw_text)
        repaired_candidate = _normalize_plan(repaired)
        if _is_valid_plan(repaired_candidate) and _has_schema_signals(repaired):
            plan = repaired_candidate
            break

    # Last attempt: ask planner again with a minimal fallback prompt.
    if plan["route"] == "general" and not plan["target_file"] and not plan["target_dir"]:
        fallback = _fallback_plan_with_llm(query, memory_summary, recent_brief)
        if _is_valid_plan(fallback) and (
            fallback["route"] != "general" or fallback["target_file"] or fallback["target_dir"]
        ):
            plan = fallback
            used_llm_fallback = True

    used_refine = False
    if _is_valid_plan(plan) and _plan_needs_refine(plan):
        refined = _refine_plan_with_llm(query, plan, memory_summary)
        if _is_valid_plan(refined):
            plan = refined
            used_refine = True

    if plan["route"] == "file" and plan["requires_code"] and not plan["target_file"]:
        refined = _refine_plan_with_llm(query, plan, memory_summary)
        if _is_valid_plan(refined) and refined.get("target_file", ""):
            plan = refined
            used_refine = True

    guard_reasons: List[str] = []
    plan, pre_guard_reason = _enforce_codegen_intent_guard(plan, query)
    if pre_guard_reason:
        guard_reasons.append(pre_guard_reason)

    plan_before_stabilize = dict(plan)
    plan = _stabilize_plan_with_consistency(plan, query)
    plan, inherit_reason = _inherit_file_target_from_recent(plan, recent_turns)
    plan, post_guard_reason = _enforce_codegen_intent_guard(plan, query)
    if post_guard_reason and post_guard_reason not in guard_reasons:
        guard_reasons.append(post_guard_reason)

    state["route"] = plan["route"]
    state["target_file"] = plan["target_file"]
    state["target_dir"] = plan["target_dir"]
    state["requires_code"] = plan["requires_code"]
    state["wants_requirements"] = plan["wants_requirements"]
    state["code_spec"] = plan["code_spec"]
    state["run_after_write"] = bool(plan.get("run_after_write", False))
    state["run_entrypoint"] = str(plan.get("run_entrypoint", "")).strip()
    state["expected_artifacts"] = plan.get("expected_artifacts", []) if isinstance(
        plan.get("expected_artifacts", []), list
    ) else []

    state["generated_code"] = ""
    state["generated_extra_files"] = {}
    state["codegen_error"] = ""
    state["file_write_succeeded"] = False
    state["file_stage"] = ""
    state["staged_data_files"] = {}
    state["direct_action_done"] = False
    state["execution_result"] = {}
    state["needs_web_verify"] = False
    state["did_web_verify"] = False
    state["verify_round"] = 0
    state["subtasks"] = _build_subtasks(plan, query=query, constraints=constraints)
    state["local_time"] = local_time
    state["local_location"] = local_location
    state["source_registry"] = SOURCE_REGISTRY
    state["constraints"] = constraints

    state["logs"].append(f"supervisor.route={plan['route']}")
    state["logs"].append(f"supervisor.local_time={local_time.get('iso', '')}")
    state["logs"].append(f"supervisor.agent_catalog_version={agent_catalog.get('version', '')}")
    active_constraints = _active_constraints(constraints)
    if active_constraints:
        state["logs"].append(
            "supervisor.constraints="
            + json.dumps(active_constraints, ensure_ascii=False, separators=(",", ":"))
        )
    if plan["route"] == "file" or plan["requires_code"] or plan["wants_requirements"]:
        state["logs"].append(f"supervisor.requires_code={plan['requires_code']}")
        state["logs"].append(f"supervisor.wants_requirements={plan['wants_requirements']}")
    if plan.get("run_after_write", False):
        state["logs"].append("supervisor.run_after_write=True")
        if plan.get("run_entrypoint", ""):
            state["logs"].append(f"supervisor.run_entrypoint={plan['run_entrypoint']}")
        if plan.get("expected_artifacts", []):
            state["logs"].append(
                "supervisor.expected_artifacts="
                + json.dumps(plan.get("expected_artifacts", []), ensure_ascii=False, separators=(",", ":"))
            )
    if (
        plan_before_stabilize.get("route") != plan.get("route")
        or plan_before_stabilize.get("target_file") != plan.get("target_file")
        or plan_before_stabilize.get("target_dir") != plan.get("target_dir")
        or plan_before_stabilize.get("requires_code") != plan.get("requires_code")
    ):
        state["logs"].append(
            "supervisor.stabilized="
            f"route:{plan_before_stabilize.get('route')}->{plan.get('route')},"
            f"requires_code:{plan_before_stabilize.get('requires_code')}->{plan.get('requires_code')},"
            f"target_file:{plan_before_stabilize.get('target_file')}->{plan.get('target_file')},"
            f"target_dir:{plan_before_stabilize.get('target_dir')}->{plan.get('target_dir')}"
        )
    if inherit_reason:
        state["logs"].append(f"supervisor.inherit_target={inherit_reason}")
    if had_parse_repair:
        state["logs"].append("supervisor.parse=repair_or_partial")
    if used_llm_fallback:
        state["logs"].append("supervisor.plan_fallback=llm_fallback")
    if used_refine:
        state["logs"].append("supervisor.plan_refined=true")
    if guard_reasons:
        for reason in guard_reasons:
            state["logs"].append(f"supervisor.codegen_guard={reason}")
    if plan["route"] == "general" and not plan["target_file"] and not plan["target_dir"]:
        state["logs"].append("supervisor.plan_fallback=general_default")
    if plan["target_file"] and (plan["route"] == "file" or plan["requires_code"]):
        state["logs"].append(f"supervisor.target_file={plan['target_file']}")
    if plan["target_dir"] and (plan["route"] == "file" or plan["requires_code"]):
        state["logs"].append(f"supervisor.target_dir={plan['target_dir']}")

    append_agent_event(
        state,
        agent_name="supervisor",
        input_payload={
            "query": query,
            "memory_summary": memory_summary[:300],
            "recent_turns": recent_brief,
            "agent_catalog_version": agent_catalog.get("version", ""),
            "constraints": constraints,
        },
        output_payload=plan,
    )
    return state
