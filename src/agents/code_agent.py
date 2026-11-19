import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.utils import append_agent_event
from src.config import settings
from src.models.lmstudio_client import build_llm

PRIMARY_MODEL = (settings.code_model_name.strip() or settings.model_name).strip()
FALLBACK_MODEL = settings.model_name.strip()
CODE_TIMEOUT_SEC = max(60.0, float(settings.lm_timeout_sec))

primary_llm = build_llm(
    temp=0.0,
    model_override=PRIMARY_MODEL,
    max_tokens=1400,
    timeout_sec=CODE_TIMEOUT_SEC,
    max_retries=0,
)
fallback_llm = None
if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
    fallback_llm = build_llm(
        temp=0.0,
        model_override=FALLBACK_MODEL,
        max_tokens=1200,
        timeout_sec=CODE_TIMEOUT_SEC,
        max_retries=0,
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _strip_code_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw + ("\n" if raw and not raw.endswith("\n") else "")


def _content_to_text(msg: Any) -> str:
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


def _invoke_text(prompt: str) -> Tuple[str, str]:
    clients: List[Tuple[str, Any]] = [(PRIMARY_MODEL, primary_llm)]
    if fallback_llm is not None:
        clients.append((FALLBACK_MODEL, fallback_llm))

    for model_name, client in clients:
        try:
            text = _content_to_text(client.invoke(prompt))
            if text.strip():
                return text, model_name
        except Exception:
            continue
    return "", ""


def _compact_value(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)[:180]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 8:
                break
            out[str(k)[:64]] = _compact_value(v, depth + 1)
        return out
    if isinstance(value, list):
        return [_compact_value(x, depth + 1) for x in value[:6]]
    if isinstance(value, str):
        s = re.sub(r"\s+", " ", value).strip()
        return s[:240]
    return value


def _compact_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    compact: List[Dict[str, str]] = []
    for e in evidence[:5]:
        compact.append(
            {
                "title": str(e.get("title", "")).strip()[:90],
                "url": str(e.get("url", "")).strip()[:180],
                "snippet": str(e.get("snippet", "")).strip().replace("\n", " ")[:130],
                "source": str(e.get("source", "")).strip()[:50],
            }
        )
    return compact


def _extract_partial_code_from_jsonish(text: str) -> str:
    clean = _strip_code_fence(_strip_think(text))
    if not clean:
        return ""

    m = re.search(r'"code"\s*:\s*"(?P<body>.*)$', clean, flags=re.DOTALL)
    if not m:
        return ""

    body = m.group("body")
    for marker in ['","extra_files"', '"},', '"}\n', '"\n', '"}']:
        idx = body.find(marker)
        if idx >= 0:
            body = body[:idx]
            break

    body = body.rstrip("\\").strip()
    if body.endswith('"'):
        body = body[:-1]
    if not body:
        return ""

    try:
        decoded = bytes(body, "utf-8").decode("unicode_escape")
    except Exception:
        decoded = body.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    return _strip_code_fence(decoded)


def _extract_payload(text: str) -> Tuple[str, Dict[str, str], str]:
    clean = _strip_think(text)

    def _coerce_extra(value) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        out: Dict[str, str] = {}
        for k, v in value.items():
            key = str(k).strip()
            val = _strip_code_fence(str(v))
            key_lower = key.lower()
            val_lower = val.strip().lower()
            # Ignore common schema placeholders copied by smaller models.
            if key_lower in {"path", "filepath", "file_path"} and val_lower in {"content", "<content>"}:
                continue
            if key and val.strip():
                out[key] = val
        return out

    candidates = [clean]
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        candidates.append(clean[start : end + 1])

    for c in candidates:
        try:
            data = json.loads(c)
            if not isinstance(data, dict):
                continue
            code = _strip_code_fence(str(data.get("code", "")))
            extras = _coerce_extra(data.get("extra_files", {}))
            err = str(data.get("error", "") or "").strip()
            if code.strip():
                return code, extras, ""
            if err:
                return "", {}, err
        except Exception:
            continue

    partial_code = _extract_partial_code_from_jsonish(clean)
    if partial_code.strip():
        return partial_code, {}, ""
    return "", {}, ""


def _best_effort_code(text: str) -> str:
    clean = _strip_think(text)
    if not clean:
        return ""

    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", clean, flags=re.DOTALL)
    if blocks:
        return _strip_code_fence(blocks[-1])
    if clean.startswith("```"):
        return _strip_code_fence(clean)
    partial = _extract_partial_code_from_jsonish(clean)
    if partial.strip():
        return partial
    if "\n" in clean and len(clean.splitlines()) >= 2:
        return _strip_code_fence(clean)
    return ""


def _is_viable_code(
    code: str,
    data_file_hint: str,
    expected_artifacts: List[str],
    wants_requirements: bool,
) -> bool:
    body = str(code or "").strip()
    if not body:
        return False
    if re.fullmatch(r"[\d\W_]+", body):
        return False

    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False

    lowered = body.lower()
    if not wants_requirements and any(
        t in lowered
        for t in ["import pandas", "from pandas", "import numpy", "from numpy", "import requests", "import seaborn"]
    ):
        return False
    has_program_shape = any(
        token in lowered for token in ["import ", "def ", "if __name__", "plt.", "open(", "csv.reader"]
    )
    if not has_program_shape:
        return False

    if data_file_hint:
        data_name = Path(data_file_hint).name
        if data_name and data_name not in body and data_file_hint not in body:
            return False

    if expected_artifacts:
        artifact_names = [Path(str(p)).name for p in expected_artifacts if str(p).strip()]
        if artifact_names and not any(name and name in body for name in artifact_names):
            return False

    return True


def _build_csv_plot_fallback(data_file_hint: str, artifact_path: str) -> str:
    data_path = str(data_file_hint).strip().replace("\\", "/")
    output_path = str(artifact_path).strip().replace("\\", "/")
    return (
        "import csv\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        f"DATA_FILE = Path(r\"{data_path}\")\n"
        f"OUTPUT_FILE = Path(r\"{output_path}\")\n"
        "\n"
        "if not DATA_FILE.exists():\n"
        "    raise FileNotFoundError(f\"CSV not found: {DATA_FILE}\")\n"
        "\n"
        "with DATA_FILE.open(\"r\", encoding=\"utf-8\", newline=\"\") as f:\n"
        "    reader = csv.DictReader(f)\n"
        "    fields = reader.fieldnames or []\n"
        "    if len(fields) < 2:\n"
        "        raise RuntimeError(f\"CSV requires at least 2 columns, got: {fields}\")\n"
        "    x_col, y_col = fields[0], fields[1]\n"
        "    xs, ys = [], []\n"
        "    for row in reader:\n"
        "        xv = str(row.get(x_col, \"\")).strip()\n"
        "        yv_raw = str(row.get(y_col, \"\")).replace(\",\", \"\").strip()\n"
        "        if not xv or not yv_raw:\n"
        "            continue\n"
        "        try:\n"
        "            yv = float(yv_raw)\n"
        "        except Exception:\n"
        "            continue\n"
        "        xs.append(xv)\n"
        "        ys.append(yv)\n"
        "\n"
        "if not xs:\n"
        "    raise RuntimeError(\"No plottable numeric rows found in CSV.\")\n"
        "\n"
        "plt.figure(figsize=(10, 5))\n"
        "plt.plot(range(len(xs)), ys, marker=\"o\")\n"
        "plt.xticks(range(len(xs)), xs, rotation=45, ha=\"right\")\n"
        "plt.title(\"Trend from CSV Data\")\n"
        "plt.xlabel(x_col)\n"
        "plt.ylabel(y_col)\n"
        "plt.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)\n"
        "plt.savefig(OUTPUT_FILE, dpi=200)\n"
    )


def code_agent_node(state):
    spec = state.get("code_spec", "") or state["query"]
    target_file = state.get("target_file", "")
    wants_requirements = bool(state.get("wants_requirements", False))
    run_after_write = bool(state.get("run_after_write", False))
    expected_artifacts = state.get("expected_artifacts", []) if isinstance(state.get("expected_artifacts", []), list) else []
    route = state.get("route", "general")
    evidence = state.get("evidence", []) or []
    context_payload = state.get("context_payload", {}) or {}
    if str(state.get("file_stage", "") or "").strip().lower() == "data":
        state["file_stage"] = ""
    data_file_hint = ""
    web_ctx = context_payload.get("web", {}) if isinstance(context_payload.get("web", {}), dict) else {}
    if isinstance(web_ctx, dict):
        data_file_hint = str(web_ctx.get("data_file", "") or "").strip()
    compact_evidence = _compact_evidence(evidence)
    compact_context = _compact_value(context_payload)
    compact_evidence_json = json.dumps(compact_evidence, ensure_ascii=False)
    compact_context_json = json.dumps(compact_context, ensure_ascii=False)
    if len(compact_evidence_json) > 2200:
        compact_evidence_json = compact_evidence_json[:2200]
    if len(compact_context_json) > 1800:
        compact_context_json = compact_context_json[:1800]

    prompts = [
        (
            "You are CodeAgent. Return strict JSON only with schema:\n"
            '{"code":"<file content>","extra_files":{},"error":""}\n'
            "Rules:\n"
            "1) No markdown fence or extra narration.\n"
            "2) Write complete executable code for target_file.\n"
            "3) Use context_evidence/context_payload only as factual inputs.\n"
            "4) If run_after_write=true and expected_artifacts is not empty, code must create those files.\n"
            "5) When wants_requirements=false, avoid third-party data libraries (pandas/numpy/requests).\n"
            "6) If data_file_hint is provided, read data from that CSV and do not crawl websites.\n"
            "7) If generation fails, set code empty and provide error.\n"
            f"wants_requirements={str(wants_requirements).lower()}\n"
            f"run_after_write={str(run_after_write).lower()}\n"
            f"expected_artifacts={expected_artifacts}\n"
            f"data_file_hint={data_file_hint}\n"
            f"route={route}\n"
            f"target_file={target_file}\n"
            f"requirement={spec}\n"
            f"context_evidence={compact_evidence_json}\n"
            f"context_payload={compact_context_json}\n"
        ),
        (
            "Retry. Output JSON object only with keys: code, extra_files, error.\n"
            "Keep code complete and runnable.\n"
            f"run_after_write={str(run_after_write).lower()}\n"
            f"expected_artifacts={expected_artifacts}\n"
            f"data_file_hint={data_file_hint}\n"
            f"route={route}\n"
            f"target_file={target_file}\n"
            f"requirement={spec}\n"
            f"context_evidence={compact_evidence_json}\n"
            f"context_payload={compact_context_json}\n"
        ),
        (
            "Last retry: output ONLY target_file code text (no JSON, no fence).\n"
            f"run_after_write={str(run_after_write).lower()}\n"
            f"expected_artifacts={expected_artifacts}\n"
            f"data_file_hint={data_file_hint}\n"
            f"route={route}\n"
            f"target_file={target_file}\n"
            f"requirement={spec}\n"
            f"context_evidence={compact_evidence_json}\n"
            f"context_payload={compact_context_json}\n"
        ),
    ]

    generated = ""
    extras: Dict[str, str] = {}
    error_msg = ""
    used_models: List[str] = []
    for idx, prompt in enumerate(prompts):
        raw, used_model = _invoke_text(prompt)
        if used_model and used_model not in used_models:
            used_models.append(used_model)

        if idx < 2:
            generated, extras, err = _extract_payload(raw)
            if err and not error_msg:
                error_msg = err
        else:
            generated = _best_effort_code(raw)
            extras = {}
        if generated.strip():
            if not _is_viable_code(
                generated,
                data_file_hint=data_file_hint,
                expected_artifacts=expected_artifacts,
                wants_requirements=wants_requirements,
            ):
                generated = ""
                extras = {}
                if not error_msg:
                    error_msg = "low_quality_code"
                continue
            error_msg = ""
            break

    if (
        not generated.strip()
        and data_file_hint
        and expected_artifacts
        and not wants_requirements
    ):
        generated = _build_csv_plot_fallback(
            data_file_hint=data_file_hint,
            artifact_path=str(expected_artifacts[0]),
        )
        extras = {}
        error_msg = ""
        state["logs"].append("code_agent.fallback=csv_plot_template")

    state["logs"].append("code_agent.action=generate_code")
    state["logs"].append(f"code_agent.model={PRIMARY_MODEL}")
    state["logs"].append(f"code_agent.evidence_in={len(evidence)}")
    if context_payload:
        state["logs"].append(
            "code_agent.context_keys="
            + ",".join(sorted([str(k) for k in context_payload.keys()])[:8])
        )
    if used_models and (len(used_models) > 1 or used_models[0] != PRIMARY_MODEL):
        state["logs"].append("code_agent.models_used=" + ",".join(used_models))

    if not generated.strip():
        state["generated_code"] = ""
        state["generated_extra_files"] = {}
        state["codegen_error"] = error_msg or "code_generation_failed"
        state["logs"].append("code_agent.error=code_generation_failed")
        append_agent_event(
            state,
            agent_name="code_agent",
            input_payload={
                "target_file": target_file,
                "wants_requirements": wants_requirements,
                "code_spec": spec[:400],
                "model": PRIMARY_MODEL,
                "models_used": used_models,
                "run_after_write": run_after_write,
                "expected_artifacts": expected_artifacts[:10],
                "data_file_hint": data_file_hint,
                "evidence_count": len(evidence),
                "context_payload_keys": sorted([str(k) for k in context_payload.keys()]),
            },
            output_payload={"status": "error", "error": state["codegen_error"]},
        )
        return state

    state["generated_code"] = generated
    state["generated_extra_files"] = extras
    state["codegen_error"] = ""
    append_agent_event(
        state,
        agent_name="code_agent",
        input_payload={
            "target_file": target_file,
            "wants_requirements": wants_requirements,
            "code_spec": spec[:400],
            "model": PRIMARY_MODEL,
            "models_used": used_models,
            "run_after_write": run_after_write,
            "expected_artifacts": expected_artifacts[:10],
            "data_file_hint": data_file_hint,
            "evidence_count": len(evidence),
            "context_payload_keys": sorted([str(k) for k in context_payload.keys()]),
        },
        output_payload={
            "status": "ok",
            "generated_code_preview": generated[:300],
            "extra_files": list(extras.keys()),
        },
    )
    return state
