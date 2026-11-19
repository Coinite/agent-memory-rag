import os
import re
from pathlib import Path

from src.agents.utils import append_agent_event
from src.config import settings
from src.tools.fs_tools import list_files, make_dir, read_file, write_file


FILE_PATH_PATTERN = re.compile(r"([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)")


ZH_LIST = "\u5217\u51fa"
ZH_READ = "\u8bfb\u53d6"
ZH_CREATE = "\u521b\u5efa"
ZH_NEW = "\u65b0\u5efa"
ZH_MODIFY = "\u4fee\u6539"
ZH_CHANGE_TO = "\u6539\u6210"
ZH_CHANGE_AS = "\u6539\u4e3a"


def _extract_file_path(query: str) -> str:
    m = FILE_PATH_PATTERN.search(query)
    return m.group(1).replace("\\", "/") if m else ""


def _sanitize_code_content(content: str) -> str:
    raw = content.strip()
    if not raw:
        return ""

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    elif "```" in raw:
        blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", raw, flags=re.DOTALL)
        if blocks:
            raw = blocks[-1].strip()

    return raw + ("\n" if raw and not raw.endswith("\n") else "")


def _norm_path(raw_path: str) -> str:
    x = str(raw_path or "").strip().replace("\\", "/").strip("\"'")
    for prefix in ["/current/project/root/", "current/project/root/", "/workspace/", "workspace/"]:
        if x.lower().startswith(prefix):
            x = x[len(prefix) :]
            break

    workspace = Path(settings.workspace_root).resolve()
    # On Windows, convert pseudo-root paths like /stock/file.py into workspace-relative.
    if workspace.drive and x.startswith("/") and not re.match(r"^[A-Za-z]:/", x):
        x = x.lstrip("/")
    ws_name = workspace.name.lower()
    if x.lower().startswith(ws_name + "/"):
        x = x[len(workspace.name) + 1 :]

    if re.match(r"^[A-Za-z]:/", x) or x.startswith("/"):
        try:
            abs_path = Path(x).resolve()
            try:
                x = abs_path.relative_to(workspace).as_posix()
            except Exception:
                rel_parent = abs_path.relative_to(workspace.parent)
                parts = list(rel_parent.parts)
                if parts and parts[0].lower() == workspace.name.lower():
                    parts = parts[1:]
                x = Path(*parts).as_posix() if parts else ""
        except Exception:
            pass

    if x.startswith("./"):
        x = x[2:]
    return x


def _resolve_target_path(query: str, state) -> str:
    path = _norm_path(state.get("target_file", "") or _extract_file_path(query))
    target_dir = _norm_path(state.get("target_dir", ""))
    if target_dir and path and "/" not in path:
        return f"{target_dir}/{path}"
    return path


def _ensure_csv_path(path: str, target_dir: str, fallback_name: str = "retrieved_data.csv") -> str:
    p = _norm_path(path)
    if not p:
        p = f"{target_dir}/{fallback_name}" if target_dir else fallback_name
    suffix = Path(p).suffix.lower()
    if suffix != ".csv":
        p = f"{Path(p).with_suffix('.csv').as_posix()}"
    if target_dir and "/" not in p:
        p = f"{target_dir}/{p}"
    p = re.sub(r"/{2,}", "/", p)
    return p


def _normalize_csv_text(content: str) -> str:
    x = str(content or "")
    x = re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL | re.IGNORECASE).strip()
    if not x:
        return ""
    return x + ("\n" if not x.endswith("\n") else "")


def file_agent_node(state):
    query = state["query"]
    q_lower = query.lower()
    target_dir = _norm_path(state.get("target_dir", ""))
    state["file_write_succeeded"] = False
    path = _resolve_target_path(query, state)
    file_stage = str(state.get("file_stage", "") or "").strip().lower()

    if file_stage == "data":
        staged = state.get("staged_data_files", {}) or {}
        if not isinstance(staged, dict) or not staged:
            state["final_answer"] = "Data staging requested but no staged data files were provided."
            state["logs"].append("file_agent.action=data_stage_empty")
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "write_data_csv"},
                output_payload={"status": "error", "error": "empty_staged_data"},
            )
            return state

        written: list[str] = []
        fallback_name = "retrieved_data.csv"
        tf = _norm_path(state.get("target_file", ""))
        if tf:
            fallback_name = f"{Path(tf).stem}_data.csv"

        for raw_path, raw_content in staged.items():
            csv_path = _ensure_csv_path(str(raw_path), target_dir=target_dir, fallback_name=fallback_name)
            csv_content = _normalize_csv_text(str(raw_content))
            if not csv_content:
                continue
            write_file(csv_path, csv_content)
            written.append(csv_path)
            state["logs"].append(f"file_agent.action=write_data_csv:{csv_path}")

        if not written:
            state["final_answer"] = "Data staging failed: no valid CSV content to write."
            state["logs"].append("file_agent.action=data_stage_no_valid_content")
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "write_data_csv"},
                output_payload={"status": "error", "error": "no_valid_csv_content"},
            )
            return state

        context_payload = state.get("context_payload", {}) or {}
        web_ctx = context_payload.get("web", {}) if isinstance(context_payload.get("web", {}), dict) else {}
        web_ctx["data_file"] = written[0]
        context_payload["web"] = web_ctx
        context_payload["data_files"] = written
        state["context_payload"] = context_payload
        state["staged_data_files"] = {}
        state["file_write_succeeded"] = True
        state["final_answer"] = f"Structured data saved as CSV: {', '.join(written)}."
        append_agent_event(
            state,
            agent_name="file_agent",
            input_payload={"query": query, "action": "write_data_csv"},
            output_payload={"status": "ok", "written_files": written},
        )
        return state

    if (
        ("list" in q_lower or ZH_LIST in query)
        and not bool(state.get("requires_code", False))
        and not path
        and not target_dir
    ):
        files = list_files(".")
        state["final_answer"] = "\n".join(files[:200]) if files else "No files found."
        state["logs"].append("file_agent.action=list_files")
        append_agent_event(
            state,
            agent_name="file_agent",
            input_payload={"query": query, "action": "list"},
            output_payload={"file_count": len(files)},
        )
        return state

    if ((ZH_CREATE in query or ZH_NEW in query or "create" in q_lower or "mkdir" in q_lower) and not path and target_dir):
        try:
            make_dir(target_dir)
            state["final_answer"] = f"Created directory `{target_dir}`."
            state["logs"].append(f"file_agent.action=mkdir:{target_dir}")
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "mkdir", "path": target_dir},
                output_payload={"status": "ok"},
            )
        except Exception as e:
            state["final_answer"] = f"Create directory `{target_dir}` failed: {e}"
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "mkdir", "path": target_dir},
                output_payload={"status": "error", "error": str(e)},
            )
        return state

    if (ZH_READ in query or "read" in q_lower) and path:
        try:
            content = read_file(path)
            state["final_answer"] = f"Read `{path}` successfully.\n\n{content}"
            state["logs"].append(f"file_agent.action=read:{path}")
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "read", "path": path},
                output_payload={"status": "ok", "content_preview": content[:200]},
            )
        except Exception as e:
            state["final_answer"] = f"Read `{path}` failed: {e}"
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "read", "path": path},
                output_payload={"status": "error", "error": str(e)},
            )
        return state

    if (
        (bool(state.get("requires_code", False)) and bool(path))
        or (
        ZH_CREATE in query
        or ZH_MODIFY in query
        or ZH_CHANGE_TO in query
        or ZH_CHANGE_AS in query
        or "write" in q_lower
        or "create" in q_lower
        or "update" in q_lower
        or "modify" in q_lower
        or "edit" in q_lower
        or "save" in q_lower
        or "store" in q_lower
        or "download" in q_lower
    )
    ) and path:
        content = _sanitize_code_content(state.get("generated_code", ""))
        if bool(state.get("requires_code", False)) and not content:
            err = str(state.get("codegen_error", "")).strip() or "code_generation_failed"
            state["final_answer"] = (
                f"Write skipped for `{path}` because code generation failed: {err}. "
                "Please refine the requirement and retry."
            )
            state["logs"].append("file_agent.action=skip_no_code")
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "write", "path": path},
                output_payload={"status": "error", "error": err},
            )
            return state
        if not content:
            content = ""

        created_files = []
        try:
            saved = write_file(path, content)
            created_files.append(path)
            state["logs"].append(f"file_agent.action=write:{path}")
            state["file_write_succeeded"] = True

            base_dir = os.path.dirname(path.replace("\\", "/"))
            extra_files = state.get("generated_extra_files", {}) or {}
            for extra_name, extra_content in extra_files.items():
                extra_path = str(extra_name).replace("\\", "/")
                if "/" not in extra_path and base_dir:
                    extra_path = f"{base_dir}/{extra_path}"
                write_file(extra_path, _sanitize_code_content(str(extra_content)))
                created_files.append(extra_path)
                state["logs"].append(f"file_agent.action=write:{extra_path}")

            if len(created_files) > 1:
                state["final_answer"] = (
                    f"Created `{path}` at `{saved}`.\n"
                    f"Also created: {', '.join(created_files[1:])}"
                )
            else:
                state["final_answer"] = f"Created `{path}` at `{saved}`."
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "write", "path": path},
                output_payload={"status": "ok", "created_files": created_files},
            )
        except Exception as e:
            state["final_answer"] = f"Write `{path}` failed: {e}"
            state["file_write_succeeded"] = False
            append_agent_event(
                state,
                agent_name="file_agent",
                input_payload={"query": query, "action": "write", "path": path},
                output_payload={"status": "error", "error": str(e)},
            )
        return state

    state["final_answer"] = (
        "File Agent received the task but could not map it to read/write/list. "
        "Try specifying an explicit file path and action."
    )
    state["logs"].append("file_agent.action=unsupported")
    append_agent_event(
        state,
        agent_name="file_agent",
        input_payload={"query": query, "action": "unsupported"},
        output_payload={"status": "ignored"},
    )
    return state
