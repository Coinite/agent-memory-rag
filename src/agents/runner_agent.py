import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from src.agents.utils import append_agent_event
from src.config import settings

ROOT = Path(settings.workspace_root).resolve()


def _to_workspace_path(path: str) -> Path:
    p = str(path or "").strip().replace("\\", "/").strip("\"'")
    if ROOT.drive and p.startswith("/") and len(p) > 1 and ":" not in p[:3]:
        p = p.lstrip("/")
    raw = Path(p)
    if not raw.is_absolute():
        return (ROOT / raw).resolve()
    return raw.resolve()


def _safe_rel(path: str) -> Path:
    p = _to_workspace_path(path)
    if p != ROOT and ROOT not in p.parents:
        raise PermissionError("Path out of workspace allowlist.")
    return p


def _collect_artifacts(paths: List[str]) -> Dict[str, List[str]]:
    found: List[str] = []
    missing: List[str] = []
    for rel in paths:
        try:
            p = _safe_rel(rel)
            if p.exists():
                found.append(str(p.relative_to(ROOT)).replace("\\", "/"))
            else:
                missing.append(str(rel).replace("\\", "/"))
        except Exception:
            missing.append(str(rel).replace("\\", "/"))
    return {"found": found, "missing": missing}


def runner_agent_node(state):
    if not bool(state.get("run_after_write", False)):
        return state
    if not bool(state.get("file_write_succeeded", False)):
        state["final_answer"] = "Runner skipped: file write did not succeed in this turn."
        state["logs"].append("runner_agent.run=skip_no_fresh_write")
        return state

    entry = str(state.get("run_entrypoint", "") or state.get("target_file", "")).strip()
    expected = state.get("expected_artifacts", []) if isinstance(state.get("expected_artifacts", []), list) else []
    if not entry:
        state["final_answer"] = "Runner skipped: no run_entrypoint provided."
        state["logs"].append("runner_agent.run=skip_no_entrypoint")
        return state

    try:
        entry_path = _safe_rel(entry)
    except Exception as e:
        state["final_answer"] = f"Runner failed: invalid entrypoint `{entry}` ({e})."
        state["logs"].append("runner_agent.run=failed_invalid_entry")
        return state

    if not entry_path.exists():
        state["final_answer"] = f"Runner failed: entrypoint `{entry}` not found."
        state["logs"].append("runner_agent.run=failed_missing_entry")
        return state

    cmd = [sys.executable, str(entry_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as e:
        state["final_answer"] = f"Runner failed to execute `{entry}`: {e}"
        state["logs"].append("runner_agent.run=failed_exec_exception")
        append_agent_event(
            state,
            agent_name="runner_agent",
            input_payload={"entrypoint": entry, "expected_artifacts": expected},
            output_payload={"status": "error", "error": str(e)},
        )
        return state

    artifact_state = _collect_artifacts(expected)
    state["execution_result"] = {
        "entrypoint": entry,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-1200:],
        "stderr": proc.stderr[-1200:],
        "artifacts_found": artifact_state["found"],
        "artifacts_missing": artifact_state["missing"],
    }

    if proc.returncode != 0:
        state["final_answer"] = (
            f"Executed `{entry}` with return code {proc.returncode}.\n"
            f"stderr:\n{proc.stderr[-600:] or '(empty)'}"
        )
        state["logs"].append("runner_agent.run=failed_returncode")
        append_agent_event(
            state,
            agent_name="runner_agent",
            input_payload={"entrypoint": entry, "expected_artifacts": expected},
            output_payload={"status": "error", "execution_result": state["execution_result"]},
        )
        return state

    if artifact_state["missing"]:
        state["final_answer"] = (
            f"Executed `{entry}` successfully, but some artifacts are missing: "
            f"{', '.join(artifact_state['missing'])}."
        )
        state["logs"].append("runner_agent.run=ok_missing_artifacts")
    else:
        state["final_answer"] = (
            f"Executed `{entry}` successfully."
            + (
                f" Artifacts: {', '.join(artifact_state['found'])}."
                if artifact_state["found"]
                else ""
            )
        )
        state["logs"].append("runner_agent.run=ok")

    append_agent_event(
        state,
        agent_name="runner_agent",
        input_payload={"entrypoint": entry, "expected_artifacts": expected},
        output_payload={"status": "ok", "execution_result": state["execution_result"]},
    )
    return state
