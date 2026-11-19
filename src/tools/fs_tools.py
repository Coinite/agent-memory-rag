from pathlib import Path
import re

from src.config import settings

ROOT = Path(settings.workspace_root).resolve()


def _to_workspace_path(path: str) -> Path:
    raw_text = str(path or "").strip().replace("\\", "/")
    # On Windows, treat "/foo/bar" from LLM output as workspace-relative path.
    if ROOT.drive and raw_text.startswith("/") and not re.match(r"^[A-Za-z]:/", raw_text):
        return (ROOT / raw_text.lstrip("/")).resolve()
    ws_name = ROOT.name.lower()
    if raw_text.lower().startswith(ws_name + "/"):
        raw_text = raw_text[len(ROOT.name) + 1 :]

    raw = Path(raw_text)
    if not raw.is_absolute():
        return (ROOT / raw).resolve()

    p = raw.resolve()
    if p == ROOT or ROOT in p.parents:
        return p

    # Common LLM mistake: absolute path under ROOT.parent but missing workspace folder segment.
    try:
        rel_parent = p.relative_to(ROOT.parent)
        parts = list(rel_parent.parts)
        if parts and parts[0].lower() == ROOT.name.lower():
            parts = parts[1:]
        return ROOT.joinpath(*parts).resolve()
    except Exception:
        return p


def _safe(path: str) -> Path:
    p = _to_workspace_path(path)
    if ROOT not in p.parents and p != ROOT:
        raise PermissionError("Path out of workspace allowlist.")
    return p


def read_file(path: str) -> str:
    return _safe(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p)


def write_bytes(path: str, content: bytes) -> str:
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return str(p)


def make_dir(path: str) -> str:
    p = _safe(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def list_files(path: str = "."):
    p = _safe(path)
    return [str(x.relative_to(ROOT)) for x in p.rglob("*") if x.is_file()]
