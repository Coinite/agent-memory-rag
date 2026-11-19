import re
import urllib.request
from html import unescape

from ddgs import DDGS

from src.tools.fs_tools import write_bytes


def web_search(query: str, k: int = 5):
    rows = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            rows.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "duckduckgo",
                }
            )
    return rows


def _arxiv_abs_to_pdf(url: str) -> str:
    u = (url or "").strip()
    m = re.search(r"https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(v\d+)?", u)
    if not m:
        return ""
    ver = m.group(2) or ""
    return f"https://arxiv.org/pdf/{m.group(1)}{ver}.pdf"


def to_pdf_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.lower().endswith(".pdf"):
        return u
    arxiv_pdf = _arxiv_abs_to_pdf(u)
    if arxiv_pdf:
        return arxiv_pdf
    return ""


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LocalAgent/1.0)",
            "Accept": "application/pdf,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _safe_pdf_name(name: str) -> str:
    n = re.sub(r"[^\w\-. ]+", "", str(name or "").strip()).replace(" ", "_")
    n = n.strip("._")
    return n[:120] if n else "paper"


def infer_pdf_filename(title: str, rank: int) -> str:
    base = _safe_pdf_name(title)
    return f"{rank:02d}_{base}.pdf"


def download_pdf_to_workspace(url: str, rel_path: str) -> dict:
    pdf_url = to_pdf_url(url)
    if not pdf_url:
        return {"ok": False, "error": "no_direct_pdf_url", "url": url}
    try:
        data = _download_bytes(pdf_url)
        saved = write_bytes(rel_path, data)
        return {"ok": True, "url": pdf_url, "saved_path": saved, "size": len(data)}
    except Exception as e:
        return {"ok": False, "error": str(e), "url": pdf_url}


def _strip_html_to_text(html: str) -> str:
    x = re.sub(r"(?is)<script.*?>.*?</script>", " ", html or "")
    x = re.sub(r"(?is)<style.*?>.*?</style>", " ", x)
    x = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", x)
    x = re.sub(r"(?is)<[^>]+>", " ", x)
    x = unescape(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def fetch_web_text(url: str, timeout: int = 20, max_chars: int = 12000) -> dict:
    try:
        req = urllib.request.Request(
            str(url or "").strip(),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=max(5, int(timeout))) as resp:
            data = resp.read()
            ctype = str(resp.headers.get("Content-Type", ""))
        charset = "utf-8"
        m = re.search(r"charset=([a-zA-Z0-9._-]+)", ctype, flags=re.IGNORECASE)
        if m:
            charset = m.group(1).strip()
        html = data.decode(charset, errors="ignore")
        text = _strip_html_to_text(html)
        return {"ok": True, "url": url, "text": text[: max(500, int(max_chars))]}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}
