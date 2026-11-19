import csv
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List


def _http_get(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LocalAgent/1.0)",
            "Accept": "application/json,text/xml,application/xml,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def get_weather_open_meteo(city: str) -> dict:
    q = urllib.parse.quote(city)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
    geo_raw = _http_get(geo_url)
    geo_data = json.loads(geo_raw)
    results = geo_data.get("results", []) if isinstance(geo_data, dict) else []
    if not results:
        return {"ok": False, "error": f"City not found: {city}", "source": geo_url}

    top = results[0]
    lat = top.get("latitude")
    lon = top.get("longitude")
    tz = top.get("timezone", "auto")
    city_name = top.get("name", city)

    weather_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,"
        "precipitation,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max&timezone="
        f"{urllib.parse.quote(str(tz))}"
    )
    weather_raw = _http_get(weather_url)
    weather_data = json.loads(weather_raw)
    return {
        "ok": True,
        "city": city_name,
        "timezone": tz,
        "data": weather_data,
        "source": weather_url,
    }


def get_news_google_rss(topic_query: str, limit: int = 8) -> dict:
    q = urllib.parse.quote(topic_query)
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    raw = _http_get(rss_url)
    root = ET.fromstring(raw)
    items = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source = ""
        src_el = item.find("source")
        if src_el is not None and src_el.text:
            source = src_el.text.strip()
        items.append({"title": title, "url": link, "published_at": pub_date, "source": source})
    return {"ok": True, "items": items, "source": rss_url}


def get_stock_stooq(symbol: str) -> dict:
    # Stooq symbol format: aapl.us, msft.us, ^spx not always supported.
    sym = symbol.strip().lower()
    if "." not in sym:
        sym = f"{sym}.us"
    url = f"https://stooq.com/q/l/?s={urllib.parse.quote(sym)}&i=d"
    raw = _http_get(url)
    reader = csv.DictReader(StringIO(raw))
    rows = list(reader)
    if not rows:
        return {"ok": False, "error": f"No data for symbol: {symbol}", "source": url}
    row = rows[0]
    return {"ok": True, "symbol": sym, "quote": row, "source": url}


def _safe_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _normalize_symbol(symbol: str) -> str:
    sym = symbol.strip().lower()
    if "." not in sym:
        sym = f"{sym}.us"
    return sym


def get_stock_stooq_history(symbol: str, lookback_days: int = 5) -> Dict[str, Any]:
    sym = _normalize_symbol(symbol)
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(sym)}&i=d"
    raw = _http_get(url)
    reader = csv.DictReader(StringIO(raw))
    rows = list(reader)
    if not rows:
        return {"ok": False, "error": f"No history for symbol: {symbol}", "source": url}

    parsed: List[Dict[str, Any]] = []
    for r in rows:
        date_str = str(r.get("Date", "")).strip()
        if not date_str:
            continue
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        close_v = _safe_float(r.get("Close", ""))
        open_v = _safe_float(r.get("Open", ""))
        high_v = _safe_float(r.get("High", ""))
        low_v = _safe_float(r.get("Low", ""))
        vol_v = _safe_float(r.get("Volume", ""))
        parsed.append(
            {
                "Date": date_obj.isoformat(),
                "Open": open_v,
                "High": high_v,
                "Low": low_v,
                "Close": close_v,
                "Volume": vol_v,
            }
        )

    if not parsed:
        return {"ok": False, "error": f"Unable to parse history for symbol: {symbol}", "source": url}

    parsed.sort(key=lambda x: x.get("Date", ""))
    days = max(1, int(lookback_days))
    sliced = parsed[-days:]

    return {
        "ok": True,
        "symbol": sym,
        "rows": sliced,
        "source": url,
        "resolution": "1d",
    }
