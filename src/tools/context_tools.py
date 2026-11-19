import os
from datetime import datetime


def get_time() -> dict:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "timezone": str(now.tzinfo),
    }


def get_location() -> dict:
    # Local-only hints; users can override via env for better accuracy.
    city = os.getenv("AGENT_CITY", "").strip()
    region = os.getenv("AGENT_REGION", "").strip()
    country = os.getenv("AGENT_COUNTRY", "").strip()
    label = ", ".join([x for x in [city, region, country] if x]) or "unknown"
    return {"city": city, "region": region, "country": country, "label": label}
