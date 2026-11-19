import json
import re
from typing import Any, Dict

from src.agents.utils import append_agent_event
from src.models.lmstudio_client import build_llm
from src.tools.context_tools import get_location, get_time
from src.tools.realtime_tools import get_weather_open_meteo

llm = build_llm(temp=0.0)


def _extract_json_object(text: str) -> Dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
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


def _extract_city(query: str, local_location_label: str) -> str:
    prompt = (
        "Return JSON only: {\"city\":\"...\"}.\n"
        "Extract target city for weather query. If absent, use local location label.\n"
        f"local_location={local_location_label}\n"
        f"query={query}\n"
    )
    for _ in range(2):
        data = _extract_json_object(_invoke_text(prompt))
        city = str(data.get("city", "")).strip()
        if city:
            return city
    return local_location_label if local_location_label and local_location_label != "unknown" else ""


def weather_agent_node(state):
    query = state["query"]
    local_time = state.get("local_time") or get_time()
    local_location = state.get("local_location") or get_location()
    city = _extract_city(query, local_location.get("label", "unknown"))
    if not city:
        state["draft_answer"] = "Weather query is missing a resolvable location."
        state["evidence"] = []
        state["logs"].append("weather_agent.fetch=missing_location")
        return state
    weather = get_weather_open_meteo(city)

    if not weather.get("ok"):
        state["draft_answer"] = f"Could not retrieve weather for {city}: {weather.get('error', 'unknown error')}"
        state["evidence"] = []
        state["logs"].append("weather_agent.fetch=failed")
        return state

    data = weather.get("data", {})
    current = data.get("current", {})
    daily = data.get("daily", {})
    summary = (
        f"{weather.get('city', city)} weather for {local_time.get('date','today')}: "
        f"current {current.get('temperature_2m', 'N/A')} C, humidity {current.get('relative_humidity_2m', 'N/A')}%, "
        f"wind {current.get('wind_speed_10m', 'N/A')} km/h, precipitation {current.get('precipitation', 'N/A')} mm. "
        f"Today high/low: {daily.get('temperature_2m_max', ['N/A'])[0]}/{daily.get('temperature_2m_min', ['N/A'])[0]} C."
    )
    source = weather.get("source", "")
    state["evidence"] = [{"title": "Open-Meteo", "url": source, "snippet": summary, "source": "open-meteo"}]
    state["draft_answer"] = summary + (f"\nSources: {source}" if source else "")
    state["logs"].append("weather_agent.fetch=ok")
    append_agent_event(
        state,
        agent_name="weather_agent",
        input_payload={"query": query, "city": city, "date": local_time.get("date", "")},
        output_payload={"source": source, "draft_preview": state["draft_answer"][:300]},
    )
    return state

