from typing import Any, Dict, List, Optional


def append_agent_event(
    state: Dict[str, Any],
    agent_name: str,
    input_payload: Dict[str, Any],
    output_payload: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    events = state.get("agent_events")
    if not isinstance(events, list):
        events = []
        state["agent_events"] = events
    events.append(
        {
            "agent_name": agent_name,
            "input": input_payload,
            "output": output_payload,
            "tool_calls": tool_calls or [],
        }
    )
