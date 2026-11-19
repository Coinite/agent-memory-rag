from langgraph.graph import END, StateGraph

from src.agent.state import AgentState
from src.agents.code_agent import code_agent_node
from src.agents.file_agent import file_agent_node
from src.agents.judge_agent import judge_agent_node
from src.agents.news_agent import news_agent_node
from src.agents.runner_agent import runner_agent_node
from src.agents.stock_agent import stock_agent_node
from src.agents.supervisor import supervisor_node
from src.agents.weather_agent import weather_agent_node
from src.agents.web_agent import web_agent_node


def _dispatch_from_supervisor(state: AgentState) -> str:
    route = state.get("route", "general")
    query = str(state.get("query", "") or "").lower()
    constraints = state.get("constraints", {}) or {}
    has_retrieval_need = bool(constraints.get("must_cite", False) or constraints.get("need_data", False))
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
    query_retrieval_intent = any(t in query for t in retrieval_terms)
    try:
        if int(constraints.get("min_sources", 0) or 0) > 0:
            has_retrieval_need = True
    except Exception:
        pass

    if route == "general" and state.get("requires_code", False) and (
        state.get("target_file", "") or state.get("target_dir", "")
    ):
        return "file_with_code"
    if route == "general" and (state.get("target_file", "") or state.get("target_dir", "")):
        return "file"
    if route == "file":
        download_pdf_intent = (
            any(t in query for t in ["download", "save", "store", "fetch", "retrieve", "下载", "保存"])
            and any(t in query for t in ["pdf", "paper", "papers", "论文", "arxiv"])
        )
        if download_pdf_intent:
            return "web"
        if state.get("requires_code", False) and (has_retrieval_need or query_retrieval_intent):
            return "file_with_research"
        return "file_with_code" if state.get("requires_code", False) else "file"
    if route == "weather":
        return "weather"
    if route == "news":
        return "news"
    if route == "stock":
        return "stock"
    if route in {"research", "general"}:
        return "web"
    return "web"


def _after_context_agent(state: AgentState) -> str:
    if bool(state.get("direct_action_done", False)):
        return "done"
    if (
        str(state.get("route", "")).lower() == "file"
        and str(state.get("file_stage", "")).strip().lower() == "data"
        and bool(state.get("staged_data_files", {}))
    ):
        return "data_file"
    if state.get("requires_code", False) and (state.get("target_file", "") or state.get("target_dir", "")):
        return "code"
    return "judge"


def _after_judge(state: AgentState) -> str:
    verify_round = int(state.get("verify_round", 0))
    if (
        state.get("needs_web_verify", False)
        and not state.get("did_web_verify", False)
        and verify_round < 1
    ):
        return "verify"
    return "done"


def _after_file(state: AgentState) -> str:
    if str(state.get("route", "")).lower() == "file" and str(state.get("file_stage", "")).strip().lower() == "data":
        return "code" if bool(state.get("file_write_succeeded", False)) else "done"
    if bool(state.get("run_after_write", False)) and bool(state.get("file_write_succeeded", False)):
        return "run"
    return "done"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("code_agent", code_agent_node)
    g.add_node("file_agent", file_agent_node)
    g.add_node("web_agent", web_agent_node)
    g.add_node("weather_agent", weather_agent_node)
    g.add_node("news_agent", news_agent_node)
    g.add_node("stock_agent", stock_agent_node)
    g.add_node("judge_agent", judge_agent_node)
    g.add_node("runner_agent", runner_agent_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        _dispatch_from_supervisor,
        {
            "file_with_code": "code_agent",
            "file_with_research": "web_agent",
            "file": "file_agent",
            "weather": "weather_agent",
            "news": "news_agent",
            "stock": "stock_agent",
            "web": "web_agent",
        },
    )

    g.add_edge("code_agent", "file_agent")
    g.add_conditional_edges(
        "file_agent",
        _after_file,
        {
            "code": "code_agent",
            "run": "runner_agent",
            "done": END,
        },
    )
    g.add_edge("runner_agent", END)

    for node_name in ["weather_agent", "news_agent", "stock_agent", "web_agent"]:
        routes = {
            "done": END,
            "code": "code_agent",
            "judge": "judge_agent",
        }
        if node_name == "web_agent":
            routes["data_file"] = "file_agent"
        g.add_conditional_edges(node_name, _after_context_agent, routes)

    g.add_conditional_edges(
        "judge_agent",
        _after_judge,
        {
            "verify": "web_agent",
            "done": END,
        },
    )

    return g.compile()
