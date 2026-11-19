from typing import Dict, List, Literal, TypedDict


RouteType = Literal["file", "research", "general", "weather", "news", "stock"]


class AgentState(TypedDict):
    session_id: str
    turn_id: int
    tenant_id: str
    brand_id: str
    query: str
    route: RouteType
    requires_code: bool
    wants_requirements: bool
    target_file: str
    target_dir: str
    code_spec: str
    run_after_write: bool
    run_entrypoint: str
    expected_artifacts: List[str]
    generated_code: str
    generated_extra_files: Dict[str, str]
    codegen_error: str
    file_write_succeeded: bool
    file_stage: str
    staged_data_files: Dict[str, str]
    direct_action_done: bool
    context_payload: Dict[str, object]
    execution_result: Dict[str, object]
    needs_web_verify: bool
    did_web_verify: bool
    verify_round: int
    local_time: Dict[str, str]
    local_location: Dict[str, str]
    source_registry: Dict[str, List[str]]
    constraints: Dict[str, object]
    memory_context: Dict[str, object]
    agent_events: List[Dict[str, object]]
    subtasks: List[str]
    search_queries: List[str]
    evidence: List[Dict[str, str]]
    draft_answer: str
    scores: Dict[str, float]
    final_answer: str
    logs: List[str]
