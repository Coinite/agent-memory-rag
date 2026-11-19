from src.agent.graph import build_graph
from src.config import settings
from src.memory.memory_manager import MemoryManager
from src.memory.session_store import SessionStore


def run(app, memory: MemoryManager, session_id: str, query: str):
    turn_id = memory.store.create_turn(session_id=session_id, user_query=query)
    tenant_id = settings.default_tenant_id
    brand_id = settings.default_brand_id
    memory_context = memory.build_context_pack(
        session_id=session_id,
        recent_limit=5,
        query=query,
        tenant_id=tenant_id,
        brand_id=brand_id,
        turn_id=turn_id,
    )

    init_state = {
        "session_id": session_id,
        "turn_id": turn_id,
        "tenant_id": tenant_id,
        "brand_id": brand_id,
        "query": query,
        "route": "general",
        "requires_code": False,
        "wants_requirements": False,
        "target_file": "",
        "target_dir": "",
        "code_spec": "",
        "run_after_write": False,
        "run_entrypoint": "",
        "expected_artifacts": [],
        "generated_code": "",
        "generated_extra_files": {},
        "codegen_error": "",
        "file_write_succeeded": False,
        "file_stage": "",
        "staged_data_files": {},
        "direct_action_done": False,
        "context_payload": {},
        "execution_result": {},
        "needs_web_verify": False,
        "did_web_verify": False,
        "verify_round": 0,
        "local_time": {},
        "local_location": {},
        "source_registry": {},
        "constraints": {},
        "memory_context": memory_context,
        "agent_events": [],
        "subtasks": [],
        "search_queries": [],
        "evidence": [],
        "draft_answer": "",
        "scores": {},
        "final_answer": "",
        "logs": [],
    }

    try:
        out = app.invoke(init_state)
    except Exception as e:
        out = {
            "route": "general",
            "final_answer": (
                "Agent execution failed due to model/backend connectivity.\n"
                "Please check LM Studio server status, model loading, and LMSTUDIO_BASE_URL.\n"
                f"Error: {e.__class__.__name__}: {str(e)[:300]}"
            ),
            "scores": {},
            "logs": [f"runtime.error={e.__class__.__name__}", f"runtime.error_detail={str(e)[:500]}"],
            "subtasks": [],
            "agent_events": [],
        }

    memory.update_after_turn(
        session_id=session_id,
        turn_id=turn_id,
        user_query=query,
        final_state=out,
        tenant_id=tenant_id,
        brand_id=brand_id,
    )
    return out["final_answer"], out["route"], out["scores"], out["logs"], out["subtasks"], turn_id


def _print_help() -> None:
    print("Commands:")
    print("- :help                       Show this help")
    print("- :session                    Show current session")
    print("- :session <id>               Switch to session id")
    print("- :new [id]                   Create/switch to a new session")
    print("- :sessions [n]               List recent sessions")
    print("- :summary                    Show current session summary")
    print("- :history [n]                Show recent turns in current session")
    print("- :back [n]                   Alias of :history (回溯)")
    print("- :trace <turn_id>            Show agent trace of a turn")
    print("- :where                      Show current session and db path")
    print("- 再见 / 退出 / exit / quit     End session")


def _handle_command(cmd: str, memory: MemoryManager, current_session_id: str):
    parts = cmd.split()
    name = parts[0].lower()
    args = parts[1:]

    if name == ":help":
        _print_help()
        return current_session_id, True

    if name == ":session":
        if not args:
            print(f"Current session: {current_session_id}")
            return current_session_id, True
        new_sid = args[0].strip()
        memory.set_current_session_id(new_sid)
        print(f"Switched session: {new_sid}")
        return new_sid, True

    if name == ":new":
        new_sid = memory.new_session(args[0].strip() if args else "")
        print(f"New session: {new_sid}")
        return new_sid, True

    if name == ":sessions":
        limit = 10
        if args and args[0].isdigit():
            limit = max(1, min(100, int(args[0])))
        rows = memory.list_sessions(limit=limit)
        if not rows:
            print("No sessions found.")
            return current_session_id, True
        print("Recent sessions:")
        for r in rows:
            sid = r.get("session_id", "")
            goal = str(r.get("active_goal", ""))[:80]
            updated = r.get("updated_at", "")
            marker = "*" if sid == current_session_id else " "
            print(f"{marker} {sid} | updated={updated} | goal={goal}")
        return current_session_id, True

    if name in {":summary"}:
        s = memory.get_summary(current_session_id)
        print(f"Session: {current_session_id}")
        print(f"Active goal: {s.get('active_goal', '')}")
        print("Summary:")
        print(s.get("summary_text", "") or "(empty)")
        return current_session_id, True

    if name in {":history", ":back"}:
        limit = 5
        if args and args[0].isdigit():
            limit = max(1, min(50, int(args[0])))
        rows = memory.get_history(current_session_id, limit=limit)
        if not rows:
            print("No turns yet.")
            return current_session_id, True
        print(f"Recent {len(rows)} turns (newest first):")
        for t in rows:
            print(f"- turn={t.get('turn_id')} route={t.get('route')} query={str(t.get('user_query',''))[:120]}")
        return current_session_id, True

    if name == ":trace":
        if not args or not args[0].isdigit():
            print("Usage: :trace <turn_id>")
            return current_session_id, True
        turn_id = int(args[0])
        events = memory.get_trace(current_session_id, turn_id)
        if not events:
            print(f"No events found for turn {turn_id}.")
            return current_session_id, True
        print(f"Trace for turn {turn_id}:")
        for e in events:
            agent = e.get("agent_name", "")
            inp = str(e.get("input", {}))[:160]
            out = str(e.get("output", {}))[:160]
            print(f"- {agent} | input={inp} | output={out}")
        return current_session_id, True

    if name == ":where":
        print(f"Current session: {current_session_id}")
        print(f"Memory DB: {memory.store.db_path}")
        return current_session_id, True

    print("Unknown command. Use :help")
    return current_session_id, True


if __name__ == "__main__":
    app = build_graph()
    memory = MemoryManager(SessionStore(settings.memory_db_path))
    session_id = memory.load_or_create_session_id()

    print(f"Session: {session_id}")
    print("Type :help for commands.")
    exit_words = {"再见", "退出", "quit", "exit", "bye", "goodbye"}

    while True:
        q = input("Query: ").strip()
        if not q:
            continue

        if q.startswith(":"):
            session_id, _ = _handle_command(q, memory, session_id)
            continue

        if q.lower() in exit_words or q in exit_words:
            print("Session ended.")
            break

        ans, route, scores, logs, subtasks, turn_id = run(app, memory, session_id, q)
        print(f"\nSession: {session_id}")
        print(f"\nTurn: {turn_id}")
        print(f"Route: {route}")
        if subtasks:
            print("\nSubtasks:")
            for item in subtasks:
                print(f"- {item}")
        print("\nAnswer:\n", ans)
        if scores:
            print("\nScores:\n", scores)
        if logs:
            print("\nLogs:")
            for line in logs:
                print(f"- {line}")
