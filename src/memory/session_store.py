import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class SessionStore:
    def __init__(self, db_path: str = ".agent_memory/session.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL DEFAULT '',
                    active_goal TEXT NOT NULL DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    route TEXT NOT NULL DEFAULT '',
                    final_answer TEXT NOT NULL DEFAULT '',
                    scores_json TEXT NOT NULL DEFAULT '{}',
                    logs_json TEXT NOT NULL DEFAULT '[]',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    agent_name TEXT NOT NULL,
                    input_json TEXT NOT NULL DEFAULT '{}',
                    output_json TEXT NOT NULL DEFAULT '{}',
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def ensure_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id) VALUES(?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at=CURRENT_TIMESTAMP
                """,
                (session_id,),
            )

    def get_session(self, session_id: str) -> Dict[str, Any]:
        self.ensure_session(session_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, summary_text, active_goal FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if not row:
            return {"session_id": session_id, "summary_text": "", "active_goal": ""}
        return dict(row)

    def update_session_summary(self, session_id: str, summary_text: str, active_goal: str = "") -> None:
        self.ensure_session(session_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET summary_text=?, active_goal=?, updated_at=CURRENT_TIMESTAMP
                WHERE session_id=?
                """,
                (summary_text, active_goal, session_id),
            )

    def create_turn(self, session_id: str, user_query: str) -> int:
        self.ensure_session(session_id)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO turns(session_id, user_query) VALUES(?, ?)",
                (session_id, user_query),
            )
            return int(cur.lastrowid)

    def finalize_turn(
        self,
        session_id: str,
        turn_id: int,
        route: str,
        final_answer: str,
        scores: Dict[str, Any],
        logs: List[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE turns
                SET route=?, final_answer=?, scores_json=?, logs_json=?
                WHERE session_id=? AND turn_id=?
                """,
                (
                    route,
                    final_answer,
                    json.dumps(scores, ensure_ascii=False),
                    json.dumps(logs, ensure_ascii=False),
                    session_id,
                    turn_id,
                ),
            )

    def add_agent_event(
        self,
        session_id: str,
        turn_id: int,
        agent_name: str,
        input_payload: Dict[str, Any],
        output_payload: Dict[str, Any],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_events(session_id, turn_id, agent_name, input_json, output_json, tool_calls_json)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    agent_name,
                    json.dumps(input_payload, ensure_ascii=False),
                    json.dumps(output_payload, ensure_ascii=False),
                    json.dumps(tool_calls or [], ensure_ascii=False),
                ),
            )

    def get_recent_turns(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turn_id, user_query, route, final_answer, scores_json, logs_json, created_at
                FROM turns
                WHERE session_id=?
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in reversed(rows):
            d = dict(r)
            d["scores"] = json.loads(d.pop("scores_json") or "{}")
            d["logs"] = json.loads(d.pop("logs_json") or "[]")
            out.append(d)
        return out

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, active_goal, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_turns(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turn_id, user_query, route, final_answer, scores_json, logs_json, created_at
                FROM turns
                WHERE session_id=?
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["scores"] = json.loads(d.pop("scores_json") or "{}")
            d["logs"] = json.loads(d.pop("logs_json") or "[]")
            out.append(d)
        return out

    def get_agent_events(self, session_id: str, turn_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT agent_name, input_json, output_json, tool_calls_json, created_at
                FROM agent_events
                WHERE session_id=? AND turn_id=?
                ORDER BY id ASC
                """,
                (session_id, turn_id),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["input"] = json.loads(d.pop("input_json") or "{}")
            d["output"] = json.loads(d.pop("output_json") or "{}")
            d["tool_calls"] = json.loads(d.pop("tool_calls_json") or "[]")
            out.append(d)
        return out
