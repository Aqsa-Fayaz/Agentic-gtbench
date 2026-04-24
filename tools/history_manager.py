"""
History Manager Tool.

Persistent JSON store for completed game sessions. Supports saving, loading,
listing, and per-agent statistics. Usable as a LangChain tool or plain class.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

from langchain.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from config.settings import settings


class HistoryManagerInput(BaseModel):
    operation: str = Field(
        description="One of: 'save', 'load', 'list', 'stats'"
    )
    session_id: Optional[str] = Field(default=None, description="Session id for save/load")
    agent_id: Optional[str] = Field(default=None, description="Agent id for stats")
    game_log: Optional[dict] = Field(default=None, description="Full session payload for save")


class HistoryManagerTool(BaseTool):
    """
    Lightweight persistence layer for the orchestrator / evaluator.

    Operations:
        save  — write a session log to <store_dir>/<session_id>.json
        load  — read back a previously saved session
        list  — list all session ids currently on disk
        stats — aggregate win/loss/draw counts for a given agent_id
    """

    name: str = "game_history"
    description: str = (
        "Persist and query completed game sessions. "
        "Operations: save | load | list | stats. "
        "Save requires session_id + game_log; load requires session_id; "
        "stats requires agent_id; list takes no extra args."
    )
    args_schema: type = HistoryManagerInput
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, store_dir: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        default_dir = settings.results_dir / "sessions"
        chosen = Path(store_dir) if store_dir else default_dir
        chosen.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "_store_dir", chosen)

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    def _run(
        self,
        operation: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        game_log: Optional[dict] = None,
    ) -> str:
        op = operation.lower()
        try:
            if op == "save":
                return json.dumps(self.save_game(session_id or "", game_log or {}))
            if op == "load":
                return json.dumps(self.load_game(session_id or ""), default=str)
            if op == "list":
                return json.dumps({"sessions": self.list_sessions()})
            if op == "stats":
                return json.dumps(self.get_stats(agent_id or ""))
            return json.dumps({"error": f"unknown operation '{operation}'"})
        except Exception as exc:  # pragma: no cover
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use synchronous _run")

    def save_game(self, session_id: str, game_log: dict) -> dict:
        if not session_id:
            session_id = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S_%f")
        payload = {
            "session_id": session_id,
            "saved_at": _utcnow_iso(),
            **game_log,
        }
        path = self._store_dir / f"{session_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return {"saved": True, "session_id": session_id, "path": str(path)}

    def load_game(self, session_id: str) -> dict:
        path = self._store_dir / f"{session_id}.json"
        if not path.exists():
            return {"error": f"session '{session_id}' not found"}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self._store_dir.glob("*.json"))

    def get_stats(self, agent_id: str) -> dict:
        stats = defaultdict(int)
        sessions = 0
        for path in self._store_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except json.JSONDecodeError:
                continue
            a_id = data.get("player_a_id")
            b_id = data.get("player_b_id")
            if agent_id not in (a_id, b_id):
                continue
            sessions += 1
            winner = data.get("winner")
            if winner == agent_id:
                stats["wins"] += 1
            elif winner == "draw":
                stats["draws"] += 1
            elif winner in (a_id, b_id):
                stats["losses"] += 1
            else:
                stats["other"] += 1
        total = max(sessions, 1)
        return {
            "agent_id": agent_id,
            "sessions": sessions,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "win_rate": stats["wins"] / total,
        }
