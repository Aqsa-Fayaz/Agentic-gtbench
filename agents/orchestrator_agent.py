"""
OrchestratorAgent — session manager & coordinator.

The orchestrator does NOT play games itself. Its responsibilities are:
    * select_game    — pick the next game from a configured pool
    * assign_players — pair two PlayerAgent instances for a matchup
    * manage_turn    — request a decision from the currently active player
    * handle_invalid_sequence — apply the retry / forfeit policy
    * log_decision   — keep a structured audit trail of every turn
    * broadcast_state — distribute game state to each player
    * summarize_game — post-game qualitative LLM analysis

The LangGraph workflow (`orchestration.game_graph`) is the runtime executor.
The orchestrator provides the helper logic that the graph nodes can call,
and also powers a simple non-LangGraph `run_session()` harness used by tests.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from agents.player_agent import PlayerAgent
from config.settings import settings
from games import load_game

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Coordinates game sessions and (optionally) runs an LLM-powered post-hoc
    analysis of each completed session.
    """

    SYSTEM_PROMPT = (
        "You are a game-theoretic research orchestrator. "
        "Given a fully played game log between two LLM agents, describe in 3-5 "
        "concise sentences: (a) each player's apparent strategy, (b) key "
        "decision points, (c) whether the outcome matches game-theoretic "
        "equilibrium predictions."
    )

    def __init__(
        self,
        agent_id: str = "orchestrator",
        model: str = None,
        temperature: float = 0.2,
        max_retries: int = None,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(agent_id=agent_id, model=model, temperature=temperature)
        self.max_retries = max_retries if max_retries is not None else settings.max_retries
        self.rng = random.Random(rng_seed)
        self.decision_log: list[dict] = []
        self._llm: Optional[ChatOpenAI] = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = settings.build_chat_openai_client(
                model=self.model,
                temperature=self.temperature,
            )
        return self._llm

    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        """Orchestrators do not play games, but BaseAgent demands a decide()."""
        raise NotImplementedError("OrchestratorAgent does not play games.")

    def new_session_id(self) -> str:
        return f"sess_{uuid.uuid4().hex[:10]}"

    def select_game(self, game_pool: list[str]) -> str:
        """Return the next game to play from `game_pool`. Random by default."""
        if not game_pool:
            raise ValueError("game_pool is empty.")
        return self.rng.choice(game_pool)

    def assign_players(
        self,
        player_a: PlayerAgent,
        player_b: PlayerAgent,
        game_name: str,
    ) -> dict:
        """Return an assignment descriptor that can be logged."""
        record = {
            "session_id": self.new_session_id(),
            "game": game_name,
            "player_a": player_a.agent_id,
            "player_b": player_b.agent_id,
            "player_a_strategy": player_a.strategy_name,
            "player_b_strategy": player_b.strategy_name,
            "timestamp": _utcnow_iso(),
        }
        logger.info(f"Assignment: {record}")
        return record

    def manage_turn(self, game, player: PlayerAgent) -> Optional[dict]:
        """
        Request a single decision from `player` for the current turn of `game`.
        Returns the proposed move dict or None if the player forfeits.
        """
        key = game.get_current_player()
        legal = game.get_legal_moves(key)
        state = game.get_state_for_player(key)
        try:
            return player.decide(state, game.name, legal)
        except RuntimeError as exc:
            logger.error(f"[{player.agent_id}] gave up — {exc}")
            return None

    def handle_invalid_sequence(
        self, player: PlayerAgent, retries_used: int
    ) -> str:
        """Return either 'retry' or 'forfeit' based on retry budget."""
        return "forfeit" if retries_used >= self.max_retries else "retry"

    def log_decision(
        self,
        session_id: str,
        turn: int,
        player: str,
        state: dict,
        move: Optional[dict],
        valid: bool,
        reason: Optional[str] = None,
    ) -> None:
        entry = {
            "session_id": session_id,
            "turn": turn,
            "player": player,
            "state_summary": {k: v for k, v in state.items() if k != "board"},
            "move": move,
            "valid": valid,
            "reason": reason,
            "timestamp": _utcnow_iso(),
        }
        self.decision_log.append(entry)
        logger.debug(f"Decision logged: {entry}")

    def broadcast_state(self, game, player_a: PlayerAgent, player_b: PlayerAgent) -> dict:
        """Produce per-player state views that can be handed to each agent."""
        return {
            "player_a": game.get_state_for_player("player_a"),
            "player_b": game.get_state_for_player("player_b"),
        }

    def summarize_game(self, game_log: dict) -> str:
        """
        Ask the LLM for a short qualitative post-mortem of a finished session.
        Skips the LLM call if no API key is configured — returns a stub string
        so tests / offline runs don't fail.
        """
        if not settings.has_llm_credentials():
            return "[summary unavailable — set OPENAI_API_KEY or OPENROUTER_API_KEY]"
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(game_log, indent=2, default=str)),
            ]
            response = self.llm.invoke(messages)
            return (response.content or "").strip()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"summarize_game failed: {exc}")
            return f"[summary error: {exc}]"

    def run_session(
        self,
        game_name: str,
        player_a: PlayerAgent,
        player_b: PlayerAgent,
        *,
        game_kwargs: Optional[dict] = None,
        max_turns: int = 200,
    ) -> dict:
        """
        Minimal non-LangGraph session runner (useful for tests / debugging).
        """
        game = load_game(game_name, **(game_kwargs or {}))
        game.reset()
        assignment = self.assign_players(player_a, player_b, game_name)
        players = {"player_a": player_a, "player_b": player_b}
        forfeited_by: Optional[str] = None

        turn = 0
        while not game.is_terminal() and turn < max_turns:
            turn += 1
            current_key = game.get_current_player()
            current_player = players[current_key]
            move = self.manage_turn(game, current_player)
            if move is None:
                forfeited_by = current_key
                break
            valid, reason = game.is_valid_move(current_key, move)
            self.log_decision(
                assignment["session_id"], turn, current_key,
                game.get_state_for_player(current_key), move, valid, reason,
            )
            if not valid:
                forfeited_by = current_key
                break
            game.make_move(current_key, move)

        winner = game.get_winner()
        if forfeited_by:
            winner = "player_b" if forfeited_by == "player_a" else "player_a"

        result = {
            **assignment,
            "winner": winner,
            "turns": turn,
            "forfeited_by": forfeited_by,
            "final_state": game.get_state(),
        }
        return result
