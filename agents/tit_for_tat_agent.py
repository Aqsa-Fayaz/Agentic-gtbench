"""
TitForTatAgent — mirrors the opponent's last action.

Paper-aligned baseline for Iterated Prisoner's Dilemma (paper §3.2 / Appendix
A3.1). Cooperates on the first round, then plays whatever the opponent played
last round.

Outside Prisoner's Dilemma, the strategy degenerates to "play the first legal
move" — kept as a safe fallback so the agent never crashes if dropped into an
unsupported game.

Makes ZERO LLM calls.
"""

from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseAgent
from games.prisoners_dilemma import PrisonersDilemma


class TitForTatAgent(BaseAgent):
    """Stateless TfT player. Conforms to the PlayerAgent interface."""

    SUPPORTED_STRATEGIES = ("tit_for_tat",)

    def __init__(
        self,
        agent_id: str,
        # Accept-but-ignore fields so this can be used wherever a PlayerAgent is.
        strategy: str = "tit_for_tat",
        model: str = "tit_for_tat",
        temperature: float = 0.0,
        provider: str = "none",
        tools: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(agent_id=agent_id, model=model, temperature=temperature)
        self.strategy_name = "tit_for_tat"
        self.provider = provider
        self.tools = tools or []
        self.invalid_move_count = 0
        self.total_moves = 0
        self.move_history: list[dict] = []

    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        if not legal_moves:
            raise RuntimeError(f"{self.agent_id}: no legal moves available.")
        self.total_moves += 1

        if game_name == "prisoners_dilemma":
            return self._tft_action(game_state)
        # Fallback for unsupported games — first legal move keeps the harness
        # alive without claiming TfT semantics.
        return legal_moves[0]

    def _tft_action(self, game_state: dict) -> dict:
        history = game_state.get("round_history") or []
        if not history:
            return {"action": PrisonersDilemma.COOPERATE}
        # Determine which side I'm playing from the state's current_player
        # (the LangGraph node passes that side's view here).
        me = game_state.get("current_player", "player_a")
        opp = "player_b" if me == "player_a" else "player_a"
        last_round = history[-1]
        opp_action = last_round.get(f"{opp}_action", PrisonersDilemma.COOPERATE)
        return {"action": opp_action}

    def reset_session(self) -> None:
        pass

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "strategy": "tit_for_tat",
            "model": self.model,
            "total_moves": self.total_moves,
            "invalid_moves": 0,
            "invalid_rate": 0.0,
        }
