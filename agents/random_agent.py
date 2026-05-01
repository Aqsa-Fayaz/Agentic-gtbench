"""
RandomAgent — picks uniformly from legal_moves.
Used as the paper's baseline opponent (paper §3.2 / Appendix A3.1).
Makes ZERO LLM calls — useful for sanity-checking the harness.
"""

from __future__ import annotations

import random
from typing import Optional

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Stateless random-policy player. Conforms to the PlayerAgent interface."""

    SUPPORTED_STRATEGIES = ("random",)

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        # Accept-but-ignore fields so this can be used wherever a PlayerAgent is.
        strategy: str = "random",
        model: str = "random",
        temperature: float = 0.0,
        provider: str = "none",
        tools: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(agent_id=agent_id, model=model, temperature=temperature)
        self.strategy_name = "random"
        self.provider = provider
        self.tools = tools or []
        self._rng = random.Random(seed)
        self.invalid_move_count = 0
        self.total_moves = 0
        self.move_history: list[dict] = []

    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        if not legal_moves:
            raise RuntimeError(f"{self.agent_id}: no legal moves available.")
        self.total_moves += 1
        return self._rng.choice(legal_moves)

    def reset_session(self) -> None:
        # Nothing to reset for a stateless random agent.
        pass

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "strategy": "random",
            "model": self.model,
            "total_moves": self.total_moves,
            "invalid_moves": self.invalid_move_count,
            "invalid_rate": 0.0,
        }
